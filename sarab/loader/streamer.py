"""Layer streaming with an LRU residency window and a background prefetch thread.

The runtime asks for one transformer block at a time. The streamer:
  * keeps only `resident_layers` blocks decompressed in RAM (LRU eviction),
  * runs a prefetch thread that loads block N+1..N+depth off disk (and dequantizes it)
    while the main thread computes block N — so the disk read hides behind compute.

A "block" here is an opaque dict of named NumPy arrays produced by a user-supplied
`build_fn(layer_idx) -> dict[str, np.ndarray]`. The runtime supplies that function; the
streamer stays agnostic to model architecture.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Callable, Dict, Optional

import numpy as np

LayerWeights = Dict[str, np.ndarray]
BuildFn = Callable[[int], LayerWeights]


class LayerStreamer:
    """Streams transformer blocks on demand with prefetch + LRU residency.

    Thread-safety: `get(i)` is called only from the main compute thread. The prefetch
    worker mutates `_pending`/`_cache` under `_lock`. The build function may run
    concurrently for different indices, so it must not share mutable state.
    """

    def __init__(
        self,
        n_layers: int,
        build_fn: BuildFn,
        resident_layers: int = 2,
        prefetch_depth: int = 1,
        prefetch: bool = True,
    ) -> None:
        if resident_layers < 1:
            raise ValueError("resident_layers must be >= 1")
        self.n_layers = n_layers
        self._build = build_fn
        self._resident = resident_layers
        self._depth = max(0, prefetch_depth)
        self._prefetch_enabled = prefetch and self._depth > 0

        self._lock = threading.Lock()
        self._cache: "OrderedDict[int, LayerWeights]" = OrderedDict()  # LRU: oldest first
        self._pending: Dict[int, Future] = {}
        self._worker: Optional[threading.Thread] = None
        self._wakeup = threading.Event()
        self._stop = False
        self._next_hint = 0  # next layer the compute loop is likely to want

        self.stats = {"hits": 0, "misses": 0, "prefetched": 0, "evictions": 0}

        if self._prefetch_enabled:
            self._worker = threading.Thread(
                target=self._prefetch_loop, name="sarab-prefetch", daemon=True
            )
            self._worker.start()

    # -- public API ----------------------------------------------------------------
    def get(self, i: int) -> LayerWeights:
        """Return weights for block `i`, building (or awaiting prefetch) as needed."""
        if not (0 <= i < self.n_layers):
            raise IndexError(f"layer {i} out of range [0,{self.n_layers})")

        with self._lock:
            if i in self._cache:
                self._cache.move_to_end(i)
                self.stats["hits"] += 1
                self._next_hint = i + 1
                self._wakeup.set()
                return self._cache[i]
            fut = self._pending.pop(i, None)

        if fut is not None:
            # Prefetch already in flight: wait for it instead of building twice.
            weights = fut.result()
            self.stats["prefetched"] += 1
        else:
            self.stats["misses"] += 1
            weights = self._build(i)

        with self._lock:
            self._insert_locked(i, weights)
            self._next_hint = i + 1
            self._wakeup.set()
        return weights

    def reset(self) -> None:
        """Hint that a new sequence is starting; re-point prefetch at layer 0."""
        with self._lock:
            self._next_hint = 0
            self._wakeup.set()

    def close(self) -> None:
        self._stop = True
        self._wakeup.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
        with self._lock:
            self._cache.clear()
            self._pending.clear()

    def __enter__(self) -> "LayerStreamer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- internals -----------------------------------------------------------------
    def _insert_locked(self, i: int, weights: LayerWeights) -> None:
        self._cache[i] = weights
        self._cache.move_to_end(i)
        # Evict LRU until within the residency window.
        while len(self._cache) > self._resident:
            old, _ = self._cache.popitem(last=False)
            self.stats["evictions"] += 1

    def _prefetch_loop(self) -> None:
        while not self._stop:
            self._wakeup.wait()
            self._wakeup.clear()
            if self._stop:
                break
            with self._lock:
                start = self._next_hint
                targets = [
                    j
                    for j in range(start, min(start + self._depth, self.n_layers))
                    if j not in self._cache and j not in self._pending
                ]
                for j in targets:
                    self._pending[j] = Future()
            for j in targets:
                if self._stop:
                    break
                fut = self._pending.get(j)
                if fut is None:
                    continue
                try:
                    weights = self._build(j)
                    # Stage the result ONLY in the future. get() will move it into the
                    # LRU cache when consumed. Inserting here too would let a block live
                    # in both _cache and _pending and blow the residency bound.
                    fut.set_result(weights)
                except Exception as e:  # surface build errors to the waiter
                    fut.set_exception(e)
                    with self._lock:
                        self._pending.pop(j, None)
