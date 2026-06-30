"""A compact, length-bounded KV cache.

One cache per attention layer. Keys/values are appended as tokens are produced and the
oldest entries are dropped once `max_context` is exceeded, so memory for the cache stays
bounded no matter how long generation runs. This is deliberately simple float32 storage;
KV-cache *compression* is a later-phase lever noted in the plan.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class KVCache:
    """Per-layer rolling key/value store of shape [len, n_kv_heads*head_dim]."""

    def __init__(self, n_layers: int, max_context: int) -> None:
        self.max_context = max_context
        self._k: List[Optional[np.ndarray]] = [None] * n_layers
        self._v: List[Optional[np.ndarray]] = [None] * n_layers
        self._len = 0

    @property
    def length(self) -> int:
        """Number of cached positions (KV length seen by the next step)."""
        return self._len

    def reset(self) -> None:
        for i in range(len(self._k)):
            self._k[i] = None
            self._v[i] = None
        self._len = 0

    def extend(
        self, layer: int, k_new: np.ndarray, v_new: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append this step's k/v for `layer` and return the full k/v seen so far.

        The first layer to be extended on a given step advances the shared length and
        applies the rolling window; subsequent layers in the same step inherit it.
        """
        if self._k[layer] is None:
            full_k, full_v = k_new, v_new
        else:
            full_k = np.concatenate([self._k[layer], k_new], axis=0)
            full_v = np.concatenate([self._v[layer], v_new], axis=0)

        if full_k.shape[0] > self.max_context:
            drop = full_k.shape[0] - self.max_context
            full_k = full_k[drop:]
            full_v = full_v[drop:]

        self._k[layer] = full_k
        self._v[layer] = full_v
        # track length off layer 0 so it advances once per step
        if layer == 0:
            self._len = full_k.shape[0]
        return full_k, full_v

    def offset(self, layer: int) -> int:
        """KV length *before* this step for `layer` (the causal offset)."""
        prev = self._k[layer]
        return 0 if prev is None else prev.shape[0]
