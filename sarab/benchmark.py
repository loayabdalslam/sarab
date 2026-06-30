"""Throughput + peak-RAM benchmark.

Proves the two claims that matter: (1) generation runs at a usable token rate, and
(2) peak resident memory stays under the configured budget even though the model on disk
is far larger. Peak RSS is sampled in a background thread via psutil.
"""

from __future__ import annotations

import threading
import time

from .config import RuntimeConfig


class _PeakRSS:
    """Samples this process's resident set size on a thread, records the peak."""

    def __init__(self, interval: float = 0.02) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop:
            rss = self._proc.memory_info().rss
            if rss > self._peak:
                self._peak = rss
            time.sleep(self._interval)

    def __enter__(self) -> "_PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop = True
        self._thread.join(timeout=1.0)

    @property
    def peak_gb(self) -> float:
        return self._peak / 1024**3


def run_benchmark(model_id: str, cfg: RuntimeConfig, prompt: str, max_new_tokens: int):
    from .hub import load

    print(f"[bench] loading {model_id} (quant={cfg.quant})")
    rt = load(model_id, config=cfg)
    try:
        on_disk = rt.store.total_bytes() / 1024**3
        n_layers = rt.model_config.n_layers
        resident = rt.runtime.resident_layers
        mode = "ALL resident (fits RAM)" if resident >= n_layers else f"streaming {resident}/{n_layers}"
        # Estimate resident float32 footprint up front so a big model doesn't silently
        # eat all RAM during the build. A resident layer lives as float32 (~2x its fp16
        # on-disk size), regardless of quant.
        est_resident_gb = (on_disk / max(1, n_layers)) * resident * 2.0
        print(f"[bench] {n_layers} layers | {mode} | est. resident ~{est_resident_gb:.1f} GB "
              f"| building (one-time)...")
        if est_resident_gb > budget:
            print(f"[bench] WARNING: estimated resident RAM (~{est_resident_gb:.1f} GB) exceeds "
                  f"budget ({budget:.1f} GB). Lower --ram-budget to force streaming, "
                  f"or this run may swap/OOM.")
        # Pre-build/quantize all resident layers with a visible progress bar, so the
        # one-time cost isn't mistaken for a hang.
        rt.runtime.prewarm(progress=True)
        # Short warm-up generation to prime BLAS threads; measure the run after it.
        _ = list(rt.generate(prompt, max_new_tokens=4, stream=True))
        with _PeakRSS() as peak:
            t0 = time.perf_counter()
            out = list(
                rt.generate(prompt, max_new_tokens=max_new_tokens, stream=True)
            )
            dt = time.perf_counter() - t0
        n = len(out)
        budget = cfg.ram_budget_gb
        within = "OK" if peak.peak_gb <= budget else "OVER"
        print("\n=== SARAB benchmark ===")
        print(f"model on disk      : {on_disk:6.2f} GB")
        print(f"resident layers    : {resident}/{n_layers}  ({mode})")
        print(f"peak resident RSS  : {peak.peak_gb:6.2f} GB  (budget {budget:.1f} GB) [{within}]")
        print(f"tokens generated   : {n}")
        print(f"throughput (decode): {n/max(dt,1e-9):6.2f} tok/s")
        print(f"streamer stats     : {rt.runtime.streamer.stats}")
    finally:
        rt.close()
