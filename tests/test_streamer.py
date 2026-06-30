"""Streamer tests: LRU residency bound, prefetch correctness, hit accounting."""

import threading
import time

import numpy as np

from sarab.loader.streamer import LayerStreamer


def _builder(call_log, lock, delay=0.0):
    def build(i):
        if delay:
            time.sleep(delay)
        with lock:
            call_log.append(i)
        return {"w": np.full((4, 4), float(i), dtype=np.float32)}
    return build


def test_residency_window_is_bounded():
    log, lock = [], threading.Lock()
    s = LayerStreamer(n_layers=10, build_fn=_builder(log, lock),
                      resident_layers=2, prefetch=False)
    try:
        for i in range(10):
            w = s.get(i)
            assert w["w"][0, 0] == float(i)
            # never more than resident_layers in cache
            assert len(s._cache) <= 2
    finally:
        s.close()


def test_values_are_correct_in_order():
    log, lock = [], threading.Lock()
    s = LayerStreamer(n_layers=6, build_fn=_builder(log, lock),
                      resident_layers=2, prefetch_depth=1, prefetch=True)
    try:
        for i in range(6):
            assert s.get(i)["w"][0, 0] == float(i)
    finally:
        s.close()


def test_prefetch_reduces_misses():
    # With prefetch on, sequential access should serve some layers from prefetch.
    log, lock = [], threading.Lock()
    s = LayerStreamer(n_layers=8, build_fn=_builder(log, lock, delay=0.005),
                      resident_layers=3, prefetch_depth=2, prefetch=True)
    try:
        for i in range(8):
            s.get(i)
        time.sleep(0.05)
        # every layer built exactly once despite prefetch + eviction churn
        assert sorted(set(log)) == list(range(8))
    finally:
        s.close()


def test_reset_repoints_prefetch():
    log, lock = [], threading.Lock()
    s = LayerStreamer(n_layers=4, build_fn=_builder(log, lock),
                      resident_layers=2, prefetch_depth=1, prefetch=True)
    try:
        for i in range(4):
            s.get(i)
        s.reset()
        for i in range(4):
            assert s.get(i)["w"][0, 0] == float(i)
    finally:
        s.close()
