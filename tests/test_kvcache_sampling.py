"""Tests for the KV cache and the sampling logic."""

import numpy as np

from sarab.engine.kvcache import KVCache
from sarab.generate import SamplingParams, _sample_logits


def test_kvcache_grows_and_returns_full():
    kv = KVCache(n_layers=1, max_context=10)
    assert kv.length == 0
    k1 = np.ones((2, 4), dtype=np.float32)
    v1 = np.ones((2, 4), dtype=np.float32)
    fk, fv = kv.extend(0, k1, v1)
    assert fk.shape == (2, 4) and kv.length == 2
    k2 = np.ones((1, 4), dtype=np.float32) * 2
    fk2, _ = kv.extend(0, k2, k2)
    assert fk2.shape == (3, 4) and kv.length == 3
    assert kv.offset(0) == 3  # offset reflects what's stored now


def test_kvcache_rolls_at_max_context():
    kv = KVCache(n_layers=1, max_context=3)
    for step in range(5):
        k = np.full((1, 2), float(step), dtype=np.float32)
        fk, _ = kv.extend(0, k, k)
    # never exceeds max_context
    assert fk.shape[0] == 3
    # oldest dropped: should hold steps 2,3,4
    np.testing.assert_array_equal(fk[:, 0], np.array([2.0, 3.0, 4.0]))


def test_kvcache_reset():
    kv = KVCache(n_layers=2, max_context=10)
    kv.extend(0, np.ones((2, 4), np.float32), np.ones((2, 4), np.float32))
    kv.reset()
    assert kv.length == 0
    assert kv.offset(0) == 0


def test_greedy_is_deterministic_argmax():
    logits = np.array([0.1, 5.0, 0.2, -1.0])
    p = SamplingParams(temperature=0.0)
    rng = np.random.default_rng(0)
    assert _sample_logits(logits, p, rng) == 1


def test_temperature_sampling_in_range():
    logits = np.zeros(10)
    p = SamplingParams(temperature=1.0, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        tok = _sample_logits(logits, p, rng)
        assert 0 <= tok < 10


def test_top_k_restricts_support():
    # one dominant logit + top_k=1 must always pick it
    logits = np.array([0.0, 0.0, 9.0, 0.0])
    p = SamplingParams(temperature=1.0, top_k=1, seed=1)
    rng = np.random.default_rng(1)
    picks = {_sample_logits(logits, p, rng) for _ in range(10)}
    assert picks == {2}


def test_top_p_keeps_top_token():
    logits = np.array([10.0, 0.0, 0.0])
    p = SamplingParams(temperature=1.0, top_p=0.01, seed=2)
    rng = np.random.default_rng(2)
    # nucleus 0.01 collapses to the single most-likely token
    assert _sample_logits(logits, p, rng) == 0
