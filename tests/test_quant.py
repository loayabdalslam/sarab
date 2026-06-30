"""Tests for group-wise int8/int4 quantization accuracy and packing."""

import numpy as np

from sarab.quant.int4 import quantize, dequantize, quantized_matmul


def _max_rel_err(a, b):
    denom = np.maximum(np.abs(a), 1e-6)
    return np.max(np.abs(a - b) / denom)


def test_int8_roundtrip_accurate():
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((48, 128)) * 0.1).astype(np.float32)
    qt = quantize(w, bits=8, group_size=64)
    w2 = dequantize(qt)
    assert w2.shape == w.shape
    # int8 group-wise should reconstruct to within a few percent
    assert np.mean(np.abs(w - w2)) < 0.01


def test_int4_roundtrip_reasonable():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((32, 96)) * 0.1).astype(np.float32)
    qt = quantize(w, bits=4, group_size=32)
    w2 = dequantize(qt)
    assert w2.shape == w.shape
    # int4 is coarser but should still track the weight
    assert np.mean(np.abs(w - w2)) < 0.05


def test_int4_packs_to_half():
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((16, 128)) * 0.1).astype(np.float32)
    q8 = quantize(w, bits=8, group_size=64)
    q4 = quantize(w, bits=4, group_size=64)
    # 4-bit codes are packed two-per-byte -> roughly half the code bytes
    assert q4.q.nbytes <= q8.q.nbytes // 2 + q8.q.shape[0]


def test_quantized_matmul_matches_float():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((40, 64)) * 0.1).astype(np.float32)
    x = (rng.standard_normal((5, 64)) * 0.5).astype(np.float32)
    ref = x @ w.T
    qt = quantize(w, bits=8, group_size=64)
    got = quantized_matmul(x, qt)
    assert got.shape == ref.shape
    assert _max_rel_err(ref, got) < 0.1


def test_non_divisible_group_size():
    # in_features (100) not divisible by group_size (64) must still round-trip
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((8, 100)) * 0.1).astype(np.float32)
    qt = quantize(w, bits=8, group_size=64)
    w2 = dequantize(qt)
    assert w2.shape == (8, 100)
    assert np.mean(np.abs(w - w2)) < 0.01
