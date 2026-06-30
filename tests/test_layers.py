"""Unit tests for the math primitives in engine.layers."""

import numpy as np

from sarab.engine import layers as L


def test_rms_norm_unit_variance():
    x = np.array([[3.0, 4.0, 0.0, 0.0]], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = L.rms_norm(x, w, eps=0.0)
    # rms of [3,4,0,0] = sqrt((9+16)/4)=2.5 ; normalized values /2.5
    np.testing.assert_allclose(out, x / 2.5, rtol=1e-5)


def test_rms_norm_add_unit():
    x = np.ones((1, 4), dtype=np.float32)
    w = np.zeros(4, dtype=np.float32)
    plain = L.rms_norm(x, w, eps=0.0, add_unit=False)
    gemma = L.rms_norm(x, w, eps=0.0, add_unit=True)
    np.testing.assert_allclose(plain, np.zeros_like(plain))
    np.testing.assert_allclose(gemma, L.rms_norm(x, np.ones(4, np.float32), 0.0))


def test_softmax_sums_to_one():
    x = np.random.randn(3, 7).astype(np.float32)
    p = L.softmax(x, axis=-1)
    np.testing.assert_allclose(p.sum(axis=-1), np.ones(3), rtol=1e-6)
    assert np.all(p >= 0)


def test_silu_and_gelu_shapes():
    x = np.linspace(-3, 3, 11).astype(np.float32)
    assert L.silu(x).shape == x.shape
    assert L.gelu_tanh(x).shape == x.shape
    # silu(0)=0, gelu(0)=0
    assert abs(float(L.silu(np.array([0.0]))[0])) < 1e-6
    assert abs(float(L.gelu_tanh(np.array([0.0]))[0])) < 1e-6


def test_linear_with_bias():
    x = np.ones((2, 3), dtype=np.float32)
    w = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # [out=2, in=3]
    b = np.array([10.0, 20.0], dtype=np.float32)
    out = L.linear(x, w, b)
    np.testing.assert_allclose(out, np.array([[11, 21], [11, 21]]))


def test_rope_is_norm_preserving():
    # rotary embeddings are a rotation -> they preserve vector norm
    rng = np.random.default_rng(0)
    q = rng.standard_normal((2, 5, 8)).astype(np.float32)  # [heads, seq, dim]
    k = q.copy()
    rope = L.RopeCache(head_dim=8, max_pos=16, theta=10000.0)
    cos, sin = rope.get(np.arange(5))
    q2, _ = L.apply_rope(q, k, cos, sin)
    np.testing.assert_allclose(
        np.linalg.norm(q, axis=-1), np.linalg.norm(q2, axis=-1), rtol=1e-4
    )


def test_attention_causality():
    # a future-masked token must not influence an earlier query's output
    rng = np.random.default_rng(1)
    seq, d = 4, 8
    q = rng.standard_normal((seq, d)).astype(np.float32)
    k = rng.standard_normal((seq, d)).astype(np.float32)
    v = rng.standard_normal((seq, d)).astype(np.float32)
    out1 = L.attention(q, k, v, n_heads=1, n_kv_heads=1, causal_offset=0)
    # change only the LAST token's value; first row output must be unchanged
    v2 = v.copy()
    v2[-1] += 100.0
    out2 = L.attention(q, k, v2, n_heads=1, n_kv_heads=1, causal_offset=0)
    np.testing.assert_allclose(out1[0], out2[0], rtol=1e-5)
    assert not np.allclose(out1[-1], out2[-1])  # last row DID change


def test_attention_gqa_grouping():
    # n_heads multiple of n_kv_heads must run and return the right shape
    rng = np.random.default_rng(2)
    seq, hd = 3, 4
    n_heads, n_kv = 4, 2
    q = rng.standard_normal((seq, n_heads * hd)).astype(np.float32)
    k = rng.standard_normal((seq, n_kv * hd)).astype(np.float32)
    v = rng.standard_normal((seq, n_kv * hd)).astype(np.float32)
    out = L.attention(q, k, v, n_heads, n_kv, causal_offset=0)
    assert out.shape == (seq, n_heads * hd)


def test_attention_logit_softcap_bounds_scores():
    # with a tiny softcap, attention should still be finite and valid
    rng = np.random.default_rng(3)
    q = rng.standard_normal((2, 8)).astype(np.float32) * 100
    k = rng.standard_normal((2, 8)).astype(np.float32) * 100
    v = rng.standard_normal((2, 8)).astype(np.float32)
    out = L.attention(q, k, v, 1, 1, causal_offset=0, logit_softcap=5.0)
    assert np.all(np.isfinite(out))
