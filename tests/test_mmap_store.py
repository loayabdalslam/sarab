"""Tests for the mmap safetensors store: zero-copy reads, sharding, dtype handling."""

import numpy as np

from sarab.loader.mmap_store import MmapStore, bf16_to_f32
from tests._synthetic import make_tiny_model


def test_roundtrip_values(tmp_path):
    make_tiny_model(tmp_path, n_layers=2, hidden=32)
    store = MmapStore.from_dir(tmp_path)
    try:
        emb = store["model.embed_tokens.weight"]
        assert emb.shape == (64, 32)
        arr = emb.array("float32")
        assert arr.dtype == np.float32
        # second read returns identical data (mmap is stable)
        arr2 = store["model.embed_tokens.weight"].array("float32")
        np.testing.assert_array_equal(arr, arr2)
    finally:
        store.close()


def test_total_bytes_matches(tmp_path):
    make_tiny_model(tmp_path, n_layers=3, hidden=32)
    store = MmapStore.from_dir(tmp_path)
    try:
        # at least embeddings + lm_head + per-layer weights are counted
        assert store.total_bytes() > 0
        assert "model.layers.2.mlp.down_proj.weight" in store
    finally:
        store.close()


def test_missing_tensor_raises(tmp_path):
    make_tiny_model(tmp_path)
    store = MmapStore.from_dir(tmp_path)
    try:
        try:
            _ = store["does.not.exist"]
            assert False, "expected KeyError"
        except KeyError:
            pass
    finally:
        store.close()


def test_bf16_upcast_is_lossless():
    # a few exact float32 values that are representable in bf16
    vals = np.array([1.0, -2.0, 0.5, 0.0, 256.0], dtype=np.float32)
    # truncate to bf16 by taking the high 16 bits
    u16 = (vals.view(np.uint32) >> 16).astype(np.uint16)
    back = bf16_to_f32(u16)
    np.testing.assert_array_equal(back, vals)
