"""Tests for the architecture registry and family-specific forward paths."""

import json

import numpy as np
import pytest

from sarab.config import RuntimeConfig
from sarab.engine.architectures import resolve_arch
from sarab.engine.model import ModelConfig, SarabModel
from sarab.engine.runtime import Runtime
from sarab.loader.mmap_store import MmapStore
from tests._synthetic import make_tiny_model


def test_resolve_known_families():
    assert resolve_arch({"model_type": "llama"}).name == "llama"
    assert resolve_arch({"model_type": "qwen2"}).attn_bias is True
    assert resolve_arch({"model_type": "qwen3"}).qk_norm is True
    g = resolve_arch({"model_type": "gemma"})
    assert g.embed_scale and g.norm_add_unit and g.activation == "gelu_tanh"


def test_unknown_family_warns_but_defaults():
    with pytest.warns(UserWarning):
        spec = resolve_arch({"model_type": "totally-new-llm"})
    assert spec.name == "default"


def test_explicitly_unsupported_raises():
    with pytest.raises(NotImplementedError):
        resolve_arch({"model_type": "t5"})


def test_attention_bias_flag_respected():
    spec = resolve_arch({"model_type": "llama", "attention_bias": True})
    assert spec.attn_bias is True


def _run(tmp_path, **mk):
    cfg = make_tiny_model(tmp_path, **mk)
    mc = ModelConfig.from_hf(cfg)
    rc = RuntimeConfig(quant="none", prefetch=False)
    store = MmapStore.from_dir(tmp_path)
    rt = Runtime(SarabModel(mc, store, rc), rc)
    rt.reset()
    logits = rt.forward(np.asarray([1, 2, 3, 4], dtype=np.int64), all_logits=True)
    rt.close(); store.close()
    return logits


def test_qwen2_with_bias_runs(tmp_path):
    logits = _run(tmp_path, model_type="qwen2", attn_bias=True, hidden=32, vocab=64)
    assert logits.shape == (4, 64)
    assert np.all(np.isfinite(logits))


def test_qwen3_with_qk_norm_runs(tmp_path):
    logits = _run(tmp_path, model_type="qwen3", qk_norm=True, hidden=32, vocab=64)
    assert logits.shape == (4, 64)
    assert np.all(np.isfinite(logits))


def test_default_llama_runs(tmp_path):
    logits = _run(tmp_path, model_type="llama", hidden=32, vocab=64)
    assert np.all(np.isfinite(logits))
