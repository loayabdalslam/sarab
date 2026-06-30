"""End-to-end tests through the public API (`load`) and the generation loop, plus a
resident-memory bound check — all on a synthetic local checkpoint, no downloads."""

import numpy as np

from sarab import RuntimeConfig, load
from sarab.generate import SamplingParams, generate_ids
from tests._synthetic import make_tiny_model


def _toy_tokenizer(tmp_path):
    """Write a minimal whitespace tokenizer.json so load() finds a tokenizer."""
    import json
    vocab = {str(i): i for i in range(64)}
    tj = {
        "version": "1.0",
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "0"},
        "pre_tokenizer": {"type": "Whitespace"},
    }
    with open(tmp_path / "tokenizer.json", "w", encoding="utf-8") as f:
        json.dump(tj, f)


def test_load_and_generate_text(tmp_path):
    make_tiny_model(tmp_path, n_layers=2, hidden=32, vocab=64)
    _toy_tokenizer(tmp_path)
    rt = load(str(tmp_path), config=RuntimeConfig(quant="none", prefetch=True))
    try:
        text = rt.generate("1 2 3", max_new_tokens=5, temperature=0.0)
        assert isinstance(text, str)
        # streaming should yield the same number of pieces
        pieces = list(rt.generate("1 2 3", max_new_tokens=5, temperature=0.0, stream=True))
        assert len(pieces) == 5
    finally:
        rt.close()


def test_generate_ids_respects_max_tokens(tmp_path):
    make_tiny_model(tmp_path, n_layers=2, hidden=32, vocab=64)
    cfg = RuntimeConfig(quant="none", prefetch=False)
    from sarab.engine.model import ModelConfig, SarabModel
    from sarab.engine.runtime import Runtime
    from sarab.loader.mmap_store import MmapStore
    import json
    with open(tmp_path / "config.json") as f:
        mc = ModelConfig.from_hf(json.load(f))
    store = MmapStore.from_dir(tmp_path)
    rt = Runtime(SarabModel(mc, store, cfg), cfg)
    try:
        out = list(generate_ids(rt, [1, 2, 3], SamplingParams(max_new_tokens=7)))
        assert len(out) == 7
        assert all(0 <= t < 64 for t in out)
    finally:
        rt.close(); store.close()


def test_resident_memory_stays_bounded(tmp_path):
    """The whole pitch: with resident_layers=1 the cache never holds >1 block,
    regardless of model depth. Proven structurally on the streamer's own accounting."""
    make_tiny_model(tmp_path, n_layers=8, hidden=64, inter=128, vocab=64)
    cfg = RuntimeConfig(quant="int8", resident_layers=1, prefetch=False)
    rt = load(str(tmp_path), config=cfg) if (tmp_path / "tokenizer.json").exists() else None
    # build via engine directly (no tokenizer needed)
    import json
    from sarab.engine.model import ModelConfig, SarabModel
    from sarab.engine.runtime import Runtime
    from sarab.loader.mmap_store import MmapStore
    with open(tmp_path / "config.json") as f:
        mc = ModelConfig.from_hf(json.load(f))
    store = MmapStore.from_dir(tmp_path)
    runtime = Runtime(SarabModel(mc, store, cfg), cfg)
    try:
        runtime.reset()
        max_resident = 0
        # monkey-tap: after each layer the cache size must stay <= resident_layers
        orig_get = runtime.streamer.get
        def traced(i):
            w = orig_get(i)
            nonlocal max_resident
            max_resident = max(max_resident, len(runtime.streamer._cache))
            return w
        runtime.streamer.get = traced
        runtime.forward(np.asarray([1, 2, 3, 4], dtype=np.int64))
        assert max_resident <= cfg.resident_layers
        assert runtime.streamer.stats["evictions"] > 0
    finally:
        runtime.close(); store.close()
