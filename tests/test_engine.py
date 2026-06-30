"""End-to-end engine tests on a synthetic checkpoint (no downloads).

These pin the invariants that make the streaming design trustworthy:
  * incremental decode == full-sequence forward (KV cache is correct),
  * resident_layers=1 (extreme streaming) == keeping every layer resident,
  * int8 quantization stays close to the float reference,
  * prefetch on/off produces identical results.
If any of these break, the "runs huge models on 8GB" claim is hollow — so they are the
real proof, not the marketing.
"""

import numpy as np
import pytest

from sarab.config import RuntimeConfig
from sarab.engine.model import ModelConfig, SarabModel
from sarab.engine.runtime import Runtime
from sarab.loader.mmap_store import MmapStore
from tests._synthetic import make_tiny_model


def _runtime(tmp_path, **cfg_kw):
    import json
    with open(tmp_path / "config.json") as f:
        hf = json.load(f)
    mc = ModelConfig.from_hf(hf)
    rc = RuntimeConfig(**cfg_kw)
    store = MmapStore.from_dir(tmp_path)
    model = SarabModel(mc, store, rc)
    return Runtime(model, rc), store


def _logits_full(rt, ids):
    rt.reset()
    return rt.forward(np.asarray(ids, dtype=np.int64), all_logits=True)


def _logits_incremental(rt, ids):
    rt.reset()
    last = None
    for t in ids:
        last = rt.forward(np.asarray([t], dtype=np.int64))
    return last


def test_forward_produces_finite_logits(tmp_path):
    make_tiny_model(tmp_path, n_layers=3, hidden=32, vocab=64)
    rt, store = _runtime(tmp_path, quant="none", prefetch=False)
    try:
        logits = _logits_full(rt, [1, 5, 9, 13])
        assert logits.shape == (4, 64)
        assert np.all(np.isfinite(logits))
    finally:
        rt.close(); store.close()


def test_incremental_decode_matches_full(tmp_path):
    """The KV cache must make step-by-step decode equal a single full forward."""
    make_tiny_model(tmp_path, n_layers=3, hidden=32, vocab=64)
    rt, store = _runtime(tmp_path, quant="none", prefetch=False)
    try:
        ids = [3, 7, 11, 2, 8]
        full_last = _logits_full(rt, ids)[-1]
        incr_last = _logits_incremental(rt, ids)
        np.testing.assert_allclose(full_last, incr_last, rtol=1e-4, atol=1e-4)
    finally:
        rt.close(); store.close()


def test_streaming_window_is_output_invariant(tmp_path):
    """resident_layers=1 (max streaming) gives the SAME logits as all-resident."""
    make_tiny_model(tmp_path, n_layers=4, hidden=32, vocab=64)
    ids = [1, 2, 3, 4, 5]

    rt1, s1 = _runtime(tmp_path, quant="none", resident_layers=1, prefetch=False)
    rt2, s2 = _runtime(tmp_path, quant="none", resident_layers=4, prefetch=True,
                       prefetch_depth=2)
    try:
        a = _logits_full(rt1, ids)
        b = _logits_full(rt2, ids)
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-5)
        # confirm rt1 really evicted (more than `resident` builds happened)
        assert rt1.streamer.stats["evictions"] > 0
    finally:
        rt1.close(); s1.close(); rt2.close(); s2.close()


def test_prefetch_does_not_change_output(tmp_path):
    make_tiny_model(tmp_path, n_layers=4, hidden=32, vocab=64)
    ids = [9, 8, 7, 6]
    rt_off, s_off = _runtime(tmp_path, quant="none", prefetch=False, resident_layers=2)
    rt_on, s_on = _runtime(tmp_path, quant="none", prefetch=True, prefetch_depth=2,
                           resident_layers=2)
    try:
        np.testing.assert_allclose(
            _logits_full(rt_off, ids), _logits_full(rt_on, ids), rtol=1e-6, atol=1e-6
        )
    finally:
        rt_off.close(); s_off.close(); rt_on.close(); s_on.close()


@pytest.mark.parametrize("quant", ["int8", "int4"])
def test_quant_stays_close_to_float(tmp_path, quant):
    make_tiny_model(tmp_path, n_layers=3, hidden=64, vocab=64, inter=128)
    ids = [1, 2, 3, 4]
    rt_f, s_f = _runtime(tmp_path, quant="none", prefetch=False)
    rt_q, s_q = _runtime(tmp_path, quant=quant, quant_group_size=64, prefetch=False)
    try:
        ref = _logits_full(rt_f, ids)[-1]
        got = _logits_full(rt_q, ids)[-1]
        # argmax of next-token should usually agree; logits should correlate strongly
        corr = np.corrcoef(ref, got)[0, 1]
        tol = 0.95 if quant == "int8" else 0.80
        assert corr > tol, f"{quant} correlation {corr:.3f} below {tol}"
    finally:
        rt_f.close(); s_f.close(); rt_q.close(); s_q.close()
