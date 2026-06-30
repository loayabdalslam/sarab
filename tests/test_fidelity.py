"""Reference-fidelity test against Hugging Face transformers on a REAL small model.

This is the ultimate accuracy proof: SARAB and transformers must agree on the greedy
continuation of a prompt. It is skipped unless `torch` + `transformers` are installed and
the model can be fetched, so the core test suite stays fast and offline. Run it explicitly
with:  SARAB_FIDELITY_MODEL=HuggingFaceTB/SmolLM2-135M pytest tests/test_fidelity.py -s
"""

import os

import numpy as np
import pytest

MODEL = os.environ.get("SARAB_FIDELITY_MODEL")

pytestmark = pytest.mark.skipif(
    not MODEL, reason="set SARAB_FIDELITY_MODEL to a small HF model id to run"
)


def _have_transformers():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_transformers(), reason="torch/transformers not installed")
def test_greedy_matches_transformers():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sarab import RuntimeConfig, load

    prompt = "The capital of France is"
    n = 12

    # --- reference ---
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    ref_model.eval()
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        ref_out = ref_model.generate(ids, max_new_tokens=n, do_sample=False)
    ref_text = tok.decode(ref_out[0][ids.shape[1]:], skip_special_tokens=True)

    # --- SARAB (float, no quant -> should match exactly in argmax) ---
    rt = load(MODEL, config=RuntimeConfig(quant="none", prefetch=True, resident_layers=2))
    try:
        got_text = rt.generate(prompt, max_new_tokens=n, temperature=0.0)
    finally:
        rt.close()

    print(f"\nREF  : {ref_text!r}\nSARAB: {got_text!r}")
    # greedy decoding with float weights should reproduce the reference continuation
    assert got_text.strip()[:40] == ref_text.strip()[:40]
