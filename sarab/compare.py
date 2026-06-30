"""Head-to-head comparison: SARAB vs vLLM (or transformers) on the SAME model.

Philosophy of the comparison — this is the honest framing, not marketing:

  vLLM is the gold standard for *GPU* throughput: it loads all weights into VRAM and
  batches aggressively. SARAB targets the opposite regime — a CPU box with as little as
  8GB RAM, where the model does NOT fit in memory and is streamed off disk. So the two
  are not competitors on the same axis; the benchmark measures each where it lives:

    * peak memory footprint   (SARAB's whole point: it stays tiny)
    * cold-start load time     (SARAB starts generating without loading the full model)
    * tokens/sec               (vLLM wins raw speed on GPU; SARAB shows it's *usable* on CPU)

We call vLLM through its own public API; NONE of vLLM's code or design is copied. If vLLM
isn't installed (e.g. no GPU), we fall back to a Hugging Face `transformers` CPU baseline,
which is the realistic "ordinary way" to run a model on the same 8GB machine.

Run:
    python -m sarab.compare meta-llama/Llama-3.2-1B-Instruct \
        --prompt "The capital of Egypt is" --max-new-tokens 64
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass, asdict
from typing import Optional

from .config import RuntimeConfig


@dataclass
class Result:
    engine: str
    ok: bool
    load_s: float = 0.0
    gen_s: float = 0.0
    tokens: int = 0
    tok_per_s: float = 0.0
    peak_gb: float = 0.0
    note: str = ""
    text: str = ""


class _PeakRSS:
    def __init__(self, interval: float = 0.02):
        import threading

        import psutil
        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop:
            self._peak = max(self._peak, self._proc.memory_info().rss)
            time.sleep(self._interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._t.join(timeout=1.0)

    @property
    def peak_gb(self):
        return self._peak / 1024**3


# --------------------------------------------------------------------------- SARAB
def bench_sarab(model_id: str, prompt: str, n: int, cfg: RuntimeConfig) -> Result:
    from .hub import load

    try:
        with _PeakRSS() as peak:
            t0 = time.perf_counter()
            rt = load(model_id, config=cfg)
            load_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            out = list(rt.generate(prompt, max_new_tokens=n, temperature=0.0, stream=True))
            gen_s = time.perf_counter() - t1
            text = "".join(out)
            rt.close()
        return Result("SARAB (CPU stream)", True, load_s, gen_s, len(out),
                      len(out) / max(gen_s, 1e-9), peak.peak_gb,
                      note=f"quant={cfg.quant}, resident<={cfg.resident_layers}",
                      text=text)
    except Exception as e:  # pragma: no cover - environment dependent
        return Result("SARAB (CPU stream)", False, note=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------- vLLM
def bench_vllm(model_id: str, prompt: str, n: int) -> Result:
    try:
        from vllm import LLM, SamplingParams
    except Exception as e:
        return Result("vLLM", False, note=f"not available ({type(e).__name__})")

    try:
        with _PeakRSS() as peak:
            t0 = time.perf_counter()
            llm = LLM(model=model_id, enforce_eager=True)
            load_s = time.perf_counter() - t0
            sp = SamplingParams(temperature=0.0, max_tokens=n)
            t1 = time.perf_counter()
            out = llm.generate([prompt], sp)
            gen_s = time.perf_counter() - t1
        text = out[0].outputs[0].text
        toks = len(out[0].outputs[0].token_ids)
        return Result("vLLM", True, load_s, gen_s, toks,
                      toks / max(gen_s, 1e-9), peak.peak_gb, text=text)
    except Exception as e:  # pragma: no cover
        return Result("vLLM", False, note=f"{type(e).__name__}: {e}")


# ------------------------------------------------------- transformers (CPU baseline)
def bench_transformers(model_id: str, prompt: str, n: int) -> Result:
    """The 'ordinary way' to run a model on a CPU box: load it all into RAM."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        return Result("transformers (CPU, full-load)", False,
                      note=f"not available ({type(e).__name__})")

    try:
        with _PeakRSS() as peak:
            t0 = time.perf_counter()
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            model.eval()
            load_s = time.perf_counter() - t0
            ids = tok(prompt, return_tensors="pt").input_ids
            t1 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=n, do_sample=False)
            gen_s = time.perf_counter() - t1
        new = out[0][ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        return Result("transformers (CPU, full-load)", True, load_s, gen_s, len(new),
                      len(new) / max(gen_s, 1e-9), peak.peak_gb, text=text)
    except Exception as e:  # pragma: no cover
        return Result("transformers (CPU, full-load)", False,
                      note=f"{type(e).__name__}: {e}")


def _print_table(results):
    print("\n" + "=" * 78)
    print(f"{'engine':<28}{'load(s)':>9}{'gen(s)':>9}{'tok/s':>9}{'peakGB':>9}{'ok':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r.engine:<28}{r.load_s:>9.1f}{r.gen_s:>9.1f}{r.tok_per_s:>9.2f}"
              f"{r.peak_gb:>9.2f}{('yes' if r.ok else 'NO'):>6}")
        if not r.ok:
            print(f"    -> {r.note}")
    print("=" * 78)
    for r in results:
        if r.ok and r.text:
            print(f"\n[{r.engine}] {r.text[:200]!r}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Compare SARAB vs vLLM / transformers")
    p.add_argument("model")
    p.add_argument("--prompt", default="The capital of Egypt is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--quant", choices=["none", "int8", "int4"], default="int8")
    p.add_argument("--ram-budget", type=float, default=4.0)
    p.add_argument("--resident-layers", type=int, default=2)
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--skip-transformers", action="store_true")
    a = p.parse_args(argv)

    cfg = RuntimeConfig(ram_budget_gb=a.ram_budget, resident_layers=a.resident_layers,
                        quant=a.quant)

    results = []
    print(f"[compare] model={a.model}  prompt={a.prompt!r}  max_new_tokens={a.max_new_tokens}")

    results.append(bench_sarab(a.model, a.prompt, a.max_new_tokens, cfg))
    gc.collect()
    if not a.skip_vllm:
        results.append(bench_vllm(a.model, a.prompt, a.max_new_tokens))
        gc.collect()
    if not a.skip_transformers:
        results.append(bench_transformers(a.model, a.prompt, a.max_new_tokens))

    _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
