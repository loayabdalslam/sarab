<div align="center">

# 🌅 SARAB · سراب

### Run GPU-class language models on an 8 GB CPU box.

*A from-scratch streaming inference runtime that treats a model as a **stream of computation flowing off disk** — not a static block that must fit in memory.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Backend](https://img.shields.io/badge/compute-NumPy%20·%20CPU-orange.svg)]()
[![HF](https://img.shields.io/badge/🤗%20Hugging%20Face-compatible-yellow.svg)](https://huggingface.co/)
[![API](https://img.shields.io/badge/API-OpenAI%20compatible-412991.svg)]()
[![Status](https://img.shields.io/badge/status-proof%20of%20concept-blueviolet.svg)]()

[Why SARAB](#-why-sarab) · [How it works](#-how-it-works) · [Benchmarks](#-benchmarks) · [Install](#-install) · [Usage](#-usage) · [Architectures](#-supported-architectures) · [Testing](#-testing)

</div>

---

## ✨ TL;DR

```python
from sarab import load

rt = load("Qwen/Qwen2.5-0.5B-Instruct", quant="int8", ram_budget_gb=4)
print(rt.generate("The capital of Egypt is", max_new_tokens=32))
```

> **Every other runtime says:** *"the model must fit entirely in memory."*
> **SARAB says:** *"no — I stream it off disk, block by block, so a tiny RAM budget serves any model."*

---

## 🤔 Why SARAB

The dominant assumption in LLM serving is that **all weights must be resident in memory** before you can generate a single token. That is why a 13B / 70B model "needs" a big GPU (or many).

That is an **assumption, not a law of physics**. At any single decode step, the *active path* through the weights is small. SARAB is built around that fact.

|                              | **vLLM**         | **llama.cpp**   | **transformers** | **🌅 SARAB**                         |
| ---------------------------- | ---------------- | --------------- | ---------------- | ----------------------------------- |
| Primary target               | GPU throughput   | CPU/Mac, C++    | reference impl   | **model bigger than RAM, on CPU**   |
| Weight residency             | all in VRAM      | all in RAM      | all in RAM       | **one block at a time, off disk**   |
| If model > memory            | ❌ OOM           | ❌ OOM / swap   | ❌ OOM           | ✅ streams and runs                 |
| Peak RAM                     | ≈ model size     | ≈ model size    | ≈ model size     | **≈ constant (one resident block)** |
| Raw speed                    | highest (GPU)    | high (SIMD)     | low              | moderate (NumPy; disk-bound if huge)|
| Built from scratch / hackable| —                | —               | —                | ✅ clean NumPy core                 |

> **SARAB is not competing on raw speed.** It competes on *"can this even run on a modest machine?"* — where the others fail with Out-Of-Memory, SARAB keeps generating.

---

## 🧠 How it works

Three mechanisms working together:

### 1. 🗺️ Memory-mapped weights (`mmap`)
Weights are read from `safetensors` through `mmap`. The OS pages in **only the bytes you touch** and reclaims them under pressure. The model is fully present *on disk* yet barely resident *in RAM*.

### 2. 📖 Layer-by-layer streaming + LRU window
The forward pass walks the transformer **one block at a time**. Only `resident_layers` blocks are ever decompressed in RAM. Whether the model is 1 B or 70 B, the resident footprint stays bounded.

```
        ┌──────────── disk (SSD/NVMe) ────────────┐
        │  L0  L1  L2  L3  ...  L22  L23  (full model) │
        └──────────────────────────────────────────┘
                       │ stream on demand
                       ▼
        RAM window:  [ L_n ][ L_n+1 ]   ← only this is resident
                       │ compute
                       ▼
                   next token
```

### 3. 🔮 Prefetch thread (the oracle)
A background thread fetches block `N+1` off disk **while block `N` is computing**, so disk latency hides behind compute. *This is what removes the stalls.*

### ➕ Group-wise quantization (int8 / int4)
Shrinks the on-disk + resident footprint further. Resident layers dequantize **once** (cached); streamed layers dequantize per visit and the cache dies with the eviction — no memory blow-up.

### 🧭 Auto-sizing
If the (quantized) model fits the RAM budget → **all layers stay resident** (built once, fast). Only genuinely over-budget models stream. You never hand-tune this.

---

## 📊 Benchmarks

> Measured with `sarab bench` on a CPU-only machine (NumPy backend, no SIMD/Rust yet).
> Methodology: warm-up run discarded, steady-state **decode** throughput, peak RSS sampled every 20 ms.

### `loaiabdalslam/SLM-FRIDGE-ICED-0.5B-32BQWEN` · 128 tokens · 4 GB budget

| Metric                 | `--quant none` | `--quant int8` |
| ---------------------- | -------------: | -------------: |
| Model on disk          |       0.92 GB  |       0.92 GB  |
| Resident layers        |       24 / 24  |       24 / 24  |
| **Peak RSS**           |       3.55 GB  |   **3.23 GB**  |
| Tokens generated       |          128   |          128   |
| **Throughput (decode)**|     4.76 tok/s |  **6.79 tok/s**|
| Cache hit rate         |        99.6 %  |       99.97 %  |
| Evictions              |            0   |            0   |

```text
=== SARAB benchmark ===
model on disk      :   0.92 GB
resident layers    : 24/24  (ALL resident (fits RAM))
peak resident RSS  :   3.23 GB  (budget 4.0 GB) [OK]
tokens generated   : 128
throughput (decode):   6.79 tok/s
streamer stats     : {'hits': 3216, 'misses': 1, 'prefetched': 23, 'evictions': 0}
```

**Reading the streamer stats:**
- `hits: 3216 / misses: 1` → **99.97 %** of layer requests served from cache.
- `prefetched: 23` → the oracle staged 23 of 24 blocks **before** they were needed.
- `evictions: 0` → the whole model lived in RAM; nothing was thrown away and rebuilt.

> 💡 For small models that fit RAM, `--quant none` and `int8` use similar memory — the int8 win on RAM only materializes for **models larger than RAM** (where evicted layers drop their float cache). The int8 *speed* win here comes from the one-time dequant cache.

### What a streaming run looks like (model > RAM)

When the model doesn't fit, you'd see the streaming signature instead:

```text
resident layers    :  6/40   (streaming 6/40)
peak resident RSS  :  3.8 GB  (budget 4.0 GB) [OK]
streamer stats     : {'hits': ..., 'misses': 40, 'prefetched': 39, 'evictions': 5440}
```

Peak RAM stays under budget **even though the model on disk is many times larger** — that's the whole point.

---

## 📦 Install

```bash
git clone <your-repo-url> sarab && cd sarab
pip install -e ".[server]"        # core + OpenAI-compatible server
# or with dev/test extras (adds torch + transformers as a reference oracle only):
pip install -e ".[dev,server]"
```

**Requirements:** Python 3.9+, NumPy. `torch`/`transformers` are used **only** as a test-time accuracy reference — never on the inference path.

---

## 🚀 Usage

### CLI

```bash
# one-shot generation with live stats
sarab run Qwen/Qwen2.5-0.5B-Instruct --prompt "Write a haiku about the desert" --max-new-tokens 64

# benchmark: throughput + peak RAM
sarab bench Qwen/Qwen2.5-0.5B-Instruct --max-new-tokens 128

# OpenAI-compatible server
sarab serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

Or via the module form (no install needed):

```bash
python -m sarab bench <model-id> --max-new-tokens 128 --quant int8
```

**Flags:** `--quant {none,int8,int4}` · `--ram-budget <GB>` · `--resident-layers <N>` · `--prefetch-depth <N>` · `--no-prefetch` · `--max-context <N>`

### Python API

```python
from sarab import load, RuntimeConfig

rt = load("Qwen/Qwen2.5-0.5B-Instruct",
          config=RuntimeConfig(quant="int8", ram_budget_gb=4, prefetch=True))

# streaming generation
for piece in rt.generate("Once upon a time", max_new_tokens=80, stream=True):
    print(piece, end="", flush=True)

rt.close()
```

### Comparison harness (vs vLLM / transformers)

```bash
python -m sarab.compare Qwen/Qwen2.5-0.5B-Instruct \
    --prompt "The capital of Egypt is" --max-new-tokens 64
```

Calls vLLM through its **public API only** (zero code copied). Falls back to a `transformers` CPU baseline if vLLM isn't installed. Prints load time, tok/s, and peak RAM for each engine side by side.

### OpenAI client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="sarab", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

---

## 🧩 Supported architectures

Most modern decoder LLMs share one skeleton (pre-norm block + RoPE + gated MLP) and differ only in small knobs, captured by `ArchSpec` in [`sarab/engine/architectures.py`](sarab/engine/architectures.py).

| Family                       | Status | Notes                                              |
| ---------------------------- | :----: | -------------------------------------------------- |
| Llama 2 / 3 / 3.1 / 3.2      |   ✅   | incl. llama3 RoPE frequency rescaling              |
| Mistral                      |   ✅   |                                                    |
| Qwen2                        |   ✅   | attention bias                                     |
| Qwen3                        |   ✅   | per-head q/k RMSNorm                                |
| Gemma / Gemma2               |   ✅   | embed scaling, (1+w) norm, gelu-tanh, logit softcap|
| Phi-3, StableLM, Starcoder2  |   ✅   |                                                    |
| Unknown `model_type`         |   ⚠️   | assumes standard Llama layout (with a warning)     |
| GPT-2 / T5 / Mamba           |   ❌   | different skeleton → clear error, not silent garbage|

---

## 🗂️ Project layout

```
sarab/
├── config.py              RuntimeConfig — RAM budget, quant, prefetch, auto-sizing
├── hub.py                 Hugging Face integration · load() · SarabRuntime (tokenizer)
├── loader/
│   ├── mmap_store.py      zero-copy safetensors over mmap (+ row-gather for embeddings)
│   └── streamer.py        layer streaming · LRU window · prefetch thread
├── quant/int4.py          group-wise int8/int4 · cached dequant · vectorized
├── engine/
│   ├── architectures.py   architecture registry (Llama/Qwen/Gemma/…)
│   ├── layers.py          RMSNorm · RoPE · GQA attention · gated MLP (NumPy)
│   ├── kvcache.py         bounded rolling KV cache
│   ├── model.py           build model from HF config · per-layer factory
│   └── runtime.py         the streaming forward pass
├── generate.py            sampling (greedy / temperature / top-p / top-k)
├── server/api.py          OpenAI-compatible FastAPI server (SSE streaming)
├── benchmark.py           throughput + peak-RAM benchmark
├── compare.py             SARAB vs vLLM / transformers
└── cli.py                 sarab run | serve | bench
```

---

## ✅ Testing

Accuracy is **measured, not promised**.

```bash
pip install -e ".[dev,server]"

# fast suite — runs on a synthetic checkpoint, no downloads
pytest -q --ignore=tests/test_fidelity.py

# fidelity vs a REAL Hugging Face model (greedy must match transformers)
SARAB_FIDELITY_MODEL=HuggingFaceTB/SmolLM2-135M pytest tests/test_fidelity.py -s
```

Key invariants pinned by the suite:

- 🎯 **`test_incremental_decode_matches_full`** — step-by-step decode equals a single full forward (KV cache is correct).
- 🎯 **`test_streaming_window_is_output_invariant`** — `resident_layers=1` (max streaming) yields **identical** logits to keeping every layer resident. *Streaming costs nothing in accuracy.*
- 🎯 **`test_quant_stays_close_to_float`** — int8/int4 outputs stay strongly correlated with the float reference.
- Plus unit tests for mmap, quantization pack/unpack, streamer LRU + prefetch, attention causality & GQA, sampling, and every architecture family.

---

## ⚠️ Honest limitations

- This is a **proof-of-concept on a pure-NumPy CPU core.** It is not faster than vLLM on a GPU, and not yet as fast as llama.cpp's hand-tuned SIMD on CPU.
- For models **larger than RAM**, throughput is bounded by your **disk speed** (NVMe strongly recommended). The prefetcher hides what it can, but physics is physics — the win is that it **runs at all**.
- Current `int8`/`int4` matmul dequantizes to float for BLAS, so quantization saves *space when streaming* but doesn't yet speed up the matmul itself.

---

## 🛣️ Roadmap

- ⚙️ Move the hot kernels (matmul / dequant / streamer) to **Rust via PyO3** for SIMD — expected 3–5× speedup; true fused int8 GEMM (no dequant).
- 🧬 **MoE support** — stream only the *active experts* per token (huge lever for 100B+ models).
- ⚡ **Speculative decoding** — small draft model + big verifier for multi-x speedup at zero accuracy loss.
- 🪶 **Ternary (1.58-bit)** quantization for models trained for it.
- 📦 Pre-quantized sidecar files (quantize once, load instantly).

---

## 📄 License

Apache-2.0

<div align="center">

**🌅 SARAB — making the impossible run on the modest.**

*A model that "needs thousands of GPUs"... breathing on a CPU.*

</div>
