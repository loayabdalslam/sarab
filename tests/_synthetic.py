"""Shared test fixtures: build a tiny synthetic Llama checkpoint on disk.

This lets the whole streaming engine be exercised end-to-end — mmap store, streamer,
quant, attention, KV cache, generation — without downloading anything. The synthetic
model is real safetensors with HF-correct tensor names, so the same code path that runs
Llama-3 runs here.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _st_write(path: Path, tensors: dict) -> None:
    """Write a dict of {name: np.ndarray(float32)} as a safetensors file."""
    header = {}
    offset = 0
    blobs = []
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        data = arr.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        blobs.append(data)
    header_json = json.dumps(header).encode("utf-8")
    pad = (8 - len(header_json) % 8) % 8           # safetensors wants 8-byte alignment
    header_json += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for b in blobs:
            f.write(b)


def make_tiny_model(dirpath: Path, *, n_layers=2, hidden=32, n_heads=4, n_kv_heads=2,
                    inter=64, vocab=64, seed=0, model_type="llama",
                    attn_bias=False, qk_norm=False) -> dict:
    """Create config.json + model.safetensors for a tiny decoder model.

    `attn_bias` adds q/k/v projection biases (Qwen2-style); `qk_norm` adds per-head q/k
    RMSNorm weights (Qwen3-style). Returns the HF config dict used.
    """
    rng = np.random.default_rng(seed)
    head_dim = hidden // n_heads
    scale = 0.02

    def r(*shape):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": r(vocab, hidden),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "lm_head.weight": r(vocab, hidden),
    }
    for i in range(n_layers):
        p = f"model.layers.{i}"
        tensors[f"{p}.self_attn.q_proj.weight"] = r(n_heads * head_dim, hidden)
        tensors[f"{p}.self_attn.k_proj.weight"] = r(n_kv_heads * head_dim, hidden)
        tensors[f"{p}.self_attn.v_proj.weight"] = r(n_kv_heads * head_dim, hidden)
        tensors[f"{p}.self_attn.o_proj.weight"] = r(hidden, n_heads * head_dim)
        tensors[f"{p}.mlp.gate_proj.weight"] = r(inter, hidden)
        tensors[f"{p}.mlp.up_proj.weight"] = r(inter, hidden)
        tensors[f"{p}.mlp.down_proj.weight"] = r(hidden, inter)
        tensors[f"{p}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        if attn_bias:
            tensors[f"{p}.self_attn.q_proj.bias"] = r(n_heads * head_dim)
            tensors[f"{p}.self_attn.k_proj.bias"] = r(n_kv_heads * head_dim)
            tensors[f"{p}.self_attn.v_proj.bias"] = r(n_kv_heads * head_dim)
        if qk_norm:
            tensors[f"{p}.self_attn.q_norm.weight"] = np.ones(head_dim, dtype=np.float32)
            tensors[f"{p}.self_attn.k_norm.weight"] = np.ones(head_dim, dtype=np.float32)

    dirpath.mkdir(parents=True, exist_ok=True)
    _st_write(dirpath / "model.safetensors", tensors)

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": model_type,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_hidden_layers": n_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": vocab,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
    }
    if attn_bias:
        cfg["attention_bias"] = True
    with open(dirpath / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    return cfg
