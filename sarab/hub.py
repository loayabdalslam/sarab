"""Hugging Face integration and the public `load()` entry point.

`load(model_id)` resolves a checkpoint (local path or HF hub id), opens it with the
mmap store, parses the config, and returns a `SarabRuntime` — a thin, friendly facade
over the streaming engine with a tokenizer attached. No weights are read into RAM at
load time; only the config, tokenizer, and the safetensors *index* are touched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .config import RuntimeConfig
from .engine.model import ModelConfig, SarabModel
from .engine.runtime import Runtime
from .generate import SamplingParams, generate_ids
from .loader.mmap_store import MmapStore


def _resolve_model_dir(model_id: str, cache_dir: Optional[str]) -> Path:
    """Return a local directory holding config + safetensors for `model_id`.

    If `model_id` is an existing local path, use it. Otherwise download the needed files
    from the HF hub (config, tokenizer, and all safetensors shards / index).
    """
    p = Path(model_id)
    if p.is_dir():
        return p

    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "*.safetensors",
            "*.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",
        ],
    )
    return Path(local)


class SarabRuntime:
    """User-facing runtime: tokenize -> stream-generate -> detokenize."""

    def __init__(self, model_dir: Path, config: RuntimeConfig) -> None:
        self.model_dir = model_dir
        self.config = config

        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            hf_cfg = json.load(f)
        # some configs nest under "text_config" (multimodal); prefer it if present
        if "text_config" in hf_cfg and "num_hidden_layers" in hf_cfg["text_config"]:
            hf_cfg = {**hf_cfg, **hf_cfg["text_config"]}
        self.model_config = ModelConfig.from_hf(hf_cfg)

        self.store = MmapStore.from_dir(model_dir)
        self.model = SarabModel(self.model_config, self.store, config)
        self.runtime = Runtime(self.model, config)

        self.tokenizer = self._load_tokenizer(model_dir)
        self._gen_defaults = self._load_generation_config(model_dir)

    # -- tokenizer -----------------------------------------------------------------
    @staticmethod
    def _load_tokenizer(model_dir: Path):
        from tokenizers import Tokenizer

        tj = model_dir / "tokenizer.json"
        if tj.is_file():
            return Tokenizer.from_file(str(tj))
        # fall back to transformers' loader for SentencePiece-only checkpoints
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(str(model_dir))

    def encode(self, text: str) -> List[int]:
        enc = self.tokenizer.encode(text)
        return enc.ids if hasattr(enc, "ids") else enc

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @staticmethod
    def _load_generation_config(model_dir: Path) -> dict:
        gc = model_dir / "generation_config.json"
        if gc.is_file():
            with open(gc, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _eos_ids(self) -> List[int]:
        eos = self._gen_defaults.get("eos_token_id")
        if eos is None:
            return []
        return eos if isinstance(eos, list) else [eos]

    # -- generation ----------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        seed: int = 0,
        stream: bool = False,
    ):
        """Generate text from a raw prompt. Returns a string, or a token iterator if
        `stream=True`."""
        params = SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_token_ids=self._eos_ids(),
        )
        prompt_ids = self.encode(prompt)
        token_iter = generate_ids(self.runtime, prompt_ids, params)

        if stream:
            return self._stream_text(token_iter)
        return self.decode(list(token_iter))

    def _stream_text(self, token_iter):
        buffer: List[int] = []
        for tok in token_iter:
            buffer.append(tok)
            yield self.decode([tok])

    def logits_for(self, prompt: str) -> np.ndarray:
        """All-position logits for a prompt — used by the fidelity test."""
        ids = np.asarray(self.encode(prompt), dtype=np.int64)
        self.runtime.reset()
        return self.runtime.forward(ids, all_logits=True)

    def prewarm(self, progress: bool = True) -> "SarabRuntime":
        """Pre-build/quantize the resident layers with a progress bar. Returns self."""
        self.runtime.prewarm(progress=progress)
        return self

    def close(self) -> None:
        self.runtime.close()
        self.store.close()

    def __enter__(self) -> "SarabRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load(
    model_id: str,
    *,
    config: Optional[RuntimeConfig] = None,
    **overrides,
) -> SarabRuntime:
    """Load a model by HF id or local path into a streaming SARAB runtime.

    Extra keyword args are applied as `RuntimeConfig` overrides, e.g.
        load("meta-llama/Llama-3.2-1B-Instruct", quant="int4", ram_budget_gb=2)
    """
    if config is not None:
        cfg = config
    else:
        cfg = RuntimeConfig(**overrides)
    model_dir = _resolve_model_dir(model_id, cfg.cache_dir)
    return SarabRuntime(model_dir, cfg)
