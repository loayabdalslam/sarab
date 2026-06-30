"""Model construction from a Hugging Face config + an mmap'd checkpoint.

`SarabModel` does *not* hold any layer weights. It holds:
  * the parsed architecture (ModelConfig) + family knobs (ArchSpec),
  * the MmapStore index (zero-copy handles, not data),
  * persistent small tensors (embeddings, final norm, lm_head) which it streams on use,
  * and a `build_layer(i)` factory the streamer calls to materialize one block at a time.

When quantization is enabled, each weight matrix is quantized the first time its layer is
built. That cost is paid once per eviction cycle; with a warm LRU window a re-used layer
is free. (Pre-quantizing to a sidecar file is a natural later optimization.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..config import RuntimeConfig
from ..loader.mmap_store import MmapStore
from ..quant.int4 import quantize
from .architectures import ArchSpec, resolve_arch
from .layers import RopeCache


@dataclass
class ModelConfig:
    """The slice of an HF decoder config the engine needs, plus the family ArchSpec."""

    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab_size: int
    rms_eps: float
    rope_theta: float
    rope_scaling: Optional[dict]
    max_pos: int
    tie_word_embeddings: bool
    arch: ArchSpec
    attn_scale: Optional[float] = None       # query_pre_attn_scalar override (gemma2)

    @classmethod
    def from_hf(cls, cfg: dict) -> "ModelConfig":
        n_heads = cfg["num_attention_heads"]
        hidden = cfg["hidden_size"]
        head_dim = cfg.get("head_dim", hidden // n_heads)
        arch = resolve_arch(cfg)
        attn_scale = None
        if cfg.get("query_pre_attn_scalar"):
            attn_scale = 1.0 / float(cfg["query_pre_attn_scalar"]) ** 0.5
        return cls(
            hidden_size=hidden,
            intermediate_size=cfg["intermediate_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads),
            head_dim=head_dim,
            vocab_size=cfg["vocab_size"],
            rms_eps=cfg.get("rms_norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", 10000.0),
            rope_scaling=cfg.get("rope_scaling"),
            max_pos=cfg.get("max_position_embeddings", 4096),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            arch=arch,
            attn_scale=attn_scale,
        )


# matrices we quantize; norms/biases stay full-precision (cheap, accuracy-sensitive)
_QUANTIZABLE = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
# optional tensors that may or may not exist depending on the family
_OPTIONAL = {"q_bias", "k_bias", "v_bias", "o_bias", "q_norm", "k_norm",
             "pre_ff_ln", "post_ff_ln"}
_REQUIRED = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
             "input_ln", "post_ln"}


class SarabModel:
    """Holds architecture + checkpoint index; materializes layers on demand."""

    def __init__(self, cfg: ModelConfig, store: MmapStore, rc: RuntimeConfig) -> None:
        self.cfg = cfg
        self.arch = cfg.arch
        self.store = store
        self.rc = rc
        self.rope = RopeCache(
            head_dim=cfg.head_dim,
            # +64 margin so positions at the max_context boundary never index past the
            # precomputed table during long generation.
            max_pos=max(cfg.max_pos, rc.max_context) + 64,
            theta=cfg.rope_theta,
            scaling=cfg.rope_scaling,
        )
        self._embed_scale = float(np.sqrt(cfg.hidden_size)) if self.arch.embed_scale else 1.0

    # -- persistent (non-block) tensors --------------------------------------------
    def embed_tokens(self, ids: np.ndarray) -> np.ndarray:
        """Gather token embeddings as float32 [len, hidden] — touches only rows used.

        Indexes the embedding table through the mmap so only the requested token rows
        fault into memory. Materializing the whole [vocab, hidden] table here (vocab can be
        150k+) would cost hundreds of MB *per step* — the difference between snappy and
        seemingly-hung generation.
        """
        view = self.store[self.arch.tmpl["embed"]]
        out = view.rows(ids, dtype="float32")
        if self._embed_scale != 1.0:
            out = out * self._embed_scale
        return out

    def final_norm_weight(self) -> np.ndarray:
        return self.store[self.arch.tmpl["final_norm"]].array("float32")

    def lm_head_weight(self) -> np.ndarray:
        """Output projection [vocab, hidden]; falls back to tied embeddings."""
        head = self.arch.tmpl["lm_head"]
        if head in self.store:
            return self.store[head].array("float32")
        return self.store[self.arch.tmpl["embed"]].array("float32")

    # -- the streamer's per-block factory ------------------------------------------
    def build_layer(self, i: int) -> Dict[str, object]:
        """Materialize (and quantize) one transformer block's weights.

        Returns a dict of named arrays / QuantizedTensors consumed by `runtime`. Optional
        tensors (biases, qk-norms, gemma2 extra norms) are included only if present in the
        checkpoint, so the same code path serves many families.
        """
        out: Dict[str, object] = {}
        tmpl = self.arch.tmpl
        for short in _REQUIRED:
            view = self.store[tmpl[short].format(i=i)]
            w = view.array("float32")
            if self.rc.quant != "none" and short in _QUANTIZABLE:
                bits = 8 if self.rc.quant == "int8" else 4
                out[short] = quantize(w, bits=bits, group_size=self.rc.quant_group_size)
            else:
                out[short] = w
        for short in _OPTIONAL:
            name = tmpl[short].format(i=i)
            if name in self.store:
                out[short] = self.store[name].array("float32")
        return out
