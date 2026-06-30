"""Architecture registry — lets SARAB run many model families, not just Llama.

Most modern open decoder LLMs share one skeleton (pre-norm transformer block with rotary
attention + gated MLP) and differ only in a handful of knobs: whether the attention
projections carry a bias, which activation the MLP uses, whether queries/keys are
RMS-normed, and whether embeddings are scaled. `ArchSpec` captures exactly those knobs,
and `resolve_arch()` maps an HF config to the right spec by `model_type`.

Supported out of the box (decoder-only, RoPE + RMSNorm family):
    llama, mistral, qwen2, qwen3, gemma, gemma2, phi3, stablelm, starcoder2, ...
Anything that declares the standard Llama tensor names works via the default spec; an
unknown `model_type` falls back to that default with a warning rather than failing, so a
new model usually "just works". Genuinely different skeletons (GPT-2 learned-pos,
encoder-decoder, Mamba/SSM, MoE routing) are out of scope for this spec and are reported
clearly instead of silently producing garbage.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class ArchSpec:
    """The per-family knobs the engine consults during the forward pass."""

    name: str
    attn_bias: bool = False             # qkv (and sometimes o) projections have a bias
    o_bias: bool = False
    mlp_bias: bool = False
    activation: str = "silu"            # "silu" | "gelu" | "gelu_tanh"
    qk_norm: bool = False               # per-head RMSNorm on q and k (qwen3, gemma2-ish)
    embed_scale: bool = False           # multiply embeddings by sqrt(hidden) (gemma)
    norm_add_unit: bool = False         # RMSNorm uses (1 + weight) (gemma)
    final_logit_softcap: Optional[float] = None   # gemma2 logit soft-capping
    attn_logit_softcap: Optional[float] = None

    # weight-name templates (HF convention by default; override per family if needed)
    tmpl: Dict[str, str] = field(default_factory=dict)


# Canonical HF Llama-style tensor names shared by almost every supported family.
_LLAMA_TMPL: Dict[str, str] = {
    "q_proj": "model.layers.{i}.self_attn.q_proj.weight",
    "k_proj": "model.layers.{i}.self_attn.k_proj.weight",
    "v_proj": "model.layers.{i}.self_attn.v_proj.weight",
    "o_proj": "model.layers.{i}.self_attn.o_proj.weight",
    "q_bias": "model.layers.{i}.self_attn.q_proj.bias",
    "k_bias": "model.layers.{i}.self_attn.k_proj.bias",
    "v_bias": "model.layers.{i}.self_attn.v_proj.bias",
    "o_bias": "model.layers.{i}.self_attn.o_proj.bias",
    "q_norm": "model.layers.{i}.self_attn.q_norm.weight",
    "k_norm": "model.layers.{i}.self_attn.k_norm.weight",
    "gate_proj": "model.layers.{i}.mlp.gate_proj.weight",
    "up_proj": "model.layers.{i}.mlp.up_proj.weight",
    "down_proj": "model.layers.{i}.mlp.down_proj.weight",
    "input_ln": "model.layers.{i}.input_layernorm.weight",
    "post_ln": "model.layers.{i}.post_attention_layernorm.weight",
    # gemma2 extra norms (optional; engine uses them only if present)
    "pre_ff_ln": "model.layers.{i}.pre_feedforward_layernorm.weight",
    "post_ff_ln": "model.layers.{i}.post_feedforward_layernorm.weight",
    "embed": "model.embed_tokens.weight",
    "final_norm": "model.norm.weight",
    "lm_head": "lm_head.weight",
}


def _spec(**kw) -> ArchSpec:
    tmpl = dict(_LLAMA_TMPL)
    return ArchSpec(tmpl=tmpl, **kw)


# model_type -> spec
_REGISTRY: Dict[str, ArchSpec] = {
    "llama": _spec(name="llama"),
    "mistral": _spec(name="mistral"),
    "mixtral": _spec(name="mistral"),          # dense path; MoE routing not yet handled
    "qwen2": _spec(name="qwen2", attn_bias=True),
    "qwen3": _spec(name="qwen3", qk_norm=True),
    "phi3": _spec(name="phi3"),
    "stablelm": _spec(name="stablelm"),
    "starcoder2": _spec(name="starcoder2", attn_bias=True, o_bias=True, mlp_bias=True),
    "gemma": _spec(name="gemma", activation="gelu_tanh", embed_scale=True,
                   norm_add_unit=True),
    "gemma2": _spec(name="gemma2", activation="gelu_tanh", embed_scale=True,
                    norm_add_unit=True),
}

_DEFAULT = _spec(name="default")

# model_types that share the skeleton but we know need work we haven't done -> hard error
_UNSUPPORTED = {
    "gpt2": "learned positional embeddings + LayerNorm (not RoPE/RMSNorm)",
    "gpt_neox": "different block layout",
    "bert": "encoder-only",
    "t5": "encoder-decoder",
    "mamba": "state-space, not attention",
}


def resolve_arch(hf_cfg: dict) -> ArchSpec:
    """Pick an ArchSpec from an HF config dict (by `model_type`)."""
    mt = (hf_cfg.get("model_type") or "").lower()

    if mt in _UNSUPPORTED:
        raise NotImplementedError(
            f"architecture '{mt}' is out of scope for SARAB's RoPE+RMSNorm engine "
            f"({_UNSUPPORTED[mt]}). It needs a dedicated adapter."
        )

    spec = _REGISTRY.get(mt)
    if spec is not None:
        # honor an explicit attention_bias flag even on families that usually lack it
        if hf_cfg.get("attention_bias") and not spec.attn_bias:
            spec = ArchSpec(**{**spec.__dict__, "attn_bias": True})
        return spec

    warnings.warn(
        f"unknown model_type '{mt}'; assuming standard Llama-style decoder. "
        f"If outputs look wrong, this architecture needs its own ArchSpec.",
        stacklevel=2,
    )
    # respect config-declared bias even for unknown types
    if hf_cfg.get("attention_bias"):
        return ArchSpec(**{**_DEFAULT.__dict__, "attn_bias": True})
    return _DEFAULT
