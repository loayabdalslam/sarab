"""The inference engine: NumPy implementations of the transformer building blocks."""

from .architectures import ArchSpec, resolve_arch
from .layers import (
    rms_norm,
    per_head_rms_norm,
    softmax,
    silu,
    gelu,
    gelu_tanh,
    activation,
    gated_mlp,
    swiglu_mlp,
    apply_rope,
    RopeCache,
    attention,
    linear,
)
from .kvcache import KVCache
from .model import ModelConfig, SarabModel
from .runtime import Runtime

__all__ = [
    "ArchSpec",
    "resolve_arch",
    "rms_norm",
    "per_head_rms_norm",
    "softmax",
    "silu",
    "gelu",
    "gelu_tanh",
    "activation",
    "gated_mlp",
    "swiglu_mlp",
    "apply_rope",
    "RopeCache",
    "attention",
    "linear",
    "KVCache",
    "ModelConfig",
    "SarabModel",
    "Runtime",
]
