"""Stateless transformer primitives in pure NumPy (Llama-family semantics).

Everything here accumulates in float32 to protect accuracy regardless of how the weights
were stored. A "weight" passed to `linear` may be a plain float ndarray or a
`QuantizedTensor`; `linear` dispatches so the rest of the engine never branches on it.

The reference semantics match Hugging Face's Llama implementation: RMSNorm (no bias),
rotary position embeddings (optionally with the "llama3" frequency rescaling), grouped
-query attention, and a SwiGLU MLP.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from ..quant.int4 import QuantizedTensor, quantized_matmul

Weight = Union[np.ndarray, QuantizedTensor]


def linear(x: np.ndarray, w: Weight, bias: Optional[np.ndarray] = None) -> np.ndarray:
    """y = x @ W.T (+ bias) for a [out, in] weight, accepting float or quantized W."""
    if isinstance(w, QuantizedTensor):
        y = quantized_matmul(x, w)
    else:
        y = x @ w.T
    if bias is not None:
        y = y + bias.astype(np.float32, copy=False)
    return y


def rms_norm(
    x: np.ndarray, weight: np.ndarray, eps: float, add_unit: bool = False
) -> np.ndarray:
    """RMSNorm in float32. `add_unit=True` uses (1 + weight) scaling (Gemma)."""
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 / np.sqrt(var + eps)
    w = weight.astype(np.float32, copy=False)
    if add_unit:
        w = w + 1.0
    return normed * w


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU via the error function."""
    from math import sqrt
    # 0.5*x*(1+erf(x/sqrt(2))); use np.erf if available else tanh approx fallback
    try:
        from scipy.special import erf  # type: ignore
        return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
    except Exception:
        return gelu_tanh(x)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    """tanh-approximation GELU (what Gemma/most HF 'gelu_pytorch_tanh' use)."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def activation(name: str):
    """Resolve an activation function by name (silu/gelu/gelu_tanh), default silu."""
    return {"silu": silu, "gelu": gelu, "gelu_tanh": gelu_tanh}.get(name, silu)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    np.exp(x, out=x)
    x /= np.sum(x, axis=axis, keepdims=True)
    return x


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def gated_mlp(
    x: np.ndarray,
    gate_w: Weight,
    up_w: Weight,
    down_w: Weight,
    act=silu,
    gate_b: Optional[np.ndarray] = None,
    up_b: Optional[np.ndarray] = None,
    down_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gated MLP: down( act(x@gate) * (x@up) ). act defaults to SiLU (SwiGLU)."""
    g = act(linear(x, gate_w, gate_b))
    u = linear(x, up_w, up_b)
    return linear(g * u, down_w, down_b)


def swiglu_mlp(
    x: np.ndarray, gate_w: Weight, up_w: Weight, down_w: Weight
) -> np.ndarray:
    """SwiGLU convenience wrapper (SiLU-gated MLP)."""
    return gated_mlp(x, gate_w, up_w, down_w, act=silu)


class RopeCache:
    """Precomputed rotary cos/sin tables, with optional llama3 frequency rescaling.

    Llama 3.x rescales the inverse frequencies so the model extrapolates to long context;
    reproducing it exactly is required to match HF logits on Llama-3.2 checkpoints.
    """

    def __init__(
        self,
        head_dim: int,
        max_pos: int,
        theta: float = 10000.0,
        scaling: Optional[dict] = None,
    ) -> None:
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        if scaling and scaling.get("rope_type", scaling.get("type")) == "llama3":
            inv_freq = self._llama3_rescale(inv_freq, scaling)
        positions = np.arange(max_pos, dtype=np.float64)
        freqs = np.outer(positions, inv_freq)                    # [max_pos, head_dim/2]
        emb = np.concatenate([freqs, freqs], axis=-1)            # [max_pos, head_dim]
        self.cos = np.cos(emb).astype(np.float32)
        self.sin = np.sin(emb).astype(np.float32)

    @staticmethod
    def _llama3_rescale(inv_freq: np.ndarray, cfg: dict) -> np.ndarray:
        factor = cfg.get("factor", 8.0)
        low_freq_factor = cfg.get("low_freq_factor", 1.0)
        high_freq_factor = cfg.get("high_freq_factor", 4.0)
        old_ctx = cfg.get("original_max_position_embeddings", 8192)

        low_wavelen = old_ctx / low_freq_factor
        high_wavelen = old_ctx / high_freq_factor
        wavelen = 2 * math.pi / inv_freq

        new_freq = inv_freq.copy()
        # long wavelengths (low freq): divide by factor
        mask_low = wavelen > low_wavelen
        new_freq[mask_low] = inv_freq[mask_low] / factor
        # medium band: smooth interpolation
        smooth = (old_ctx / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        interp = (1 - smooth) * (inv_freq / factor) + smooth * inv_freq
        mask_mid = (~mask_low) & (wavelen >= high_wavelen)
        new_freq[mask_mid] = interp[mask_mid]
        return new_freq

    def get(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """cos/sin for the given absolute positions -> [len, head_dim]."""
        return self.cos[positions], self.sin[positions]


def per_head_rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """RMSNorm applied over the head_dim of [n_heads, seq, head_dim] (Qwen3-style qk-norm)."""
    x32 = x.astype(np.float32, copy=False)
    var = np.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 / np.sqrt(var + eps)
    return normed * weight.astype(np.float32, copy=False)


def _rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope(
    q: np.ndarray, k: np.ndarray, cos: np.ndarray, sin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rotary embeddings. q,k: [n_heads, seq, head_dim]; cos,sin: [seq, head_dim]."""
    cos = cos[None, :, :]
    sin = sin[None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_heads: int,
    n_kv_heads: int,
    causal_offset: int,
    scale: Optional[float] = None,
    logit_softcap: Optional[float] = None,
) -> np.ndarray:
    """Grouped-query scaled-dot-product attention.

    q : [seq, n_heads*head_dim]   k,v : [total, n_kv_heads*head_dim]
    `causal_offset` is the absolute position of the first query (= KV length before this
    step), so the causal mask aligns query i with keys [0 .. causal_offset+i].
    `scale` overrides 1/sqrt(head_dim) (some models use a query_pre_attn_scalar).
    `logit_softcap` applies tanh soft-capping to scores (Gemma2). Returns
    [seq, n_heads*head_dim].
    """
    seq = q.shape[0]
    total = k.shape[0]
    head_dim = q.shape[-1] // n_heads
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    group = n_heads // n_kv_heads

    qh = q.reshape(seq, n_heads, head_dim).transpose(1, 0, 2)        # [H, seq, d]
    kh = k.reshape(total, n_kv_heads, head_dim).transpose(1, 0, 2)   # [KV, total, d]
    vh = v.reshape(total, n_kv_heads, head_dim).transpose(1, 0, 2)

    out = np.empty((n_heads, seq, head_dim), dtype=np.float32)
    # causal mask: query row i (absolute pos causal_offset+i) attends keys j<=offset+i
    q_pos = causal_offset + np.arange(seq)[:, None]
    k_pos = np.arange(total)[None, :]
    mask = k_pos > q_pos                                              # True = disallow

    for h in range(n_heads):
        kv = h // group
        scores = (qh[h] @ kh[kv].T) * scale                          # [seq, total]
        if logit_softcap is not None:
            scores = logit_softcap * np.tanh(scores / logit_softcap)
        scores[mask] = -np.inf
        probs = softmax(scores, axis=-1)
        out[h] = probs @ vh[kv]                                       # [seq, d]

    return out.transpose(1, 0, 2).reshape(seq, n_heads * head_dim)
