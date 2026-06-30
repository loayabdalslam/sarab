"""Group-wise int8 / int4 weight quantization with vectorized dequant.

Why quantize at all: it shrinks the on-disk and resident footprint of every weight
matrix, which is exactly the budget SARAB is fighting for. int8 halves float16 and keeps
accuracy essentially intact; int4 halves again at a small, measured fidelity cost.

Scheme — symmetric, group-wise along the input dimension:
  * A weight row is split into contiguous groups of `group_size` elements.
  * Each group gets one float32 `scale = max(|w|)/qmax`.
  * Stored value `q = round(w / scale)` clamped to the signed range.
  * Reconstruction `w ~= q * scale`.

int4 values are packed two-per-byte so the stored array is genuinely half the size of
int8. Dequant is a single vectorized multiply broadcast over groups — SIMD-friendly and
trivial to port to a Rust/C++ kernel later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

QBits = Literal[4, 8]


@dataclass
class QuantizedTensor:
    """A quantized 2-D weight matrix plus the metadata to reconstruct it.

    Shapes (logical matrix is [out_features, in_features]):
        q       : int8 codes. For 4-bit, packed two-per-byte along the last axis, so its
                  stored shape is [out_features, in_features // 2].
        scales  : float32 [out_features, n_groups]  (one scale per group)
        bits    : 4 or 8
        group_size, in_features : needed to unpack/broadcast correctly.

    `float()` lazily dequantizes once and caches the result, so a *resident* layer pays
    the dequant cost a single time instead of on every token. A layer that is streamed and
    evicted is rebuilt fresh each time, so its cache is naturally short-lived — no memory
    blow-up. The cache is excluded from equality/repr.
    """

    q: np.ndarray
    scales: np.ndarray
    bits: QBits
    group_size: int
    in_features: int
    out_features: int
    _float_cache: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )

    @property
    def nbytes(self) -> int:
        return int(self.q.nbytes + self.scales.nbytes)

    @property
    def compression_vs_f16(self) -> float:
        return (self.in_features * self.out_features * 2) / max(1, self.nbytes)

    def float(self) -> np.ndarray:
        """Dequantized float32 matrix [out, in], computed once and cached."""
        if self._float_cache is None:
            self._float_cache = dequantize(self)
        return self._float_cache


def _qmax(bits: QBits) -> int:
    # symmetric signed range: int8 -> 127, int4 -> 7
    return (1 << (bits - 1)) - 1


def quantize(w: np.ndarray, bits: QBits = 8, group_size: int = 64) -> QuantizedTensor:
    """Quantize a 2-D weight matrix [out_features, in_features].

    `in_features` need not be divisible by `group_size`; the final short group is handled
    by padding the scale broadcast (the padded tail contributes nothing because it is
    sliced off after dequant).
    """
    if w.ndim != 2:
        raise ValueError(f"expected 2-D weight, got shape {w.shape}")
    w = np.ascontiguousarray(w, dtype=np.float32)
    out_features, in_features = w.shape
    qmax = _qmax(bits)

    n_groups = (in_features + group_size - 1) // group_size
    pad = n_groups * group_size - in_features
    if pad:
        w_pad = np.pad(w, ((0, 0), (0, pad)), mode="constant")
    else:
        w_pad = w

    # reshape to [out, n_groups, group_size] and derive a per-group scale
    grouped = w_pad.reshape(out_features, n_groups, group_size)
    absmax = np.abs(grouped).max(axis=2)                       # [out, n_groups]
    scales = (absmax / qmax).astype(np.float32)
    scales[scales == 0.0] = 1.0                                # avoid div-by-zero

    q = np.round(grouped / scales[:, :, None]).astype(np.int32)
    q = np.clip(q, -qmax, qmax).astype(np.int8)
    q = q.reshape(out_features, n_groups * group_size)[:, :in_features]

    if bits == 4:
        q = _pack_int4(q)

    return QuantizedTensor(
        q=np.ascontiguousarray(q),
        scales=np.ascontiguousarray(scales),
        bits=bits,
        group_size=group_size,
        in_features=in_features,
        out_features=out_features,
    )


def dequantize(qt: QuantizedTensor, dtype=np.float32) -> np.ndarray:
    """Reconstruct the float matrix [out_features, in_features]."""
    q = _unpack_int4(qt.q, qt.in_features) if qt.bits == 4 else qt.q
    out_features, in_features = qt.out_features, qt.in_features
    gs = qt.group_size

    n_groups = qt.scales.shape[1]
    pad = n_groups * gs - in_features
    if pad:
        q = np.pad(q.astype(np.int16), ((0, 0), (0, pad)), mode="constant")
    grouped = q.reshape(out_features, n_groups, gs).astype(np.float32)
    w = grouped * qt.scales[:, :, None]
    w = w.reshape(out_features, n_groups * gs)[:, :in_features]
    return w.astype(dtype, copy=False)


def quantized_matmul(x: np.ndarray, qt: QuantizedTensor) -> np.ndarray:
    """Compute x @ W.T where W is the quantized [out, in] matrix.

    Dequant-then-matmul: we reconstruct W in float32 and hand off to BLAS. This keeps
    accuracy identical to a float reference up to the quantization error of W itself, and
    BLAS is far faster than any hand-rolled integer GEMM in pure NumPy. (A fused int GEMM
    is exactly the kind of kernel we'd later move to Rust/SIMD.)

    x : [..., in_features]  ->  returns [..., out_features]
    """
    w = qt.float()                           # [out, in], dequantized once and cached
    return x @ w.T


# -- int4 packing --------------------------------------------------------------------
def _pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack signed int4 values (range [-7,7]) two-per-byte.

    We bias by +8 into [1,15] (0..15 fits a nibble), low nibble = even column.
    """
    out_features, in_features = q.shape
    if in_features % 2 == 1:
        q = np.pad(q, ((0, 0), (0, 1)), mode="constant")
    biased = (q.astype(np.int16) + 8).astype(np.uint8)         # [-7,7] -> [1,15]
    low = biased[:, 0::2]
    high = biased[:, 1::2]
    packed = (low | (high << 4)).astype(np.uint8)
    return np.ascontiguousarray(packed)


def _unpack_int4(packed: np.ndarray, in_features: int) -> np.ndarray:
    """Inverse of `_pack_int4`, returning signed int8 codes trimmed to in_features."""
    low = (packed & 0x0F).astype(np.int16) - 8
    high = ((packed >> 4) & 0x0F).astype(np.int16) - 8
    out = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out[:, :in_features]
