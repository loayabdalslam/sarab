"""Runtime configuration for SARAB.

A single dataclass that every component reads from, so the RAM budget and streaming
behaviour are tuned in one place rather than scattered through the engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal, Optional

QuantKind = Literal["none", "int8", "int4"]


@dataclass
class RuntimeConfig:
    """Knobs that govern how aggressively SARAB streams and quantizes.

    Defaults target an 8GB-RAM machine: keep peak resident weights small, prefetch one
    block ahead, and quantize to int8 (safe accuracy) by default.
    """

    # --- memory budget ---------------------------------------------------------------
    ram_budget_gb: float = 4.0
    """Soft ceiling for resident model weights. The streamer's LRU window is sized so
    peak weight residency stays under this. KV cache and activations are accounted
    separately and are tiny by comparison for short contexts."""

    # --- streaming -------------------------------------------------------------------
    resident_layers: int = 2
    """How many transformer blocks may be decompressed in RAM at once (LRU window).
    1 = absolute minimum residency; 2 lets the prefetcher stage the next block while the
    current one computes without immediately evicting it. Acts as a *floor*: when
    `auto_resident` is on it may be raised so a model that fits the budget stays fully
    resident (avoiding pointless per-token re-quantization)."""

    auto_resident: bool = True
    """If True, size the resident window from the model size and `ram_budget_gb`: keep
    ALL layers resident when the (quantized) model fits the budget, otherwise stream with
    as many resident layers as fit. This is the difference between a small model running
    fast and re-quantizing every layer on every token. Set False to force exactly
    `resident_layers`."""

    prefetch_depth: int = 1
    """How many layers ahead the prefetch thread loads off disk. 0 disables prefetch
    (useful to measure the I/O stall the prefetcher hides)."""

    prefetch: bool = True
    """Master switch for the background prefetch thread."""

    # --- numeric ---------------------------------------------------------------------
    quant: QuantKind = "int8"
    """Weight quantization applied at load time. int8 is the accuracy-safe default;
    int4 halves disk/RAM again at a small fidelity cost."""

    quant_group_size: int = 64
    """Group size for group-wise quantization (per-group scale/zero-point). Smaller =
    more accurate, slightly larger metadata."""

    compute_dtype: str = "float32"
    """dtype used for the actual matmuls after dequant. float32 is the safe reference;
    we keep accumulation in float32 regardless to protect accuracy."""

    # --- generation ------------------------------------------------------------------
    max_context: int = 4096
    """Hard cap on KV-cache length; older tokens are dropped past this."""

    seed: int = 0

    # --- paths -----------------------------------------------------------------------
    cache_dir: Optional[str] = None
    """Where HF artifacts live. None -> default huggingface_hub cache."""

    device: str = "cpu"
    """SARAB is a CPU runtime; this exists for forward-compat and assertions."""

    threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    """Worker threads available to the compute kernels (NumPy/BLAS honour this)."""

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("SARAB is a CPU-first runtime; device must be 'cpu'.")
        if self.resident_layers < 1:
            raise ValueError("resident_layers must be >= 1")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be >= 0")
        if self.quant not in ("none", "int8", "int4"):
            raise ValueError(f"unknown quant kind: {self.quant}")

    @property
    def ram_budget_bytes(self) -> int:
        return int(self.ram_budget_gb * 1024**3)
