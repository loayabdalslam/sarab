"""The streaming forward pass — where the whole design comes together.

For each decode step we walk the transformer one block at a time. Each block's weights
arrive from `LayerStreamer.get(i)` (built lazily off the mmap, prefetched one ahead),
are used, and become eligible for eviction. Peak resident weights stay at
`resident_layers` blocks no matter how large the model is — that is the property that
lets a GPU-class model run inside an 8GB budget.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import RuntimeConfig
from ..loader.streamer import LayerStreamer
from . import layers as L
from .kvcache import KVCache
from .model import SarabModel


class Runtime:
    """Owns the streamer + KV cache and runs the forward pass.

    Stateless across sequences except for the KV cache, which `reset()` clears.
    """

    def __init__(self, model: SarabModel, rc: RuntimeConfig) -> None:
        self.model = model
        self.rc = rc
        self.cfg = model.cfg
        self.arch = model.arch
        self._act = L.activation(self.arch.activation)
        resident = self._effective_resident(rc)
        self.resident_layers = resident
        self.streamer = LayerStreamer(
            n_layers=self.cfg.n_layers,
            build_fn=model.build_layer,
            resident_layers=resident,
            prefetch_depth=rc.prefetch_depth,
            prefetch=rc.prefetch,
        )
        self.kv = KVCache(self.cfg.n_layers, rc.max_context)
        # cached persistent tensors (small): final norm + lm head
        self._final_norm: Optional[np.ndarray] = None
        self._lm_head: Optional[np.ndarray] = None

    def _effective_resident(self, rc: RuntimeConfig) -> int:
        """Choose how many blocks stay resident.

        A block is built (read off mmap + quantized) every time it is needed but not
        cached. With a tiny window, an evicted block is rebuilt — and re-quantized — on
        *every* token, which is catastrophically slow for a model that would have fit in
        RAM anyway. So when the whole (quantized) model fits ~60% of the budget we keep
        every layer resident and pay the build cost exactly once. Only genuinely
        over-budget models stream. `resident_layers` is always honoured as a floor.
        """
        floor = rc.resident_layers
        if not rc.auto_resident:
            return floor
        n = self.cfg.n_layers
        total_disk = self.model.store.total_bytes()
        per_layer = total_disk / max(1, n)
        # bytes a built layer occupies vs its fp16 on-disk size
        q_factor = {"none": 2.0, "int8": 0.5, "int4": 0.25}.get(rc.quant, 1.0)
        est_layer = per_layer * q_factor
        budget = rc.ram_budget_bytes * 0.6                  # leave room for kv/embeds/acts
        if est_layer * n <= budget:
            return n                                        # whole model fits -> no eviction
        fits = int(budget // max(1.0, est_layer))
        return max(floor, min(n, max(1, fits)))

    def reset(self) -> None:
        self.kv.reset()
        self.streamer.reset()

    def prewarm(self, progress: bool = True) -> None:
        """Build (and quantize) the resident layers up front, with a tqdm progress bar.

        Without this, every layer is built lazily inside the first forward pass, so a slow
        one-time quantize looks like a hang. Prewarming makes the cost visible and bounded.
        Only the resident layers are built (building more than fit would just evict them).
        """
        count = min(self.resident_layers, self.cfg.n_layers)
        iterator = range(count)
        if progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=count, unit="layer",
                                desc=f"building layers ({self.rc.quant})")
            except Exception:
                pass
        for i in iterator:
            self.streamer.get(i)
        # materialize the persistent tensors too (embeddings table is touched on first use)
        if self._final_norm is None:
            self._final_norm = self.model.final_norm_weight()
        if self._lm_head is None:
            self._lm_head = self.model.lm_head_weight()

    def close(self) -> None:
        self.streamer.close()

    # -- forward -------------------------------------------------------------------
    def forward(self, input_ids: np.ndarray, all_logits: bool = False) -> np.ndarray:
        """Run `input_ids` (1-D int array) through the model.

        Returns logits for the last position [vocab] by default, or for every position
        [seq, vocab] when `all_logits=True` (used by the fidelity test).
        """
        cfg = self.cfg
        arch = self.arch
        add_unit = arch.norm_add_unit
        ids = np.asarray(input_ids, dtype=np.int64).reshape(-1)
        seq = ids.shape[0]
        pos0 = self.kv.length
        positions = np.arange(pos0, pos0 + seq)
        cos, sin = self.model.rope.get(positions)                # [seq, head_dim]

        h = self.model.embed_tokens(ids).astype(np.float32)       # [seq, hidden]

        for i in range(cfg.n_layers):
            w = self.streamer.get(i)

            # --- attention block ---
            residual = h
            x = L.rms_norm(h, w["input_ln"], cfg.rms_eps, add_unit=add_unit)
            q = L.linear(x, w["q_proj"], w.get("q_bias"))
            k = L.linear(x, w["k_proj"], w.get("k_bias"))
            v = L.linear(x, w["v_proj"], w.get("v_bias"))
            q, k = self._rope_flat(q, k, cos, sin, w)
            offset = self.kv.offset(i)
            full_k, full_v = self.kv.extend(i, k, v)
            attn = L.attention(
                q, full_k, full_v, cfg.n_heads, cfg.n_kv_heads, causal_offset=offset,
                scale=cfg.attn_scale, logit_softcap=arch.attn_logit_softcap,
            )
            attn_out = L.linear(attn, w["o_proj"], w.get("o_bias"))
            if "post_ff_ln" in w and "pre_ff_ln" in w:
                # gemma2 layout: norm AFTER the attention sublayer, before residual add
                attn_out = L.rms_norm(attn_out, w["post_ln"], cfg.rms_eps, add_unit=add_unit)
            h = residual + attn_out

            # --- mlp block ---
            residual = h
            if "pre_ff_ln" in w:
                x = L.rms_norm(h, w["pre_ff_ln"], cfg.rms_eps, add_unit=add_unit)
            else:
                x = L.rms_norm(h, w["post_ln"], cfg.rms_eps, add_unit=add_unit)
            mlp_out = L.gated_mlp(
                x, w["gate_proj"], w["up_proj"], w["down_proj"], act=self._act
            )
            if "post_ff_ln" in w:
                mlp_out = L.rms_norm(mlp_out, w["post_ff_ln"], cfg.rms_eps, add_unit=add_unit)
            h = residual + mlp_out

        if self._final_norm is None:
            self._final_norm = self.model.final_norm_weight()
        h = L.rms_norm(h, self._final_norm, cfg.rms_eps, add_unit=add_unit)

        if self._lm_head is None:
            self._lm_head = self.model.lm_head_weight()           # [vocab, hidden]

        logits = (h @ self._lm_head.T) if all_logits else (h[-1] @ self._lm_head.T)
        if arch.final_logit_softcap is not None:
            cap = arch.final_logit_softcap
            logits = cap * np.tanh(logits / cap)
        return logits

    def _rope_flat(self, q_flat, k_flat, cos, sin, w):
        cfg = self.cfg
        seq = q_flat.shape[0]
        hd = cfg.head_dim
        q = q_flat.reshape(seq, cfg.n_heads, hd).transpose(1, 0, 2)
        k = k_flat.reshape(seq, cfg.n_kv_heads, hd).transpose(1, 0, 2)
        # optional Qwen3-style per-head qk RMSNorm (applied before RoPE)
        if "q_norm" in w:
            q = L.per_head_rms_norm(q, w["q_norm"], cfg.rms_eps)
        if "k_norm" in w:
            k = L.per_head_rms_norm(k, w["k_norm"], cfg.rms_eps)
        q_rot, k_rot = L.apply_rope(q, k, cos, sin)
        q_out = q_rot.transpose(1, 0, 2).reshape(seq, cfg.n_heads * hd)
        k_out = k_rot.transpose(1, 0, 2).reshape(seq, cfg.n_kv_heads * hd)
        return q_out, k_out
