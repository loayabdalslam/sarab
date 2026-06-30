"""Token sampling and the autoregressive generation loop.

Kept separate from the engine so the runtime stays a pure logits producer and sampling
strategy can evolve independently. Greedy decoding is the default because it is what the
fidelity test compares against the HF reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional

import numpy as np

from .engine.runtime import Runtime


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0          # 0.0 -> greedy (argmax)
    top_p: float = 1.0
    top_k: int = 0                    # 0 -> disabled
    seed: int = 0
    stop_token_ids: Optional[List[int]] = None


def _sample_logits(logits: np.ndarray, p: SamplingParams, rng: np.random.Generator) -> int:
    if p.temperature <= 0.0:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64) / p.temperature
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    if p.top_k and p.top_k < probs.shape[0]:
        cut = np.argpartition(probs, -p.top_k)[:-p.top_k]
        probs[cut] = 0.0

    if p.top_p < 1.0:
        order = np.argsort(probs)[::-1]
        cumulative = np.cumsum(probs[order])
        keep = cumulative <= p.top_p
        keep[0] = True                # always keep the top token
        mask = np.zeros_like(probs, dtype=bool)
        mask[order[keep]] = True
        probs[~mask] = 0.0

    total = probs.sum()
    if total <= 0:
        return int(np.argmax(logits))
    probs /= total
    return int(rng.choice(probs.shape[0], p=probs))


def generate_ids(
    runtime: Runtime,
    prompt_ids: List[int],
    params: SamplingParams,
) -> Iterator[int]:
    """Yield generated token ids one at a time (prefill then decode)."""
    rng = np.random.default_rng(params.seed)
    runtime.reset()

    # Prefill: process the whole prompt in one forward, get logits for next token.
    logits = runtime.forward(np.asarray(prompt_ids, dtype=np.int64))
    stop = set(params.stop_token_ids or [])

    for _ in range(params.max_new_tokens):
        tok = _sample_logits(logits, params, rng)
        yield tok
        if tok in stop:
            return
        # Decode: feed back just the new token; KV cache carries the context.
        logits = runtime.forward(np.asarray([tok], dtype=np.int64))
