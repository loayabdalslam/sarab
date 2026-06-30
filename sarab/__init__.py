"""SARAB — a streaming LLM runtime.

The premise that breaks the "you need thousands of GPUs" assumption: at any single
decode step only a small *active path* of the weights is needed. SARAB treats a model
as a **stream of computation flowing off disk**, not a static block that must fit in RAM.

Three mechanisms, working together:
  1. mmap'd weights      — the OS pages in only what is touched (see loader.mmap_store)
  2. layer-by-layer LRU  — one transformer block resident at a time (see loader.streamer)
  3. a prefetch thread   — layer N+1 loads off disk while layer N computes

Public surface:
    >>> from sarab import load
    >>> rt = load("meta-llama/Llama-3.2-1B-Instruct")
    >>> print(rt.generate("The capital of Egypt is", max_new_tokens=16))
"""

from .config import RuntimeConfig
from .hub import load

__all__ = ["RuntimeConfig", "load"]
__version__ = "0.1.0"
