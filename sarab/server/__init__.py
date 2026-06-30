"""OpenAI-compatible HTTP server — the vLLM-style serving surface."""

from .api import build_app, serve

__all__ = ["build_app", "serve"]
