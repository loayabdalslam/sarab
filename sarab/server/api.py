"""An OpenAI-compatible FastAPI server over the streaming runtime.

Mirrors the subset of the OpenAI API that clients actually use — `/v1/models`,
`/v1/completions`, and `/v1/chat/completions` — with optional SSE streaming. The runtime
is single-instance and not thread-safe (one KV cache, one streamer), so requests are
serialized behind a lock. That is the honest constraint of a from-scratch CPU runtime;
batching/continuous-batching is a later-phase lever, not faked here.
"""

from __future__ import annotations

import json
import threading
import time
from typing import List, Optional

from ..config import RuntimeConfig


def build_app(model_id: str, cfg: RuntimeConfig):
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from ..hub import load

    app = FastAPI(title="SARAB", version="0.1.0")
    rt = load(model_id, config=cfg)
    lock = threading.Lock()           # serialize access to the single runtime

    class CompletionRequest(BaseModel):
        model: Optional[str] = model_id
        prompt: str
        max_tokens: int = 64
        temperature: float = 0.0
        top_p: float = 1.0
        stream: bool = False

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        model: Optional[str] = model_id
        messages: List[ChatMessage]
        max_tokens: int = 64
        temperature: float = 0.0
        top_p: float = 1.0
        stream: bool = False

    def _render_chat(messages: List[ChatMessage]) -> str:
        """Apply the tokenizer's chat template if present, else a simple fallback."""
        tok = rt.tokenizer
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        apply = getattr(tok, "apply_chat_template", None)
        if callable(apply):
            try:
                return apply(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        rendered = "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in msgs)
        return rendered + "<|assistant|>\n"

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_id, "object": "model"}]}

    @app.get("/health")
    def health():
        return {"status": "ok", "streamer": rt.runtime.streamer.stats}

    def _generate(prompt: str, max_tokens: int, temperature: float, top_p: float):
        return rt.generate(
            prompt, max_new_tokens=max_tokens, temperature=temperature,
            top_p=top_p, stream=True,
        )

    def _sse(pieces, kind: str):
        created = int(time.time())
        for piece in pieces:
            if kind == "chat":
                delta = {"choices": [{"index": 0, "delta": {"content": piece}}]}
            else:
                delta = {"choices": [{"index": 0, "text": piece}]}
            yield f"data: {json.dumps(delta)}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        with lock:
            if req.stream:
                pieces = _generate(req.prompt, req.max_tokens, req.temperature, req.top_p)
                return StreamingResponse(_sse(pieces, "text"),
                                         media_type="text/event-stream")
            text = "".join(_generate(req.prompt, req.max_tokens,
                                     req.temperature, req.top_p))
        return {
            "object": "text_completion",
            "model": model_id,
            "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
        }

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        prompt = _render_chat(req.messages)
        with lock:
            if req.stream:
                pieces = _generate(prompt, req.max_tokens, req.temperature, req.top_p)
                return StreamingResponse(_sse(pieces, "chat"),
                                         media_type="text/event-stream")
            text = "".join(_generate(prompt, req.max_tokens,
                                     req.temperature, req.top_p))
        return {
            "object": "chat.completion",
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    @app.on_event("shutdown")
    def _shutdown():
        rt.close()

    return app


def serve(model_id: str, cfg: RuntimeConfig, host: str = "127.0.0.1", port: int = 8000):
    import uvicorn

    app = build_app(model_id, cfg)
    print(f"[sarab] serving {model_id} on http://{host}:{port} "
          f"(OpenAI-compatible)")
    uvicorn.run(app, host=host, port=port)
