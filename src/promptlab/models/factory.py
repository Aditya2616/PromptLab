from __future__ import annotations

import os

from .base import ChatBackend

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def make_backend(
    kind: str | None = None, model: str | None = None, device: str | None = None, dtype: str | None = None
) -> ChatBackend:
    kind = kind or os.environ.get("PROMPTLAB_BACKEND", "local")
    model = model or os.environ.get("PROMPTLAB_MODEL") or DEFAULT_MODEL
    if kind == "local":
        from .local_hf import LocalHFBackend

        return LocalHFBackend(model, device=device, dtype=dtype)
    if kind == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(model)
    if kind == "fake":
        from .fake import SimBackend

        return SimBackend()
    raise ValueError(f"unknown backend {kind!r}; use local, openai or fake")
