from __future__ import annotations

from collections.abc import Sequence

from .base import ChatBackend, ChatRequest


class CountingBackend:
    """Pass-through wrapper that counts how many completions were requested (mutator accounting)."""

    def __init__(self, inner: ChatBackend) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.calls = 0

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        self.calls += len(requests)
        return self.inner.complete(requests)
