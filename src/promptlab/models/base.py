from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChatRequest:
    """One chat completion request. Messages are OpenAI-style dicts."""

    messages: list[dict[str, str]]
    max_tokens: int = 16
    temperature: float = 0.0
    meta: dict = field(default_factory=dict, compare=False)


class ChatBackend(Protocol):
    name: str
    model: str

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        """Return one completion string per request, in order."""
        ...
