from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

from .base import ChatRequest


def _u(*parts: str) -> float:
    h = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


class ScriptedBackend:
    """Backend driven by a plain function; used in unit tests."""

    name = "scripted"
    model = "scripted"

    def __init__(self, fn: Callable[[ChatRequest], str]) -> None:
        self.fn = fn
        self.calls = 0

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        self.calls += len(requests)
        return [self.fn(r) for r in requests]


class SimBackend:
    """A deterministic stand-in for a model, for tests and `--backend fake`.

    It is NOT a language model and its numbers mean nothing. Accuracy rises with
    format-constraining wording and few-shot turns, so the evolutionary loop has
    something to climb. Call ``bind(task)`` so it knows the gold answers.
    """

    name = "sim"
    model = "sim-v1"

    def __init__(self) -> None:
        self.gold: dict[str, str] = {}
        self.shot_fmt: Callable[[str], str] = lambda g: g

    def bind(self, task) -> None:
        for ex in [*task.pool, *task.dev, *task._test]:
            self.gold[ex.question] = task.shot_answer(ex)

    def _answer(self, req: ChatRequest) -> str:
        system = req.messages[0]["content"] if req.messages[0]["role"] == "system" else ""
        user = req.messages[-1]["content"]
        if "Output only the new instruction" in user:
            base = user.split("Instruction:")[-1].strip().split("\n")[0][:120] if "Instruction:" in user else "Answer."
            extra = ["Reply with only the answer.", "Be concise.", "Do not explain."]
            return base + " " + extra[int(_u(user) * 3) % 3]
        gold = self.gold.get(user, "A")
        shots = sum(1 for m in req.messages if m["role"] == "assistant")
        p = 0.18 + 0.30 * ("only" in system or "nothing else" in system) + 0.10 * min(shots, 3)
        p += 0.06 * ("concise" in system.lower())
        return gold if _u(system, user, str(shots)) < min(p, 0.95) else f"I think it might be {gold}x"

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        return [self._answer(r) for r in requests]
