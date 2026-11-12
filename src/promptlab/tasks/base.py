from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..prompts import Prompt

# Task-agnostic instruction clauses the "add_clause" operator may append.
GENERIC_CLAUSES = [
    "Be concise.",
    "Do not explain your answer.",
    "Read the question carefully.",
    "Reply with only the final answer.",
    "Do not add any extra text.",
    "If you are unsure, give your best guess.",
    "Answer in as few words as possible.",
    "Do not use punctuation in your answer.",
    "Be precise.",
    "Focus on the key information.",
]


@dataclass(frozen=True)
class Example:
    id: str
    question: str
    gold: str
    meta: dict = field(default_factory=dict, compare=False, hash=False)


class Task(ABC):
    """A dataset plus a deterministic scorer. Splits: few-shot pool, dev (search), test (report)."""

    name: str = "task"
    title: str = "Task"
    description: str = ""
    max_new_tokens: int = 12
    source: str = ""

    def __init__(self, pool: list[Example], dev: list[Example], test: list[Example]) -> None:
        self.pool = pool
        self.dev = dev
        self._test = test
        self._locked = False
        self._by_id = {e.id: e for e in [*pool, *dev, *test]}

    # -- test split guard: the search loop must never see it -----------------
    def lock_test(self) -> None:
        self._locked = True

    def unlock_test(self) -> None:
        self._locked = False

    @property
    def test(self) -> list[Example]:
        if self._locked:
            raise PermissionError("test split is locked until the final report")
        return self._test

    @property
    def n_test(self) -> int:
        return len(self._test)

    def example(self, ex_id: str) -> Example:
        return self._by_id[ex_id]

    # -- task definition -----------------------------------------------------
    @abstractmethod
    def seeds(self) -> list[Prompt]: ...

    @abstractmethod
    def format_bank(self) -> list[str]: ...

    @abstractmethod
    def score(self, output: str, ex: Example) -> int: ...

    def shot_answer(self, ex: Example) -> str:
        return ex.gold

    def clause_bank(self) -> list[str]:
        return GENERIC_CLAUSES

    def messages(self, prompt: Prompt, ex: Example) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if prompt.system:
            msgs.append({"role": "system", "content": prompt.system})
        for sid in prompt.shots:
            s = self.example(sid)
            msgs.append({"role": "user", "content": s.question})
            msgs.append({"role": "assistant", "content": self.shot_answer(s)})
        msgs.append({"role": "user", "content": ex.question})
        return msgs

    def prompt_text(self, prompt: Prompt) -> str:
        """Human-readable rendering used in the UI and in diffs."""
        lines = [prompt.system] if prompt.system else ["(empty system prompt)"]
        for sid in prompt.shots:
            s = self.example(sid)
            q = s.question.replace("\n", " / ")
            q = q if len(q) <= 90 else q[:87] + "..."
            lines.append(f"Example: {q}  =>  {self.shot_answer(s)}")
        return "\n".join(lines)

    def majority_rate(self) -> float | None:
        return None


def split_indices(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(n)
