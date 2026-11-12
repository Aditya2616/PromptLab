from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


def split_sentences(text: str) -> tuple[str, ...]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return tuple(p.strip() for p in parts if p.strip())


@dataclass(frozen=True)
class Prompt:
    """A prompt variant: instruction sentences, an optional format constraint and few-shot ids.

    Keeping the instruction as a tuple of sentences is what lets deterministic
    operators (reorder, drop, crossover) work without an LLM.
    """

    sentences: tuple[str, ...]
    fmt: str = ""
    shots: tuple[str, ...] = ()

    @property
    def system(self) -> str:
        parts = [*self.sentences, self.fmt] if self.fmt else list(self.sentences)
        return " ".join(p for p in parts if p)

    @property
    def id(self) -> str:
        blob = json.dumps([list(self.sentences), self.fmt, list(self.shots)], ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]

    @property
    def n_words(self) -> int:
        return len(self.system.split())

    @classmethod
    def from_text(cls, text: str, fmt: str = "", shots: tuple[str, ...] = ()) -> Prompt:
        return cls(split_sentences(text), fmt, shots)
