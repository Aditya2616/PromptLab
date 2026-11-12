from __future__ import annotations

import re

import pandas as pd

from ..prompts import Prompt
from .base import Example, Task, split_indices

_YN = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)


def extract_yes_no(output: str) -> str | None:
    m = _YN.search(output)
    if not m:
        return None
    return "yes" if m.group(1).lower() in ("yes", "true") else "no"


class BoolQ(Task):
    name = "boolq"
    title = "BoolQ"
    description = "Yes/no questions about a short passage. Score: first yes/no/true/false in the reply equals the gold label."
    max_new_tokens = 8
    source = "google/boolq (CC BY-SA 3.0)"

    @classmethod
    def from_frames(
        cls, train: pd.DataFrame, validation: pd.DataFrame, dev_n: int = 80, test_n: int = 200,
        pool_n: int = 60, seed: int = 1234,
    ) -> BoolQ:
        def rows(df: pd.DataFrame, prefix: str) -> list[Example]:
            out = []
            for i in split_indices(len(df), seed):
                r = df.iloc[int(i)]
                passage = str(r["passage"])
                passage = passage if len(passage) <= 500 else passage[:500].rsplit(" ", 1)[0] + " ..."
                q = f"Passage: {passage}\nQuestion: {str(r['question']).strip()}?"
                out.append(Example(f"{prefix}{int(i)}", q, "yes" if bool(r["answer"]) else "no"))
            return out

        tr = rows(train, "tr")
        va = rows(validation, "va")
        return cls(tr[:pool_n], tr[pool_n : pool_n + dev_n], va[:test_n])

    def seeds(self) -> list[Prompt]:
        texts = [
            "You are a helpful assistant.",
            "Answer the question about the passage.",
            "Read the passage and answer the question.",
            "Decide whether the passage supports a yes or a no.",
            "You are a careful reader. Use only the passage to answer.",
            "Answer the question using the information in the passage.",
        ]
        return [Prompt.from_text(t) for t in texts]

    def format_bank(self) -> list[str]:
        return [
            "Reply with only 'yes' or 'no'.",
            "Answer yes or no and nothing else.",
            "Your whole reply must be one word: yes or no.",
            "Output yes or no only.",
            "",
        ]

    def score(self, output: str, ex: Example) -> int:
        return int(extract_yes_no(output) == ex.gold)

    def majority_rate(self) -> float | None:
        ys = sum(e.gold == "yes" for e in self._test)
        return max(ys, len(self._test) - ys) / max(1, len(self._test))
