from __future__ import annotations

import re

import pandas as pd

from ..prompts import Prompt
from .base import Example, Task, split_indices

_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*\(?([A-E])\b", re.IGNORECASE)
_LEADING = re.compile(r"^\W*\(?([A-E])\b")


def extract_letter(output: str) -> str | None:
    """Strict-ish extractor: 'Answer: B' / 'the answer is B' anywhere, else a leading letter."""
    m = _ANSWER_IS.search(output)
    if m:
        return m.group(1).upper()
    m = _LEADING.match(output)
    return m.group(1) if m else None


class CommonsenseQA(Task):
    name = "csqa"
    title = "CommonsenseQA"
    description = "Five-way commonsense multiple choice. Score: extracted option letter equals the gold letter."
    max_new_tokens = 12
    source = "tau/commonsense_qa (MIT)"

    @classmethod
    def from_frames(
        cls, train: pd.DataFrame, validation: pd.DataFrame, dev_n: int = 80, test_n: int = 200,
        pool_n: int = 60, seed: int = 1234,
    ) -> CommonsenseQA:
        def rows(df: pd.DataFrame, prefix: str) -> list[Example]:
            out = []
            for i in split_indices(len(df), seed):
                r = df.iloc[int(i)]
                opts = "\n".join(
                    f"{lab}. {txt}"
                    for lab, txt in zip(r["choices"]["label"], r["choices"]["text"], strict=True)
                )
                out.append(Example(f"{prefix}{r['id']}", f"{r['question']}\n{opts}", str(r["answerKey"])))
            return out

        tr = rows(train, "tr")
        va = rows(validation, "va")
        return cls(tr[:pool_n], tr[pool_n : pool_n + dev_n], va[:test_n])

    def seeds(self) -> list[Prompt]:
        texts = [
            "You are a helpful assistant.",
            "Answer the question.",
            "Answer the following multiple-choice question.",
            "Think carefully and choose the best option.",
            "You are an expert at commonsense reasoning. Pick the most plausible answer.",
            "Read the question and the options, then respond with the correct option.",
        ]
        return [Prompt.from_text(t) for t in texts]

    def format_bank(self) -> list[str]:
        return [
            "Reply with only the letter of the correct option.",
            "Answer with a single letter (A-E) and nothing else.",
            "Respond in the form 'Answer: X'.",
            "Output the option letter only.",
            "",
        ]

    def score(self, output: str, ex: Example) -> int:
        return int(extract_letter(output) == ex.gold)
