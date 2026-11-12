"""Mutation operators. Deterministic ones edit the structured prompt; LLM ones ask the model to rewrite.

Every operator returns a new ``Prompt`` or ``None`` (could not apply). Guards live in
``guard`` so an operator can stay simple and the loop can reject junk uniformly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .models.base import ChatBackend, ChatRequest
from .prompts import Prompt, split_sentences
from .tasks.base import Task

MAX_SHOTS = 3
MAX_WORDS = 70


@dataclass
class Parent:
    prompt: Prompt
    scores: np.ndarray | None = None
    outputs: list[str] | None = None


@dataclass
class OpContext:
    task: Task
    backend: ChatBackend
    rng: np.random.Generator
    max_words: int = MAX_WORDS


class Operator:
    name = "operator"
    kind = "deterministic"  # or "llm"
    arity = 1
    needs_feedback = False

    def apply(self, parents: list[Parent], ctx: OpContext) -> Prompt | None:
        raise NotImplementedError


def guard(child: Prompt | None, parents: list[Parent], max_words: int = MAX_WORDS) -> Prompt | None:
    """Reject empty, oversized, or unchanged children."""
    if child is None:
        return None
    if not child.sentences and not child.fmt:
        return None
    if child.n_words > max_words or child.n_words < 2:
        return None
    if any(child.id == p.prompt.id for p in parents):
        return None
    return child


class Reorder(Operator):
    name = "reorder"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        if len(p.sentences) < 2:
            return None
        perm = ctx.rng.permutation(len(p.sentences))
        return Prompt(tuple(p.sentences[i] for i in perm), p.fmt, p.shots)


class DropClause(Operator):
    name = "drop_clause"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        if len(p.sentences) < 2:
            return None
        i = int(ctx.rng.integers(len(p.sentences)))
        return Prompt(p.sentences[:i] + p.sentences[i + 1 :], p.fmt, p.shots)


class AddClause(Operator):
    name = "add_clause"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        bank = [c for c in ctx.task.clause_bank() if c not in p.sentences]
        if not bank:
            return None
        clause = bank[int(ctx.rng.integers(len(bank)))]
        pos = int(ctx.rng.integers(len(p.sentences) + 1))
        s = list(p.sentences)
        s.insert(pos, clause)
        return Prompt(tuple(s), p.fmt, p.shots)


class SwapFormat(Operator):
    name = "swap_format"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        bank = [f for f in ctx.task.format_bank() if f != p.fmt]
        if not bank:
            return None
        return Prompt(p.sentences, bank[int(ctx.rng.integers(len(bank)))], p.shots)


class AddShot(Operator):
    name = "add_shot"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        if len(p.shots) >= MAX_SHOTS:
            return None
        free = [e.id for e in ctx.task.pool if e.id not in p.shots]
        if not free:
            return None
        return Prompt(p.sentences, p.fmt, (*p.shots, free[int(ctx.rng.integers(len(free)))]))


class RemoveShot(Operator):
    name = "remove_shot"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        if not p.shots:
            return None
        i = int(ctx.rng.integers(len(p.shots)))
        return Prompt(p.sentences, p.fmt, p.shots[:i] + p.shots[i + 1 :])


class Crossover(Operator):
    name = "crossover"
    arity = 2

    def apply(self, parents, ctx):
        if len(parents) < 2:
            return None
        a, b = parents[0].prompt, parents[1].prompt
        ca = int(ctx.rng.integers(0, len(a.sentences) + 1))
        cb = int(ctx.rng.integers(0, len(b.sentences) + 1))
        sentences = list(dict.fromkeys(a.sentences[:ca] + b.sentences[cb:]))
        fmt = a.fmt if ctx.rng.random() < 0.5 else b.fmt
        pool = list(dict.fromkeys(a.shots + b.shots))
        keep = int(ctx.rng.integers(0, min(MAX_SHOTS, len(pool)) + 1)) if pool else 0
        order = ctx.rng.permutation(len(pool))[:keep] if pool else []
        shots = tuple(pool[int(i)] for i in sorted(order))
        return Prompt(tuple(sentences), fmt, shots)


REWRITE_META = [
    "Rewrite the following instruction in different words. Keep the meaning.",
    "Make the following instruction shorter and clearer.",
    "Make the following instruction more specific about how the answer must be written.",
    "Rewrite the following instruction as a short list of rules in one paragraph.",
]
_SYSTEM = "You edit instructions that are given to a language model."
_PREFIX = re.compile(r"^(?:new |improved |rewritten )?instruction\s*:\s*", re.IGNORECASE)


def clean_instruction(raw: str, max_sentences: int = 3) -> tuple[str, ...]:
    text = raw.strip().strip("`").strip()
    text = text.splitlines()[0] if text.count("\n") > 3 else text.replace("\n", " ")
    text = _PREFIX.sub("", text).strip().strip("\"'")
    return split_sentences(text)[:max_sentences]


class LLMRewrite(Operator):
    name = "llm_rewrite"
    kind = "llm"

    def apply(self, parents, ctx):
        p = parents[0].prompt
        meta = REWRITE_META[int(ctx.rng.integers(len(REWRITE_META)))]
        user = f"{meta} Output only the new instruction.\n\nInstruction: {' '.join(p.sentences)}"
        raw = ctx.backend.complete(
            [ChatRequest([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}], 60)]
        )[0]
        s = clean_instruction(raw)
        return Prompt(s, p.fmt, p.shots) if s else None


class LLMReflect(Operator):
    name = "llm_reflect"
    kind = "llm"
    needs_feedback = True

    def apply(self, parents, ctx):
        par = parents[0]
        if par.scores is None or par.outputs is None:
            return None
        wrong = np.flatnonzero(par.scores == 0)
        if len(wrong) == 0:
            return None
        pick = ctx.rng.choice(wrong, size=min(3, len(wrong)), replace=False)
        lines = []
        for i in pick:
            ex = ctx.task.dev[int(i)]
            lines.append(
                f"Question: {ex.question[:200]}\nAssistant replied: {par.outputs[int(i)][:80]}\n"
                f"Correct answer: {ex.gold}"
            )
        user = (
            f"A language model was given this instruction:\n{' '.join(par.prompt.sentences)}\n\n"
            "It answered these questions wrongly:\n\n" + "\n\n".join(lines) +
            "\n\nWrite an improved instruction that makes the model reply correctly and in the "
            "right format. Output only the new instruction."
        )
        raw = ctx.backend.complete(
            [ChatRequest([{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}], 60)]
        )[0]
        s = clean_instruction(raw)
        return Prompt(s, par.prompt.fmt, par.prompt.shots) if s else None


OPERATORS: list[tuple[Operator, float]] = [
    (Reorder(), 1.0),
    (DropClause(), 1.0),
    (AddClause(), 2.0),
    (SwapFormat(), 2.0),
    (AddShot(), 1.5),
    (RemoveShot(), 1.0),
    (Crossover(), 1.5),
    (LLMRewrite(), 1.5),
    (LLMReflect(), 1.5),
]


def choose_operator(
    rng: np.random.Generator, allow_feedback: bool = True, parents_available: int = 2
) -> Operator:
    ops = [
        (o, w)
        for o, w in OPERATORS
        if (allow_feedback or not o.needs_feedback) and o.arity <= parents_available
    ]
    w = np.array([w for _, w in ops], dtype=float)
    return ops[int(rng.choice(len(ops), p=w / w.sum()))][0]
