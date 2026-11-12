"""Selection maths (NumPy). Scores are matrices of shape (population, n_examples) of 0/1."""

from __future__ import annotations

import numpy as np


def rank_order(acc: np.ndarray, n_words: np.ndarray) -> np.ndarray:
    """Indices best-first: higher accuracy, then shorter prompt, then earlier index."""
    return np.lexsort((np.arange(len(acc)), n_words, -acc))


def truncation(acc: np.ndarray, n_words: np.ndarray, k: int) -> np.ndarray:
    return rank_order(acc, n_words)[:k]


def tournament(acc: np.ndarray, rng: np.random.Generator, size: int = 2) -> int:
    size = min(size, len(acc))
    cand = rng.choice(len(acc), size=size, replace=False)
    return int(cand[np.argmax(acc[cand])])


def pareto_front(scores: np.ndarray) -> np.ndarray:
    """Indices of candidates not dominated on the per-example score vectors (GEPA-style)."""
    s = np.asarray(scores, dtype=np.int16)
    n = len(s)
    front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i != j and np.all(s[j] >= s[i]) and np.any(s[j] > s[i]):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return np.array(front, dtype=int)


def pareto_pick(scores: np.ndarray, rng: np.random.Generator) -> int:
    """Pick a parent with probability proportional to the number of examples it is best on."""
    s = np.asarray(scores)
    best = s.max(axis=0)
    wins = ((s == best) & (best[None, :] > 0)).sum(axis=1).astype(float)
    front = pareto_front(s)
    w = np.zeros(len(s))
    w[front] = np.maximum(wins[front], 1.0)
    return int(rng.choice(len(s), p=w / w.sum()))


def pick_parent(
    method: str, scores: np.ndarray, rng: np.random.Generator, tournament_size: int = 2
) -> int:
    if method == "pareto":
        return pareto_pick(scores, rng)
    if method == "tournament":
        return tournament(scores.mean(axis=1), rng, tournament_size)
    raise ValueError(f"unknown parent selection {method!r}")
