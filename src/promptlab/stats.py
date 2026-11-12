from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean with a percentile bootstrap CI."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> dict[str, float]:
    """Paired difference mean(a - b) with a bootstrap CI and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_boot": float(min(1.0, p))}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Exact McNemar test on paired 0/1 outcomes: a_only vs b_only discordant pairs."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    p = 1.0 if n == 0 else float(binomtest(a_only, n, 0.5).pvalue)
    return {"a_only": a_only, "b_only": b_only, "p_mcnemar": p}


def compare(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Everything reported for 'a beats b' on a shared test set."""
    return {**paired_bootstrap(a, b, seed=seed), **mcnemar_exact(a, b)}
