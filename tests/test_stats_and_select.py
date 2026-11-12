import numpy as np
import pytest

from promptlab import select, stats


def test_wilson_known_value():
    lo, hi = stats.wilson(50, 100)
    assert lo == pytest.approx(0.4038, abs=1e-3)
    assert hi == pytest.approx(0.5962, abs=1e-3)
    assert stats.wilson(0, 0) == (0.0, 1.0)


def test_mcnemar_exact_matches_binomial():
    a = np.array([1] * 8 + [0] * 2 + [1] * 5)
    b = np.array([0] * 8 + [1] * 2 + [1] * 5)
    r = stats.mcnemar_exact(a, b)
    assert (r["a_only"], r["b_only"]) == (8, 2)
    assert r["p_mcnemar"] == pytest.approx(0.109375)  # two-sided exact, n=10, k=8


def test_mcnemar_no_discordance():
    x = np.ones(10)
    assert stats.mcnemar_exact(x, x)["p_mcnemar"] == 1.0


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    x = np.array([1] * 60 + [0] * 40)
    m, lo, hi = stats.bootstrap_ci(x, n_boot=4000, seed=1)
    assert m == 0.6 and lo < 0.6 < hi
    assert (lo, hi) == stats.bootstrap_ci(x, n_boot=4000, seed=1)[1:]


def test_paired_bootstrap_detects_clear_gain_and_null():
    rng = np.random.default_rng(0)
    base = (rng.random(200) < 0.4).astype(int)
    better = np.maximum(base, (rng.random(200) < 0.3).astype(int))
    r = stats.paired_bootstrap(better, base)
    assert r["diff"] > 0.1 and r["lo"] > 0 and r["p_boot"] < 0.01
    null = stats.paired_bootstrap(base, base)
    assert null["diff"] == 0 and null["p_boot"] == 1.0


def test_rank_order_tiebreaks_shorter_then_earlier():
    acc = np.array([0.5, 0.7, 0.7, 0.7])
    words = np.array([5, 9, 4, 4])
    assert list(select.rank_order(acc, words)) == [2, 3, 1, 0]
    assert list(select.truncation(acc, words, 2)) == [2, 3]


def test_pareto_front_keeps_complementary_candidates():
    s = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 1]])
    assert list(select.pareto_front(s)) == [3]
    s2 = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]])
    assert list(select.pareto_front(s2)) == [0, 1]


def test_pick_parent_methods_return_valid_index():
    rng = np.random.default_rng(0)
    s = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    for m in ("tournament", "pareto"):
        picks = {select.pick_parent(m, s, rng) for _ in range(50)}
        assert picks <= {0, 1, 2}
    # pareto never picks the dominated candidate 0
    assert 0 not in {select.pick_parent("pareto", s, rng) for _ in range(100)}
    with pytest.raises(ValueError):
        select.pick_parent("nope", s, rng)
