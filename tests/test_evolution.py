import json

import numpy as np

from promptlab.baselines import random_search
from promptlab.evolve import EvoConfig, run_evolution
from promptlab.models.cache import CachedBackend
from promptlab.models.fake import SimBackend
from promptlab.pipeline import run_experiment

CFG = EvoConfig(generations=3, pop=4, children=5, seed=2)


def test_evolution_climbs_and_lineage_is_consistent(task, backend):
    res = run_evolution(task, backend, CFG)
    best = [g["best_dev"] for g in res.generations]
    assert best == sorted(best), "elitist truncation: best dev score never drops"
    assert best[-1] > best[0], "the simulated model rewards format wording, so evolution must find it"
    for n in res.nodes.values():
        assert all(p in res.nodes for p in n.parents)
        assert n.gen == 0 or n.parents
    final = set(res.final_population)
    assert len(final) == CFG.pop
    assert all(res.nodes[i].culled_at is None for i in final)
    culled = [n for n in res.nodes.values() if n.culled_at is not None]
    assert culled and all(n.id not in final for n in culled)
    for g in res.generations:
        assert len(g["survivors"]) == CFG.pop


def test_search_never_touches_the_test_split(task, sim):
    test_questions = {e.question for e in task._test}
    seen_test = []
    inner = sim.complete

    def spy(reqs):
        seen_test.extend(r for r in reqs if r.messages[-1]["content"] in test_questions)
        return inner(reqs)

    sim.complete = spy
    cb = CachedBackend(sim)
    res = run_evolution(task, cb, CFG)
    random_search(task, cb, CFG, res)
    assert seen_test == []


def test_same_seed_reproduces_and_warm_cache_needs_no_model_calls(task, sim, tmp_path):
    path = tmp_path / "c.sqlite"
    a = run_evolution(task, CachedBackend(sim, path), CFG)
    cold_calls = CachedBackend(sim, path)
    b = run_evolution(task, cold_calls, CFG)
    assert list(a.nodes) == list(b.nodes)
    assert cold_calls.misses == 0  # resume: everything replayed from the cache
    c = run_evolution(task, CachedBackend(sim), EvoConfig(3, 4, 5, seed=99))
    assert list(c.nodes) != list(a.nodes)


def test_random_search_matches_budget_and_uses_no_feedback(task, backend):
    res = run_evolution(task, backend, CFG)
    cands, best = random_search(task, backend, CFG, res)
    n_seed = sum(n.origin == "seed" for n in res.nodes.values())
    assert len(cands) == res.evaluated - n_seed
    assert all(c.origin == "random" for c in cands)
    assert best.acc >= max(n.acc for n in res.nodes.values() if n.origin == "seed")


def test_pareto_selection_also_runs(task, backend):
    res = run_evolution(task, backend, EvoConfig(2, 4, 4, seed=1, selection="pareto"))
    assert len(res.generations) == 3


def test_full_pipeline_writes_schema_and_stats(task, backend, tmp_path):
    out = tmp_path / "run.json"
    r = run_experiment(task, backend, CFG, out=out, state_path=tmp_path / "state.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == 1 and data["recorded"] is True
    ids = {n["id"] for n in data["nodes"]}
    assert data["winner"] in ids
    assert {"seed0", "best_seed", "evolved", "random"} == set(data["report"]["arms"])
    arm = data["report"]["arms"]["evolved"]
    assert 0 <= arm["test_ci"][0] <= arm["test_acc"] <= arm["test_ci"][1] <= 1
    assert len(arm["test_bits"]) == data["task"]["n_test"]
    comp = data["report"]["comparisons"]["evolved_vs_seed0"]
    assert {"diff", "lo", "hi", "p_boot", "p_mcnemar", "a_only", "b_only"} <= set(comp)
    assert data["git"].keys() >= {"commit", "dirty"}
    assert json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))["status"] == "running"
    assert r["operator_stats"]
    assert data["report"]["arms"]["evolved"]["dev_acc"] >= data["report"]["arms"]["seed0"]["dev_acc"]
    assert not np.isnan(comp["p_mcnemar"])


def test_sim_backend_is_deterministic(task):
    a, b = SimBackend(), SimBackend()
    a.bind(task)
    b.bind(task)
    from promptlab.models.base import ChatRequest

    r = ChatRequest(task.messages(task.seeds()[0], task.dev[0]))
    assert a.complete([r]) == b.complete([r])
