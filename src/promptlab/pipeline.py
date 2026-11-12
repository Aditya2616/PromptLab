"""One experiment end to end: evolve, run the random-search baseline, report on test."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from .baselines import random_search
from .evolve import EvoConfig, EvoResult, node_to_dict, run_evolution
from .models.base import ChatBackend
from .models.cache import CachedBackend
from .models.counting import CountingBackend
from .report import build_run_json, final_report, git_info, write_json
from .tasks.base import Task


def partial_state(task: Task, backend_name: str, model: str, cfg: EvoConfig, evo: EvoResult) -> dict:
    """What the live UI streams after every generation (no test numbers yet)."""
    return {
        "schema": 1,
        "recorded": False,
        "status": "running",
        "backend": backend_name,
        "model": model,
        "seed": cfg.seed,
        "config": cfg.__dict__,
        "task": {"name": task.name, "title": task.title, "n_dev": len(task.dev), "n_test": task.n_test},
        "nodes": [node_to_dict(task, n) for n in evo.nodes.values()],
        "generations": [dict(g) for g in evo.generations],
    }


def run_experiment(
    task: Task,
    backend: CachedBackend,
    cfg: EvoConfig,
    out: str | Path | None = None,
    state_path: str | Path | None = None,
    on_state: Callable[[dict], None] | None = None,
    mutator: ChatBackend | None = None,
    alt_mutator: ChatBackend | None = None,
    analysis_n: int = 250,
) -> dict:
    """Evolve, run the equal-budget random baseline, optionally evolve again with a stronger
    mutator, then score the winners on test. ``mutator`` defaults to the task model."""
    t0 = time.time()
    stats0 = backend.stats()
    mut = CountingBackend(mutator or backend)

    def emit(evo: EvoResult) -> None:
        st = partial_state(task, backend.name, backend.model, cfg, evo)
        if state_path:
            write_json(st, state_path)
        if on_state:
            on_state(st)

    evo = run_evolution(task, backend, cfg, on_generation=emit, mutator=mut)
    evo_mut_calls = mut.calls
    cands, rand_best = random_search(task, backend, cfg, evo, mutator=mut)
    rand_mut_calls = mut.calls - evo_mut_calls
    alt = alt_info = None
    if alt_mutator is not None:
        alt_mut = CountingBackend(alt_mutator)
        alt = run_evolution(task, backend, cfg, mutator=alt_mut)
        alt_info = {
            "mutator_model": alt_mutator.model, "mutator_calls": alt_mut.calls, "winner": alt.best().id,
            "nodes": [node_to_dict(task, n) for n in alt.nodes.values()],
            "generations": alt.generations,
        }
    rep = final_report(task, backend, evo, rand_best, alt=alt, analysis_n=analysis_n)
    result = build_run_json(
        task, backend.name, backend.model, cfg, evo, cands, rand_best, rep,
        {k: v - stats0[k] for k, v in backend.stats().items()}, time.time() - t0, git_info(),
        device=getattr(backend.inner, "device", "n/a"), dtype=getattr(backend.inner, "dtype", "n/a"),
        mutator={
            "model": (mutator or backend).model, "evolution_calls": evo_mut_calls,
            "random_calls": rand_mut_calls,
            "dev_evaluations": {"evolution": evo.evaluated - sum(n.origin == "seed" for n in evo.nodes.values()),
                                "random": len(cands)},
        },
        alt=alt_info,
    )
    if out:
        write_json(result, out)
    if on_state:
        on_state({**result, "status": "done"})
    return result
