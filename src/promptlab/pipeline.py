"""One experiment end to end: evolve, run the random-search baseline, report on test."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from .baselines import random_search
from .evolve import EvoConfig, EvoResult, node_to_dict, run_evolution
from .models.cache import CachedBackend
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
) -> dict:
    t0 = time.time()

    def emit(evo: EvoResult) -> None:
        st = partial_state(task, backend.name, backend.model, cfg, evo)
        if state_path:
            write_json(st, state_path)
        if on_state:
            on_state(st)

    evo = run_evolution(task, backend, cfg, on_generation=emit)
    cands, rand_best = random_search(task, backend, cfg, evo)
    rep = final_report(task, backend, evo, rand_best)
    result = build_run_json(
        task, backend.name, backend.model, cfg, evo, cands, rand_best, rep,
        backend.stats(), time.time() - t0, git_info(),
    )
    if out:
        write_json(result, out)
    if on_state:
        on_state({**result, "status": "done"})
    return result
