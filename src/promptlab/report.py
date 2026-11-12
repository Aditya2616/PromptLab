"""Final reporting: the one place the held-out test split is touched."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import __version__
from .evaluate import evaluate
from .evolve import EvoConfig, EvoResult, Node, node_to_dict
from .models.base import ChatBackend
from .select import rank_order
from .stats import bootstrap_ci, compare
from .tasks.base import Task


def git_info(cwd: Path | None = None) -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True, cwd=cwd, timeout=10, check=False
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    return {"commit": run("rev-parse", "--short", "HEAD") or None, "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def operator_stats(evo: EvoResult) -> list[dict]:
    rows: dict[str, dict] = {}
    for n in evo.nodes.values():
        if n.origin != "evolution" or n.gen == 0:
            continue
        best_parent = max(evo.nodes[p].acc for p in n.parents)
        r = rows.setdefault(n.operator, {"operator": n.operator, "n": 0, "improved": 0, "survived": 0, "delta": 0.0})
        r["n"] += 1
        r["improved"] += int(n.acc > best_parent)
        r["survived"] += int(n.culled_at != n.gen)
        r["delta"] += n.acc - best_parent
    out = []
    for r in rows.values():
        r["mean_delta"] = r.pop("delta") / r["n"]
        out.append(r)
    return sorted(out, key=lambda r: -r["n"])


def _arm(task: Task, node: Node, test_scores: np.ndarray, label: str) -> dict:
    mean, lo, hi = bootstrap_ci(test_scores)
    return {
        "label": label,
        "id": node.id,
        "dev_acc": node.acc,
        "test_acc": mean,
        "test_ci": [lo, hi],
        "gap": node.acc - mean,
        "test_bits": "".join(str(int(x)) for x in test_scores),
        "text": task.prompt_text(node.prompt),
    }


def final_report(
    task: Task, backend: ChatBackend, evo: EvoResult, rand_best: Node
) -> dict:
    """Evaluate the candidate winners once on the test split and compare them."""
    task.unlock_test()
    try:
        seeds = [n for n in evo.nodes.values() if n.origin == "seed"]
        first_seed = evo.nodes[task.seeds()[0].id]
        acc = np.array([n.acc for n in seeds])
        words = np.array([n.prompt.n_words for n in seeds])
        best_seed = seeds[int(rank_order(acc, words)[0])]
        arms_nodes = {
            "seed0": first_seed,
            "best_seed": best_seed,
            "evolved": evo.best(),
            "random": rand_best,
        }
        cache: dict[str, np.ndarray] = {}
        for node in arms_nodes.values():
            if node.id not in cache:
                cache[node.id], _ = evaluate(task, node.prompt, task.test, backend)
        arms = {k: _arm(task, n, cache[n.id], k) for k, n in arms_nodes.items()}
        ev = cache[arms_nodes["evolved"].id]
        comps = {
            f"evolved_vs_{k}": compare(ev, cache[arms_nodes[k].id])
            for k in ("seed0", "best_seed", "random")
        }
        return {
            "n_test": len(task.test),
            "majority_rate": task.majority_rate(),
            "arms": arms,
            "comparisons": comps,
        }
    finally:
        task.lock_test()


def build_run_json(
    task: Task,
    backend_name: str,
    model: str,
    cfg: EvoConfig,
    evo: EvoResult,
    rand_cands: list[Node],
    rand_best: Node,
    report: dict,
    calls: dict,
    seconds: float,
    git: dict | None = None,
) -> dict:
    winner = evo.best()
    return {
        "schema": 1,
        "recorded": True,
        "created": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "promptlab_version": __version__,
        "git": git or git_info(),
        "backend": backend_name,
        "model": model,
        "seed": cfg.seed,
        "config": asdict(cfg),
        "task": {
            "name": task.name,
            "title": task.title,
            "description": task.description,
            "source": task.source,
            "n_pool": len(task.pool),
            "n_dev": len(task.dev),
            "n_test": task.n_test,
        },
        "nodes": [node_to_dict(task, n) for n in evo.nodes.values()],
        "generations": evo.generations,
        "winner": winner.id,
        "baseline": {
            "nodes": [node_to_dict(task, n) for n in rand_cands],
            "winner": rand_best.id,
            "budget": len(rand_cands),
        },
        "operator_stats": operator_stats(evo),
        "report": report,
        "calls": calls,
        "seconds": round(seconds, 1),
    }


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
