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


def operator_heldout(
    task: Task, backend: ChatBackend, evo: EvoResult, cache: dict[str, np.ndarray], n: int
) -> list[dict]:
    """Per operator: children that beat their best parent on dev, and how they did on held-out.

    Diagnostic only, computed after the search is over on the first ``n`` test items; nothing
    here feeds back into selection. A dev improvement that does not hold up on test is selection noise.
    """
    sub = task.test[:n]

    def test_scores(node: Node) -> np.ndarray:
        if node.id not in cache:
            cache[node.id], _ = evaluate(task, node.prompt, sub, backend)
        return cache[node.id]

    rows: dict[str, dict] = {}
    for node in evo.nodes.values():
        if node.origin != "evolution" or node.gen == 0:
            continue
        parent = max((evo.nodes[p] for p in node.parents), key=lambda x: x.acc)
        row = rows.setdefault(
            node.operator,
            {"operator": node.operator, "children": 0, "beat_parent_dev": 0, "dev_gain": 0.0,
             "test_gain": 0.0, "beat_parent_test": 0, "survived": 0},
        )
        row["children"] += 1
        if node.acc > parent.acc:
            row["beat_parent_dev"] += 1
            row["dev_gain"] += node.acc - parent.acc
            tg = float(test_scores(node).mean() - test_scores(parent).mean())
            row["test_gain"] += tg
            row["beat_parent_test"] += int(tg > 0)
            row["survived"] += int(node.culled_at is None or node.culled_at > node.gen)
    out = []
    for row in rows.values():
        k = row["beat_parent_dev"]
        dev_sum, test_sum = row.pop("dev_gain"), row.pop("test_gain")
        row["mean_dev_gain"] = dev_sum / k if k else None
        row["mean_test_gain"] = test_sum / k if k else None
        out.append(row)
    return sorted(out, key=lambda x: -x["children"])


def final_report(
    task: Task, backend: ChatBackend, evo: EvoResult, rand_best: Node,
    alt: EvoResult | None = None, analysis_n: int = 250,
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
        if alt is not None:
            arms_nodes["evolved_alt"] = alt.best()
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
        if alt is not None:
            alt_s = cache[arms_nodes["evolved_alt"].id]
            comps["evolved_alt_vs_random"] = compare(alt_s, cache[arms_nodes["random"].id])
            comps["evolved_alt_vs_evolved"] = compare(alt_s, ev)
        # diagnostic pass over children, after all arm scores above are fixed
        op_cache: dict[str, np.ndarray] = {}
        ops_held = operator_heldout(task, backend, evo, op_cache, analysis_n)
        ops_alt = operator_heldout(task, backend, alt, op_cache, analysis_n) if alt is not None else None
        return {
            "n_test": len(task.test),
            "majority_rate": task.majority_rate(),
            "arms": arms,
            "comparisons": comps,
            "operators_heldout": {"n_test_used": min(analysis_n, len(task.test)), "main": ops_held, "alt": ops_alt},
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
    device: str = "n/a",
    dtype: str = "n/a",
    mutator: dict | None = None,
    alt: dict | None = None,
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
        "device": device,
        "dtype": dtype,
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
        "mutator": mutator,
        "alt": alt,
        "operator_stats": operator_stats(evo),
        "report": report,
        "calls": calls,
        "seconds": round(seconds, 1),
    }


def write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
