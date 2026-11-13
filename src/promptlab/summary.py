"""Aggregate the per-seed result files into the tables the README and the viewer show.

Everything published about the real-scale runs is computed here from ``results/<task>-s<seed>.json``
and nothing else: ``scripts/check_numbers.py`` renders the README from this, the static viewer
reads the same output (``docs/data/summary.json``), and a fast test fails if they drift apart.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .stats import paired_bootstrap

TASK_ORDER = ["csqa", "synthetic", "boolq"]
ARMS = ["seed0", "best_seed", "random", "evolved", "evolved_alt"]
ARM_LABEL = {
    "seed0": "First-draft seed prompt",
    "best_seed": "Best seed prompt (picked on dev)",
    "random": "Random-search winner (equal budget)",
    "evolved": "Evolved winner",
    "evolved_alt": "Evolved winner, 1.5B mutator",
}
_FILE = re.compile(r"^([a-z]+)-s(\d+)\.json$")


def load_runs(results_dir: str | Path) -> dict[str, dict[int, dict]]:
    """``{task: {seed: run}}`` for every finished run file directly inside ``results_dir``."""
    out: dict[str, dict[int, dict]] = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        m = _FILE.match(f.name)
        if not m:
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("schema") != 1 or not d.get("report"):
            continue
        out.setdefault(m.group(1), {})[int(m.group(2))] = d
    return out


def _bits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8) - ord("0")


def _sd(x: list[float]) -> float:
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _pooled(a: np.ndarray, b: np.ndarray, seed: int = 0) -> dict:
    """Paired comparison pooled over seeds: per test item, the difference averaged over seeds,
    then a bootstrap over items. Also how many seeds had a positive per-seed difference."""
    d = (a - b).mean(axis=0)
    res = paired_bootstrap(d, np.zeros_like(d), seed=seed)
    per_seed = (a - b).mean(axis=1)
    return {
        "diff": res["diff"], "lo": res["lo"], "hi": res["hi"], "p_boot": res["p_boot"],
        "seed_diffs": [float(x) for x in per_seed],
        "seeds_ahead": int((per_seed > 0).sum()), "seeds_behind": int((per_seed < 0).sum()),
        "seed_sd": _sd([float(x) for x in per_seed]),
    }


def _operators(runs: dict[int, dict], which: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for run in runs.values():
        for r in (run["report"]["operators_heldout"].get(which) or []):
            t = rows.setdefault(r["operator"], {"operator": r["operator"], "children": 0, "beat_dev": 0,
                                                "beat_test": 0, "dev_gain": 0.0, "test_gain": 0.0})
            k = r["beat_parent_dev"]
            t["children"] += r["children"]
            t["beat_dev"] += k
            t["beat_test"] += r["beat_parent_test"]
            t["dev_gain"] += (r["mean_dev_gain"] or 0.0) * k
            t["test_gain"] += (r["mean_test_gain"] or 0.0) * k
    out = []
    for t in rows.values():
        k = t["beat_dev"]
        out.append({
            "operator": t["operator"], "children": t["children"], "beat_dev": k,
            "beat_dev_rate": k / t["children"] if t["children"] else 0.0,
            "beat_test": t["beat_test"],
            "beat_test_rate": t["beat_test"] / k if k else None,
            "mean_dev_gain": t["dev_gain"] / k if k else None,
            "mean_test_gain": t["test_gain"] / k if k else None,
        })
    return sorted(out, key=lambda r: (-r["children"], r["operator"]))


def summarise_task(runs: dict[int, dict]) -> dict:
    seeds = sorted(runs)
    first = runs[seeds[0]]
    arms_present = [a for a in ARMS if all(a in runs[s]["report"]["arms"] for s in seeds)]
    arms: dict[str, dict] = {}
    bits: dict[str, np.ndarray] = {}
    for a in arms_present:
        per = [runs[s]["report"]["arms"][a] for s in seeds]
        bits[a] = np.stack([_bits(p["test_bits"]) for p in per]).astype(float)
        test = [p["test_acc"] for p in per]
        pooled = bits[a].mean(axis=0)
        rng = np.random.default_rng(0)
        means = pooled[rng.integers(0, len(pooled), size=(10_000, len(pooled)))].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        arms[a] = {
            "label": ARM_LABEL[a],
            "ids": [p["id"] for p in per],
            "dev": [p["dev_acc"] for p in per],
            "test": test,
            "test_ci": [p["test_ci"] for p in per],
            "gap": [p["gap"] for p in per],
            "test_mean": float(np.mean(test)), "test_sd": _sd(test),
            "test_min": float(min(test)), "test_max": float(max(test)),
            "dev_mean": float(np.mean([p["dev_acc"] for p in per])),
            "gap_mean": float(np.mean([p["gap"] for p in per])),
            "pooled_ci": [float(lo), float(hi)],
        }
    comps: dict[str, dict] = {}
    for key, (a, b) in {
        "evolved_vs_random": ("evolved", "random"),
        "evolved_vs_seed0": ("evolved", "seed0"),
        "evolved_vs_best_seed": ("evolved", "best_seed"),
        "evolved_alt_vs_random": ("evolved_alt", "random"),
        "evolved_alt_vs_evolved": ("evolved_alt", "evolved"),
    }.items():
        if a in bits and b in bits:
            c = _pooled(bits[a], bits[b])
            c["per_seed"] = [runs[s]["report"]["comparisons"][key] for s in seeds]
            comps[key] = c
    return {
        "name": first["task"]["name"], "title": first["task"]["title"], "source": first["task"]["source"],
        "seeds": seeds,
        "n_dev": first["task"]["n_dev"], "n_test": first["task"]["n_test"], "n_pool": first["task"]["n_pool"],
        "majority_rate": first["report"]["majority_rate"],
        "config": first["config"],
        "device": first["device"], "dtype": first["dtype"], "model": first["model"],
        "mutator_model": (first.get("alt") or {}).get("mutator_model"),
        "commits": sorted({(r["git"] or {}).get("commit") or "n/a" for r in runs.values()}),
        "dev_evaluations": [r["mutator"]["dev_evaluations"] for r in runs.values()],
        "arms": arms, "comparisons": comps,
        "operators": _operators(runs, "main"),
        "operators_alt": _operators(runs, "alt"),
        "operator_n_test": first["report"]["operators_heldout"]["n_test_used"],
        "calls": sum(r["calls"]["model_calls"] for r in runs.values()),
        "cache_hits": sum(r["calls"]["cache_hits"] for r in runs.values()),
        "seconds": sum(r["seconds"] for r in runs.values()),
    }


def summarise(results_dir: str | Path) -> dict:
    runs = load_runs(results_dir)
    tasks = [t for t in TASK_ORDER if t in runs] + [t for t in sorted(runs) if t not in TASK_ORDER]
    out = {"tasks": {t: summarise_task(runs[t]) for t in tasks}, "order": tasks}
    out["totals"] = {
        "runs": sum(len(runs[t]) for t in tasks),
        "calls": sum(out["tasks"][t]["calls"] for t in tasks),
        "seconds": sum(out["tasks"][t]["seconds"] for t in tasks),
    }
    return out


def _round(x, nd: int = 6):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    return x


def viewer_files(results_dir: str | Path) -> dict[str, bytes]:
    """Everything the static viewer reads, as ``{file name in docs/data: bytes}``: a copy of each
    finished run, an index of them, and the cross-seed summary."""
    runs = load_runs(results_dir)
    summary = _round(summarise(results_dir))
    files: dict[str, bytes] = {}
    index = []
    for task in summary["order"]:
        for seed in sorted(runs[task]):
            name = f"{task}-s{seed}.json"
            run = runs[task][seed]
            files[name] = (Path(results_dir) / name).read_bytes()
            index.append({
                "file": name, "name": task, "task": run["task"]["title"], "seed": seed,
                "model": run["model"], "device": run["device"], "dtype": run["dtype"],
                "commit": (run["git"] or {}).get("commit"), "dirty": bool((run["git"] or {}).get("dirty")),
            })
    files["index.json"] = (json.dumps({"runs": index, "summary": "summary.json"}, indent=1) + "\n").encode()
    files["summary.json"] = (json.dumps(summary, indent=1) + "\n").encode()
    return files


def export_demo(results_dir: str | Path, docs_dir: str | Path) -> list[str]:
    """Write :func:`viewer_files` into ``<docs_dir>/data`` and drop stale run files there."""
    dst = Path(docs_dir) / "data"
    dst.mkdir(parents=True, exist_ok=True)
    files = viewer_files(results_dir)
    for old in dst.glob("*.json"):
        if old.name not in files:
            old.unlink()
    for name, data in files.items():
        (dst / name).write_bytes(data)
    return sorted(files)
