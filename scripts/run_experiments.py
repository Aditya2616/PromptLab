"""Reproduce the recorded experiments on one machine, one process, one device and dtype.

    python scripts/run_experiments.py                 # 3 tasks x 3 seeds
    python scripts/run_experiments.py csqa --seeds 0  # a subset

Per task and seed: evolution (task model also mutates), random search at an equal number of dev
evaluations, and a second evolution whose LLM operators use a stronger mutator model. Winners are
scored on the held-out test split once. Every model call is cached in runs/ (keyed on model,
device and dtype), so an interrupted run resumes and finished results are skipped.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("PROMPTLAB_THREADS", "2")

from promptlab.evolve import EvoConfig  # noqa: E402
from promptlab.models import CachedBackend, cache_path  # noqa: E402
from promptlab.models.local_hf import LocalHFBackend  # noqa: E402
from promptlab.pipeline import run_experiment  # noqa: E402
from promptlab.report import write_json  # noqa: E402
from promptlab.tasks import load_task  # noqa: E402

TASK_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MUTATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEV_N, TEST_N = 200, 500
GENERATIONS, POP, CHILDREN = 10, 10, 10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=["csqa", "boolq", "synthetic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results")
    ap.add_argument("--cache", default="runs")
    ap.add_argument("--no-alt", action="store_true", help="skip the stronger-mutator arm")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    args = ap.parse_args()

    raw = LocalHFBackend(TASK_MODEL, batch_size=32, device=args.device, dtype=args.dtype)
    backend = CachedBackend(raw, cache_path(args.cache, raw))
    alt = None
    if not args.no_alt:
        a = LocalHFBackend(MUTATOR_MODEL, batch_size=8, device=args.device, dtype=args.dtype)
        alt = CachedBackend(a, cache_path(args.cache, a))
    print(f"device {raw.device} dtype {raw.dtype}", flush=True)

    tasks = {name: load_task(name, dev_n=DEV_N, test_n=TEST_N) for name in args.tasks}
    t_all = time.time()
    for seed in args.seeds:  # seeds outermost: every task gets a complete seed before the next starts
        for name, task in tasks.items():
            out = Path(args.out) / f"{name}-s{seed}.json"
            if out.exists():
                print(f"skip {out} (exists)", flush=True)
                continue
            t0 = time.time()
            cfg = EvoConfig(generations=GENERATIONS, pop=POP, children=CHILDREN, seed=seed)
            res = run_experiment(task, backend, cfg, out=out, state_path=Path(args.cache) / f"{name}-s{seed}-state.json",
                                 alt_mutator=alt)
            arms = res["report"]["arms"]
            print(f"{name} s{seed}: " + "  ".join(f"{k} {a['test_acc']:.3f}" for k, a in arms.items())
                  + f"  | {(time.time() - t0) / 60:.1f} min, {res['calls']['model_calls']} new model calls", flush=True)
            write_json({"seconds": res["seconds"], "calls": res["calls"]}, Path(args.cache) / f"{name}-s{seed}-timing.json")
    print(f"all done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
