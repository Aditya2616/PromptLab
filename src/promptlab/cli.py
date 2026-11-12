from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _setup_threads() -> None:
    n = os.environ.get("PROMPTLAB_THREADS") or str(max(1, (os.cpu_count() or 4) // 4))
    os.environ.setdefault("OMP_NUM_THREADS", n)
    os.environ.setdefault("MKL_NUM_THREADS", n)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def cmd_download(args: argparse.Namespace) -> int:
    from .tasks import TASK_NAMES, load_task

    for name in args.tasks or TASK_NAMES:
        t = load_task(name)
        print(f"{name}: pool={len(t.pool)} dev={len(t.dev)} test={t.n_test}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    _setup_threads()
    from .evolve import EvoConfig
    from .models import CachedBackend, make_backend
    from .pipeline import run_experiment
    from .tasks import load_task

    task = load_task(args.task, dev_n=args.dev_n, test_n=args.test_n)
    raw = make_backend(args.backend, args.model)
    if hasattr(raw, "bind"):
        raw.bind(task)
    backend = CachedBackend(raw, Path(args.cache) / f"{raw.name}-{raw.model.replace('/', '_')}.sqlite")
    cfg = EvoConfig(
        generations=args.generations, pop=args.pop, children=args.children, seed=args.seed,
        selection=args.selection,
    )
    out = Path(args.out) / f"{args.task}-s{args.seed}.json"
    res = run_experiment(task, backend, cfg, out=out, state_path=Path(args.cache) / f"{args.task}-state.json")
    rep = res["report"]["arms"]
    for k, a in rep.items():
        print(f"{k:10s} dev={a['dev_acc']:.3f} test={a['test_acc']:.3f} "
              f"[{a['test_ci'][0]:.3f}, {a['test_ci'][1]:.3f}]")
    print(f"wrote {out}  ({res['calls']['model_calls']} model calls, {res['seconds']}s)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Copy finished run files into docs/data and write the index the static site reads."""
    src, dst = Path(args.results), Path(args.docs) / "data"
    dst.mkdir(parents=True, exist_ok=True)
    runs = []
    for f in sorted(src.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("schema") != 1 or "report" not in d:
            continue
        shutil.copy(f, dst / f.name)
        runs.append({"file": f.name, "task": d["task"]["title"], "name": d["task"]["name"], "model": d["model"]})
    (dst / "index.json").write_text(json.dumps({"runs": runs}, indent=1), encoding="utf-8")
    print(f"exported {len(runs)} runs to {dst}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("promptlab.api:app", host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="promptlab", description="Evolutionary prompt optimiser")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch datasets into the Hugging Face cache")
    d.add_argument("tasks", nargs="*")
    d.set_defaults(fn=cmd_download)

    r = sub.add_parser("run", help="run one experiment")
    r.add_argument("--task", default="synthetic", choices=["csqa", "boolq", "synthetic"])
    r.add_argument("--backend", default=None, help="local | openai | fake (default: $PROMPTLAB_BACKEND or local)")
    r.add_argument("--model", default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--generations", type=int, default=4)
    r.add_argument("--pop", type=int, default=5)
    r.add_argument("--children", type=int, default=6)
    r.add_argument("--dev-n", type=int, default=80)
    r.add_argument("--test-n", type=int, default=200)
    r.add_argument("--selection", default="tournament", choices=["tournament", "pareto"])
    r.add_argument("--out", default="results")
    r.add_argument("--cache", default="runs")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("export-demo", help="copy results/*.json into docs/data for the static demo")
    e.add_argument("--results", default="results")
    e.add_argument("--docs", default="docs")
    e.set_defaults(fn=cmd_export)

    s = sub.add_parser("serve", help="serve the UI and API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8103)
    s.set_defaults(fn=cmd_serve)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
