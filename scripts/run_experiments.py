"""Reproduce the recorded experiments: one run per task with the local Qwen2.5-0.5B-Instruct model.

    python scripts/run_experiments.py            # all three tasks
    python scripts/run_experiments.py csqa       # one task

Sizes are deliberately small (CPU only, shared machine): see README "Limitations". Every model
call is cached in runs/, so re-running finished work is instant and an interrupted run resumes.
"""

from __future__ import annotations

import sys
import time

from promptlab.cli import main

SETTINGS = {
    "synthetic": ["--dev-n", "40", "--test-n", "100"],
    "csqa": ["--dev-n", "40", "--test-n", "100"],
    "boolq": ["--dev-n", "40", "--test-n", "100"],
}
COMMON = ["--backend", "local", "--seed", "0", "--generations", "3", "--pop", "4", "--children", "4"]


def run(tasks: list[str]) -> int:
    for task in tasks:
        t0 = time.time()
        print(f"=== {task}", flush=True)
        code = main(["run", "--task", task, *COMMON, *SETTINGS[task]])
        print(f"=== {task} finished in {(time.time() - t0) / 60:.1f} min (exit {code})", flush=True)
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:] or list(SETTINGS)))
