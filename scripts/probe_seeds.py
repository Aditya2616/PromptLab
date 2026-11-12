"""Score the hand-written seed prompts on each task's dev split (sanity check before experiments)."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 4)))

from promptlab.evaluate import evaluate  # noqa: E402
from promptlab.models import CachedBackend, make_backend  # noqa: E402
from promptlab.tasks import TASK_NAMES, load_task  # noqa: E402


def main() -> None:
    names = sys.argv[1:] or TASK_NAMES
    backend = CachedBackend(make_backend("local"), "runs/local-Qwen_Qwen2.5-0.5B-Instruct.sqlite")
    for name in names:
        task = load_task(name)
        task.lock_test()
        print(f"== {name}  dev={len(task.dev)}", flush=True)
        for p in task.seeds():
            t0 = time.time()
            s, outs = evaluate(task, p, task.dev, backend)
            print(f"  {s.mean():.3f}  {time.time() - t0:5.1f}s  {p.system!r}  e.g. {outs[0][:40]!r}", flush=True)


if __name__ == "__main__":
    main()
