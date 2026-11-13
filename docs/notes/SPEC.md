# PromptLab design notes

## Goal
An evolutionary prompt optimiser: mutate, score on a dev split, select, and report once on a held-out test split. Runs a small local model (CUDA float16 when a GPU is available, CPU float32 otherwise; the fast tests use a scripted backend on the CPU), records every run, and replays it in a viewer that is the full-size version of the `evolve` scene on my portfolio.

## Layout
```
src/promptlab/
  models/      base.py (ChatBackend protocol), openai_compat.py (OpenAI-compatible client, tested on a stub transport),
               local_hf.py (transformers, batched greedy, out-of-memory retry), cache.py (SQLite call cache), fake.py (test doubles)
  tasks/       base.py (Task: splits, rendering, scoring, test-split lock), csqa.py, boolq.py, synthetic.py
  prompts.py   Prompt: instruction sentences, format constraint, few-shot ids; content-hash id
  operators.py deterministic: reorder, drop clause, add clause, swap format, add/remove shot, crossover
               LLM: rewrite, reflect on failed dev examples; guards for length, empty, unchanged
  evaluate.py  per-example 0/1 score vectors
  select.py    NumPy: ranking with a length tie-break, tournament, Pareto parent selection
  evolve.py    the generation loop (seeded, test split locked)
  baselines.py random search at equal evaluation budget
  stats.py     Wilson and bootstrap CIs, paired bootstrap, exact McNemar
  summary.py   cross-seed aggregation, the viewer's data files (docs/data)
  report.py    the one place the test split is unlocked; builds run.json, per-operator held-out diagnostic
  pipeline.py  evolve -> baseline -> report
  api.py       FastAPI: /health, live runs over SSE, static viewer
  cli.py       promptlab download | run | export-demo | serve
docs/          the viewer (served by FastAPI and as a static site); docs/data holds recorded runs
tests/  scripts/ (run_experiments.py, check_numbers.py, check_ui.py, probe_seeds.py)
results/       one file per task and seed (the real-scale runs); results/pilot/ holds the earlier small CPU runs
```

## Data flow
Task splits (few-shot pool, dev, test; disjoint and seeded) -> seed prompts -> score on dev -> loop: pick parents (tournament or Pareto) -> apply an operator -> guard -> score the child on dev -> keep the top `pop` with a shorter-prompt tie-break -> log the generation. After the last generation the dev winner, the first seed, the best seed, the random-search winner (and, in the optional arm, the winner evolved with a 1.5B mutator) are each scored once on test; a per-operator diagnostic then scores the children that beat their parent on dev on part of the test split, after the arms are fixed. The loop never receives the test split: `Task.test` raises while the task is locked, and a test asserts it.

## Backends
`PROMPTLAB_BACKEND=local|openai|fake`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PROMPTLAB_MODEL`. Only the local path was exercised for the recorded results (device and dtype are part of the cache key and are recorded in every result file). The OpenAI-compatible client is adapter code tested against `httpx.MockTransport`. Every call goes through the SQLite cache, so runs are reproducible and an interrupted run resumes.

## Statistics
Wilson interval for dev scores, percentile bootstrap (10,000 resamples) for test accuracy, paired bootstrap for differences, exact McNemar on discordant pairs. The winner's-curse gap is dev accuracy minus test accuracy of the dev winner.
Control: random search at an equal evaluation budget (same operators, no feedback, only the final pick uses dev scores).

## Viewer
One SVG column per generation (the columns scroll sideways inside the graph on a phone), one node per variant (vertical position is dev score), parent-child curves, survivors lime, culled nodes fade, the winning lineage highlighted. A run picker chooses task and seed. Below it, best dev score per generation, a separate chart of the held-out test results (seed prompt, random-search winner, evolved winner, with intervals), the cross-seed table and the per-operator table; all of them read `docs/data`, which `scripts/check_numbers.py` generates from `results/`. Tapping a node opens its prompt as a word diff against its parent (a bottom sheet on a phone). Locally the page can start a live run streamed over SSE; the static `docs/` version replays recorded runs and says so.

## Tests
Offline and fast: operator determinism and guards, selection maths, statistics against known values, cache hit/miss and resume equivalence, the OpenAI client on a stub transport, the loop on a deterministic fake backend (score climbs, lineage consistent, test locked), the API endpoints. Real-model tests are behind `@pytest.mark.slow`.
