# PromptLab design notes

## Goal
An evolutionary prompt optimiser: mutate, score on a dev split, select, and report once on a held-out test split. Runs on a CPU with a small local model, records every run, and replays it in a viewer that is the full-size version of the `evolve` scene on my portfolio.

## Layout
```
src/promptlab/
  models/      base.py (ChatBackend protocol), openai_compat.py (OpenAI-compatible client, tested on a stub transport),
               local_hf.py (transformers, CPU, batched greedy), cache.py (SQLite call cache), fake.py (test doubles)
  tasks/       base.py (Task: splits, rendering, scoring, test-split lock), csqa.py, boolq.py, synthetic.py
  prompts.py   Prompt: instruction sentences, format constraint, few-shot ids; content-hash id
  operators.py deterministic: reorder, drop clause, add clause, swap format, add/remove shot, crossover
               LLM: rewrite, reflect on failed dev examples; guards for length, empty, unchanged
  evaluate.py  per-example 0/1 score vectors
  select.py    NumPy: ranking with a length tie-break, tournament, Pareto parent selection
  evolve.py    the generation loop (seeded, test split locked)
  baselines.py random search at equal evaluation budget
  stats.py     Wilson and bootstrap CIs, paired bootstrap, exact McNemar
  report.py    the one place the test split is unlocked; builds run.json
  pipeline.py  evolve -> baseline -> report
  api.py       FastAPI: /health, live runs over SSE, static viewer
  cli.py       promptlab download | run | export-demo | serve
docs/          the viewer (served by FastAPI and as a static site); docs/data holds recorded runs
tests/  scripts/ (run_experiments.py, make_screenshots.py, probe_seeds.py)
```

## Data flow
Task splits (few-shot pool, dev, test; disjoint and seeded) -> seed prompts -> score on dev -> loop: pick parents (tournament or Pareto) -> apply an operator -> guard -> score the child on dev -> keep the top `pop` with a shorter-prompt tie-break -> log the generation. After the last generation the dev winner, the first seed, the best seed and the random-search winner are each scored once on test. The loop never receives the test split: `Task.test` raises while the task is locked, and a test asserts it.

## Backends
`PROMPTLAB_BACKEND=local|openai|fake`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PROMPTLAB_MODEL`. Only the local path was exercised for the recorded results. The OpenAI-compatible client is adapter code tested against `httpx.MockTransport`. Every call goes through the SQLite cache, so runs are reproducible and an interrupted run resumes.

## Statistics
Wilson interval for dev scores, percentile bootstrap (10,000 resamples) for test accuracy, paired bootstrap for differences, exact McNemar on discordant pairs. The winner's-curse gap is dev accuracy minus test accuracy of the dev winner.
Control: random search at an equal evaluation budget (same operators, no feedback, only the final pick uses dev scores).

## Viewer
One SVG column per generation, one node per variant (vertical position is dev score), parent-child curves, survivors lime, culled nodes fade, the winning lineage highlighted. Below it, best dev score per generation with the held-out test markers drawn separately. Tapping a node opens its prompt as a word diff against its parent (a bottom sheet on a phone). Locally the page can start a live run streamed over SSE; the static `docs/` version replays recorded runs and says so.

## Tests
Offline and fast: operator determinism and guards, selection maths, statistics against known values, cache hit/miss and resume equivalence, the OpenAI client on a stub transport, the loop on a deterministic fake backend (score climbs, lineage consistent, test locked), the API endpoints. Real-model tests are behind `@pytest.mark.slow`.
