# PromptLab spec

## Goal
Evolutionary prompt optimiser: mutate, score on a dev split, select, and report once on a held-out test split. Real runs on CPU with a local model, recorded and replayed in a UI that is the full-size version of the site's `evolve` scene.

## Layout
```
src/promptlab/
  models/      base.py (ChatBackend protocol, ChatRequest/Response)
               openai_compat.py (OpenAI API client, any base_url; stub-transport tests)
               local_hf.py (transformers, CPU, batched greedy)
               cache.py (SQLite; key = sha256(backend, model, messages, params))
  tasks/       base.py (Task: load, splits, render(prompt, example), score)
               commonsenseqa.py  synthetic.py  (third task decided after licence check)
  prompts.py   Prompt dataclass: instruction, few-shot ids, format constraint, id, parent ids, operator
  operators.py deterministic: reorder sentences, add/remove few-shot, swap format constraint, drop clause, crossover
               LLM: rewrite, simplify, reflect on 3 failed dev examples (GEPA-style)
               guards: length, non-empty, differs, dedupe
  evaluate.py  batch scoring; per-example 0/1 vectors cached
  select.py    NumPy: truncation, tournament, Pareto parent selection, elitism
  evolve.py    generation loop, seeded RNG, resumable from run dir, call budget accounting
  stats.py     bootstrap CI, paired bootstrap, exact McNemar, winner's-curse gap
  baselines.py seed only; random search at equal evaluation budget
  report.py    run -> run.json (population, lineage, scores, per-generation stats, test report)
  cli.py       promptlab run | resume | report | export-demo | serve
  api.py       FastAPI: /health, /runs, /runs/{id}, SSE /runs/{id}/stream, POST /runs
web/ (served) and docs/ (static demo replaying a recorded run.json)
tests/   scripts/ (download_data.py, run_experiments.py -> results/*.json, make_screenshots.py)
```

## Data flow
Task splits (few-shot pool / dev ~150 / test ~300, disjoint, seeded) -> seed prompts -> evaluate on dev -> loop: select parents -> apply operators -> guard -> evaluate children on dev (cached) -> survivors = top-k with elitism -> log generation -> after the last generation evaluate the dev-best and the seed once on test -> stats -> run.json. The loop never receives test examples; a test asserts this.

## Backends
`PROMPTLAB_BACKEND=local|openai`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `PROMPTLAB_MODEL`. Local is the default and the only path exercised for real; the OpenAI-compatible client is tested against `httpx.MockTransport`. Every call is cached, so runs are reproducible and resumable. `.env.example` provided.

## Experiments (real, scripted)
Three tasks, each evolution vs random-search baseline at equal budget, 3 seeds where time allows (about 4 h CPU total; shrink dev size or generations first). Report test accuracy of seed vs evolved with CIs, p-values, dev-test gap, model-call cost, and honest null results. Optional transfer check on SmolLM2-360M.

## UI
Dark void, lime survivors, orange for culls and failures, mono labels. Main view: SVG graph, one column per generation, one node per variant (vertical order by dev score), parent-child edges as curves, survivors lime with glow, culled nodes fade and shrink, elites ringed. Below, a best-dev-score curve with the held-out test score as a separate marker so overfitting is visible. Controls: play/pause, scrub, speed. Click or tap a node: side panel (bottom sheet on phone) with the prompt word-diffed against its parent, operator, dev score with CI, agreement with parent, and for the winner the held-out result. Extra tabs: experiments table, operator win rates. Demo mode replays recorded `run.json`, labelled "Recorded run"; local mode can launch a live run over SSE. Phone first at 390 px (graph scrolls inside its own container only), 44 px targets, reduced-motion respected, pause off-screen, DPR capped, no hover dependence.

## Tests (offline, fast)
Operator determinism and guards; selection maths; stats against known values; cache hit/miss and resume equivalence; OpenAI client on a stub transport; evolution loop with a fake deterministic backend (score climbs, lineage consistent, test untouched); FastAPI endpoints; run.json schema. Model tests behind `@pytest.mark.slow`.

## Done means
Fresh-clone quickstart works; pytest and ruff pass; three real experiments recorded with the reproducing script; README numbers equal stored results; UI screenshot-verified at desktop and phone; static `docs/` demo works; MIT licence, credits, limitations; clean local git history.
