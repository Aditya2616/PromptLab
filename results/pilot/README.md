# Pilot run (CPU, 3 generations, 40 dev items)

A first, much smaller set of runs, kept here for reference. They are not mixed into any table in the main README: one seed, a different device and dtype, and a tenth of the budget.

<!-- BEGIN pilot -->
Pilot settings: cpu float32, 3 generations, population 4, 4 children, 40 dev items, 100 test items, seed 0, commit `530c703`. Test accuracy (%) with 95% bootstrap CI.

| Task | First-draft seed | Best seed (dev) | Random search | Evolved winner |
|---|---|---|---|---|
| CommonsenseQA | 34.0 [25.0, 43.0] | 48.0 [38.0, 58.0] | 55.0 [45.0, 65.0] | 55.0 [45.0, 65.0] |
| BoolQ | 66.0 [57.0, 75.0] | 71.0 [62.0, 80.0] | 69.0 [60.0, 78.0] | 70.0 [61.0, 79.0] |
| Synthetic transforms | 7.0 [2.0, 12.0] | 12.0 [6.0, 19.0] | 36.0 [27.0, 45.0] | 38.0 [29.0, 48.0] |
<!-- END pilot -->

Regenerate with `python scripts/check_numbers.py --write`; the fast tests check that this table matches the JSON files next to it.
