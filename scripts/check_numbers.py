"""Keep every published number tied to the results files.

    python scripts/check_numbers.py --write   # regenerate README blocks and docs/data from results/
    python scripts/check_numbers.py --check   # fail if anything is out of sync (a fast test runs this)

Generated, never hand-typed: the tables and figures between ``<!-- BEGIN name -->`` and
``<!-- END name -->`` markers in README.md, and the files in docs/data/ that the viewer reads.
``CLAIMS`` lists the qualitative statements the README makes; each is verified against the data.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from promptlab.summary import ARM_LABEL, summarise, viewer_files

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"
PILOT = ROOT / "results" / "pilot"
DOCS = ROOT / "docs"

pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
pts = lambda x: f"{100 * x:+.1f}"  # noqa: E731


def fp(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _int(n: int) -> str:
    return f"{n:,}"


def b_setup(S: dict) -> str:
    t0 = S["tasks"][S["order"][0]]
    c = t0["config"]
    seeds = {len(S["tasks"][t]["seeds"]) for t in S["order"]}
    n_seeds = seeds.pop() if len(seeds) == 1 else None
    devs = sorted({d["evolution"] for t in S["order"] for d in S["tasks"][t]["dev_evaluations"]})
    rand = sorted({d["random"] for t in S["order"] for d in S["tasks"][t]["dev_evaluations"]})
    tot = S["totals"]
    return (
        f"Task model {t0['model']}, greedy decoding, {t0['device']} {t0['dtype']} for every run "
        f"(recorded as `device` and `dtype` in each `results/*.json`). "
        f"Per task and seed: {c['generations']} generations, population {c['pop']}, {c['children']} children per "
        f"generation, {t0['n_dev']} dev items, {t0['n_test']} held-out test items, {n_seeds} seeds "
        f"({', '.join(map(str, t0['seeds']))}). Evolution scored {', '.join(map(str, devs))} children on dev and the "
        f"random-search baseline scored {', '.join(map(str, rand))} candidates, so the dev-evaluation budget is equal. "
        f"All {tot['runs']} runs: {_int(tot['calls'])} new model calls and "
        f"{tot['seconds'] / 3600:.1f} h of recorded wall time on one GPU; calls already in the cache when a run was "
        f"resumed are not counted again."
    )


def b_headline(S: dict) -> str:
    rows = ["| Task | Prompt | Held-out test per seed (%) | Mean ± SD over seeds (%) | 95% CI, items resampled (%) | Test − dev, mean (pts) |",
            "|---|---|---|---|---|---|"]
    for t in S["order"]:
        T = S["tasks"][t]
        maj = f" (majority answer {pct(T['majority_rate'])}%)" if T["majority_rate"] is not None else ""
        first = True
        for a in ("seed0", "best_seed", "random", "evolved", "evolved_alt"):
            if a not in T["arms"]:
                continue
            A = T["arms"][a]
            label = ARM_LABEL[a]
            if a == "evolved":
                label = f"**{label}**"
            name = f"{T['title']}{maj}" if first else ""
            first = False
            rows.append(
                f"| {name} | {label} | {' / '.join(pct(x) for x in A['test'])} | {pct(A['test_mean'])} ± {pct(A['test_sd'])} "
                f"| {pct(A['pooled_ci'][0])} to {pct(A['pooled_ci'][1])} | {pts(-A['gap_mean'])} |"
            )
    return "\n".join(rows)


def b_paired(S: dict) -> str:
    rows = ["| Task | Evolved winner minus | Mean difference (pts), 95% CI | Per seed (pts) | Exact McNemar p per seed | Seeds ahead / behind |",
            "|---|---|---|---|---|---|"]
    names = {"evolved_vs_random": "random-search winner", "evolved_vs_best_seed": "best seed prompt",
             "evolved_vs_seed0": "first-draft seed prompt", "evolved_alt_vs_random": "random-search winner (1.5B-mutator arm)"}
    for t in S["order"]:
        T = S["tasks"][t]
        first = True
        for k, nm in names.items():
            c = T["comparisons"].get(k)
            if c is None:
                continue
            label = f"**{nm}**" if k == "evolved_vs_random" else nm
            rows.append(
                f"| {T['title'] if first else ''} | {label} | {pts(c['diff'])} [{pts(c['lo'])}, {pts(c['hi'])}] "
                f"| {' / '.join(pts(x) for x in c['seed_diffs'])} "
                f"| {' / '.join(fp(p['p_mcnemar']) for p in c['per_seed'])} | {c['seeds_ahead']} / {c['seeds_behind']} |"
            )
            first = False
    return "\n".join(rows)


def b_verdicts(S: dict) -> str:
    out = []
    for t in S["order"]:
        T = S["tasks"][t]
        c = T["comparisons"]["evolved_vs_random"]
        if c["lo"] > 0:
            word = "the interval is above zero"
        elif c["hi"] < 0:
            word = "the interval is below zero"
        else:
            word = "the interval includes zero"
        out.append(
            f"- {T['title']}: evolved minus random search is {pts(c['diff'])} pts (95% CI {pts(c['lo'])} to {pts(c['hi'])}), "
            f"ahead in {c['seeds_ahead']} of {len(T['seeds'])} seeds; {word}."
        )
    return "\n".join(out)


def _op_table(rows: list[dict], n_test: int) -> list[str]:
    out = ["| Operator | Children | Beat parent on dev | Of those, beat parent on test | Mean gain on dev (pts) | Mean gain on test (pts) |",
           "|---|---|---|---|---|---|"]
    for r in rows:
        k = r["beat_dev"]
        held = f"{r['beat_test']} ({pct(r['beat_test_rate'])}%)" if k else "-"
        out.append(
            f"| `{r['operator']}` | {r['children']} | {k} ({pct(r['beat_dev_rate'])}%) "
            f"| {held} "
            f"| {pts(r['mean_dev_gain']) if k else '-'} | {pts(r['mean_test_gain']) if k else '-'} |"
        )
    return out


def b_operators(S: dict) -> str:
    out = []
    for t in S["order"]:
        T = S["tasks"][t]
        out.append(f"**{T['title']}** (all {len(T['seeds'])} seeds pooled; held-out columns use the first {T['operator_n_test']} test items)\n")
        out.extend(_op_table(T["operators"], T["operator_n_test"]))
        out.append("")
    return "\n".join(out).rstrip()


def b_operators_alt(S: dict) -> str:
    out = []
    for t in S["order"]:
        T = S["tasks"][t]
        if not T["operators_alt"]:
            continue
        rows = [r for r in T["operators_alt"] if r["operator"] in ("llm_rewrite", "llm_reflect")]
        main = {r["operator"]: r for r in T["operators"]}
        out.append(f"**{T['title']}**: LLM-operator children, 0.5B mutator (main arm) vs 1.5B mutator\n")
        out.append("| Operator | Mutator | Children | Beat parent on dev | Mean gain on dev (pts) |")
        out.append("|---|---|---|---|---|")
        for r in rows:
            m = main.get(r["operator"])
            if m:
                out.append(f"| `{r['operator']}` | 0.5B | {m['children']} | {m['beat_dev']} ({pct(m['beat_dev_rate'])}%) | {pts(m['mean_dev_gain']) if m['beat_dev'] else '-'} |")
            out.append(f"| `{r['operator']}` | 1.5B | {r['children']} | {r['beat_dev']} ({pct(r['beat_dev_rate'])}%) | {pts(r['mean_dev_gain']) if r['beat_dev'] else '-'} |")
        out.append("")
    return "\n".join(out).rstrip() or "Not run."


def b_pilot(_: dict) -> str:
    rows = ["| Task | First-draft seed | Best seed (dev) | Random search | Evolved winner |", "|---|---|---|---|---|"]
    meta = ""
    for t in ("csqa", "boolq", "synthetic"):
        f = PILOT / f"{t}-s0.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        arms, c = d["report"]["arms"], d["config"]
        cells = [f"{pct(arms[k]['test_acc'])} [{pct(arms[k]['test_ci'][0])}, {pct(arms[k]['test_ci'][1])}]"
                 for k in ("seed0", "best_seed", "random", "evolved")]
        rows.append(f"| {d['task']['title']} | " + " | ".join(cells) + " |")
        meta = (f"{d['device']} {d['dtype']}, {c['generations']} generations, population {c['pop']}, {c['children']} children, "
                f"{d['task']['n_dev']} dev items, {d['task']['n_test']} test items, seed {d['seed']}, commit `{d['git']['commit']}`")
    return f"Pilot settings: {meta}. Test accuracy (%) with 95% bootstrap CI.\n\n" + "\n".join(rows)


def b_curse(S: dict) -> str:
    rows = ["| Task | Prompt | Dev (%) | Test (%) | Gap (pts) |", "|---|---|---|---|---|"]
    for t in S["order"]:
        T = S["tasks"][t]
        for a in ("best_seed", "random", "evolved"):
            A = T["arms"][a]
            rows.append(f"| {T['title'] if a == 'best_seed' else ''} | {A['label']} | {pct(A['dev_mean'])} | {pct(A['test_mean'])} | {pts(A['gap_mean'])} |")
    return "Means over seeds; the gap is dev minus test of the prompt each method picked on dev.\n\n" + "\n".join(rows)


BLOCKS = {"setup": b_setup, "headline": b_headline, "paired": b_paired, "verdicts": b_verdicts,
          "operators": b_operators, "operators_alt": b_operators_alt, "curse": b_curse, "pilot": b_pilot}


def _ci(S: dict, t: str, key: str = "evolved_vs_random") -> dict:
    return S["tasks"][t]["comparisons"][key]


def _claim_ties(S: dict) -> bool:
    return all(_ci(S, t)["lo"] <= 0 <= _ci(S, t)["hi"] for t in ("csqa", "boolq") if t in S["tasks"])


def _claim_synthetic_ahead(S: dict) -> bool:
    c = _ci(S, "synthetic")
    return c["lo"] > 0 and c["seeds_ahead"] == len(S["tasks"]["synthetic"]["seeds"])


def _claim_seed_spread(S: dict) -> bool:
    A = S["tasks"]["synthetic"]["arms"]
    return A["evolved"]["test_sd"] > 2 * A["random"]["test_sd"] or min(
        p["p_mcnemar"] for p in [_ci(S, "synthetic")["per_seed"][-1]]) > 0.05


def _claim_beats_first_draft(S: dict) -> bool:
    return all(_ci(S, t, "evolved_vs_seed0")["lo"] > 0 for t in ("csqa", "synthetic"))


def _claim_curse(S: dict) -> bool:
    ok = all(S["tasks"][t]["arms"][a]["gap_mean"] > 0 for t in ("csqa", "boolq") for a in ("best_seed", "random", "evolved"))
    return ok and S["tasks"]["synthetic"]["arms"]["best_seed"]["gap_mean"] > 0


def _claim_boolq_no_headroom(S: dict) -> bool:
    T = S["tasks"]["boolq"]
    A = T["arms"]
    return (A["seed0"]["test_mean"] < T["majority_rate"]
            and A["evolved"]["test_mean"] <= T["majority_rate"] + 0.03
            and A["random"]["test_mean"] <= T["majority_rate"] + 0.03)


def _claim_format_ops(S: dict) -> bool:
    T = S["tasks"]["synthetic"]["operators"]
    rate = {r["operator"]: r["beat_dev_rate"] for r in T}
    return rate["add_shot"] > rate["llm_rewrite"] and rate["add_shot"] > rate["llm_reflect"]


# Qualitative sentences in README.md that the numbers must keep supporting. Keyed by the marker
# ``<!-- claim:name -->`` that follows the sentence.
CLAIMS = {
    "beats_first_draft": ("evolved beats the first-draft seed on CommonsenseQA and synthetic (CI above zero)", _claim_beats_first_draft),
    "ties_random": ("evolved minus random search has an interval around zero on CommonsenseQA and BoolQ", _claim_ties),
    "synthetic_ahead": ("evolved beats random search on synthetic, CI above zero and ahead in every seed", _claim_synthetic_ahead),
    "synthetic_spread": ("on synthetic the evolved arm varies more across seeds than random search, or the last seed is not significant", _claim_seed_spread),
    "boolq_no_headroom": ("BoolQ first-draft prompt is below the majority rate; evolved and random stay within 3 points of it", _claim_boolq_no_headroom),
    "winners_curse": ("dev exceeds test for every dev-picked prompt on CommonsenseQA and BoolQ, and for the best seed on synthetic", _claim_curse),
    "format_ops": ("on synthetic, add_shot beats its parent on dev more often than either LLM operator", _claim_format_ops),
}


def render(text: str, S: dict) -> str:
    for name, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- BEGIN {name} -->)\s*(.*?)\s*(<!-- END {name} -->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md is missing the <!-- BEGIN {name} --> / END markers")
        text = pat.sub(lambda m, f=fn: m.group(1) + chr(10) + f(S) + chr(10) + m.group(3), text)
    return text


def _same_json(a: bytes, b: bytes, tol: float = 2e-6) -> bool:
    def eq(x, y) -> bool:
        if isinstance(x, float) or isinstance(y, float):
            return isinstance(x, int | float) and isinstance(y, int | float) and abs(x - y) <= tol
        if isinstance(x, dict):
            return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
        if isinstance(x, list):
            return isinstance(y, list) and len(x) == len(y) and all(eq(p, q) for p, q in zip(x, y, strict=True))
        return x == y
    return eq(json.loads(a), json.loads(b))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--readme", default=str(README))
    ap.add_argument("--docs", default=str(DOCS))
    args = ap.parse_args(argv)

    S = summarise(args.results)
    problems: list[str] = []
    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    new = render(text, S)
    for name, (desc, fn) in CLAIMS.items():
        if f"<!-- claim:{name} -->" not in text:
            problems.append(f"README.md lacks the <!-- claim:{name} --> marker")
        elif not fn(S):
            problems.append(f"claim '{name}' no longer holds: {desc}")
    for m in re.finditer(r"<!-- claim:(\w+) -->", text):
        if m.group(1) not in CLAIMS:
            problems.append(f"unknown claim marker {m.group(1)}")

    data_dir = Path(args.docs) / "data"
    files = viewer_files(args.results)
    if args.write:
        from promptlab.summary import export_demo
        export_demo(args.results, args.docs)
        if new != text:
            readme.write_text(new, encoding="utf-8")
        print("README blocks and docs/data regenerated")
    else:
        if new != text:
            problems.append("README.md tables are out of sync with results/ (run scripts/check_numbers.py --write)")
        have = {p.name for p in data_dir.glob("*.json")} if data_dir.exists() else set()
        if have != set(files):
            problems.append(f"docs/data files differ: extra {sorted(have - set(files))}, missing {sorted(set(files) - have)}")
        for name, data in files.items():
            f = data_dir / name
            if f.exists() and not (data == f.read_bytes().replace(b"
", b"
") or (name == "summary.json" and _same_json(data, f.read_bytes()))):
                problems.append(f"docs/data/{name} is out of sync with results/")
    for p in problems:
        print("PROBLEM:", p)
    if not problems and args.check:
        print("numbers in sync")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
