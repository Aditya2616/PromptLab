import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from promptlab.cli import main
from promptlab.summary import export_demo, summarise, viewer_files

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def two_seed_results(tmp_path_factory):
    base = tmp_path_factory.mktemp("res")
    for seed in (0, 1):
        code = main(["run", "--task", "synthetic", "--backend", "fake", "--generations", "2", "--pop", "3",
                     "--children", "3", "--dev-n", "20", "--test-n", "40", "--seed", str(seed),
                     "--out", str(base / "results"), "--cache", str(base / "cache")])
        assert code == 0
    return base / "results"


def test_summary_arithmetic(two_seed_results):
    S = summarise(two_seed_results)
    T = S["tasks"]["synthetic"]
    assert T["seeds"] == [0, 1] and S["totals"]["runs"] == 2
    for arm in T["arms"].values():
        assert len(arm["test"]) == 2
        assert arm["test_mean"] == pytest.approx(np.mean(arm["test"]))
        assert arm["test_sd"] == pytest.approx(np.std(arm["test"], ddof=1))
        assert arm["gap_mean"] == pytest.approx(np.mean(arm["gap"]))
        lo, hi = arm["pooled_ci"]
        assert lo <= arm["test_mean"] <= hi
    c = T["comparisons"]["evolved_vs_random"]
    assert c["diff"] == pytest.approx(np.mean(c["seed_diffs"]))
    assert c["seeds_ahead"] + c["seeds_behind"] <= 2
    assert c["lo"] <= c["diff"] <= c["hi"]
    for op in T["operators"]:
        assert op["beat_dev"] <= op["children"]
        assert op["beat_test"] <= op["beat_dev"]
    assert S["totals"]["calls"] == T["calls"]


def test_viewer_files_and_export_round_trip(two_seed_results, tmp_path):
    files = viewer_files(two_seed_results)
    assert set(files) == {"index.json", "summary.json", "synthetic-s0.json", "synthetic-s1.json"}
    index = json.loads(files["index.json"])
    assert [(r["name"], r["seed"]) for r in index["runs"]] == [("synthetic", 0), ("synthetic", 1)]
    assert all(r["device"] and r["dtype"] for r in index["runs"])
    export_demo(two_seed_results, tmp_path)
    assert {p.name for p in (tmp_path / "data").glob("*.json")} == set(files)
    (tmp_path / "data" / "stale-s9.json").write_text("{}", encoding="utf-8")
    export_demo(two_seed_results, tmp_path)
    assert not (tmp_path / "data" / "stale-s9.json").exists()


def test_readme_and_viewer_data_match_results():
    """The published README tables and docs/data are generated from results/ and must be in sync."""
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_numbers.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
