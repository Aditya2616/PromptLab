import json

from promptlab.cli import main


def test_run_then_export_demo(tmp_path, capsys):
    res, cache, docs = tmp_path / "results", tmp_path / "cache", tmp_path / "docs"
    code = main(["run", "--task", "synthetic", "--backend", "fake", "--generations", "2", "--pop", "3",
                 "--children", "3", "--dev-n", "20", "--test-n", "30", "--out", str(res), "--cache", str(cache)])
    assert code == 0
    out = capsys.readouterr().out
    assert "evolved" in out and "wrote" in out
    run_file = res / "synthetic-s0.json"
    assert json.loads(run_file.read_text(encoding="utf-8"))["task"]["name"] == "synthetic"
    assert main(["export-demo", "--results", str(res), "--docs", str(docs)]) == 0
    index = json.loads((docs / "data" / "index.json").read_text(encoding="utf-8"))
    assert index["runs"][0]["file"] == "synthetic-s0.json"
    assert (docs / "data" / "synthetic-s0.json").exists()


def test_export_skips_non_run_files(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "notes.json").write_text("{}", encoding="utf-8")
    assert main(["export-demo", "--results", str(res), "--docs", str(tmp_path / "docs")]) == 0
    index = json.loads((tmp_path / "docs" / "data" / "index.json").read_text(encoding="utf-8"))
    assert index["runs"] == []
