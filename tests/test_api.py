import json
import time

from fastapi.testclient import TestClient

from promptlab import api

client = TestClient(api.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_validation_rejects_bad_request():
    assert client.post("/api/live", json={"task": "nope"}).status_code == 422
    assert client.post("/api/live", json={"generations": 0}).status_code == 422
    assert client.get("/api/live/missing").status_code == 404


def test_live_run_streams_generations_then_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "CACHE", tmp_path)
    r = client.post("/api/live", json={"task": "synthetic", "backend": "fake", "generations": 2,
                                       "pop": 3, "children": 3, "dev_n": 20, "test_n": 20})
    assert r.status_code == 202
    run_id = r.json()["id"]
    events, ended = [], False
    with client.stream("GET", f"/api/live/{run_id}/stream") as s:
        for line in s.iter_lines():
            if line.startswith("event: end"):
                ended = True
            elif line.startswith("data: ") and not ended:
                events.append(json.loads(line[6:]))
    assert ended
    assert [len(e["generations"]) for e in events[:3]] == [1, 2, 3]
    assert events[-1]["status"] == "done" and "report" in events[-1]
    for _ in range(20):
        if client.get(f"/api/live/{run_id}").json()["status"] == "done":
            break
        time.sleep(0.1)
    assert client.get(f"/api/live/{run_id}").json()["status"] == "done"
