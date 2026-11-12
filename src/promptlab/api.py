"""FastAPI service: health, recorded runs, and optional live runs streamed over SSE."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__

ROOT = Path(os.environ.get("PROMPTLAB_ROOT", Path(__file__).resolve().parents[2]))
DOCS = ROOT / "docs"
CACHE = ROOT / "runs"

app = FastAPI(title="PromptLab", version=__version__)


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    backend: str


class LiveRequest(BaseModel):
    task: Literal["csqa", "boolq", "synthetic"] = "synthetic"
    backend: Literal["fake", "local", "openai"] = "fake"
    seed: int = 0
    generations: int = Field(4, ge=1, le=10)
    pop: int = Field(5, ge=2, le=10)
    children: int = Field(6, ge=1, le=12)
    dev_n: int = Field(60, ge=20, le=200)
    test_n: int = Field(100, ge=20, le=400)


class LiveStatus(BaseModel):
    id: str
    status: Literal["running", "done", "error"]
    error: str | None = None


_RUNS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _worker(run_id: str, req: LiveRequest) -> None:
    from .evolve import EvoConfig
    from .models import CachedBackend, make_backend
    from .pipeline import run_experiment
    from .tasks import load_task

    slot = _RUNS[run_id]
    try:
        task = load_task(req.task, dev_n=req.dev_n, test_n=req.test_n)
        raw = make_backend(req.backend)
        if hasattr(raw, "bind"):
            raw.bind(task)
        backend = CachedBackend(raw, CACHE / f"{raw.name}-{raw.model.replace('/', '_')}.sqlite")
        cfg = EvoConfig(req.generations, req.pop, req.children, req.seed)

        def push(state: dict) -> None:
            slot["events"].append(state)

        run_experiment(task, backend, cfg, on_state=push)
        slot["status"] = "done"
    except Exception as exc:  # surfaced to the client, not swallowed
        slot["status"] = "error"
        slot["error"] = f"{type(exc).__name__}: {exc}"


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(version=__version__, backend=os.environ.get("PROMPTLAB_BACKEND", "local"))


@app.post("/api/live", response_model=LiveStatus, status_code=202)
def start_live(req: LiveRequest) -> LiveStatus:
    with _LOCK:
        if any(r["status"] == "running" for r in _RUNS.values()):
            raise HTTPException(409, "a live run is already in progress")
        run_id = uuid.uuid4().hex[:8]
        _RUNS[run_id] = {"status": "running", "error": None, "events": []}
    threading.Thread(target=_worker, args=(run_id, req), daemon=True).start()
    return LiveStatus(id=run_id, status="running")


@app.get("/api/live/{run_id}", response_model=LiveStatus)
def live_status(run_id: str) -> LiveStatus:
    slot = _RUNS.get(run_id)
    if slot is None:
        raise HTTPException(404, "unknown run")
    return LiveStatus(id=run_id, status=slot["status"], error=slot["error"])


@app.get("/api/live/{run_id}/stream")
async def live_stream(run_id: str) -> StreamingResponse:
    slot = _RUNS.get(run_id)
    if slot is None:
        raise HTTPException(404, "unknown run")

    async def gen():
        sent = 0
        while True:
            events = slot["events"]
            while sent < len(events):
                yield f"data: {json.dumps(events[sent])}\n\n"
                sent += 1
            if slot["status"] != "running" and sent >= len(slot["events"]):
                yield f"event: end\ndata: {json.dumps({'status': slot['status'], 'error': slot['error']})}\n\n"
                return
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="text/event-stream")


if DOCS.exists():
    app.mount("/", StaticFiles(directory=DOCS, html=True), name="site")
