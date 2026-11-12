from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from .base import ChatBackend, ChatRequest


def request_key(backend: str, model: str, req: ChatRequest, variant: str = "") -> str:
    """Cache key. ``variant`` (device and dtype for local models) keeps outputs from different
    numerics apart, so one run never mixes CPU float32 and GPU float16 completions."""
    payload = json.dumps(
        [backend, model, req.messages, req.max_tokens, req.temperature, variant],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedBackend:
    """Wraps a backend with a persistent SQLite cache of every model call.

    The cache is what makes runs reproducible and resumable: re-running an
    interrupted experiment replays finished calls from disk.
    """

    def __init__(self, inner: ChatBackend, path: str | Path = ":memory:") -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.variant = getattr(inner, "variant", "")
        self._lock = threading.Lock()
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS calls (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()
        self.hits = 0
        self.misses = 0

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        keys = [request_key(self.name, self.model, r, self.variant) for r in requests]
        out: list[str | None] = [None] * len(requests)
        todo: dict[str, list[int]] = {}
        with self._lock:
            for i, k in enumerate(keys):
                row = self._db.execute("SELECT v FROM calls WHERE k=?", (k,)).fetchone()
                if row is not None:
                    out[i] = row[0]
                    self.hits += 1
                else:
                    todo.setdefault(k, []).append(i)
        if todo:
            idxs = [v[0] for v in todo.values()]
            results = self.inner.complete([requests[i] for i in idxs])
            with self._lock:
                for (k, ids), text in zip(todo.items(), results, strict=True):
                    self._db.execute("INSERT OR REPLACE INTO calls VALUES (?, ?)", (k, text))
                    for i in ids:
                        out[i] = text
                    self.misses += 1
                    self.hits += len(ids) - 1
                self._db.commit()
        return [o if o is not None else "" for o in out]

    def stats(self) -> dict[str, int]:
        return {"model_calls": self.misses, "cache_hits": self.hits}

    def close(self) -> None:
        self._db.close()


def cache_path(directory: str | Path, backend: ChatBackend) -> Path:
    """One cache file per backend, model and (for local models) device and dtype."""
    tag = "-".join(x for x in (backend.name, backend.model.replace("/", "_"), getattr(backend, "variant", "")) if x)
    return Path(directory) / f"{tag}.sqlite"
