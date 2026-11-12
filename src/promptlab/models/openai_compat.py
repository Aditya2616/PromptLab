from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import httpx
from openai import OpenAI

from .base import ChatRequest


class OpenAICompatBackend:
    """Chat completions over the OpenAI API, or any OpenAI-compatible server.

    Set ``base_url`` to a local llama.cpp / vLLM / Ollama endpoint to stay offline.
    ``http_client`` lets tests inject an ``httpx.MockTransport``.
    """

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        workers: int = 4,
    ) -> None:
        self.model = model
        self.workers = workers
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            http_client=http_client,
            max_retries=2,
        )

    def _one(self, req: ChatRequest) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=req.messages,  # type: ignore[arg-type]
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return resp.choices[0].message.content or ""

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, requests))
