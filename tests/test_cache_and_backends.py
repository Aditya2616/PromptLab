import json

import httpx

from promptlab.models.base import ChatRequest
from promptlab.models.cache import CachedBackend, request_key
from promptlab.models.fake import ScriptedBackend
from promptlab.models.openai_compat import OpenAICompatBackend


def req(text: str, **kw) -> ChatRequest:
    return ChatRequest([{"role": "user", "content": text}], **kw)


def test_cache_hits_and_misses(tmp_path):
    inner = ScriptedBackend(lambda r: r.messages[-1]["content"].upper())
    cb = CachedBackend(inner, tmp_path / "c.sqlite")
    assert cb.complete([req("a"), req("b"), req("a")]) == ["A", "B", "A"]
    assert inner.calls == 2  # duplicate inside one batch is only computed once
    assert cb.complete([req("a"), req("c")]) == ["A", "C"]
    assert inner.calls == 3
    assert cb.stats()["model_calls"] == 3


def test_cache_persists_across_instances(tmp_path):
    path = tmp_path / "c.sqlite"
    inner = ScriptedBackend(lambda r: "x")
    CachedBackend(inner, path).complete([req("q")])
    inner2 = ScriptedBackend(lambda r: "y")
    assert CachedBackend(inner2, path).complete([req("q")]) == ["x"]
    assert inner2.calls == 0


def test_key_depends_on_params():
    a = request_key("b", "m", req("q", max_tokens=4))
    assert a != request_key("b", "m", req("q", max_tokens=8))
    assert a != request_key("b", "m2", req("q", max_tokens=4))
    assert a == request_key("b", "m", req("q", max_tokens=4))


def test_openai_backend_with_stub_transport():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((str(request.url), body))
        return httpx.Response(
            200,
            json={
                "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok:" + body["messages"][-1]["content"]}}],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    b = OpenAICompatBackend("gpt-test", api_key="k", base_url="http://local.test/v1", http_client=client)
    out = b.complete([req("one", max_tokens=5), req("two")])
    assert out == ["ok:one", "ok:two"]
    assert all(u.endswith("/chat/completions") for u, _ in seen)
    assert {s[1]["max_tokens"] for s in seen} == {5, 16}
