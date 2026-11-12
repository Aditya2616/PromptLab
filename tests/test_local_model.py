import pytest

from promptlab.models.base import ChatRequest

pytestmark = pytest.mark.slow


def test_local_backend_answers_with_real_model():
    from promptlab.models.local_hf import LocalHFBackend

    b = LocalHFBackend("Qwen/Qwen2.5-0.5B-Instruct", batch_size=2)
    msgs = [{"role": "system", "content": "Reply with only the answer."},
            {"role": "user", "content": "What is 2 + 2?"}]
    out = b.complete([ChatRequest(msgs, max_tokens=4), ChatRequest(msgs, max_tokens=4)])
    assert len(out) == 2 and "4" in out[0] and out[0] == out[1]
