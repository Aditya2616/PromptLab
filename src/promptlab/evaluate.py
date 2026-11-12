from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .models.base import ChatBackend, ChatRequest
from .prompts import Prompt
from .tasks.base import Example, Task


def evaluate(
    task: Task, prompt: Prompt, examples: Sequence[Example], backend: ChatBackend
) -> tuple[np.ndarray, list[str]]:
    """Run a prompt over examples. Returns (0/1 score vector, raw outputs)."""
    reqs = [
        ChatRequest(task.messages(prompt, ex), max_tokens=task.max_new_tokens) for ex in examples
    ]
    outputs = backend.complete(reqs)
    scores = np.array([task.score(o, ex) for o, ex in zip(outputs, examples, strict=True)], dtype=np.int8)
    return scores, outputs
