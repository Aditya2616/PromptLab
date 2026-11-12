from __future__ import annotations

import os
from collections.abc import Sequence

from .base import ChatRequest


def default_threads() -> int:
    env = os.environ.get("PROMPTLAB_THREADS")
    return int(env) if env else max(1, (os.cpu_count() or 4) // 4)


class LocalHFBackend:
    """Greedy chat completions from a small transformers model on CPU."""

    name = "local"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", batch_size: int = 16) -> None:
        n = default_threads()
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        import torch

        torch.set_num_threads(n)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.batch_size = batch_size
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        texts = [
            self._tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
            for r in requests
        ]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        for s in range(0, len(order), self.batch_size):
            ids = order[s : s + self.batch_size]
            batch = self._tok([texts[i] for i in ids], return_tensors="pt", padding=True)
            new = max(requests[i].max_tokens for i in ids)
            with self._torch.no_grad():
                gen = self._lm.generate(
                    **batch, max_new_tokens=new, do_sample=False, pad_token_id=self._tok.pad_token_id
                )
            width = batch["input_ids"].shape[1]
            dec = self._tok.batch_decode(gen[:, width:], skip_special_tokens=True)
            for i, d in zip(ids, dec, strict=True):
                out[i] = d
        return out
