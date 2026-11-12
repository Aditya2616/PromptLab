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

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        batch_size: int = 16,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        n = default_threads()
        os.environ.setdefault("OMP_NUM_THREADS", str(n))
        import torch

        torch.set_num_threads(n)
        device = device or os.environ.get("PROMPTLAB_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or os.environ.get("PROMPTLAB_DTYPE") or ("float16" if device.startswith("cuda") else "float32")
        self.device, self.dtype = device, dtype
        self.variant = f"{device}-{dtype}"
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = model
        self.batch_size = self.max_batch = batch_size
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(model)
        self._tok.padding_side = "left"
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        self._lm = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).eval().to(device)

    def _generate(self, texts: list[str], new: int) -> list[str]:
        batch = self._tok(texts, return_tensors="pt", padding=True).to(self.device)
        with self._torch.no_grad():
            gen = self._lm.generate(
                **batch, max_new_tokens=new, do_sample=False, pad_token_id=self._tok.pad_token_id
            )
        width = batch["input_ids"].shape[1]
        return self._tok.batch_decode(gen[:, width:].cpu(), skip_special_tokens=True)

    def complete(self, requests: Sequence[ChatRequest]) -> list[str]:
        texts = [
            self._tok.apply_chat_template(r.messages, tokenize=False, add_generation_prompt=True)
            for r in requests
        ]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        self.batch_size = self.max_batch  # an earlier out-of-memory retry only shrinks batches temporarily
        s = 0
        while s < len(order):
            ids = order[s : s + self.batch_size]
            try:
                dec = self._generate([texts[i] for i in ids], max(requests[i].max_tokens for i in ids))
            except self._torch.cuda.OutOfMemoryError:
                if self.batch_size == 1:
                    raise
                self._torch.cuda.empty_cache()
                self.batch_size = max(1, self.batch_size // 2)  # retry the same slice smaller
                continue
            for i, d in zip(ids, dec, strict=True):
                out[i] = d
            s += len(ids)
        return out
