from __future__ import annotations

import os

import pandas as pd

from .base import Example, Task
from .boolq import BoolQ
from .csqa import CommonsenseQA
from .synthetic import SyntheticTask

TASK_NAMES = ["csqa", "boolq", "synthetic"]

_PARQUET = {
    "csqa": (
        "tau/commonsense_qa",
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ),
    "boolq": (
        "google/boolq",
        "data/train-00000-of-00001.parquet",
        "data/validation-00000-of-00001.parquet",
    ),
}


def fetch_frames(name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or reuse from the Hugging Face cache) the parquet files for a task."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import hf_hub_download

    repo, tr, va = _PARQUET[name]
    return (
        pd.read_parquet(hf_hub_download(repo, tr, repo_type="dataset")),
        pd.read_parquet(hf_hub_download(repo, va, repo_type="dataset")),
    )


def load_task(name: str, dev_n: int = 80, test_n: int = 200) -> Task:
    if name == "synthetic":
        return SyntheticTask.generate(dev_n=dev_n, test_n=test_n)
    if name not in _PARQUET:
        raise ValueError(f"unknown task {name!r}; choose from {TASK_NAMES}")
    train, val = fetch_frames(name)
    cls = CommonsenseQA if name == "csqa" else BoolQ
    return cls.from_frames(train, val, dev_n=dev_n, test_n=test_n)


__all__ = ["BoolQ", "CommonsenseQA", "Example", "SyntheticTask", "TASK_NAMES", "Task", "load_task"]
