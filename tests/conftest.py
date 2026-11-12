import pytest

from promptlab.models.cache import CachedBackend
from promptlab.models.fake import SimBackend
from promptlab.tasks import SyntheticTask


@pytest.fixture()
def task():
    return SyntheticTask.generate(dev_n=40, test_n=60, pool_n=12, seed=3)


@pytest.fixture()
def sim(task):
    b = SimBackend()
    b.bind(task)
    return b


@pytest.fixture()
def backend(sim):
    return CachedBackend(sim)
