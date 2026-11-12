"""Random-search baseline at an equal evaluation budget: same operators, no selection feedback."""

from __future__ import annotations

import numpy as np

from .evolve import EvoConfig, EvoResult, Node, score_node
from .models.base import ChatBackend
from .operators import OpContext, Parent, choose_operator, guard
from .prompts import Prompt
from .select import rank_order
from .tasks.base import Task


def random_search(
    task: Task, backend: ChatBackend, cfg: EvoConfig, evo: EvoResult
) -> tuple[list[Node], Node]:
    """Score the same seeds, then ``budget`` random multi-step mutants of random seeds.

    Candidates are produced without looking at any score (chains of 1-3 mutations from a
    random seed), so only the final pick uses the dev split. ``llm_reflect`` is excluded
    because it needs scores. The budget equals the number of variants evolution scored.
    """
    task.lock_test()
    rng = np.random.default_rng(cfg.seed + 10_000)
    ctx = OpContext(task, backend, rng, cfg.max_words)
    seed_nodes = [n for n in evo.nodes.values() if n.origin == "seed"]
    budget = max(0, evo.evaluated - len(seed_nodes))
    seen = {n.id for n in seed_nodes}
    cands: list[Node] = []
    attempts = 0
    while len(cands) < budget and attempts < budget * 40:
        attempts += 1
        cur: Prompt = seed_nodes[int(rng.integers(len(seed_nodes)))].prompt
        for _ in range(int(rng.integers(1, 4))):
            op = choose_operator(rng, allow_feedback=False, parents_available=1)
            parents = [Parent(cur)]
            nxt = guard(op.apply(parents, ctx), parents, cfg.max_words)
            if nxt is not None:
                cur = nxt
        if cur.id in seen:
            continue
        seen.add(cur.id)
        cands.append(score_node(task, backend, cur, 1, [], "random", origin="random"))
    pool = [*seed_nodes, *cands]
    acc = np.array([n.acc for n in pool])
    words = np.array([n.prompt.n_words for n in pool])
    return cands, pool[int(rank_order(acc, words)[0])]
