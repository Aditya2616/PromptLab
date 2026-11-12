"""The generation loop: select parents, mutate, score on dev, keep the strongest."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .evaluate import evaluate
from .models.base import ChatBackend
from .operators import MAX_WORDS, OpContext, Parent, choose_operator, guard
from .prompts import Prompt
from .select import pick_parent, rank_order
from .stats import wilson
from .tasks.base import Task


@dataclass
class EvoConfig:
    generations: int = 4
    pop: int = 5
    children: int = 6
    seed: int = 0
    selection: str = "tournament"  # or "pareto"
    tournament: int = 2
    max_words: int = MAX_WORDS
    attempts_per_child: int = 30


@dataclass
class Node:
    id: str
    gen: int
    parents: list[str]
    operator: str
    prompt: Prompt
    scores: np.ndarray
    outputs: list[str]
    culled_at: int | None = None
    origin: str = "evolution"

    @property
    def acc(self) -> float:
        return float(self.scores.mean())


@dataclass
class EvoResult:
    nodes: dict[str, Node] = field(default_factory=dict)
    generations: list[dict] = field(default_factory=list)
    final_population: list[str] = field(default_factory=list)
    evaluated: int = 0

    def best(self) -> Node:
        pop = [self.nodes[i] for i in self.final_population]
        accs = np.array([n.acc for n in pop])
        words = np.array([n.prompt.n_words for n in pop])
        return pop[int(rank_order(accs, words)[0])]


def node_to_dict(task: Task, n: Node) -> dict:
    k, m = int(n.scores.sum()), len(n.scores)
    lo, hi = wilson(k, m)
    return {
        "id": n.id,
        "gen": n.gen,
        "parents": n.parents,
        "operator": n.operator,
        "origin": n.origin,
        "text": task.prompt_text(n.prompt),
        "sentences": list(n.prompt.sentences),
        "fmt": n.prompt.fmt,
        "n_shots": len(n.prompt.shots),
        "dev_acc": n.acc,
        "dev_ci": [lo, hi],
        "dev_bits": "".join(str(int(x)) for x in n.scores),
        "culled_at": n.culled_at,
    }


def score_node(
    task: Task, backend: ChatBackend, prompt: Prompt, gen: int, parents: list[str], operator: str,
    origin: str = "evolution",
) -> Node:
    scores, outputs = evaluate(task, prompt, task.dev, backend)
    return Node(prompt.id, gen, parents, operator, prompt, scores, outputs, origin=origin)


def run_evolution(
    task: Task,
    backend: ChatBackend,
    cfg: EvoConfig,
    on_generation: Callable[[EvoResult], None] | None = None,
) -> EvoResult:
    """Evolve prompts using only the dev split. The test split stays locked throughout."""
    task.lock_test()
    rng = np.random.default_rng(cfg.seed)
    ctx = OpContext(task, backend, rng, cfg.max_words)
    res = EvoResult()

    seeds = task.seeds()
    for p in seeds:
        if p.id in res.nodes:
            continue
        res.nodes[p.id] = score_node(task, backend, p, 0, [], "seed", origin="seed")
        res.evaluated += 1
    born0 = list(res.nodes)
    pop = _select(res, born0, cfg.pop, gen=0)
    _log(res, 0, born0, pop)
    if on_generation:
        on_generation(res)

    for g in range(1, cfg.generations + 1):
        born: list[str] = []
        for _ in range(cfg.children):
            child = _make_child(res, pop, ctx, cfg)
            if child is None:
                continue
            prompt, parents, op = child
            res.nodes[prompt.id] = score_node(task, backend, prompt, g, parents, op)
            res.evaluated += 1
            born.append(prompt.id)
        pop = _select(res, [*pop, *born], cfg.pop, gen=g)
        _log(res, g, born, pop)
        if on_generation:
            on_generation(res)

    res.final_population = pop
    return res


def _make_child(
    res: EvoResult, pop: list[str], ctx: OpContext, cfg: EvoConfig
) -> tuple[Prompt, list[str], str] | None:
    mat = np.stack([res.nodes[i].scores for i in pop])
    for _ in range(cfg.attempts_per_child):
        a = pick_parent(cfg.selection, mat, ctx.rng, cfg.tournament)
        op = choose_operator(ctx.rng, True, 2 if len(pop) > 1 else 1)
        idxs = [a]
        if op.arity == 2:
            b = pick_parent(cfg.selection, mat, ctx.rng, cfg.tournament)
            tries = 0
            while b == a and tries < 5:
                b = pick_parent(cfg.selection, mat, ctx.rng, cfg.tournament)
                tries += 1
            if b == a:
                continue
            idxs.append(b)
        parents = [
            Parent(res.nodes[pop[i]].prompt, res.nodes[pop[i]].scores, res.nodes[pop[i]].outputs)
            for i in idxs
        ]
        child = guard(op.apply(parents, ctx), parents, cfg.max_words)
        if child is not None and child.id not in res.nodes:
            return child, [pop[i] for i in idxs], op.name
    return None


def _select(res: EvoResult, candidates: list[str], k: int, gen: int) -> list[str]:
    nodes = [res.nodes[i] for i in candidates]
    acc = np.array([n.acc for n in nodes])
    words = np.array([n.prompt.n_words for n in nodes])
    keep = rank_order(acc, words)[:k]
    kept = {candidates[int(i)] for i in keep}
    for cid in candidates:
        if cid not in kept and res.nodes[cid].culled_at is None:
            res.nodes[cid].culled_at = gen
    return [candidates[int(i)] for i in keep]


def _log(res: EvoResult, gen: int, born: list[str], pop: list[str]) -> None:
    accs = [res.nodes[i].acc for i in pop]
    res.generations.append(
        {
            "gen": gen,
            "born": born,
            "survivors": pop,
            "best_dev": float(max(accs)),
            "mean_dev": float(np.mean(accs)),
            "evaluated_total": res.evaluated,
        }
    )
    res.final_population = pop
