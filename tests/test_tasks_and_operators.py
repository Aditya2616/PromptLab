import numpy as np
import pandas as pd
import pytest

from promptlab.models.fake import ScriptedBackend
from promptlab.operators import (
    OPERATORS,
    AddClause,
    AddShot,
    Crossover,
    DropClause,
    LLMReflect,
    LLMRewrite,
    OpContext,
    Parent,
    RemoveShot,
    Reorder,
    SwapFormat,
    clean_instruction,
    guard,
)
from promptlab.prompts import Prompt
from promptlab.tasks import BoolQ, CommonsenseQA
from promptlab.tasks.boolq import extract_yes_no
from promptlab.tasks.csqa import extract_letter
from promptlab.tasks.synthetic import normalise


def test_prompt_id_is_stable_and_content_based():
    a = Prompt.from_text("Answer. Be brief.", "Only the letter.", ("s1",))
    b = Prompt.from_text("Answer. Be brief.", "Only the letter.", ("s1",))
    assert a.id == b.id and a.id != Prompt.from_text("Answer.").id
    assert a.system == "Answer. Be brief. Only the letter."


@pytest.mark.parametrize(
    "out,want",
    [("B", "B"), ("(C) because", "C"), ("The answer is D.", "D"), ("Answer: e", "E"),
     ("A person walks", "A"), ("I think maybe", None), ("", None)],
)
def test_extract_letter(out, want):
    assert extract_letter(out) == want


def test_extract_yes_no():
    assert extract_yes_no("Yes.") == "yes"
    assert extract_yes_no("The answer is No") == "no"
    assert extract_yes_no("True") == "yes" and extract_yes_no("unclear") is None


def test_synthetic_scoring_rewards_bare_answers(task):
    ex = task.dev[0]
    assert task.score(ex.gold, ex) == 1
    assert task.score(f"The answer is {ex.gold}.", ex) == 0
    assert normalise(" `7, 19, 42` .") == "7,19,42"


def test_synthetic_splits_are_disjoint_and_seeded(task):
    ids = [e.id for e in [*task.pool, *task.dev, task.test[0]]]
    assert len(ids) == len(set(ids))
    from promptlab.tasks import SyntheticTask

    again = SyntheticTask.generate(dev_n=40, test_n=60, pool_n=12, seed=3)
    assert [e.question for e in again.dev] == [e.question for e in task.dev]


def test_test_split_is_locked_until_unlocked(task):
    task.lock_test()
    with pytest.raises(PermissionError):
        _ = task.test
    task.unlock_test()
    assert len(task.test) == 60


def test_messages_include_shots_as_chat_turns(task):
    p = Prompt.from_text("Be brief.", "", (task.pool[0].id, task.pool[1].id))
    m = task.messages(p, task.dev[0])
    assert [x["role"] for x in m] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert m[-1]["content"] == task.dev[0].question


def test_csqa_and_boolq_from_frames():
    cs = pd.DataFrame(
        [{"id": str(i), "question": f"q{i}", "answerKey": "ABCDE"[i % 5],
          "choices": {"label": list("ABCDE"), "text": [f"t{j}" for j in range(5)]}} for i in range(30)]
    )
    t = CommonsenseQA.from_frames(cs, cs, dev_n=5, test_n=7, pool_n=4)
    assert (len(t.pool), len(t.dev), t.n_test) == (4, 5, 7)
    assert "A. t0" in t.dev[0].question and t.score("B", t.dev[0]) in (0, 1)
    bq = pd.DataFrame([{"question": f"is {i}", "passage": "p " * 400, "answer": bool(i % 2)} for i in range(30)])
    b = BoolQ.from_frames(bq, bq, dev_n=5, test_n=7, pool_n=4)
    assert len(b.dev[0].question) < 700 and b.majority_rate() >= 0.5


def ctx_for(task, seed=0, backend=None):
    return OpContext(task, backend or ScriptedBackend(lambda r: "Reply with only the answer."),
                     np.random.default_rng(seed))


def test_deterministic_operators_are_seeded_and_change_the_prompt(task):
    p = Prompt.from_text("Answer the question. Be careful. Think first.", "Give the answer only.", (task.pool[0].id,))
    for op in (Reorder(), DropClause(), AddClause(), SwapFormat(), AddShot(), RemoveShot()):
        a = op.apply([Parent(p)], ctx_for(task, 5))
        b = op.apply([Parent(p)], ctx_for(task, 5))
        assert a == b and a is not None, op.name
    assert DropClause().apply([Parent(Prompt.from_text("Only one."))], ctx_for(task)) is None
    assert RemoveShot().apply([Parent(Prompt.from_text("x y."))], ctx_for(task)) is None


def test_add_shot_respects_cap(task):
    p = Prompt(("Do it.",), "", tuple(e.id for e in task.pool[:3]))
    assert AddShot().apply([Parent(p)], ctx_for(task)) is None


def test_crossover_mixes_two_parents(task):
    a = Prompt.from_text("Alpha one. Alpha two.")
    b = Prompt.from_text("Beta one. Beta two.")
    kids = {Crossover().apply([Parent(a), Parent(b)], ctx_for(task, s)) for s in range(30)}
    kids.discard(None)
    assert any(set(k.sentences) & {"Alpha one."} and set(k.sentences) & {"Beta two."} for k in kids)
    assert Crossover().apply([Parent(a)], ctx_for(task)) is None


def test_llm_operators_use_backend_and_clean_output(task):
    be = ScriptedBackend(lambda r: 'Instruction: "Reply with only the answer. No extra words."')
    out = LLMRewrite().apply([Parent(Prompt.from_text("Answer."))], ctx_for(task, backend=be))
    assert out.sentences == ("Reply with only the answer.", "No extra words.")
    wrong = np.zeros(len(task.dev), dtype=np.int8)
    par = Parent(Prompt.from_text("Answer."), wrong, ["blah"] * len(task.dev))
    seen = []
    be2 = ScriptedBackend(lambda r: seen.append(r.messages[-1]["content"]) or "Be exact.")
    assert LLMReflect().apply([par], ctx_for(task, backend=be2)).sentences == ("Be exact.",)
    assert "answered these questions wrongly" in seen[0] and "Correct answer" in seen[0]
    assert LLMReflect().apply([Parent(Prompt.from_text("A."))], ctx_for(task)) is None
    perfect = Parent(Prompt.from_text("A."), np.ones(3, dtype=np.int8), ["x"] * 3)
    assert LLMReflect().apply([perfect], ctx_for(task)) is None


def test_clean_instruction_edge_cases():
    assert clean_instruction("") == ()
    assert clean_instruction("One. Two. Three. Four.") == ("One.", "Two.", "Three.")


def test_guard_rejects_empty_long_and_unchanged():
    p = Prompt.from_text("Answer the question now.")
    assert guard(None, [Parent(p)]) is None
    assert guard(Prompt((), ""), [Parent(p)]) is None
    assert guard(p, [Parent(p)]) is None
    assert guard(Prompt.from_text("word " * 100), [Parent(p)]) is None
    assert guard(Prompt.from_text("Totally different text."), [Parent(p)]) is not None


def test_operator_table_has_every_family():
    names = {o.name for o, _ in OPERATORS}
    assert {"reorder", "drop_clause", "add_clause", "swap_format", "add_shot", "remove_shot",
            "crossover", "llm_rewrite", "llm_reflect"} == names
