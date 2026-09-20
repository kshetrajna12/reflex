"""Regression checks for the Phase 1 findings in docs/MODEL_QUALITY_PLAN.md."""

from types import SimpleNamespace

import numpy as np
import torch

from reflex.engine import Engine
from reflex.eval.metrics import fit_temperature
from reflex.prompt import PromptFormat, build_branches, distinct_orders
from reflex.readout import Calibration, softmax
from reflex.schema import ChoiceQuestion, NoulAnswer, NoulQuestion, SystemOneRequest


# ---- permutations: distinct orders, balanced binary swaps, per-question seeding
def test_orders_are_distinct_and_binary_is_swapped():
    import random

    assert distinct_orders(["x", "y"], 4, random.Random(0)) == [["x", "y"], ["y", "x"]]
    orders = distinct_orders(list("abcd"), 6, random.Random(1))
    assert orders[0] == list("abcd") and len(orders) == 6
    assert len({tuple(o) for o in orders}) == 6


def test_noul_gets_swapped_order_for_both_readouts():
    q = NoulQuestion(type="noul", instructions="Is it?")
    for texts in ({}, {"noul_readout": "yesno"}):
        fmt = PromptFormat.with_texts(PromptFormat(), texts)
        a, b = build_branches("q", q, fmt, 2)
        assert a.keys == [True, False] and b.keys == [False, True]
        assert a.text != b.text
        assert len(build_branches("q", q, fmt, 1)) == 1


def test_orders_do_not_depend_on_neighbouring_questions():
    q = ChoiceQuestion(type="choice", instructions="?", criteria={k: None for k in "abcde"})
    alone = [b.keys for b in build_branches("q1", q, PromptFormat(), 3)]
    # building another question first must not change q1's orders
    build_branches("q0", q, PromptFormat(), 3)
    again = [b.keys for b in build_branches("q1", q, PromptFormat(), 3)]
    assert alone == again
    other = [b.keys for b in build_branches("q2", q, PromptFormat(), 3)]
    assert other != alone  # different question, different draw


# ---- temperature fitting honours soft targets
def test_soft_target_is_not_sharpened_to_its_argmax():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(400, 3))
    soft = np.stack([softmax(r, 1.0) for r in logits])  # predictions already match targets
    T = fit_temperature(logits, soft)
    assert abs(T - 1.0) < 0.05
    hard = soft.argmax(1)
    assert fit_temperature(logits, hard) < 0.7  # the old objective would sharpen


# ---- there is no reasoning path: the readout is one fast forward, always
class _Fake:
    """Duck-typed Engine for Engine._answer: no model, canned logits per branch."""

    def __init__(self, variants):
        self.variants = variants
        self.default_permutations = 1
        self.cal = Calibration()
        self.model_name = "fake"
        self.fast_calls = 0

    def _encode(self, text):
        return [1, 2, 3]

    def encode_state(self, state):
        return SimpleNamespace(ids=[0] * 8, n_images=0), False

    def _forward_branches(self, entry, branch_ids):
        self.fast_calls += 1
        return torch.zeros((len(branch_ids), 4))

    def restrict(self, row, br):
        return torch.tensor([3.0, 0.0] if br.keys[0] is True else [0.0, 3.0])


def test_answer_is_one_fast_forward_and_never_reasons(monkeypatch):
    def no_thinking(*a, **kw):
        raise AssertionError("the serving path must not reason")

    monkeypatch.setattr("reflex.think.think_logits", no_thinking)
    fmt = PromptFormat()
    variants = [fmt, PromptFormat.with_texts(fmt, {"question_heading": "# Q"})]
    fake = _Fake(variants)
    req = SystemOneRequest(state="s", questions={"q": {"type": "noul", "instructions": "?"}})
    resp = Engine._answer(fake, req)
    assert fake.fast_calls == 1
    assert 0.0 <= resp.answers["q"].noul <= 1.0


def test_no_reasoning_or_escalation_hooks_on_the_engine():
    import inspect

    for attr in ("think_tokens", "think_if_disagree", "escalator", "_escalate"):
        assert not hasattr(Engine, attr)
    src = inspect.getsource(Engine._answer) + inspect.getsource(Engine.label_logits_batch)
    assert "think" not in src and "escalat" not in src
    # and the response shape carries no cascade marker
    assert "path" not in NoulAnswer.model_fields
