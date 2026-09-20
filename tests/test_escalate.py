"""The escalation trigger: features, the JSON round trip, and the engine's routing."""

from types import SimpleNamespace

import numpy as np
import torch

from reflex.engine import Engine
from reflex.escalate import (
    FEATURE_NAMES,
    Escalator,
    budget_report,
    question_features,
    train_escalator,
    vector,
)
from reflex.prompt import PromptFormat
from reflex.readout import Calibration
from reflex.schema import SystemOneRequest

POLICY = (
    "Refunds are issued only if the request arrives before 2026-03-14 and the order total "
    "is under $250. Orders placed more than 30 days prior are not eligible. A 15% "
    "restocking fee applies unless the item was damaged."
)


# ---- feature extraction
def test_reasoning_cues_are_picked_up_and_plain_text_is_not():
    hard = question_features(POLICY, "noul", "Is this refundable?", 2, 64)
    easy = question_features("the cat sat on the mat", "noul", "Is it about a cat?", 2, 8)
    assert hard["state_has_date"] == 1.0 and easy["state_has_date"] == 0.0
    assert hard["state_has_currency"] == 1.0 and easy["state_has_currency"] == 0.0
    assert hard["state_has_percent"] == 1.0 and easy["state_has_percent"] == 0.0
    assert hard["log_state_numbers"] > easy["log_state_numbers"] == 0.0
    assert hard["log_state_cues"] > easy["log_state_cues"] == 0.0
    assert hard["log_state_sentences"] > easy["log_state_sentences"]


def test_json_state_field_count_and_kind_flags():
    f = question_features({"a": 1, "b": {"c": 2, "d": 3}}, "choice", "which?", 4, 40)
    assert f["log_state_fields"] > 0.0
    assert (f["is_choice"], f["is_noul"], f["is_score"]) == (1.0, 0.0, 0.0)
    assert f["n_options"] == 4.0


def test_readout_features_track_the_distribution():
    sure = question_features("s", "choice", "q", 3, 10, np.array([0.9, 0.05, 0.05]), 0.01)
    unsure = question_features("s", "choice", "q", 3, 10, np.array([0.4, 0.35, 0.25]), 0.3)
    assert sure["max_prob"] > unsure["max_prob"]
    assert sure["entropy"] < unsure["entropy"]
    assert sure["margin"] > unsure["margin"]
    assert unsure["disagreement"] == 0.3
    # with no fast-path reading, only the pre-answer features are populated
    pre = question_features("s", "choice", "q", 3, 10)
    assert pre["max_prob"] == pre["entropy"] == pre["margin"] == pre["disagreement"] == 0.0
    assert len(vector(pre)) == len(FEATURE_NAMES)


# ---- the classifier
def _toy_set(n=600, seed=0):
    """Errors concentrated where the (standardised) first feature is large."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    p = 1.0 / (1.0 + np.exp(-(2.0 * X[:, 0] - 1.5)))
    return X, (rng.random(n) < p).astype(float)


def test_trigger_ranks_errors_above_chance():
    X, y = _toy_set()
    esc = train_escalator(X, y, budget=0.15)
    scores = np.array([esc.score(dict(zip(FEATURE_NAMES, row, strict=True))) for row in X])
    rows = budget_report(scores, y, budgets=(0.15,))
    assert rows[0]["lift"] > 1.5  # better than escalating at random


def test_budget_threshold_selects_about_the_budget():
    X, y = _toy_set()
    esc = train_escalator(X, y, budget=0.12)
    fired = sum(esc.should_escalate(dict(zip(FEATURE_NAMES, row, strict=True))) for row in X)
    assert 0.09 * len(X) <= fired <= 0.15 * len(X)


def test_escalator_round_trips_through_json(tmp_path):
    X, y = _toy_set(n=200, seed=3)
    esc = train_escalator(X, y, budget=0.2)
    esc.meta = {"note": "toy"}
    path = tmp_path / "escalator.json"
    esc.save(path)
    back = Escalator.load(path)
    assert back.names == esc.names and back.meta == {"note": "toy"}
    assert back.threshold == esc.threshold
    for row in X[:25]:
        f = dict(zip(FEATURE_NAMES, row, strict=True))
        assert abs(back.score(f) - esc.score(f)) < 1e-12
        assert back.should_escalate(f) == esc.should_escalate(f)


# ---- engine routing (no model: a duck-typed Engine, as in test_phase1_fixes)
class _Fake:
    def __init__(self, escalator):
        self.variants = [PromptFormat()]
        self.default_permutations = 2
        self.cal = Calibration()
        self.think_tokens = 64
        self.think_if_disagree = None
        self.escalator = escalator
        self.model_name = "fake"

    def _encode(self, text):
        return [1, 2, 3]

    def encode_state(self, state):
        return SimpleNamespace(ids=[0] * 8, n_images=0), False

    def _forward_branches(self, entry, branch_ids):
        return torch.zeros((len(branch_ids), 4))

    def restrict(self, row, br):
        # the fast path says "yes", in whichever order this branch lists the options
        return torch.tensor([3.0, 0.0] if br.keys[0] is True else [0.0, 3.0])

    _escalate = Engine._escalate


class _AlwaysEscalate:
    threshold = 0.0

    def should_escalate(self, feats):
        assert set(feats) >= set(FEATURE_NAMES)  # the engine builds the full vector
        return True


class _NeverEscalate:
    threshold = 1.0

    def should_escalate(self, feats):
        return False


def _run(monkeypatch, escalator):
    calls = []

    def fake_think(engine, items, n):  # reasoning says "no"
        calls.append(len(items))
        return [np.array([0.0, 3.0] if br.keys[0] is True else [3.0, 0.0]) for _, br in items]

    monkeypatch.setattr("reflex.think.think_logits", fake_think)
    req = SystemOneRequest(
        state={"policy": POLICY},
        questions={"q": {"type": "noul", "instructions": "Refundable before the deadline?"}},
    )
    return calls, Engine._answer(_Fake(escalator), req)


def test_flagged_question_takes_the_reasoned_path(monkeypatch):
    calls, resp = _run(monkeypatch, _AlwaysEscalate())
    assert calls and resp.answers["q"].path == "reasoned"
    assert resp.answers["q"].noul < 0.5  # the reasoning answer won


def test_unflagged_question_stays_on_the_fast_path(monkeypatch):
    calls, resp = _run(monkeypatch, _NeverEscalate())
    assert calls == [] and resp.answers["q"].path == "fast"
    assert resp.answers["q"].noul > 0.5


def test_no_escalator_leaves_the_path_field_absent(monkeypatch):
    calls = []
    monkeypatch.setattr("reflex.think.think_logits", lambda *a: calls.append(1))
    fake = _Fake(None)
    fake.think_tokens = 0
    req = SystemOneRequest(state="s", questions={"q": {"type": "noul", "instructions": "?"}})
    resp = Engine._answer(fake, req)
    assert calls == [] and resp.answers["q"].path is None
