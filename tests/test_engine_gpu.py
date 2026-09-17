"""Packed/cached forward must match naive full-prompt forward. Needs a GPU + model download."""

import numpy as np
import pytest
import torch

from reflex.prompt import build_branches
from reflex.readout import softmax
from reflex.schema import SystemOneRequest

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")

MODEL = "Qwen/Qwen3-0.6B"

REQ = SystemOneRequest(
    state={
        "ticket": "My payouts have failed three times this week and support hasn't replied.",
        "plan": "pro",
    },
    questions={
        "queue": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "payments": "payouts, refunds, invoices",
                "account": "login, profile, permissions",
                "other": None,
            },
        },
        "escalate": {"type": "noul", "instructions": "Should this be escalated to a manager?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["can wait a week", "should be handled today", "needs attention now"],
        },
    },
    permutations=2,
)


@pytest.fixture(scope="module", params=["packed", "batched"])
def engine(request):
    """Both execution strategies must agree with the naive forward on an attention-only model."""
    from reflex.engine import Engine

    return Engine.load(MODEL, dtype=torch.float32, max_pack_tokens=2048, strategy=request.param)


def naive_probs(engine, state, branch):
    ids = engine._encode(engine.fmt.prefix(state)) + engine._encode(branch.text)
    with torch.inference_mode():
        logits = (
            engine.model(input_ids=torch.tensor([ids], device=engine.device), logits_to_keep=1)
            .logits[0, -1]
            .float()
        )
    lab = torch.tensor([engine.label_id(l) for l in branch.labels], device=logits.device)
    return softmax(logits[lab].cpu().numpy())


def test_packed_matches_naive(engine):
    resp = engine.answer(REQ)
    assert set(resp.answers) == {"queue", "escalate", "urgency"}
    assert not resp.usage.state_cache_hit

    # recompute each question naively (same permutations / rng seed as the engine)
    import random

    rng = random.Random(0)
    branches = []
    for qid, q in REQ.questions.items():
        branches.extend(build_branches(qid, q, engine.fmt, REQ.permutations, rng))
    packed = engine._forward_branches(
        engine.encode_state(REQ.state)[0], [engine._encode(b.text) for b in branches]
    )
    for b, row in zip(branches, packed):
        lab = torch.tensor([engine.label_id(l) for l in b.labels], device=row.device)
        p_packed = softmax(row[lab].cpu().numpy())
        p_naive = naive_probs(engine, REQ.state, b)
        np.testing.assert_allclose(p_packed, p_naive, atol=2e-3), (b.qid, p_packed, p_naive)

    # second call hits the state cache and returns identical numbers
    resp2 = engine.answer(REQ)
    assert resp2.usage.state_cache_hit
    assert resp2.answers == resp.answers

    # typed shapes
    q = resp.answers["queue"]
    assert abs(sum(q.probabilities.values()) - 1) < 1e-4 and q.choice in q.probabilities
    assert 0 <= resp.answers["escalate"].noul <= 1
    u = resp.answers["urgency"]
    assert 0 <= u.score <= 2 and list(u.legend) == ["0", "1", "2"]


def test_batch_matches_naive(engine):
    items = []
    for i in range(3):
        state = f"Report {i}: the export button crashes in Safari but works in Chrome."
        q = REQ.questions["urgency"]
        items.append((state, build_branches("u", q, engine.fmt)[0]))
    rows = engine.label_logits_batch(items)
    for (state, br), row in zip(items, rows):
        np.testing.assert_allclose(softmax(row), naive_probs(engine, state, br), atol=2e-3)
