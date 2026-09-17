"""The canonical System One example: route a support ticket with three parallel judgments.

uv run python examples/support_ticket.py            # in-process, default model
uv run python examples/support_ticket.py --http     # against a running reflex-serve
"""

import argparse
import json
import time

QUESTIONS = {
    "queue": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "payments": "Payouts, refunds, invoices, failed charges",
            "account": "Login, profile, permissions, 2FA",
            "other": "Anything else",
        },
    },
    "escalate": {
        "type": "noul",
        "instructions": "Should this ticket be escalated to a human manager right away?",
        "criteria": {
            "true": "Customer is at risk of churning, threatens legal action, or has been ignored",
            "false": "A normal first-line reply will do",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this ticket?",
        "criteria": [
            "Low: can wait several days",
            "Medium: should be handled today",
            "High: money or access is blocked right now",
        ],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the customer explicitly ask for a refund?",
    },
}

STATE = {
    "ticket": {
        "subject": "Payouts failing",
        "body": "My payouts have failed three times this week and nobody has replied to my two "
        "emails. I have contractors waiting on this money. If it's not fixed by Friday I'm moving "
        "to a competitor.",
    },
    "customer": {"plan": "pro", "tenure_months": 27, "prior_tickets_30d": 2},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--permutations", type=int, default=1)
    args = ap.parse_args()

    if args.http:
        from reflex.client import Reflex

        c = Reflex()
        t0 = time.perf_counter()
        out = c.systemone(STATE, QUESTIONS, permutations=args.permutations)
        t1 = time.perf_counter()
        out2 = c.systemone(STATE, QUESTIONS, permutations=args.permutations)
        t2 = time.perf_counter()
    else:
        from reflex import Engine, SystemOneRequest

        eng = Engine.load(args.model)
        req = SystemOneRequest(state=STATE, questions=QUESTIONS, permutations=args.permutations)
        t0 = time.perf_counter()
        out = eng.answer(req).model_dump()
        t1 = time.perf_counter()
        out2 = eng.answer(req).model_dump()
        t2 = time.perf_counter()

    assert out2["answers"] == out["answers"]  # cached state gives identical numbers
    print(json.dumps(out["answers"], indent=2))
    print(f"usage: {out['usage']}")
    print(f"cold: {(t1 - t0) * 1000:.0f} ms   warm (state cached): {(t2 - t1) * 1000:.0f} ms")

    # the thing Jev is for: code owns the policy, the model supplies the judgment
    a = out["answers"]
    if a["queue"]["confidence"] > 0.6:
        print(f"-> auto-route to {a['queue']['choice']}")
    else:
        print("-> ambiguous queue, hold for triage")
    if a["escalate"]["noul"] > 0.5 or a["urgency"]["score"] > 1.5:
        print("-> page on-call")


if __name__ == "__main__":
    main()
