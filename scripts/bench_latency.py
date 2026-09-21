"""Latency and throughput grid for a running reflex server, measured the way a client sees it.

    uv run python scripts/bench_latency.py --url http://host:8010 \
        --concurrency 1,2,4,8,16,32 --questions 1,3,10 --requests 48 --out bench.json

Every cell is `--requests` POSTs to `/v1/systemone` driven by `--concurrency` clients.
Two regimes per cell:

  * **warm** - every request carries the *same* state, so the server (and, behind the
    SGLang backend, its radix cache) has seen the prefix before.
  * **cold** - a fresh state per request, so nothing is reused.

Reported per cell: p50 and p95 of the per-request wall time as the client measures it
(HTTP included), requests/s and questions/s over the cell's own wall clock.

`--in-process` skips HTTP and calls the backend object directly in this process, for the
numbers the earlier spikes quoted; it needs the SGLang server, not a reflex server.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from typing import Any

import httpx

WORDS = [
    "order",
    "refund",
    "invoice",
    "shipment",
    "account",
    "payment",
    "billing",
    "password",
    "login",
    "charge",
    "dispute",
    "delivery",
    "package",
    "tracking",
    "replacement",
    "warranty",
    "subscription",
    "renewal",
    "cancel",
    "upgrade",
    "credit",
    "transaction",
    "receipt",
    "merchant",
    "courier",
    "address",
    "carrier",
    "support",
    "agent",
    "ticket",
    "escalate",
    "priority",
    "policy",
    "window",
    "return",
    "label",
    "damaged",
    "missing",
    "delayed",
    "duplicate",
]

QUEUES = ["billing", "shipping", "account", "technical"]

LEVELS = [
    "can wait a week",
    "routine, answer this week",
    "normal queue, answer within two days",
    "answer today",
    "drop everything",
]


def filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def make_state(rng: random.Random, n_tokens: int, tag: str) -> dict[str, Any]:
    """A support ticket whose rendered prefix is roughly `n_tokens` tokens.

    The prompt wrapper (chat template, instructions, the state's own keys) costs about 88
    tokens before a word of message is added, and a word of this vocabulary costs about
    1.27 tokens; the script reports the state length the server actually measured, so the
    fit only has to be close."""
    n_words = max(12, int((n_tokens - 88) / 1.27))
    return {
        "ticket": {
            "id": tag,
            "channel": "email",
            "customer_tier": rng.choice(["free", "plus", "business"]),
            "subject": filler(rng, 8),
        },
        "message": filler(rng, n_words),
    }


def make_questions(n: int) -> dict[str, Any]:
    """`n` questions cycling noul / choice / score, the mix a real caller sends."""
    out: dict[str, Any] = {}
    for i in range(n):
        kind = ("noul", "choice", "score")[i % 3]
        if kind == "noul":
            out[f"escalate_{i}"] = {
                "type": "noul",
                "instructions": "Does this ticket need a human agent rather than an automated reply?",
            }
        elif kind == "choice":
            out[f"queue_{i}"] = {
                "type": "choice",
                "instructions": "Which queue should this ticket go to?",
                "criteria": {q: None for q in QUEUES},
            }
        else:
            out[f"urgency_{i}"] = {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": list(LEVELS),
            }
    return out


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))
    return xs[i]


# --------------------------------------------------------------------------------- HTTP


async def _one(client: httpx.AsyncClient, payload: dict, headers: dict) -> tuple[float, int]:
    t0 = time.perf_counter()
    r = await client.post("/v1/systemone", json=payload, headers=headers)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return dt, int(r.json()["usage"]["state_tokens"])


async def run_cell_http(
    url: str,
    api_key: str | None,
    conc: int,
    n_questions: int,
    total: int,
    cold: bool,
    state_tokens: int,
    permutations: int | None,
    seed: int,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    questions = make_questions(n_questions)
    rng = random.Random(seed)
    warm_state = make_state(rng, state_tokens, "warm-state")

    def payload(i: int) -> dict:
        state = (
            make_state(random.Random(seed * 7919 + i), state_tokens, f"cold-{seed}-{i}")
            if cold
            else warm_state
        )
        body: dict[str, Any] = {"state": state, "questions": questions}
        if permutations:
            body["permutations"] = permutations
        return body

    limits = httpx.Limits(max_connections=max(conc, 8), max_keepalive_connections=max(conc, 8))
    async with httpx.AsyncClient(base_url=url, timeout=600.0, limits=limits) as client:
        # Warm-up outside the timed window: the shared state for the warm regime, a couple
        # of throwaways for the cold one (uvicorn/httpx first-call costs are not the model).
        for i in range(2):
            await _one(client, payload(-1 - i), headers)

        counter = iter(range(total))
        lats: list[float] = []
        observed: list[int] = []
        lock = asyncio.Lock()

        async def worker():
            while True:
                async with lock:
                    i = next(counter, None)
                if i is None:
                    return
                dt, st = await _one(client, payload(i), headers)
                lats.append(dt)
                observed.append(st)

        t0 = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(conc)])
        elapsed = time.perf_counter() - t0

    return _summary(lats, observed, elapsed, conc, n_questions, cold, permutations)


# ---------------------------------------------------------------------------- in-process


def run_cell_inproc(backend, conc, n_questions, total, cold, state_tokens, permutations, seed):
    from concurrent.futures import ThreadPoolExecutor

    from reflex.schema import SystemOneRequest

    questions = make_questions(n_questions)
    rng = random.Random(seed)
    warm_state = make_state(rng, state_tokens, "warm-state")

    def req(i: int) -> SystemOneRequest:
        state = (
            make_state(random.Random(seed * 7919 + i), state_tokens, f"cold-{seed}-{i}")
            if cold
            else warm_state
        )
        kw = {"permutations": permutations} if permutations else {}
        return SystemOneRequest(state=state, questions=questions, **kw)

    for i in range(2):
        backend.answer(req(-1 - i))

    lats: list[float] = []
    observed: list[int] = []

    def one(i: int):
        t0 = time.perf_counter()
        resp = backend.answer(req(i))
        lats.append(time.perf_counter() - t0)
        observed.append(resp.usage.state_tokens)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        list(pool.map(one, range(total)))
    elapsed = time.perf_counter() - t0
    return _summary(lats, observed, elapsed, conc, n_questions, cold, permutations)


def _summary(lats, observed, elapsed, conc, n_questions, cold, permutations) -> dict[str, Any]:
    return {
        "concurrency": conc,
        "questions": n_questions,
        "regime": "cold" if cold else "warm",
        "permutations": permutations,
        "n": len(lats),
        "state_tokens": int(statistics.median(observed)) if observed else 0,
        "p50_ms": 1000 * percentile(lats, 0.50),
        "p95_ms": 1000 * percentile(lats, 0.95),
        "req_s": len(lats) / elapsed,
        "q_s": len(lats) * n_questions / elapsed,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="reflex server base url (omit with --in-process)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--questions", default="1,3,10")
    ap.add_argument("--requests", type=int, default=48, help="requests per cell")
    ap.add_argument("--state-tokens", type=int, default=150)
    ap.add_argument(
        "--regimes",
        default="warm,cold",
        help="warm (one shared state) and/or cold (a fresh state per request)",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=None,
        help="ask the server for this many option orders (default: whatever it was launched with)",
    )
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None, help="write the rows as json")
    ap.add_argument("--seed", type=int, default=0)
    # --in-process only
    ap.add_argument("--in-process", action="store_true", help="call the SGLang backend directly")
    ap.add_argument("--model", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    args = ap.parse_args(argv)

    concs = [int(x) for x in args.concurrency.split(",") if x]
    nqs = [int(x) for x in args.questions.split(",") if x]
    regimes = [r.strip() for r in args.regimes.split(",") if r.strip()]

    backend = None
    if args.in_process:
        from transformers import AutoTokenizer

        from reflex.backends.sglang import SGLangBackend
        from reflex.prompt import PromptFormat

        tok = AutoTokenizer.from_pretrained(args.model)
        template = tok.chat_template or ""
        backend = SGLangBackend(
            args.sglang_url,
            tokenizer=tok,
            fmt=PromptFormat(chat=bool(template), no_think="enable_thinking" in template),
            default_permutations=args.permutations or 1,
            max_concurrent_calls=256,
        )
    elif not args.url:
        ap.error("--url is required without --in-process")

    rows = []
    for nq in nqs:
        for conc in concs:
            for regime in regimes:
                cold = regime == "cold"
                total = max(args.requests, conc * 2)
                try:
                    if backend is not None:
                        row = run_cell_inproc(
                            backend,
                            conc,
                            nq,
                            total,
                            cold,
                            args.state_tokens,
                            args.permutations,
                            args.seed,
                        )
                    else:
                        row = asyncio.run(
                            run_cell_http(
                                args.url,
                                args.api_key,
                                conc,
                                nq,
                                total,
                                cold,
                                args.state_tokens,
                                args.permutations,
                                args.seed,
                            )
                        )
                except (RuntimeError, OSError, httpx.HTTPError) as e:  # keep the rest of the sweep
                    print(f"{args.label} c={conc} q={nq} {regime}: FAILED {e}", flush=True)
                    rows.append(
                        {
                            "label": args.label,
                            "concurrency": conc,
                            "questions": nq,
                            "regime": regime,
                            "error": str(e)[:200],
                        }
                    )
                    continue
                row["label"] = args.label
                row["state_tokens_target"] = args.state_tokens
                rows.append(row)
                print(
                    f"{args.label} c={conc:<3} q={nq:<3} {row['regime']:<5} "
                    f"state={row['state_tokens']:<5} p50={row['p50_ms']:8.1f}ms "
                    f"p95={row['p95_ms']:8.1f}ms {row['req_s']:6.2f} req/s {row['q_s']:7.2f} q/s",
                    flush=True,
                )
                if args.out:  # written after every cell, so a late failure keeps the rest
                    with open(args.out, "w") as f:
                        json.dump(rows, f, indent=2)
    if backend is not None:
        backend.close()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"wrote {args.out}")
    return rows


if __name__ == "__main__":
    main()
