"""Raw states: short and long texts of the kinds people actually send to a decision model.

Each domain is a small streaming loader over a public dataset, yielding
`{"state": <text or dict>, "domain": <name>, "origin": <dataset>}`. No labels are used;
the teacher supplies those. Domains were chosen to be diverse and *not* to overlap the
never-trained external sets (bitext support intents, toxic-chat, MNLI, Yelp) or the
benchmark; `reflex-data check-overlap` is run on the result.
"""

from __future__ import annotations

import builtins
import hashlib
import random
from collections.abc import Callable, Iterator

MIN_CHARS, MAX_CHARS = 120, 14000


def _load(name: str, cfg: str | None, split: str, seed: int):
    from datasets import load_dataset

    ds = load_dataset(name, cfg, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=5000)


class _Dry(Exception):
    """A source stream ran out; the domain ends (a StopIteration inside a generator
    would become a RuntimeError instead)."""


def _next(it):
    try:
        return builtins.next(it)
    except StopIteration:
        raise _Dry from None


def _clip(s: str, limit: int) -> str:
    s = s.strip()
    return s if len(s) <= limit else s[:limit].rsplit("\n", 1)[0].rstrip() + "\n[...]"


# ----------------------------------------------------------------------------- domains
def support_ticket(seed: int) -> Iterator[dict]:
    for r in _load("Tobi-Bueck/customer-support-tickets", None, "train", seed):
        if r.get("language") != "en" or not r.get("body"):
            continue
        yield {
            "state": {"subject": r.get("subject") or "", "message": r["body"]},
            "origin": "customer-support-tickets",
        }


def pr_hunk(seed: int) -> Iterator[dict]:
    for r in _load("ronantakizawa/github-codereview", None, "train", seed):
        code = r.get("diff_context") or r.get("before_code") or ""
        if len(code) < 80:
            continue
        yield {
            "state": {
                "file": r.get("file_path"),
                "language": r.get("language"),
                "hunk": _clip(code, 4000),
            },
            "origin": "github-codereview",
        }


def chat_prompt(seed: int) -> Iterator[dict]:
    a = _load("HuggingFaceH4/ultrachat_200k", None, "train_sft", seed)
    b = _load("PKU-Alignment/PKU-SafeRLHF", None, "train", seed)
    c = _load("allenai/real-toxicity-prompts", None, "train", seed)
    ia, ib, ic = iter(a), iter(b), iter(c)
    while True:
        r = _next(ia)
        yield {"state": {"user_message": r["prompt"]}, "origin": "ultrachat"}
        r = _next(ib)
        yield {"state": {"user_message": r["prompt"]}, "origin": "pku-saferlhf"}
        r = _next(ic)
        yield {
            "state": {"user_message": r["prompt"]["text"] + r["continuation"]["text"]},
            "origin": "real-toxicity-prompts",
        }


def review(seed: int) -> Iterator[dict]:
    a, b = (
        iter(_load("fancyzhx/amazon_polarity", None, "train", seed)),
        iter(_load("stanfordnlp/imdb", None, "train", seed)),
    )
    while True:
        r = _next(a)
        yield {"state": {"title": r["title"], "review": r["content"]}, "origin": "amazon_polarity"}
        r = _next(b)
        yield {
            "state": {"review": _clip(r["text"].replace("<br />", "\n"), 3000)},
            "origin": "imdb",
        }


def news(seed: int) -> Iterator[dict]:
    a, b = (
        iter(_load("fancyzhx/ag_news", None, "train", seed)),
        iter(_load("abisee/cnn_dailymail", "3.0.0", "train", seed)),
    )
    while True:
        yield {"state": {"headline_and_lede": _next(a)["text"]}, "origin": "ag_news"}
        yield {"state": {"article": _clip(_next(b)["article"], 5000)}, "origin": "cnn_dailymail"}


def legal_clause(seed: int) -> Iterator[dict]:
    """Single contract provisions, and 'excerpts' of 4-6 provisions stitched together so
    some states are policy-length documents with several interacting clauses."""
    led = iter(_load("coastalcph/lex_glue", "ledgar", "train", seed))
    tos = iter(_load("coastalcph/lex_glue", "unfair_tos", "train", seed))
    rng = random.Random(seed)
    while True:
        yield {"state": {"clause": _next(led)["text"]}, "origin": "ledgar"}
        k = rng.randint(4, 6)
        yield {
            "state": {
                "contract_excerpt": "\n\n".join(f"{i + 1}. {_next(led)['text']}" for i in range(k))
            },
            "origin": "ledgar-excerpt",
        }
        yield {
            "state": {
                "terms_of_service": " ".join(_next(tos)["text"] for _ in range(rng.randint(3, 8)))
            },
            "origin": "unfair_tos",
        }


def long_document(seed: int) -> Iterator[dict]:
    a = iter(_load("ccdv/govreport-summarization", None, "train", seed))
    b = iter(_load("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", seed))
    while True:
        yield {"state": {"report": _clip(_next(a)["report"], 12000)}, "origin": "govreport"}
        r = _next(b)
        if 1500 <= len(r["text"]) <= 8000:
            yield {"state": {"document": r["text"]}, "origin": "fineweb-edu"}


def forum_post(seed: int) -> Iterator[dict]:
    a = iter(_load("sentence-transformers/reddit-title-body", None, "train", seed))
    b = iter(_load("jonathanli/legal-advice-reddit", None, "train", seed))
    while True:
        r = _next(a)
        yield {
            "state": {
                "subreddit": r["subreddit"],
                "title": r["title"],
                "post": _clip(r["body"], 3000),
            },
            "origin": "reddit-title-body",
        }
        r = _next(b)
        yield {
            "state": {"title": r["title"], "post": _clip(r["body"], 3000)},
            "origin": "legal-advice-reddit",
        }


def assistant_reply(seed: int) -> Iterator[dict]:
    """A conversation plus one candidate reply: the kind of state a guardrail judges."""
    a = iter(_load("nvidia/HelpSteer3", "preference", "train", seed))
    b = iter(_load("Anthropic/hh-rlhf", None, "train", seed))
    while True:
        r = _next(a)
        if not str(r.get("language", "en")).lower().startswith("en"):
            continue
        ctx = r["context"]
        conv = (
            "\n".join(f"{m['role']}: {m['content']}" for m in ctx)
            if isinstance(ctx, list)
            else str(ctx)
        )
        yield {
            "state": {
                "conversation": _clip(conv, 4000),
                "candidate_reply": _clip(r["response1"], 3000),
            },
            "origin": "helpsteer3",
        }
        r = _next(b)
        yield {"state": {"conversation": _clip(r["chosen"], 4000)}, "origin": "hh-rlhf"}


def qa_passage(seed: int) -> Iterator[dict]:
    a = iter(_load("hotpotqa/hotpot_qa", "distractor", "train", seed))
    b = iter(_load("databricks/databricks-dolly-15k", None, "train", seed))
    while True:
        r = _next(a)
        paras = ["".join(s) for s in r["context"]["sentences"]]
        yield {
            "state": {
                "passages": paras[:6],
                "question": r["question"],
                "proposed_answer": r["answer"],
            },
            "origin": "hotpot_qa",
        }
        r = _next(b)
        if r.get("context"):
            yield {
                "state": {
                    "context": r["context"],
                    "instruction": r["instruction"],
                    "response": r["response"],
                },
                "origin": "dolly",
            }


DOMAINS: dict[str, Callable[[int], Iterator[dict]]] = {
    "support_ticket": support_ticket,
    "pr_hunk": pr_hunk,
    "chat_prompt": chat_prompt,
    "review": review,
    "news": news,
    "legal_clause": legal_clause,
    "long_document": long_document,
    "forum_post": forum_post,
    "assistant_reply": assistant_reply,
    "qa_passage": qa_passage,
}


def _size(state) -> int:
    import json

    return len(state) if isinstance(state, str) else len(json.dumps(state))


def _iter_dry(gen):
    try:
        yield from gen
    except _Dry:
        return


def build(per_domain: int, seed: int = 0, domains: list[str] | None = None) -> Iterator[dict]:
    """`per_domain` states from each domain, de-duplicated, within length bounds, with a
    stable id so later stages can join on it."""
    seen: set[str] = set()
    for name in domains or DOMAINS:
        n = 0
        try:
            gen = DOMAINS[name](seed)
        except _Dry:
            continue
        for r in _iter_dry(gen):
            size = _size(r["state"])
            if not MIN_CHARS <= size <= MAX_CHARS:
                continue
            h = hashlib.sha1(str(r["state"]).encode()).hexdigest()[:16]
            if h in seen:
                continue
            seen.add(h)
            yield {"id": f"{name}-{h}", "domain": name, "origin": r["origin"], "state": r["state"]}
            n += 1
            if n >= per_domain:
                break
