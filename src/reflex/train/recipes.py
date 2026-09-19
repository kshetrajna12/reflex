"""Recipes: turn public labelled datasets into reflex training data.

Each recipe is a small function that shows how one dataset maps onto one primitive.
That mapping *is* the lesson of this file: routing is a Choice, "is this toxic" is a
Noul with a soft label, a 1-5 rating is a Score. Every recipe yields plain JSONL rows
(state, questions, labels, source), the same shape as an API request plus the answer.

    reflex-data mix --out runs/mix_train.jsonl --eval-out runs/mix_eval.jsonl --per-source 800
    reflex-data mix --sources banking77,civil_comments --per-source 2000 --out runs/small.jsonl

Sources (all on the Hugging Face hub):
    banking77         customer query -> 77 intents (mteb mirror)    choice   CC-BY-4.0
    clinc             query -> 150 intents + "none of these"        choice   CC-BY-3.0
    mmlu_pro          exam question, up to 10 options               choice   MIT
    civil_comments    comment -> toxic?  (fraction of annotators)   noul*    CC0
    halueval          question + answer -> hallucinated?            noul     Apache-2.0
    msmarco           query + passage -> answers the query?         noul     MS research
    helpsteer2        prompt + response -> helpfulness 0..4         score    CC-BY-4.0
    helpsteer2_soft   same, but the raters' score distribution      score*   CC-BY-4.0
    codereview        diff chunk -> needs a reviewer? + issue type  noul/choice  "other"
  synthetic, correct by construction (see the bottom of this file):
    adequacy_synth    request + response -> satisfies exactly?      noul
    rule_routing_synth  request + stated policy -> specialist       choice
    base_rates_synth  stated outcome counts -> outcome              choice*
    policy_synth      long policy document + case -> outcome        choice
  * soft label
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator

# ------------------------------------------------------------------------------ helpers


def _shuffled(ds, seed: int):
    return ds.shuffle(seed=seed)


def _choice(instructions: str, options: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def _noul(instructions: str, true: str | None = None, false: str | None = None) -> dict:
    q: dict = {"type": "noul", "instructions": instructions}
    if true or false:
        q["criteria"] = {"true": true, "false": false}
    return q


def _score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


# ------------------------------------------------------------------------------ recipes


def banking77(n: int, seed: int, split: str) -> Iterator[dict]:
    """Routing: one query, one of 77 intents. More than 26 options, so the option list
    is cut to the true intent plus 11 random distractors (a 12-way choice) – the same
    trick you would use in production when a full taxonomy does not fit one question."""
    from datasets import load_dataset

    # mteb/banking77 is a parquet mirror of PolyAI/banking77 (the original uses a
    # loading script, which the datasets library no longer runs)
    ds = load_dataset("mteb/banking77", split=split)
    names = sorted({r.replace("_", " ") for r in ds.unique("label_text")})
    rng = random.Random(seed)
    for row in _shuffled(ds, seed).select(range(min(n, len(ds)))):
        gold = row["label_text"].replace("_", " ")
        distractors = rng.sample([x for x in names if x != gold], 11)
        opts = distractors + [gold]
        rng.shuffle(opts)
        yield {
            "state": {"customer_message": row["text"]},
            "questions": {
                "intent": _choice(
                    "Which banking intent best matches the customer's message?",
                    {o: None for o in opts},
                )
            },
            "labels": {"intent": gold},
            "source": "banking77",
        }


def clinc(n: int, seed: int, split: str) -> Iterator[dict]:
    """Routing with a no-match outcome: 150 intents plus 'out of scope'. Same distractor
    trick, but 'none of these' is always offered, and is the answer for OOS rows."""
    from datasets import load_dataset

    ds = load_dataset("clinc/clinc_oos", "plus", split=split)
    names = [n_.replace("_", " ") for n_ in ds.features["intent"].names]
    oos = names.index("oos")
    rng = random.Random(seed)
    for row in _shuffled(ds, seed).select(range(min(n, len(ds)))):
        gold = "none of these" if row["intent"] == oos else names[row["intent"]]
        pool = [x for x in names if x != "oos" and x != gold]
        opts = rng.sample(pool, 10) + ([gold] if gold != "none of these" else [])
        rng.shuffle(opts)
        opts.append("none of these")
        yield {
            "state": {"user_message": row["text"]},
            "questions": {
                "intent": _choice(
                    "Which intent does the user's message express?",
                    {
                        o: (
                            "the message does not match any listed intent"
                            if o == "none of these"
                            else None
                        )
                        for o in opts
                    },
                )
            },
            "labels": {"intent": gold},
            "source": "clinc",
        }


def mmlu_pro(n: int, seed: int, split: str) -> Iterator[dict]:
    """Exam questions with up to 10 options: calibration at higher cardinality."""
    from datasets import load_dataset

    # MMLU-Pro's own validation split is only 70 rows: split its 12k test rows 80/20 instead
    ds = _shuffled(load_dataset("TIGER-Lab/MMLU-Pro", split="test"), 12345)
    cut = len(ds) * 8 // 10
    ds = ds.select(range(cut)) if split == "train" else ds.select(range(cut, len(ds)))
    for row in _shuffled(ds, seed).select(range(min(n, len(ds)))):
        opts = [str(o).strip() or "(blank)" for o in row["options"]]
        seen: dict[str, int] = {}
        keys = []
        for o in opts:  # de-duplicate identical option texts
            seen[o] = seen.get(o, 0) + 1
            keys.append(o if seen[o] == 1 else f"{o} ({seen[o]})")
        yield {
            "state": {"subject": row["category"], "question": row["question"]},
            "questions": {
                "answer": _choice("Which option is the correct answer?", {k: None for k in keys})
            },
            "labels": {"answer": keys[row["answer_index"]]},
            "source": "mmlu_pro",
        }


def civil_comments(n: int, seed: int, split: str) -> Iterator[dict]:
    """Moderation with SOFT labels: `toxicity` is the fraction of annotators who said toxic.
    A proper scoring rule trained on 0.4 learns to say 0.4, which is the whole point."""
    from datasets import load_dataset

    ds = load_dataset("google/civil_comments", split=split)
    # keep the interesting middle: plenty of clear cases exist, ambiguous ones are rarer
    ds = (
        _shuffled(ds, seed)
        .select(range(min(20 * n, len(ds))))
        .filter(lambda r: len(r["text"]) > 20)
    )
    rows = list(ds.select(range(min(n, len(ds)))))
    for row in rows:
        yield {
            "state": {"comment": row["text"]},
            "questions": {
                "toxic": _noul(
                    "Would a reasonable moderator consider this comment toxic (rude, disrespectful, or likely to make someone leave the discussion)?"
                )
            },
            "labels": {"toxic": round(float(row["toxicity"]), 4)},
            "source": "civil_comments",
        }


def halueval(n: int, seed: int, split: str) -> Iterator[dict]:
    """Verification: given a question, supporting knowledge and an answer, is the answer
    hallucinated? Each HaluEval row has a right and a hallucinated answer; we emit both."""
    from datasets import load_dataset

    ds = load_dataset("pminervini/HaluEval", "qa", split="data")
    rng = random.Random(seed)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    lo, hi = (0, len(idx) * 9 // 10) if split == "train" else (len(idx) * 9 // 10, len(idx))
    count = 0
    for i in idx[lo:hi]:
        row = ds[i]
        for answer, halluc in ((row["right_answer"], False), (row["hallucinated_answer"], True)):
            yield {
                "state": {
                    "question": row["question"],
                    "knowledge": row["knowledge"],
                    "answer": answer,
                },
                "questions": {
                    "hallucinated": _noul(
                        "Is the `answer` unsupported by, or contradicted by, the `knowledge`?",
                        "the answer invents or contradicts facts",
                        "the answer is grounded in the knowledge",
                    )
                },
                "labels": {"hallucinated": halluc},
                "source": "halueval",
            }
            count += 1
            if count >= n:
                return


def msmarco(n: int, seed: int, split: str) -> Iterator[dict]:
    """Reranking: does this passage answer the query? One noul per (query, passage)."""
    from datasets import load_dataset

    ds = load_dataset(
        "microsoft/ms_marco", "v1.1", split="train" if split == "train" else "validation"
    )
    rng = random.Random(seed)
    count = 0
    for row in _shuffled(ds, seed):
        p = row["passages"]
        if not any(p["is_selected"]):
            continue
        pos = [i for i, s in enumerate(p["is_selected"]) if s]
        neg = [i for i, s in enumerate(p["is_selected"]) if not s]
        for i, label in (
            ((rng.choice(pos), True), (rng.choice(neg), False)) if neg else ((pos[0], True),)
        ):
            yield {
                "state": {"query": row["query"], "passage": p["passage_text"][i]},
                "questions": {
                    "relevant": _noul("Does the `passage` contain the answer to the `query`?")
                },
                "labels": {"relevant": label},
                "source": "msmarco",
            }
            count += 1
            if count >= n:
                return


def helpsteer2(n: int, seed: int, split: str) -> Iterator[dict]:
    """Scoring: human 0-4 helpfulness ratings of a response. Levels are written out so
    the model judges against descriptions, not bare numbers."""
    from datasets import load_dataset

    ds = load_dataset("nvidia/HelpSteer2", split="train" if split == "train" else "validation")
    levels = [
        "not helpful at all: ignores or misunderstands the request",
        "slightly helpful: touches the request but mostly misses",
        "partially helpful: addresses the request with notable gaps",
        "mostly helpful: addresses the request well with minor gaps",
        "extremely helpful: complete, correct and well presented",
    ]
    for row in _shuffled(ds, seed).select(range(min(n, len(ds)))):
        yield {
            "state": {"prompt": row["prompt"][-4000:], "response": row["response"][-4000:]},
            "questions": {
                "helpfulness": _score("How helpful is the `response` to the `prompt`?", levels)
            },
            "labels": {"helpfulness": int(row["helpfulness"])},
            "source": "helpsteer2",
        }


def codereview(n: int, seed: int, split: str) -> Iterator[dict]:
    """Code review: a ~50-line chunk from a merged PR. Negative rows passed review with no
    comment; positive rows got a human inline comment of a known type. Two questions
    per chunk: needs a reviewer (noul) and, when it does, what kind of issue (choice).
    The type question is only labelled on positive rows."""
    from datasets import load_dataset

    # the repo holds two overlapping sets of train shards; name one set explicitly
    files = (
        "data/train/train-0000[0-3]-of-00004.parquet"
        if split == "train"
        else "data/test/test-00000-of-00001.parquet"
    )
    ds = load_dataset("ronantakizawa/github-codereview", data_files=files, split="train")
    types = [
        "bug",
        "security",
        "performance",
        "suggestion",
        "refactor",
        "style",
        "nitpick",
        "question",
    ]
    count = 0
    for row in _shuffled(ds, seed):
        neg = str(row["is_negative"]).lower() == "true"
        if not neg and row["comment_type"] not in types:
            continue
        diff = row["diff_context"] or row["before_code"]
        if not diff or len(diff) > 6000:
            continue
        labels: dict = {"needs_comment": not neg}
        if not neg:
            labels["issue_type"] = row["comment_type"]
        yield {
            "state": {
                "file": row["file_path"],
                "language": row["language"],
                "pr_title": row["pr_title"],
                "diff": diff,
            },
            "questions": {
                "needs_comment": _noul(
                    "Would an experienced reviewer leave a comment on this code?",
                    "there is something worth pointing out",
                    "it is fine as is",
                ),
                "issue_type": _choice(
                    "If a reviewer commented, what kind of issue would it most likely be?",
                    {t: None for t in types},
                ),
            },
            "labels": labels,
            "source": "codereview",
        }
        count += 1
        if count >= n:
            return


RECIPES = {
    "banking77": banking77,
    "clinc": clinc,
    "mmlu_pro": mmlu_pro,
    "civil_comments": civil_comments,
    "halueval": halueval,
    "msmarco": msmarco,
    "helpsteer2": helpsteer2,
    "codereview": codereview,
}

# ------------------------------------------------------------------ synthetic recipes
# Verifiable data you can make with code. Each generator builds a state, a question and a
# label that is correct BY CONSTRUCTION (a checker function, a stated rule, or stated
# frequencies), so the model learns a *skill* rather than a dataset. These target the
# failure shapes seen on JevBench's public items without touching any benchmark item.

_WORDS = [
    "apple",
    "river",
    "stone",
    "cloud",
    "lantern",
    "harbor",
    "meadow",
    "copper",
    "velvet",
    "signal",
    "orbit",
    "maple",
    "ember",
    "canyon",
    "ledger",
]
_TOPICS = [
    "the weather in Lisbon",
    "how to boil an egg",
    "why the sky is blue",
    "our refund policy",
    "the history of chess",
    "how to reset a password",
    "what a mutex is",
    "the moon landing",
]


def _sentences(rng, n):
    return " ".join(
        f"{rng.choice(_WORDS).capitalize()} {rng.choice(_WORDS)} {rng.choice(_WORDS)}."
        for _ in range(n)
    )


def adequacy_synth(n: int, seed: int, split: str) -> Iterator[dict]:
    """Judging whether a response satisfies a request EXACTLY. Every template has a
    programmatic checker; negatives are near-misses (off by one word, right content in
    the wrong container, a forbidden capital letter), because that is what a judge gets
    wrong. Labels: noul, hard."""
    rng = random.Random(seed + (0 if split == "train" else 7919))
    templates = [
        # (request builder, satisfying response, near-miss response)
        lambda r: (
            (
                f"Say exactly {k} words.",
                " ".join(r.sample(_WORDS, k)),
                " ".join(r.sample(_WORDS, k + r.choice([-1, 1]))),
            )
            if (k := r.randint(2, 6))
            else None
        ),
        lambda r: (
            (
                f"Return a JSON array containing the integer {v}.",
                f"[{v}]",
                r.choice([f'{{"value": {v}}}', f"[{v + 1}]", f'["{v}"]']),
            )
            if (v := r.randint(1, 99))
            else None
        ),
        lambda r: (
            (
                f"Reply in all lowercase letters about {t}.",
                _sentences(r, 2).lower(),
                _sentences(r, 2),
            )
            if (t := r.choice(_TOPICS))
            else None
        ),
        lambda r: (
            (
                f"Reply in ALL CAPITAL letters about {t}.",
                _sentences(r, 2).upper(),
                _sentences(r, 2).capitalize(),
            )
            if (t := r.choice(_TOPICS))
            else None
        ),
        lambda r: (
            (
                f"Write exactly {k} sentences about {t}.",
                _sentences(r, k),
                _sentences(r, k + r.choice([-1, 1])),
            )
            if (k := r.randint(1, 4)) and (t := r.choice(_TOPICS))
            else None
        ),
        lambda r: (
            (
                f"Answer with only one of: {', '.join(o)}.",
                r.choice(o),
                r.choice([f"I think {o[0]}", "maybe", o[0].upper() + "!"]),
            )
            if (o := r.sample(["yes", "no", "north", "south", "red", "green", "cat", "dog"], 3))
            else None
        ),
        lambda r: (
            (
                f"Include the word '{w}' in your reply about {t}.",
                _sentences(r, 1) + f" I like {w}.",
                _sentences(r, 2),
            )
            if (w := r.choice(_WORDS)) and (t := r.choice(_TOPICS))
            else None
        ),
        lambda r: (
            (
                f"Do not use the word '{w}'. Describe {t} in one sentence.",
                f"{t.capitalize()} is worth a careful look.",
                f"The {w} matters when thinking about {t}.",
            )
            if (w := r.choice(_WORDS)) and (t := r.choice(_TOPICS))
            else None
        ),
        lambda r: (
            (
                f"Reply with a single integer: what is {a} + {b}?",
                str(a + b),
                r.choice([str(a + b + 1), f"{a + b}.0", f"The answer is {a + b}"]),
            )
            if (a := r.randint(2, 60)) and (b := r.randint(2, 60))
            else None
        ),
        lambda r: (
            (
                f"End your reply with the exact phrase '{ph}'.",
                _sentences(r, 1) + " " + ph,
                ph + " " + _sentences(r, 1),
            )
            if (ph := r.choice(["Thank you.", "Over and out.", "That is all."]))
            else None
        ),
    ]
    count = 0
    while count < n:
        request, good, bad = rng.choice(templates)(rng)
        for response, ok in ((good, True), (bad, False)):
            yield {
                "state": {"request": request, "response": response},
                "questions": {
                    "satisfies": _noul(
                        "Does the `response` fully satisfy the `request`, including every explicit constraint (counts, format, case, wording)?",
                        "every constraint is met exactly",
                        "at least one constraint is violated, even slightly",
                    )
                },
                "labels": {"satisfies": ok},
                "source": "adequacy_synth",
            }
            count += 1


_SPECIALISTS = {
    "files": "working with a file the user has shared: reading, summarising, extracting",
    "code_help": "explaining or writing code snippets without touching a repository",
    "repo_agent": "changing files inside a repository and running its checks",
    "calculator": "arithmetic, finance and other quantitative questions",
    "web": "anything that needs live or very recent information",
    "chat": "open conversation, creative writing, advice",
}
_REQUESTS = [
    ("Go through the lease I shared and pull out every notice period.", "files"),
    ("What are the three main risks in the audit report I uploaded?", "files"),
    ("Turn the shared spreadsheet's first tab into a short summary.", "files"),
    ("Show me how to debounce a function in TypeScript.", "code_help"),
    ("Why does this SQL query return duplicate rows?", "code_help"),
    ("Update the retry logic in client.py and make sure the test suite passes.", "repo_agent"),
    ("Add a --verbose flag to the CLI and run the linter afterwards.", "repo_agent"),
    ("How much is 8.5 % of 2,150?", "calculator"),
    ("Compare 4.2 % compounded monthly with 4.3 % compounded yearly over ten years.", "calculator"),
    ("Is the London Underground running normally this morning?", "web"),
    ("What did the central bank announce today?", "web"),
    ("Help me word a polite reminder to a neighbour about noise.", "chat"),
    ("Give me three ideas for a rainy-day activity with kids.", "chat"),
]
_RULES = [
    (
        "Whenever the user refers to something they shared or uploaded, send it to `files`, whatever else the request asks for.",
        lambda req, lab: "files" if any(w in req for w in ("shared", "uploaded")) else lab,
    ),
    (
        "`repo_agent` is only for requests that change files and run checks; explanations and snippets stay with `code_help`.",
        lambda req, lab: lab,
    ),
    (
        "If the answer depends on today's state of the world, use `web`; never answer such questions from memory.",
        lambda req, lab: lab,
    ),
    (
        "Percentages, interest and money calculations always go to `calculator`, even when a shared file is involved.",
        lambda req, lab: "calculator" if ("%" in req or "compounded" in req) else lab,
    ),
    (
        "Requests that only need judgement or wording, with no computation or lookup, go to `chat`.",
        lambda req, lab: lab,
    ),
]


def rule_routing_synth(n: int, seed: int, split: str) -> Iterator[dict]:
    """Routing under a stated policy: the state carries an explicit rule that sometimes
    overrides the surface cue. Labels: choice, hard, decided by the rule."""
    rng = random.Random(seed + (0 if split == "train" else 7919))
    for _ in range(n):
        req, base = rng.choice(_REQUESTS)
        rule_text, apply = rng.choice(_RULES)
        yield {
            "state": {"routing_policy": rule_text, "request": req},
            "questions": {
                "specialist": _choice(
                    "Choose the specialist for the `request`, following the `routing_policy` when it applies.",
                    dict(_SPECIALISTS),
                )
            },
            "labels": {"specialist": apply(req, base)},
            "source": "rule_routing_synth",
        }


_OUTCOMES = [
    ("support ticket", ["resolved_first_contact", "escalated", "reopened"]),
    ("delivery", ["on_time", "late", "early"]),
    ("chargeback dispute", ["merchant_wins", "customer_wins", "customer_withdraws"]),
    ("appeal", ["upheld", "overturned", "modified"]),
    ("appointment", ["attended", "late_cancellation", "no_show"]),
    ("outage", ["bad_push", "upstream_provider", "database_hardware"]),
]


def base_rates_synth(n: int, seed: int, split: str) -> Iterator[dict]:
    """Reading a probability off stated evidence: the state gives counts of past outcomes
    for cases like this one; the honest answer is the frequency, not a confident pick.
    Labels: choice, SOFT (the exact frequencies)."""
    rng = random.Random(seed + (0 if split == "train" else 7919))
    for _ in range(n):
        thing, outcomes = rng.choice(_OUTCOMES)
        total = rng.choice([20, 40, 50, 80, 100])
        cuts = sorted(rng.sample(range(1, total), len(outcomes) - 1))
        counts = [b - a for a, b in zip([0] + cuts, cuts + [total])]
        rng.shuffle(counts)
        history = ", ".join(f"{c} were {o.replace('_', ' ')}" for o, c in zip(outcomes, counts))
        yield {
            "state": {
                "case": f"A new {thing} with the same profile as the historical ones below.",
                "history": f"Of the last {total} similar cases, {history}.",
            },
            "questions": {
                "outcome": _choice(
                    f"What will the outcome of this {thing} be? Answer with probabilities that reflect the `history`.",
                    {o: None for o in outcomes},
                )
            },
            "labels": {"outcome": {o: c / total for o, c in zip(outcomes, counts)}},
            "source": "base_rates_synth",
        }


def helpsteer2_soft(n: int, seed: int, split: str) -> Iterator[dict]:
    """HelpSteer2's raw per-rater scores (3-5 raters per response) turned into a SOFT
    distribution over the five helpfulness levels, so disagreement becomes the target."""
    import gzip

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "nvidia/HelpSteer2", "disagreements/disagreements.jsonl.gz", repo_type="dataset"
    )
    with gzip.open(path, "rt") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = len(rows) * 9 // 10
    rows = rows[:cut] if split == "train" else rows[cut:]
    levels = [
        "not helpful at all: ignores or misunderstands the request",
        "slightly helpful: touches the request but mostly misses",
        "partially helpful: addresses the request with notable gaps",
        "mostly helpful: addresses the request well with minor gaps",
        "extremely helpful: complete, correct and well presented",
    ]
    count = 0
    for row in rows:
        votes = [int(v) for v in row.get("helpfulness") or [] if v is not None]
        if len(votes) < 2 or len(set(votes)) == 1:
            continue  # keep the disagreements: that is the point of this recipe
        dist = [votes.count(i) / len(votes) for i in range(5)]
        yield {
            "state": {"prompt": row["prompt"][-4000:], "response": row["response"][-4000:]},
            "questions": {
                "helpfulness": _score(
                    "How helpful is the `response` to the `prompt`? Several raters scored it; answer with the distribution of their scores.",
                    levels,
                )
            },
            "labels": {"helpfulness": dist},
            "source": "helpsteer2_soft",
        }
        count += 1
        if count >= n:
            return


RECIPES.update(
    {
        "adequacy_synth": adequacy_synth,
        "rule_routing_synth": rule_routing_synth,
        "base_rates_synth": base_rates_synth,
        "helpsteer2_soft": helpsteer2_soft,
    }
)


def write(rows: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ------------------------------------------------ long policy documents (synthetic)
# The hardest family on external decision sets is "apply a 2-3k-token policy with many
# numbered rules, thresholds, exclusions and a precedence order to a concrete case".
# This generator writes such a policy from templates, a case with concrete attributes,
# and computes the outcome in code by evaluating the rules in order, so the label is
# correct by construction and the state is genuinely long.

_POLICY_DOMAINS = {
    "expense_reimbursement": {
        "attrs": {
            "amount": (5, 4000),
            "days_since": (0, 120),
            "category": ["travel", "meals", "equipment", "software", "training", "entertainment"],
            "receipt": [True, False],
            "manager_approved": [True, False],
            "employee_level": ["junior", "senior", "director"],
        },
        "outcomes": [
            "reimburse_in_full",
            "reimburse_capped",
            "reject_missing_receipt",
            "reject_late_claim",
            "reject_excluded_category",
            "escalate_to_finance",
        ],
    },
    "insurance_claim": {
        "attrs": {
            "amount": (100, 25000),
            "days_since": (0, 400),
            "category": ["water_damage", "theft", "fire", "wear_and_tear", "flood", "accidental"],
            "receipt": [True, False],
            "manager_approved": [True, False],
            "employee_level": ["standard", "premium", "platinum"],
        },
        "outcomes": [
            "pay_in_full",
            "pay_capped",
            "deny_no_documentation",
            "deny_late_notice",
            "deny_excluded_peril",
            "refer_to_adjuster",
        ],
    },
    "access_request": {
        "attrs": {
            "amount": (1, 365),
            "days_since": (0, 60),
            "category": [
                "production_db",
                "billing_console",
                "source_repo",
                "hr_records",
                "staging",
                "analytics",
            ],
            "receipt": [True, False],
            "manager_approved": [True, False],
            "employee_level": ["contractor", "employee", "lead"],
        },
        "outcomes": [
            "grant",
            "grant_time_limited",
            "deny_no_ticket",
            "deny_stale_request",
            "deny_restricted_system",
            "route_to_security_review",
        ],
    },
}
_FILLER = [
    "Definitions in this section apply throughout unless a clause states otherwise.",
    "Requests are processed in the order received; processing times are not guaranteed.",
    "Nothing in this policy creates an entitlement beyond what the applicable clause grants.",
    "Records are retained for seven years in accordance with the retention schedule.",
    "Questions about interpretation should be sent to the policy owner listed in Appendix A.",
    "This policy is reviewed annually; the version date appears in the footer.",
    "Approvers must not approve their own requests under any circumstances.",
    "Amounts are stated in the requester's local currency before tax.",
    "A request may be withdrawn by the requester at any time before a decision is issued.",
    "Decisions are communicated in writing within five working days of being made.",
    "Appeals are heard by the committee named in Appendix C and follow the appeals procedure.",
    "Supporting documents must be legible; illegible documents are treated as not provided.",
    "The policy owner may publish non-binding guidance notes on the application of this policy.",
    "Delegations of authority are recorded in the delegation register maintained by the secretariat.",
    "Where a request spans several categories, the category of the largest component applies.",
    "Dates are interpreted in the requester's local time zone.",
    "Verbal approvals are not recognised for the purposes of this policy.",
    "The secretariat may request additional information; the clock stops until it is received.",
    "Historic decisions do not bind future decisions on similar requests.",
    "Confidential information in a request is handled under the information-handling standard.",
    "A request is filed on the date it is received by the secretariat, not the date it is sent.",
    "Training on this policy is mandatory for approvers and is refreshed every two years.",
]


def _policy(rng, dom):
    """Return (policy_text, decide(case) -> outcome). Rules are checked in clause order."""
    o = dom["outcomes"]
    cap = rng.choice([250, 500, 1000, 2500])
    late = rng.choice([30, 45, 60, 90])
    excluded = rng.sample(dom["attrs"]["category"], 2)
    big = rng.choice([1500, 3000, 5000, 10000])
    senior = dom["attrs"]["employee_level"][-1]
    limited = rng.choice([90, 180])
    clauses = [
        (
            f"A request older than {late} days at the time of filing is refused as late.",
            lambda c: o[3] if c["days_since"] > late else None,
        ),
        (
            f"Requests in the categories {excluded[0]} and {excluded[1]} are excluded and refused, regardless of amount or approval.",
            lambda c: o[4] if c["category"] in excluded else None,
        ),
        (
            "A request without supporting documentation is refused, unless the requester holds the "
            f"{senior} level, in which case documentation may follow within 10 days.",
            lambda c: o[2] if (not c["receipt"] and c["employee_level"] != senior) else None,
        ),
        (
            f"Any request above {big} must be referred for a second review; it is neither granted nor refused at this stage.",
            lambda c: o[5] if c["amount"] > big else None,
        ),
        (
            f"Requests above {cap} that carry manager approval are granted up to the cap of {cap}.",
            lambda c: o[1] if (c["amount"] > cap and c["manager_approved"]) else None,
        ),
        (
            f"Requests above {cap} without manager approval are referred for a second review.",
            lambda c: o[5] if (c["amount"] > cap and not c["manager_approved"]) else None,
        ),
        (
            "All remaining requests are granted in full."
            + (
                f" For the {dom['attrs']['employee_level'][0]} level the grant is limited to {limited} days."
                if "grant" in o[0]
                else ""
            ),
            lambda c: (
                o[1]
                if ("grant" in o[0] and c["employee_level"] == dom["attrs"]["employee_level"][0])
                else o[0]
            ),
        ),
    ]
    order = list(range(len(clauses)))
    # keep the deciding order but interleave filler and re-number so the document reads like a real policy
    lines = [
        f"POLICY {rng.randint(100, 999)}: {rng.choice(['Handling of', 'Rules for', 'Procedure for'])} {list(_POLICY_DOMAINS).index(next(k for k, v in _POLICY_DOMAINS.items() if v is dom)) + 1}.0 requests\n"
    ]
    n = 1
    for i in order:
        for _ in range(rng.randint(1, 3)):
            lines.append(f"{n}. {rng.choice(_FILLER)}")
            n += 1
        lines.append(f"{n}. {clauses[i][0]}")
        n += 1
    lines.append(
        f"{n}. Clauses are applied in numerical order; the first clause whose condition is met decides the request."
    )
    text = "\n".join(lines)

    def decide(c):
        for _, rule in clauses:
            r = rule(c)
            if r:
                return r
        return o[0]

    return text, decide


def policy_synth(n: int, seed: int, split: str) -> Iterator[dict]:
    """Long policy application: a 1.5-3k-token policy document plus a concrete case;
    the outcome is computed by applying the clauses in order. Labels: choice, hard."""
    rng = random.Random(seed + (0 if split == "train" else 7919))
    counts: dict[str, int] = {}
    made = 0
    while made < n:
        name = rng.choice(list(_POLICY_DOMAINS))
        dom = _POLICY_DOMAINS[name]
        text, decide = _policy(rng, dom)
        a = dom["attrs"]
        case = {
            "amount": rng.randint(*a["amount"]),
            "days_since": rng.randint(0, 100),
            "category": rng.choice(a["category"]),
            "receipt": rng.random() < 0.75,
            "manager_approved": rng.choice(a["manager_approved"]),
            "employee_level": rng.choice(a["employee_level"]),
        }
        outcome = decide(case)
        # rejection sampling keeps the six outcomes roughly balanced
        if counts.get(outcome, 0) > min([counts.get(o, 0) for o in dom["outcomes"]]) + 2:
            continue
        counts[outcome] = counts.get(outcome, 0) + 1
        made += 1
        # appendix of notes so states reach roughly 1.5-3k tokens
        pad = "\n".join(f"Note {i + 1}: {rng.choice(_FILLER)}" for i in range(rng.randint(40, 90)))
        yield {
            "state": {
                "policy_document": text + "\n\nAppendix B: general notes\n" + pad,
                "request": case,
            },
            "questions": {
                "outcome": _choice(
                    "Applying the `policy_document` to the `request`, what is the correct outcome? Apply clauses in numerical order; the first clause whose condition holds decides.",
                    {o: None for o in dom["outcomes"]},
                )
            },
            "labels": {"outcome": outcome},
            "source": "policy_synth",
        }


RECIPES["policy_synth"] = policy_synth


def parse_counts(spec: str, default: int) -> dict[str, int]:
    """'a,b,c=1200' -> {a: default, b: default, c: 1200}"""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, num = part.partition("=")
        out[name.strip()] = int(num) if num else default
    return out


def ngrams(text: str, n: int = 8) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def check_overlap(train_path: str, against: list[str], n: int = 8) -> int:
    """Flag training rows that share an n-gram of `n` words with any benchmark item.
    Run this before training on anything you intend to evaluate on."""
    bench = set()
    for path in against:
        for row in read_jsonl_any(path):
            # states only: our question templates are generic and may echo a benchmark's wording
            bench |= ngrams(json.dumps(row.get("state", "")), n)
    hits = 0
    for i, row in enumerate(read_jsonl_any(train_path)):
        if ngrams(json.dumps(row.get("state", "")), n) & bench:
            hits += 1
            if hits <= 10:
                print(f"  overlap: row {i} source={row.get('source')}")
    return hits


def read_jsonl_any(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description="build reflex training data from public datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix", help="sample from several recipes into one JSONL")
    m.add_argument(
        "--sources",
        default=",".join(RECIPES),
        help="comma-separated recipe names, optionally name=count",
    )
    m.add_argument("--per-source", type=int, default=800)
    m.add_argument("--eval-per-source", type=int, default=200)
    m.add_argument("--out", required=True)
    m.add_argument("--eval-out", default=None)
    m.add_argument("--seed", type=int, default=0)
    c = sub.add_parser(
        "check-overlap", help="flag training rows that share 8-grams with benchmark items"
    )
    c.add_argument("--train", required=True)
    c.add_argument("--against", nargs="+", required=True)
    c.add_argument("--n", type=int, default=8)
    args = ap.parse_args(argv)

    if args.cmd == "check-overlap":
        hits = check_overlap(args.train, args.against, args.n)
        print(f"{hits} overlapping rows")
        return 1 if hits else 0

    counts = parse_counts(args.sources, args.per_source)
    for s in counts:
        if s not in RECIPES:
            ap.error(f"unknown source {s!r}; choose from {', '.join(RECIPES)}")

    def build(split: str, scale: float):
        rows = []
        for s, per in counts.items():
            got = list(RECIPES[s](max(1, int(per * scale)), args.seed, split))
            print(f"  {s:<20} {split:<6} {len(got):>6} examples")
            rows.extend(got)
        random.Random(args.seed).shuffle(rows)
        return rows

    print("train:")
    n = write(build("train", 1.0), args.out)
    print(f"wrote {n} rows to {args.out}")
    if args.eval_out:
        print(f"eval ({args.eval_per_source}/{args.per_source} of each source, held-out splits):")
        n = write(build("test", args.eval_per_source / args.per_source), args.eval_out)
        print(f"wrote {n} rows to {args.eval_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
