"""Typed questions for each raw state.

Two sources, both attached to every state:

* a small **bank** per domain: the standard operational questions a decision model gets
  asked about that kind of input (route it, escalate it, how urgent, is it safe...);
* **LLM-written** questions: an instruction-following model reads the state and writes a
  few more, in its own words, with its own option sets. This is where wording diversity
  comes from, so the student does not learn one phrasing per task.

No answers are produced here. Every question is validated as a real `SystemOneRequest`
question before it is kept.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from reflex.schema import SystemOneRequest

log = logging.getLogger("reflex.distill")


# ------------------------------------------------------------------------------ bank
def _noul(instructions: str) -> dict:
    return {"type": "noul", "instructions": instructions}


def _choice(instructions: str, options: dict) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": options}


def _score(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": levels}


_URGENCY = [
    "can wait a week or more",
    "should be handled within a few days",
    "needs attention today",
    "blocking right now",
]
_QUALITY = ["poor", "below average", "acceptable", "good", "excellent"]

BANK: dict[str, dict[str, dict]] = {
    "support_ticket": {
        "queue": _choice(
            "Which team should handle this ticket?",
            {
                "billing": "charges, invoices, refunds",
                "technical": "bugs, errors, outages",
                "account": "login, access, profile changes",
                "sales": "pricing, upgrades, purchases",
                "other": None,
            },
        ),
        "escalate": _noul(
            "Should this ticket be escalated to a human supervisor rather than handled by first-line support?"
        ),
        "urgency": _score("How urgent is this ticket?", _URGENCY),
        "angry": _noul("Is the customer expressing frustration or anger?"),
    },
    "pr_hunk": {
        "needs_review": _noul("Does this change need a careful human review before merging?"),
        "issue": _choice(
            "What is the most important concern with this change, if any?",
            {
                "bug": "incorrect behaviour",
                "security": "unsafe input handling, secrets, injection",
                "performance": None,
                "readability": "naming, structure, clarity",
                "none": "nothing worth raising",
            },
        ),
        "risk": _score("How risky is this change to ship?", ["trivial", "low", "moderate", "high"]),
    },
    "chat_prompt": {
        "harmful": _noul(
            "Is the user asking for something harmful, illegal, or against typical usage policies?"
        ),
        "category": _choice(
            "What kind of request is this?",
            {
                "information": "facts, explanations",
                "creative": "stories, poems, brainstorming",
                "coding": None,
                "advice": "personal, medical, legal, financial",
                "task": "rewrite, summarise, translate",
                "other": None,
            },
        ),
        "needs_clarification": _noul(
            "Is the request too ambiguous to answer well without asking a follow-up question?"
        ),
        "complexity": _score(
            "How much reasoning does a good answer need?",
            [
                "a one-line reply",
                "a short explanation",
                "a structured multi-part answer",
                "careful multi-step reasoning",
            ],
        ),
    },
    "review": {
        "stars": _score(
            "How would this reviewer rate the product, on a 1 to 5 star scale?",
            ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        ),
        "recommend": _noul("Would the reviewer recommend this to others?"),
        "complaint": _choice(
            "What is the main complaint, if any?",
            {
                "quality": "broke, defective, poorly made",
                "value": "overpriced",
                "service": "shipping, support, returns",
                "expectations": "not as described",
                "none": "no complaint",
            },
        ),
    },
    "news": {
        "topic": _choice(
            "What is the main topic?",
            {
                "politics": None,
                "business": None,
                "technology": None,
                "sports": None,
                "science_health": None,
                "crime_justice": None,
                "culture": None,
                "other": None,
            },
        ),
        "opinion": _noul("Is this primarily opinion or commentary rather than factual reporting?"),
        "market_moving": _noul("Could this news plausibly move a company's stock price?"),
    },
    "legal_clause": {
        "obligation": _noul("Does this text impose an obligation on the user or customer?"),
        "unfair": _noul(
            "Does this text contain a term a consumer-protection reviewer would flag as potentially unfair?"
        ),
        "party": _choice(
            "Who benefits most from this provision?",
            {
                "provider": "the company or service",
                "customer": "the user or buyer",
                "balanced": None,
            },
        ),
        "restrictiveness": _score(
            "How restrictive is this for the customer?",
            [
                "not restrictive",
                "mildly restrictive",
                "significantly restrictive",
                "severely restrictive",
            ],
        ),
    },
    "long_document": {
        "audience": _choice(
            "Who is the intended audience?",
            {
                "general_public": None,
                "specialists": "experts in the field",
                "policymakers": None,
                "students": None,
            },
        ),
        "recommends_action": _noul(
            "Does the document recommend a specific action or policy change?"
        ),
        "readability": _score(
            "How hard is this document to read?",
            ["easy", "moderate", "demanding", "very technical"],
        ),
    },
    "forum_post": {
        "seeking": _choice(
            "What does the poster want?",
            {
                "advice": None,
                "information": None,
                "venting": "sharing an experience or feelings",
                "discussion": "opinions from others",
                "other": None,
            },
        ),
        "distress": _noul("Does the poster appear to be in distress or crisis?"),
        "professional_help": _noul(
            "Should the poster be advised to consult a professional (lawyer, doctor, therapist)?"
        ),
    },
    "assistant_reply": {
        "helpful": _score("How helpful is the assistant's most recent reply?", _QUALITY),
        "harmful": _noul(
            "Does the assistant's reply contain harmful, dangerous, or policy-violating content?"
        ),
        "refused": _noul("Did the assistant decline or deflect the user's request?"),
        "hallucination_risk": _score(
            "How likely is it that the reply contains fabricated facts?",
            ["very unlikely", "possible", "likely", "almost certain"],
        ),
    },
    "qa_passage": {
        "supported": _noul(
            "Is the proposed answer or response fully supported by the provided passages or context?"
        ),
        "answerable": _noul("Can the question be answered from the provided material alone?"),
        "confidence": _score(
            "How confident should a careful reader be in the proposed answer?",
            ["not at all", "slightly", "fairly", "very"],
        ),
    },
}

# any-domain questions asked of every state, so the student sees them in many contexts
GENERAL: dict[str, dict] = {
    "needs_human": _noul(
        "Does this input need a person to look at it rather than an automated process?"
    ),
    "sentiment": _choice(
        "What is the overall tone?", {"negative": None, "neutral": None, "positive": None}
    ),
    "clarity": _score(
        "How clear and well-formed is this text?",
        ["confusing", "somewhat unclear", "clear", "very clear"],
    ),
}

# ------------------------------------------------------------------------------ LLM
GEN_PROMPT = """You write evaluation questions for a decision model. The model will read the STATE below and answer each question with a probability distribution; it cannot write text.

Write {n} questions about the STATE, as a JSON array. Each question is one of:
  {{"id": "<short_snake_case>", "type": "noul",   "instructions": "<a yes/no question>"}}
  {{"id": "<short_snake_case>", "type": "choice", "instructions": "<which one?>", "criteria": {{"<option>": "<one-line description or null>", ...}}}}   (2 to 6 options, mutually exclusive, covering the realistic answers; include a fallback option when sensible)
  {{"id": "<short_snake_case>", "type": "score",  "instructions": "<how much / how likely?>", "criteria": ["<lowest level>", ..., "<highest level>"]}}   (3 to 5 ordered levels)

Rules:
- Every question must be answerable from the STATE by judgement, not by looking up a single word. Mix easy and genuinely uncertain ones.
- Ask about things someone operating a real system would want to know: what to do with this input, what it is, how severe or good it is, whether a claim holds, what happens next.
- Vary your phrasing; do not reuse the example wording. Do not include answers. Use at least two different types.
- Output ONLY the JSON array.

STATE:
{state}
"""


def _extract_json_array(text: str):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def validate(state, questions: dict) -> dict:
    """Keep only questions that build as a real request (types, option counts, keys)."""
    good = {}
    for qid, q in questions.items():
        try:
            SystemOneRequest(state=state, questions={qid: q})
        except (ValueError, TypeError) as e:  # pydantic ValidationError is a ValueError
            log.debug("dropping question %s: %s", qid, str(e)[:120])
            continue
        if not q.get("instructions") or len(q["instructions"]) > 400:
            continue
        good[qid] = q
    return good


def clean(q: dict) -> dict:
    """Only the fields a question has; LLMs add ids and commentary."""
    item = {"type": q.get("type"), "instructions": q.get("instructions")}
    if q.get("type") in ("choice", "score"):
        item["criteria"] = q.get("criteria")
    return item


def _llm_questions(call, state, n: int, tag: str = "llm") -> dict:
    text = call(GEN_PROMPT.format(n=n, state=json.dumps(state, ensure_ascii=False)[:9000]))
    arr = _extract_json_array(text) or []
    qs = {}
    for i, q in enumerate(arr):
        if not isinstance(q, dict):
            continue
        qid = re.sub(r"[^a-z0-9_]", "_", str(q.get("id") or f"q{i}").lower())[:40] or f"q{i}"
        qid = f"{tag}_{qid}"
        qs[qid] = clean(q)
    return validate(state, qs)


def gateway_chat(base_url: str, key: str, model: str, max_tokens: int = 900):
    import httpx

    def call(prompt: str) -> str:
        for attempt in range(3):
            try:
                r = httpx.post(
                    f"{base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.8,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=300,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001
                log.warning("question LM attempt %d failed: %s", attempt + 1, e)
                time.sleep(3)
        return ""

    return call


def attach(
    rows: list[dict],
    call=None,
    per_state: int = 3,
    workers: int = 8,
    bank_max: int = 3,
    seed: int = 0,
    tag: str = "llm",
    bank: bool = True,
):
    """Bank questions (domain + general, sampled so no state carries all of them) plus
    `per_state` LLM-written ones. Returns rows with a `questions` dict."""
    import random

    rng = random.Random(seed)

    def one(row):
        pool = dict(BANK.get(row["domain"], {}))
        pool.update(GENERAL)
        keys = list(pool)
        rng2 = random.Random(f"{seed}:{row['id']}")
        chosen = rng2.sample(keys, min(bank_max, len(keys)))
        qs = validate(row["state"], {k: pool[k] for k in chosen}) if bank else {}
        if call is not None and per_state > 0:
            qs.update(_llm_questions(call, row["state"], per_state, tag))
        return {**row, "questions": qs}

    rng.shuffle(rows)
    with ThreadPoolExecutor(workers) as ex:
        for i, out in enumerate(ex.map(one, rows)):
            if (i + 1) % 200 == 0:
                log.info("questions: %d/%d states", i + 1, len(rows))
            if out["questions"]:
                yield out
