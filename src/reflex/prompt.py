"""Turn (state, question) into token ids with an exactly shared state prefix.

Layout of one request (ChatML, which Qwen instruct models are trained on):

    prefix  = <|im_start|>system ... <|im_end|>
              <|im_start|>user
              # State
              <state>

    branch_i = # Question
               <instructions>
               # Options
               A. ...
               B. ...
               Respond with only the letter.<|im_end|>
               <|im_start|>assistant
               <think>\n\n</think>\n\n          (Qwen3 "no-think" prefix)

The prefix is tokenized *once* and the branches are tokenized separately, so the
prefix token ids are byte-identical across branches and can be cached / shared.
The model never generates: we read next-token logits at the end of each branch and
restrict them to the label tokens (A/B/C…, Yes/No).
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from typing import Any

from reflex.schema import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    Text,
)

SYSTEM_PROMPT = (
    "You are a System One decision model. You read the State and answer each Question "
    "by choosing exactly one of the listed options. You never explain. You answer with "
    "the single option label only."
)

LETTERS = string.ascii_uppercase  # A..Z
YES, NO = "Yes", "No"


def render_text(x: Text | None) -> str:
    """Render instructions / criteria / state given as str or JSON-ish into prompt text."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False, indent=2)


@dataclass
class Branch:
    """One isolated question branch: text + which label tokens to read out."""

    qid: str
    kind: str  # "noul" | "choice" | "score"
    text: str
    labels: list[str]  # label strings whose next-token logit we read, in prompt order
    # For choice/score: option keys (choice) or level indices (score) in prompt order,
    # i.e. keys[i] is what labels[i] refers to. Lets us undo permutations.
    keys: list[Any] = field(default_factory=list)


# Every piece of instruction wording the model sees, as named components. A prompt
# optimiser (reflex-optimize, GEPA) edits these; nothing else about the layout changes.
DEFAULT_TEXTS = {
    "system_prompt": SYSTEM_PROMPT,
    "state_heading": "# State",
    "question_heading": "# Question",
    "options_heading": "# Options",
    "noul_ask": "Respond with only Yes or No.",
    "choice_ask": "Respond with only the letter of the best option.",
    "score_ask": "Respond with only the letter of the level that best matches.",
    "noul_true_default": "The statement is true.",
    "noul_false_default": "The statement is false.",
    "score_level_prefix": "(level {i} of {n})",
}

COMPACT_SYSTEM_PROMPT = (
    "You are given a JSON object with the state, a question about it, and lettered options. "
    "Judge the question against the state and pick the single best option. "
    "Reply with that option's letter only."
)


@dataclass
class PromptFormat:
    """How to wrap text for a given model family.

    `style` selects the prompt layout:
      * "markdown": headed sections (# State / # Question / # Options), Yes/No tokens for noul
      * "compact":  one JSON object {state, question, options:[{letter, text}]} with a short
                    instruction, and every primitive (noul included) read out as a letter.
                    Compact JSON keeps the model's attention on the content and avoids the
                    stylistic pull of a "Yes"/"No" answer token.
    """

    chat: bool = True  # ChatML wrapping (Qwen instruct); False = plain text (base models)
    no_think: bool = True  # Qwen3 hybrid models: emit empty <think> block to skip reasoning
    system_prompt: str = SYSTEM_PROMPT
    style: str = "markdown"
    texts: dict[str, str] = field(default_factory=dict)  # overrides for DEFAULT_TEXTS

    def __post_init__(self):
        if self.style not in ("markdown", "compact"):
            raise ValueError(f"unknown prompt style {self.style!r}")
        if self.style == "compact" and self.system_prompt == SYSTEM_PROMPT:
            self.system_prompt = COMPACT_SYSTEM_PROMPT
        if "system_prompt" in self.texts:
            self.system_prompt = self.texts["system_prompt"]

    def t(self, key: str) -> str:
        return self.texts.get(key, DEFAULT_TEXTS[key])

    @classmethod
    def with_texts(cls, base: PromptFormat, texts: dict[str, str]) -> PromptFormat:
        return cls(chat=base.chat, no_think=base.no_think, style=base.style, texts=dict(texts))

    def prefix(self, state: Text) -> str:
        if self.style == "compact":
            # the state is the first member of the JSON payload; branches close the object
            body = '{"state": ' + json.dumps(state, ensure_ascii=False) + ", "
        else:
            body = f"{self.t('state_heading')}\n{render_text(state)}\n\n"
        if not self.chat:
            return f"{self.system_prompt}\n\n{body}"
        return f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{body}"

    def branch(self, body: str) -> str:
        if not self.chat:
            return f"{body}Answer:"
        tail = "<|im_end|>\n<|im_start|>assistant\n"
        if self.no_think:
            tail += "<think>\n\n</think>\n\n"
        return f"{body}{tail}"


def _options_block(
    instructions: Text, labelled: list[tuple[str, str]], ask: str, fmt: PromptFormat | None = None
) -> str:
    qh = fmt.t("question_heading") if fmt else DEFAULT_TEXTS["question_heading"]
    oh = fmt.t("options_heading") if fmt else DEFAULT_TEXTS["options_heading"]
    lines = [f"{qh}\n{render_text(instructions)}\n", oh]
    for label, desc in labelled:
        lines.append(f"{label}. {desc}" if desc else f"{label}.")
    lines.append(f"\n{ask}\n")
    return "\n".join(lines)


def _compact_body(instructions: Text, labelled: list[tuple[str, str]]) -> str:
    """Close the JSON object opened by the compact prefix: question + lettered options."""
    payload = {
        "question": instructions,
        "options": [{"letter": lab, "text": desc} for lab, desc in labelled],
    }
    return json.dumps(payload, ensure_ascii=False)[1:]  # drop "{": the prefix opened it


def build_branches(
    qid: str,
    q: NoulQuestion | ChoiceQuestion | ScoreQuestion,
    fmt: PromptFormat,
    permutations: int = 1,
    rng: random.Random | None = None,
) -> list[Branch]:
    """Build 1..permutations branches for a question. Noul is never permuted."""
    rng = rng or random.Random(0)
    if fmt.style == "compact":
        return _build_compact(qid, q, fmt, permutations, rng)

    if isinstance(q, NoulQuestion):
        t = render_text(q.criteria.true) if q.criteria else ""
        f = render_text(q.criteria.false) if q.criteria else ""
        labelled = [(YES, t or fmt.t("noul_true_default")), (NO, f or fmt.t("noul_false_default"))]
        body = _options_block(q.instructions, labelled, fmt.t("noul_ask"), fmt)
        return [Branch(qid, "noul", fmt.branch(body), [YES, NO], [True, False])]

    if isinstance(q, ChoiceQuestion):
        keys = list(q.criteria.keys())
        descs = {k: (render_text(v) if v is not None else "") for k, v in q.criteria.items()}
        kind = "choice"
        ask = fmt.t("choice_ask")

        def desc_of(k):
            d = descs[k]
            return f"{k}: {d}" if d else k

    else:  # ScoreQuestion – ordered levels, never reordered semantically but may be permuted
        keys = list(range(len(q.criteria)))
        kind = "score"
        ask = fmt.t("score_ask")
        prefix = fmt.t("score_level_prefix")

        def desc_of(k):
            # plain token replacement: the prefix may be optimiser-generated and contain braces
            head = prefix.replace("{i}", str(k)).replace("{n}", str(len(keys) - 1))
            return f"{head} {render_text(q.criteria[k])}"

    out: list[Branch] = []
    for p in range(permutations):
        order = list(keys)
        if p > 0:
            rng.shuffle(order)
        labels = [LETTERS[i] for i in range(len(order))]
        labelled = [(lab, desc_of(k)) for lab, k in zip(labels, order)]
        body = _options_block(q.instructions, labelled, ask, fmt)
        out.append(Branch(qid, kind, fmt.branch(body), labels, order))
    return out


def _build_compact(
    qid, q, fmt: PromptFormat, permutations: int, rng: random.Random
) -> list[Branch]:
    """Compact style: every primitive is a lettered option list inside the JSON payload."""
    if isinstance(q, NoulQuestion):
        t = render_text(q.criteria.true) if q.criteria else ""
        f = render_text(q.criteria.false) if q.criteria else ""
        labelled = [
            ("A", "yes: " + (t or "the statement holds")),
            ("B", "no: " + (f or "the statement does not hold")),
        ]
        return [
            Branch(
                qid,
                "noul",
                fmt.branch(_compact_body(q.instructions, labelled)),
                ["A", "B"],
                [True, False],
            )
        ]
    if isinstance(q, ChoiceQuestion):
        keys = list(q.criteria.keys())
        kind = "choice"

        def desc_of(k):
            d = render_text(q.criteria[k]) if q.criteria[k] is not None else ""
            return f"{k}: {d}" if d else k
    else:
        keys = list(range(len(q.criteria)))
        kind = "score"

        def desc_of(k):
            return f"level {k} of {len(keys) - 1}: {render_text(q.criteria[k])}"

    out: list[Branch] = []
    for p in range(permutations):
        order = list(keys)
        if p > 0:
            rng.shuffle(order)
        labels = [LETTERS[i] for i in range(len(order))]
        labelled = [(lab, desc_of(k)) for lab, k in zip(labels, order, strict=True)]
        out.append(
            Branch(qid, kind, fmt.branch(_compact_body(q.instructions, labelled)), labels, order)
        )
    return out
