"""Training data: JSONL of System One requests with known answers.

One line per example:

    {"state":     <text or JSON>,
     "questions": {<qid>: <question>, ...},          # same shape as an API request
     "labels":    {<qid>: <label>, ...},             # the answer(s) we know to be right
     "source":    "banking77"}                       # optional, for per-task reporting

A label can be *hard* or *soft*:

    noul    true / false                 or a probability of "yes", e.g. 0.75
    choice  "billing"                    or {"billing": 0.7, "technical": 0.3}
    score   2 (a level index)            or {"1": 0.4, "2": 0.6}  /  [0, 0.4, 0.6]

Soft labels matter: when three annotators disagree, "0.67" is the honest target, and a
proper scoring rule trained on it learns that uncertainty instead of a coin flip.

`examples()` turns every labelled question into an `Example` holding the state, the
rendered `Branch` (the exact prompt the server uses) and a target distribution over the
branch's option keys. Training and serving share one code path.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.schema import SystemOneRequest


@dataclass
class Example:
    state: Any
    branch: Branch
    target: np.ndarray  # distribution over branch.keys, sums to 1
    source: str = ""

    @property
    def hard_label(self) -> int:
        """Index of the most likely option (used for accuracy / ECE reports)."""
        return int(self.target.argmax())


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def target_vector(kind: str, label: Any, keys: list[Any]) -> np.ndarray:
    """Hard or soft label -> probability vector aligned with `keys` (the branch's options)."""
    t = np.zeros(len(keys), dtype=np.float64)
    if kind == "noul":  # keys == [True, False]
        p = float(label) if not isinstance(label, bool) else (1.0 if label else 0.0)
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"noul label must be a bool or a probability, got {label!r}")
        t[keys.index(True)] = p
        t[keys.index(False)] = 1.0 - p
        return t
    if isinstance(label, dict):  # soft choice / score: {key: prob}
        for k, v in label.items():
            k = int(k) if kind == "score" else k
            t[keys.index(k)] = float(v)
    elif isinstance(label, list) and kind == "score":  # soft score: [p0, p1, ...]
        for i, v in enumerate(label):
            t[keys.index(i)] = float(v)
    else:  # hard label
        t[keys.index(int(label) if kind == "score" else label)] = 1.0
    if t.sum() <= 0:
        raise ValueError(f"label {label!r} puts no mass on any option")
    return t / t.sum()


def examples(path: str, fmt: PromptFormat, permutations: int = 1, seed: int = 0) -> list[Example]:
    """Flatten a JSONL file into training examples (one per labelled question, per permutation)."""
    return examples_from_rows(list(read_jsonl(path)), fmt, permutations, seed)


def examples_from_rows(
    rows: list[dict], fmt: PromptFormat, permutations: int = 1, seed: int = 0
) -> list[Example]:
    out: list[Example] = []
    for ri, row in enumerate(rows):
        req = SystemOneRequest(state=row["state"], questions=row["questions"])
        for qid, q in req.questions.items():
            if qid not in row.get("labels", {}):
                continue
            # orders are seeded per (row, question) so they never depend on neighbours
            for br in build_branches(qid, q, fmt, permutations, seed=seed * 100003 + ri):
                # br.keys is in *prompt order*, which differs per permutation: build the
                # target against those keys so shuffled option letters stay correct.
                target = target_vector(q.type, row["labels"][qid], br.keys)
                out.append(Example(req.state, br, target, row.get("source", "")))
    return out


def mmlu_to_jsonl(out_path: str, split: str = "validation", n: int | None = None, seed: int = 0):
    """Write MMLU as reflex training data (one Choice question per item)."""
    from reflex.eval.mmlu import load_items, to_question

    items = load_items(n or 10**9, seed, split)
    with open(out_path, "w") as f:
        for it in items:
            q, keys = to_question(it)
            f.write(
                json.dumps(
                    {
                        "state": it["question"],
                        "questions": {"answer": q.model_dump()},
                        "labels": {"answer": keys[it["answer"]]},
                        "source": "mmlu",
                    }
                )
                + "\n"
            )
    return len(items)
