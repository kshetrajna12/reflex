"""Training data: JSONL of System One requests with known answers.

Each line:
    {"state": <text|json>,
     "questions": {<qid>: <question>, ...},
     "labels": {<qid>: <option key | true/false | level index>, ...}}

`examples()` flattens that into (state, Branch, target_index) triples where
target_index indexes `Branch.keys` — the same objects the engine reads out at serving
time, so training and serving share one code path.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from typing import Any

from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.schema import SystemOneRequest


def read_jsonl(path: str) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def examples(
    path: str, fmt: PromptFormat, permutations: int = 1, seed: int = 0
) -> list[tuple[Any, Branch, int]]:
    rng = random.Random(seed)
    out: list[tuple[Any, Branch, int]] = []
    for row in read_jsonl(path):
        req = SystemOneRequest(state=row["state"], questions=row["questions"])
        for qid, q in req.questions.items():
            if qid not in row["labels"]:
                continue
            label = row["labels"][qid]
            if q.type == "noul":
                label = bool(label)
            elif q.type == "score":
                label = int(label)
            for br in build_branches(qid, q, fmt, permutations, rng):
                out.append((req.state, br, br.keys.index(label)))
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
                    }
                )
                + "\n"
            )
    return len(items)
