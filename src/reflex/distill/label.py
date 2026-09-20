"""A teacher answers the questions through reflex; its distributions become soft labels.

The teacher is any model reflex can load (the point is a *much* larger one than the
student). It sees exactly the prompt the student will be trained and served with, so the
targets are "what a stronger model reads off the same evidence", not a paraphrase.
`--permutations 2` averages over option orders to strip position bias from the targets.
"""

from __future__ import annotations

import json
import logging
import time

from reflex.schema import SystemOneRequest

log = logging.getLogger("reflex.distill")


def label_rows(engine, rows, permutations: int = 2, log_every: int = 100):
    t0 = time.time()
    for i, row in enumerate(rows):
        req = SystemOneRequest(
            state=row["state"], questions=row["questions"], permutations=permutations
        )
        try:
            resp = engine.answer(req)
        except Exception as e:  # noqa: BLE001 - one bad row must not kill a long run
            log.warning("row %s failed: %s", row.get("id"), str(e)[:200])
            continue
        labels, conf = {}, {}
        for qid, a in resp.answers.items():
            if a.type == "noul":
                labels[qid] = round(a.noul, 4)
            else:
                labels[qid] = {k: round(v, 4) for k, v in a.probabilities.items()}
                conf[qid] = round(a.confidence, 4)
        out = {
            **row,
            "labels": labels,
            "teacher": engine.model_name,
            "confidence": conf,
            "state_tokens": resp.usage.state_tokens,
        }
        yield out
        if (i + 1) % log_every == 0:
            log.info("labelled %d rows (%.1f s/row)", i + 1, (time.time() - t0) / (i + 1))


def to_training_rows(rows, source_prefix: str = "distill"):
    """Strip distillation metadata down to the trainer's row format."""
    for r in rows:
        yield {
            "state": r["state"],
            "questions": r["questions"],
            "labels": r["labels"],
            "source": f"{source_prefix}/{r['domain']}",
            "id": r["id"],
        }


def write_jsonl(rows, path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
