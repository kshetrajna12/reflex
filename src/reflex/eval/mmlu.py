"""Replicates the calibration experiment from the Jev write-up on our engine:
run N MMLU items as Choice questions, read the distribution directly, measure ECE.

    reflex-eval-mmlu --model Qwen/Qwen3-8B --n 1200
    reflex-eval-mmlu --model Qwen/Qwen3-8B --n 1200 --fit-temperature runs/calibration.json

With --fit-temperature the items are split in half: T is fit on the first half and
the report is on the second half, before and after scaling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np

from reflex.eval.metrics import fit_temperature, report
from reflex.prompt import build_branches
from reflex.readout import Calibration, softmax
from reflex.schema import ChoiceQuestion

log = logging.getLogger("reflex.eval.mmlu")


def load_items(n: int, seed: int, split: str = "test"):
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        row = ds[i]
        items.append(
            {
                "subject": row["subject"].replace("_", " "),
                "question": row["question"],
                "choices": row["choices"],
                "answer": int(row["answer"]),
            }
        )
    return items


def to_question(item) -> tuple[ChoiceQuestion, list[str]]:
    """MMLU item -> Choice question. Returns (question, option keys in answer order)."""
    keys, seen = [], {}
    for c in item["choices"]:
        k = str(c).strip() or "(blank)"
        if k in seen:  # duplicate option text; keep both distinguishable
            seen[k] += 1
            k = f"{k} ({seen[k]})"
        else:
            seen[k] = 1
        keys.append(k)
    q = ChoiceQuestion(
        type="choice",
        instructions=f"This is a {item['subject']} question. Which option is the correct answer?",
        criteria={k: None for k in keys},
    )
    return q, keys


def run(engine, items, permutations: int = 1):
    """Returns restricted logits [N, 4] in answer-index order and label array [N]."""
    rng = random.Random(0)
    pairs, meta = [], []
    for it in items:
        q, keys = to_question(it)
        for br in build_branches("q", q, engine.fmt, permutations, rng):
            pairs.append((it["question"], br))
            meta.append((len(meta) // permutations, br, keys))
    t0 = time.perf_counter()
    rows = engine.label_logits_batch(pairs)
    dt = time.perf_counter() - t0

    N = len(items)
    logits = np.zeros((N, 4))
    for (item_i, br, keys), row in zip(meta, rows):
        # br.keys are option keys in prompt order; map back to answer index
        for k, v in zip(br.keys, row):
            logits[item_i, keys.index(k)] += v / permutations
    labels = np.array([it["answer"] for it in items])
    log.info("%d items (%d branches) in %.1fs = %.1f items/s", N, len(pairs), dt, N / dt)
    return logits, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--calibration", default=None, help="apply an existing calibration.json")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--max-pack-tokens", type=int, default=8192)
    ap.add_argument("--fit-temperature", default=None, help="write calibration.json here")
    ap.add_argument("--dump", default=None, help="write per-item logits/labels .npz here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from reflex.engine import Engine

    engine = Engine.load(
        args.model,
        calibration_path=args.calibration,
        adapter_path=args.adapter,
        max_pack_tokens=args.max_pack_tokens,
    )
    items = load_items(args.n, args.seed, args.split)
    logits, labels = run(engine, items, args.permutations)
    if args.dump:
        np.savez(args.dump, logits=logits, labels=labels)

    T0 = engine.cal.t("choice")
    probs = np.stack([softmax(l, T0) for l in logits])
    print(f"== {args.model} on MMLU {args.split} (T={T0:.3f}) ==")
    print(report(probs, labels))

    if args.fit_temperature:
        h = len(items) // 2
        T = fit_temperature(logits[:h], labels[:h])
        p_raw = np.stack([softmax(l, 1.0) for l in logits[h:]])
        p_cal = np.stack([softmax(l, T) for l in logits[h:]])
        print(f"\n== held-out half, raw (T=1) ==\n{report(p_raw, labels[h:])}")
        print(f"\n== held-out half, temperature-scaled (T={T:.3f}) ==\n{report(p_cal, labels[h:])}")
        cal = Calibration(temperature={"noul": T, "choice": T, "score": T})
        cal.save(args.fit_temperature)
        print(f"wrote {args.fit_temperature}: {json.dumps(cal.temperature)}")


if __name__ == "__main__":
    main()
