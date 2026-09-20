"""`reflex-distill`: corpus -> questions -> label -> mix. See the package docstring."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random

from reflex.distill.label import label_rows, read_jsonl, to_training_rows, write_jsonl

log = logging.getLogger("reflex.distill")


def cmd_corpus(a):
    from reflex.distill.corpus import DOMAINS, build

    domains = a.domains.split(",") if a.domains else list(DOMAINS)
    n = write_jsonl(build(a.per_domain, a.seed, domains), a.out)
    log.info("wrote %d states to %s", n, a.out)


def cmd_questions(a):
    from reflex.distill.questions import attach, gateway_chat

    rows = list(read_jsonl(a.inp))
    if a.subset < 1.0:  # a random slice, e.g. for a slower, better question writer
        rows = [r for r in rows if _split(r["id"], a.seed + 7) < a.subset]
    call = None
    if a.per_state > 0:
        key = os.environ.get(a.key_env, "")
        if not key:
            raise SystemExit(f"set {a.key_env} for the question-writing model")
        call = gateway_chat(a.gateway, key, a.model)
    out = list(attach(rows, call, a.per_state, a.workers, a.bank_max, a.seed, a.tag, not a.no_bank))
    # the anchor slice is labelled by the frozen student, on states the teacher never
    # labels, so it can only pull toward the base model, never toward the teacher
    teach = [r for r in out if _split(r["id"], a.seed + 1) >= a.anchor_frac]
    anchor = [r for r in out if _split(r["id"], a.seed + 1) < a.anchor_frac]
    write_jsonl(teach, a.out)
    if a.anchor_out:
        write_jsonl(anchor, a.anchor_out)
    nq = sum(len(r["questions"]) for r in out)
    log.info(
        "%d states, %d questions -> %d teacher states (%s), %d anchor states (%s)",
        len(out),
        nq,
        len(teach),
        a.out,
        len(anchor),
        a.anchor_out,
    )


def cmd_label(a):
    import torch

    from reflex.engine import Engine

    engine = Engine.load(
        a.model,
        dtype=getattr(torch, a.dtype),
        adapter_path=a.adapter,
        max_pack_tokens=a.max_pack_tokens,
        prompt_style=a.prompt_style,
        prompt_texts=a.prompt_texts,
        think_tokens=a.think,
    )
    rows = list(read_jsonl(a.inp))
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {r["id"] for r in read_jsonl(a.out)}
        rows = [r for r in rows if r["id"] not in done]
        log.info("resuming: %d already labelled, %d to go", len(done), len(rows))
    mode = "a" if done else "w"
    n = 0
    with open(a.out, mode) as f:
        for r in label_rows(engine, rows, a.permutations, a.log_every):
            import json

            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    log.info("labelled %d rows -> %s", n, a.out)


def _split(row_id: str, seed: int) -> float:
    return int(hashlib.sha1(f"{seed}:{row_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def cmd_mix(a):
    """Teacher rows + anchor rows -> train / eval files in the trainer's format.

    * `--teacher` rows carry the teacher's distributions (the thing to learn);
    * `--anchor` rows carry the frozen student's own distributions on *other* states (a
      pull toward the base model; weight them with --anchor-frac by subsampling);
    * a held-out `--eval-frac` of the teacher rows measures fidelity to the teacher.
    """
    rng = random.Random(a.seed)
    teach = list(to_training_rows(read_jsonl(a.teacher), "distill"))
    anchors = list(to_training_rows(read_jsonl(a.anchor), "anchor")) if a.anchor else []
    train, ev = [], []
    for r in teach:
        (ev if _split(r["id"], a.seed) < a.eval_frac else train).append(r)
    if anchors:
        k = int(len(train) * a.anchor_frac)
        rng.shuffle(anchors)
        train += anchors[:k]
    rng.shuffle(train)
    write_jsonl(train, a.out)
    write_jsonl(ev, a.eval_out)
    log.info(
        "train %d rows (%d teacher + %d anchor), eval %d rows",
        len(train),
        len(train)
        - min(
            len(anchors), int((len(train)) * a.anchor_frac / (1 + a.anchor_frac)) if anchors else 0
        ),
        0 if not anchors else min(len(anchors), k),
        len(ev),
    )


def cmd_merge(a):
    """Union of questions (and labels, if present) for rows with the same id, across files
    written by different question writers or labelling passes."""
    merged: dict[str, dict] = {}
    for path in a.inputs:
        for r in read_jsonl(path):
            m = merged.setdefault(r["id"], {**r, "questions": {}, "labels": {}, "confidence": {}})
            m["questions"].update(r.get("questions", {}))
            m["labels"].update(r.get("labels", {}))
            m["confidence"].update(r.get("confidence", {}))
    rows = [r for r in merged.values() if r["questions"]]
    for r in rows:
        if not r["labels"]:
            r.pop("labels")
            r.pop("confidence")
    n = write_jsonl(rows, a.out)
    log.info(
        "merged %d files -> %d rows, %d questions -> %s",
        len(a.inputs),
        n,
        sum(len(r["questions"]) for r in rows),
        a.out,
    )


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(prog="reflex-distill")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="raw states from public domains")
    c.add_argument("--per-domain", type=int, default=400)
    c.add_argument("--domains", default=None, help="comma list (default: all)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_corpus)

    q = sub.add_parser("questions", help="attach bank + LLM-written questions")
    q.add_argument("--in", dest="inp", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--gateway", default="http://127.0.0.1:8000", help="OpenAI-compatible base URL")
    q.add_argument("--key-env", default="SPARK_KEY")
    q.add_argument("--model", default="default")
    q.add_argument(
        "--per-state", type=int, default=3, help="LLM-written questions per state (0 = bank only)"
    )
    q.add_argument("--bank-max", type=int, default=3)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--subset", type=float, default=1.0, help="only this random share of the states")
    q.add_argument(
        "--tag", default="llm", help="prefix for LLM-written question ids (one per writer model)"
    )
    q.add_argument("--no-bank", action="store_true", help="LLM-written questions only")
    q.add_argument(
        "--anchor-frac",
        type=float,
        default=0.2,
        help="share of states reserved for the anchor slice",
    )
    q.add_argument(
        "--anchor-out", default=None, help="where the anchor slice goes (default: dropped)"
    )
    q.set_defaults(fn=cmd_questions)

    lab = sub.add_parser("label", help="a teacher answers through reflex")
    lab.add_argument("--model", required=True)
    lab.add_argument("--adapter", default=None)
    lab.add_argument("--dtype", default="bfloat16")
    lab.add_argument("--in", dest="inp", required=True)
    lab.add_argument("--out", required=True)
    lab.add_argument("--permutations", type=int, default=2)
    lab.add_argument("--max-pack-tokens", type=int, default=4096)
    lab.add_argument("--prompt-style", default="markdown")
    lab.add_argument("--prompt-texts", default=None)
    lab.add_argument(
        "--think", type=int, default=0, help="System Two teacher: reasoning tokens per branch"
    )
    lab.add_argument("--limit", type=int, default=0)
    lab.add_argument("--resume", action="store_true")
    lab.add_argument("--log-every", type=int, default=100)
    lab.set_defaults(fn=cmd_label)

    mg = sub.add_parser("merge", help="union questions/labels per state id across files")
    mg.add_argument("inputs", nargs="+")
    mg.add_argument("--out", required=True)
    mg.set_defaults(fn=cmd_merge)

    m = sub.add_parser("mix", help="teacher + anchor rows -> train/eval files")
    m.add_argument("--teacher", required=True)
    m.add_argument("--anchor", default=None)
    m.add_argument(
        "--anchor-frac",
        type=float,
        default=0.25,
        help="anchor rows as a fraction of teacher train rows",
    )
    m.add_argument("--eval-frac", type=float, default=0.06)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    m.add_argument("--eval-out", required=True)
    m.set_defaults(fn=cmd_mix)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
