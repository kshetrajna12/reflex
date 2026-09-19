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
    codereview        diff chunk -> needs a reviewer? + issue type  noul/choice  "other"
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


def write(rows: Iterable[dict], path: str) -> int:
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description="build reflex training data from public datasets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix", help="sample from several recipes into one JSONL")
    m.add_argument("--sources", default=",".join(RECIPES), help="comma-separated recipe names")
    m.add_argument("--per-source", type=int, default=800)
    m.add_argument("--eval-per-source", type=int, default=200)
    m.add_argument("--out", required=True)
    m.add_argument("--eval-out", default=None)
    m.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    for s in sources:
        if s not in RECIPES:
            ap.error(f"unknown source {s!r}; choose from {', '.join(RECIPES)}")

    def build(split: str, per: int):
        rows = []
        for s in sources:
            got = list(RECIPES[s](per, args.seed, split))
            print(f"  {s:<16} {split:<6} {len(got):>6} examples")
            rows.extend(got)
        random.Random(args.seed).shuffle(rows)
        return rows

    print(f"train (up to {args.per_source} per source):")
    n = write(build("train", args.per_source), args.out)
    print(f"wrote {n} rows to {args.out}")
    if args.eval_out:
        print(f"eval (up to {args.eval_per_source} per source, held-out splits):")
        n = write(build("test", args.eval_per_source), args.eval_out)
        print(f"wrote {n} rows to {args.eval_out}")


if __name__ == "__main__":
    main()
