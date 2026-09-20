"""The cascade's escalation trigger: "is the fast path likely wrong about this question?"

The fast path (frozen 4B, two distinct option orders) is right about ~80 % of questions and
knows it: where its two readings scatter, it is usually unsure. But its *hard* misses are
often confident-and-wrong — a long policy read once, two dates compared badly, a fact that
lives in two places — and scatter alone does not see those (docs/results/order-averaging.md).

So the trigger looks at two kinds of evidence together:

  * **shape of the question, known before answering** — how long the state is, whether it
    carries dates, money, percentages or many numbers, whether it reads like a rule with
    deadlines, how many options there are, which primitive it is;
  * **what the fast path just produced** — its top probability, its normalised entropy, and
    its cross-order (and cross-wording) disagreement.

A logistic regression over those features, fitted on our own external sets and never on a
benchmark item, ranks questions by "probably wrong". Serving keeps a budget: escalate the
top ~12 % to the thinking readout (`reflex.think`) and answer the rest at ~200 ms.

    reflex-escalate features --val runs/external_eval.jsonl --out runs/esc_features.jsonl
    reflex-escalate fit --features runs/esc_features.jsonl --out runs/escalator.json
    reflex-serve --permutations 2 --think 768 --escalate runs/escalator.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("reflex.escalate")

# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2}\b|"
    r"\b\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|"
    r"\b\d{1,2}:\d{2}\b",
    re.IGNORECASE,
)
CURRENCY_RE = re.compile(
    r"[$€£¥]\s?\d|\b\d[\d,.]*\s?(?:usd|eur|gbp|dollars?|euros?|cents?)\b", re.IGNORECASE
)
PERCENT_RE = re.compile(r"\d\s?%|\bpercent(?:age)?\b|\bbasis points?\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[.!?\n]+")

# Words that mark the reasoning shapes the fast path misses: comparisons against a
# deadline, arithmetic over a total, per-unit rates, ordering in time.
CUE_WORDS = (
    "before",
    "after",
    "until",
    "since",
    "deadline",
    "expire",
    "expires",
    "expired",
    "days",
    "hours",
    "minutes",
    "weeks",
    "months",
    "total",
    "net",
    "gross",
    "per",
    "within",
    "prior",
    "remaining",
    "elapsed",
    "exceed",
    "exceeds",
    "least",
    "most",
    "unless",
    "except",
    "eligible",
    "threshold",
    "both",
    "neither",
    "only if",
)
CUE_RE = re.compile(
    r"\b(?:" + "|".join(w.replace(" ", r"\s") for w in CUE_WORDS) + r")\b", re.IGNORECASE
)

# The order is the feature vector's order; it is stored in the saved JSON so a model
# fitted today still lines up with a feature extractor changed tomorrow.
FEATURE_NAMES: list[str] = [
    "is_noul",
    "is_choice",
    "is_score",
    "n_options",
    "log_state_tokens",
    "log_state_chars",
    "log_instruction_chars",
    "log_state_sentences",
    "log_state_fields",
    "state_has_date",
    "state_has_currency",
    "state_has_percent",
    "log_state_numbers",
    "log_state_cues",
    "question_has_date",
    "question_has_currency",
    "question_has_percent",
    "log_question_numbers",
    "log_question_cues",
    "max_prob",
    "entropy",
    "margin",
    "disagreement",
]


def _text_of(x: Any) -> str:
    """Flatten a state / instruction (str, dict or list) to the text the model would see."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)


def _n_fields(state: Any) -> int:
    """How many leaf fields a JSON-ish state has; 0 for plain text. Many fields means the
    answer probably has to be assembled from several of them."""
    if isinstance(state, dict):
        return sum(_n_fields(v) or 1 for v in state.values())
    if isinstance(state, list):
        return sum(_n_fields(v) or 1 for v in state)
    return 0


def _log1p(x: float) -> float:
    return float(math.log1p(max(0.0, x)))


def text_features(text: str, prefix: str) -> dict[str, float]:
    """Surface cues for reasoning-shaped items, for either the state or the question."""
    return {
        f"{prefix}_has_date": float(bool(DATE_RE.search(text))),
        f"{prefix}_has_currency": float(bool(CURRENCY_RE.search(text))),
        f"{prefix}_has_percent": float(bool(PERCENT_RE.search(text))),
        f"log_{prefix}_numbers": _log1p(len(NUMBER_RE.findall(text))),
        f"log_{prefix}_cues": _log1p(len(CUE_RE.findall(text))),
    }


def question_features(
    state: Any,
    kind: str,
    instructions: Any,
    n_options: int,
    state_tokens: int,
    probs: np.ndarray | None = None,
    disagreement: float = 0.0,
) -> dict[str, float]:
    """All features for one question. `probs` / `disagreement` come from the fast path; pass
    them as None/0 to get only the features available before answering."""
    s_text = _text_of(state)
    q_text = _text_of(instructions)
    feats: dict[str, float] = {
        "is_noul": float(kind == "noul"),
        "is_choice": float(kind == "choice"),
        "is_score": float(kind == "score"),
        "n_options": float(n_options),
        "log_state_tokens": _log1p(state_tokens),
        "log_state_chars": _log1p(len(s_text)),
        "log_instruction_chars": _log1p(len(q_text)),
        "log_state_sentences": _log1p(sum(1 for s in SENTENCE_RE.split(s_text) if s.strip())),
        "log_state_fields": _log1p(_n_fields(state)),
    }
    feats.update(text_features(s_text, "state"))
    feats.update(text_features(q_text, "question"))
    if probs is None or len(probs) == 0:
        feats.update({"max_prob": 0.0, "entropy": 0.0, "margin": 0.0})
    else:
        p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
        p = p / p.sum()
        srt = np.sort(p)[::-1]
        h = -(p * np.log(p)).sum() / math.log(len(p)) if len(p) > 1 else 0.0
        feats["max_prob"] = float(srt[0])
        feats["entropy"] = float(h)
        feats["margin"] = float(srt[0] - (srt[1] if len(srt) > 1 else 0.0))
    feats["disagreement"] = float(disagreement)
    return feats


def vector(feats: dict[str, float], names: list[str] | None = None) -> np.ndarray:
    names = names or FEATURE_NAMES
    return np.array([float(feats.get(n, 0.0)) for n in names], dtype=np.float64)


# --------------------------------------------------------------------------------------
# The classifier
# --------------------------------------------------------------------------------------


def fit_logistic(
    X: np.ndarray, y: np.ndarray, *, l2: float = 1e-2, steps: int = 4000, lr: float = 0.2
) -> tuple[np.ndarray, float]:
    """Logistic regression by full-batch gradient descent with momentum.

    Dependency-free on purpose: the trigger is a ~24-weight model that ships as JSON next to
    the adapter, and a scikit-learn import in the serving path would not earn its size.
    Features are expected standardised; class weights balance the (rarer) error class so the
    ranking is not dominated by the majority.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    vw, vb = np.zeros(d), 0.0
    pos = max(1, int(y.sum()))
    sw = np.where(y > 0.5, n / (2 * pos), n / (2 * max(1, n - pos)))
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = sw * (p - y)
        gw = X.T @ g / n + l2 * w
        gb = g.sum() / n
        vw = 0.9 * vw + gw
        vb = 0.9 * vb + gb
        w -= lr * vw
        b -= lr * vb
    return w, float(b)


@dataclass
class Escalator:
    """A fitted "the fast path is probably wrong here" score, with its budget threshold."""

    names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    mean: list[float] = field(default_factory=list)
    std: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    threshold: float = 0.5
    budget: float = 0.0  # fraction of questions the threshold escalated when it was chosen
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, feats: dict[str, float]) -> float:
        x = vector(feats, self.names)
        z = (x - np.asarray(self.mean)) / np.asarray(self.std)
        return float(1.0 / (1.0 + math.exp(-(float(z @ np.asarray(self.weights)) + self.bias))))

    def should_escalate(self, feats: dict[str, float]) -> bool:
        return self.score(feats) >= self.threshold

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "names": self.names,
                    "mean": self.mean,
                    "std": self.std,
                    "weights": self.weights,
                    "bias": self.bias,
                    "threshold": self.threshold,
                    "budget": self.budget,
                    "meta": self.meta,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> Escalator:
        with open(path) as f:
            d = json.load(f)
        return cls(
            names=list(d["names"]),
            mean=list(d["mean"]),
            std=list(d["std"]),
            weights=list(d["weights"]),
            bias=float(d["bias"]),
            threshold=float(d.get("threshold", 0.5)),
            budget=float(d.get("budget", 0.0)),
            meta=dict(d.get("meta", {})),
        )


def train_escalator(
    X: np.ndarray, y: np.ndarray, *, budget: float = 0.12, names: list[str] | None = None, **kw
) -> Escalator:
    """Standardise, fit, and pick the threshold that escalates `budget` of the fitting set."""
    names = names or FEATURE_NAMES
    mean = X.mean(0)
    std = X.std(0)
    std[std < 1e-9] = 1.0
    w, b = fit_logistic((X - mean) / std, y, **kw)
    scores = 1.0 / (1.0 + np.exp(-(((X - mean) / std) @ w + b)))
    thr = float(np.quantile(scores, 1.0 - budget)) if len(scores) else 0.5
    return Escalator(
        names=list(names),
        mean=[float(v) for v in mean],
        std=[float(v) for v in std],
        weights=[float(v) for v in w],
        bias=b,
        threshold=thr,
        budget=budget,
    )


# --------------------------------------------------------------------------------------
# Budget report
# --------------------------------------------------------------------------------------

BUDGETS = (0.05, 0.10, 0.12, 0.15, 0.25)


def budget_report(scores: np.ndarray, wrong: np.ndarray, budgets=BUDGETS) -> list[dict[str, float]]:
    """At each budget, how many of the fast path's errors the top-scoring slice contains.

    Recall of errors is the number that matters: every error the trigger does not select is
    one the slow path never gets to fix. Precision says how much reasoning time is spent on
    questions that were already right.
    """
    out = []
    n = len(scores)
    n_err = max(1, int(wrong.sum()))
    order = np.argsort(-scores)
    for b in budgets:
        k = max(1, round(b * n))
        sel = order[:k]
        caught = int(wrong[sel].sum())
        out.append(
            {
                "budget": b,
                "n": k,
                "recall": caught / n_err,
                "precision": caught / k,
                "lift": (caught / k) / (wrong.mean() if wrong.mean() > 0 else 1.0),
            }
        )
    return out


def print_budget_table(title: str, rows: list[dict[str, float]]) -> None:
    print(f"\n== {title} ==")
    print("  budget    n   error recall   precision   lift")
    for r in rows:
        print(
            f"  {r['budget']:>5.0%} {r['n']:>5d}   {r['recall']:>10.3f}   {r['precision']:>9.3f}"
            f"   {r['lift']:>4.2f}x"
        )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def dump_features(args) -> None:
    """Run the fast path over a labelled set and write one JSON row of features per question."""
    from reflex.engine import Engine
    from reflex.train.calibrate import evaluate_ensemble
    from reflex.train.data import read_jsonl

    engine = Engine.load(
        args.model,
        adapter_path=args.adapter,
        calibration_path=args.calibration,
        max_pack_tokens=args.max_pack_tokens,
        prompt_style=args.prompt_style,
        prompt_texts=args.prompt_texts,
        ensemble=args.ensemble,
    )
    rows = list(read_jsonl(args.val))
    exs, probs, dis, meta = evaluate_ensemble(engine, rows, args.permutations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_wrong = 0
    with open(out, "w") as f:
        for i, (ex, m) in enumerate(zip(exs, meta, strict=True)):
            p = probs[i, : len(ex.branch.keys)]
            wrong = int(int(p.argmax()) != ex.hard_label)
            n_wrong += wrong
            feats = question_features(
                m["state"],
                m["kind"],
                m["instructions"],
                len(ex.branch.keys),
                m["state_tokens"],
                probs=p,
                disagreement=float(dis[i]),
            )
            f.write(
                json.dumps(
                    {
                        "source": ex.source,
                        "wrong": wrong,
                        "probs": [float(v) for v in p],
                        "disagreement": float(dis[i]),
                        "features": feats,
                    }
                )
                + "\n"
            )
    print(f"wrote {len(exs)} rows to {out}: {n_wrong} fast-path errors ({n_wrong / len(exs):.1%})")


def load_features(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (X, wrong, source) over one or more feature dumps."""
    X, y, src = [], [], []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                X.append(vector(r["features"]))
                y.append(float(r["wrong"]))
                src.append(r.get("source", "?"))
    return np.array(X), np.array(y), np.array(src)


def _disagreement_baseline(X: np.ndarray) -> np.ndarray:
    return X[:, FEATURE_NAMES.index("disagreement")]


def fit_cmd(args) -> None:
    X, y, src = load_features(args.features)
    print(f"{len(y)} questions, {int(y.sum())} fast-path errors ({y.mean():.1%})")

    def evaluate_split(name: str, tr: np.ndarray, te: np.ndarray) -> None:
        esc = train_escalator(X[tr], y[tr], budget=args.budget, l2=args.l2)
        z = (X[te] - np.asarray(esc.mean)) / np.asarray(esc.std)
        scores = 1.0 / (1.0 + np.exp(-(z @ np.asarray(esc.weights) + esc.bias)))
        print_budget_table(
            f"{name}: trigger (n={len(te)}, errors={int(y[te].sum())})",
            budget_report(scores, y[te]),
        )
        print_budget_table(
            f"{name}: disagreement only", budget_report(_disagreement_baseline(X[te]), y[te])
        )

    # leave-one-set-out: the honest question is whether the trigger transfers to a task
    # family it never saw, which is what serving asks of it.
    for s in sorted(set(src.tolist())):
        te = np.where(src == s)[0]
        tr = np.where(src != s)[0]
        if len(te) < 20 or len(tr) < 50:
            continue
        evaluate_split(f"leave-out {s}", tr, te)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(y))
    cut = int(0.7 * len(y))
    evaluate_split("random 70/30 split", perm[:cut], perm[cut:])

    esc = train_escalator(X, y, budget=args.budget, l2=args.l2)
    esc.meta = {
        "n_train": len(y),
        "error_rate": float(y.mean()),
        "sources": sorted(set(src.tolist())),
        "features": args.features,
    }
    esc.save(args.out)
    all_scores = np.array([esc.score(dict(zip(FEATURE_NAMES, row, strict=True))) for row in X])
    print_budget_table("fitted on everything (in-sample)", budget_report(all_scores, y))
    print(f"\nwrote {args.out}: threshold {esc.threshold:.4f} for a {args.budget:.0%} budget")
    order = np.argsort(-np.abs(np.asarray(esc.weights)))
    print("top weights: " + ", ".join(f"{esc.names[i]} {esc.weights[i]:+.2f}" for i in order[:8]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="the cascade's escalation trigger")
    sub = ap.add_subparsers(dest="cmd", required=True)

    fe = sub.add_parser("features", help="run the fast path on a labelled set, dump features")
    fe.add_argument("--val", required=True, help="labelled JSONL (reflex.train.data format)")
    fe.add_argument("--out", required=True)
    fe.add_argument("--model", default="Qwen/Qwen3.5-4B")
    fe.add_argument("--adapter", default=None)
    fe.add_argument("--calibration", default=None)
    fe.add_argument("--ensemble", default=None)
    fe.add_argument("--permutations", type=int, default=2)
    fe.add_argument("--prompt-style", default="markdown", choices=["markdown", "compact"])
    fe.add_argument("--prompt-texts", default=None)
    fe.add_argument("--max-pack-tokens", type=int, default=4096)

    ft = sub.add_parser("fit", help="fit the trigger and report budget/recall")
    ft.add_argument("--features", nargs="+", required=True)
    ft.add_argument("--out", required=True)
    ft.add_argument("--budget", type=float, default=0.12)
    ft.add_argument("--l2", type=float, default=1e-2)
    ft.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "features":
        dump_features(args)
    else:
        fit_cmd(args)


if __name__ == "__main__":
    main()
