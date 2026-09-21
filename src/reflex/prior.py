"""A position prior over option letters, and the readout step that divides it out.

A pretrained LM does not read a lettered option list neutrally: it likes some positions
more than others, whatever is written there. That is why asking the same question in two
option orders and averaging helps so much (docs/results/order-averaging.md) — the second
order cancels most of the bias. It also doubles the branch count.

This module buys the same cancellation for one order. Estimate the model's prior over
letter positions once, offline, then divide it out of every single-order reading:

    p_debiased(option at position i)  ∝  p_observed(i) / (n · prior(i)) ** strength

The prior is fitted per model, per primitive (`noul` / `choice` / `score`) and per option
count, because a 2-way yes/no pair and a 12-way intent list are biased differently. Two
estimators, both label-free:

  * ``permuted`` (PriDe, Zheng et al. 2023, "Large language models are not robust multiple
    choice selectors"): ask a few hundred questions in several orders. The order-averaged
    distribution over *options* is taken as unbiased; what each single order adds on top
    of it, per *position*, is the prior. Costs one multi-order pass over the fit corpus.
  * ``content-free`` (Zhao et al. 2021, "Calibrate before use"): ask the same questions
    with the state replaced by a content-free string ("N/A"). Whatever mass the model
    still puts on a letter is prior, not evidence. One single-order pass, no permutations.

Both write the same artefact: a probability vector per (primitive, option count) cell.
A uniform vector is an exact no-op, so a prior fitted on a model that has no position
bias costs nothing.

    reflex-calibrate fit-prior --data runs/distill_questions.jsonl --n 400 \
        --method permuted --permutations 4 --out prior/qwen3.5-4b.json
    reflex-serve --model Qwen/Qwen3.5-4B --permutations 1 --prior prior/qwen3.5-4b.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# The state a `content-free` fit shows the model instead of real evidence.
CONTENT_FREE_STATE = "N/A"

MODES = ("div", "sub")


def cell(kind: str, n_options: int) -> str:
    """Table key: one prior per primitive and option count."""
    return f"{kind}:{n_options}"


@dataclass
class PositionPrior:
    """What the model would answer if the options carried no information.

    `table` maps "<kind>:<n>" to an n-vector summing to 1. `mode` picks how it is removed:
    "div" divides in probability space (equivalently subtracts `strength · log prior` from
    the logits, the Zhao et al. form and the one we ship), "sub" subtracts the excess over
    uniform (the PriDe form). `strength` scales the correction; 0 is off, 1 is full.
    """

    table: dict[str, list[float]] = field(default_factory=dict)
    mode: str = "div"
    strength: float = 1.0
    model: str = ""
    method: str = ""
    n_fit: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown debias mode {self.mode!r}, expected one of {MODES}")

    # ------------------------------------------------------------------------ lookup
    def row(self, kind: str, n: int) -> np.ndarray | None:
        """The prior for this cell, normalised; None when nothing applies."""
        if n < 2:
            return None
        raw = self.table.get(cell(kind, n)) or self._backoff(kind, n)
        if raw is None:
            return None
        p = np.clip(np.asarray(raw, dtype=np.float64), 1e-9, None)
        return p / p.sum()

    def _backoff(self, kind: str, n: int) -> list[float] | None:
        """An unfitted option count borrows the same primitive's nearest fitted count:
        a longer row is truncated to n, a shorter one extended with its last value. Letter
        bias is mostly "the front of the list", which carries across lengths; the
        renormalisation in `row` puts the borrowed vector back on the simplex."""
        counts = [int(k.split(":")[1]) for k in self.table if k.startswith(f"{kind}:") and ":" in k]
        if not counts:
            return None
        m = min(counts, key=lambda c: (abs(c - n), -c))
        src = list(self.table[cell(kind, m)])
        return src[:n] if m >= n else src + [src[-1]] * (n - m)

    # ------------------------------------------------------------------------ apply
    def debias(self, kind: str, probs: np.ndarray) -> np.ndarray:
        """Remove the position prior from one branch's distribution, in position order."""
        p = np.asarray(probs, dtype=np.float64)
        pri = self.row(kind, len(p))
        if pri is None or self.strength == 0.0:
            return p
        n = len(p)
        if self.mode == "div":
            q = p / np.power(pri * n, self.strength)
        else:
            q = p - self.strength * (pri - 1.0 / n)
        q = np.clip(q, 1e-12, None)
        total = q.sum()
        return q / total if total > 0 else p

    # ------------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return {
            "table": {k: [float(x) for x in v] for k, v in sorted(self.table.items())},
            "mode": self.mode,
            "strength": float(self.strength),
            "model": self.model,
            "method": self.method,
            "n_fit": int(self.n_fit),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PositionPrior:
        return cls(
            table={str(k): [float(x) for x in v] for k, v in (d.get("table") or {}).items()},
            mode=str(d.get("mode", "div")),
            strength=float(d.get("strength", 1.0)),
            model=str(d.get("model", "")),
            method=str(d.get("method", "")),
            n_fit=int(d.get("n_fit", 0)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str | Path | None) -> PositionPrior | None:
        """Read a prior file. `None` (or a missing path) means "serve without one"."""
        if path is None:
            return None
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no prior file at {p}")
        with open(p) as f:
            return cls.from_dict(json.load(f))

    def describe(self) -> str:
        lines = [f"position prior ({self.method or '?'}, mode={self.mode}, n_fit={self.n_fit})"]
        for k, v in sorted(self.table.items(), key=lambda kv: (kv[0].split(":")[0], len(kv[1]))):
            lines.append(f"  {k:<12} " + " ".join(f"{x:.3f}" for x in v))
        return "\n".join(lines)


# ---------------------------------------------------------------------------- fitting


class PriorFitter:
    """Accumulate position statistics per (primitive, option count), then close the table.

    Feed it either `add_permuted` (several orders of one question) or `add_content_free`
    (one reading of a content-free state). Both land in the same accumulator, so a fit
    never mixes methods by accident — the caller runs one or the other.
    """

    def __init__(self) -> None:
        self._sum: dict[str, np.ndarray] = {}
        self._count: dict[str, int] = defaultdict(int)
        self.questions = 0

    def _add(self, kind: str, vec: np.ndarray) -> None:
        k = cell(kind, len(vec))
        if k not in self._sum:
            self._sum[k] = np.zeros(len(vec), dtype=np.float64)
        self._sum[k] += vec
        self._count[k] += 1

    def add_content_free(self, kind: str, probs: np.ndarray) -> None:
        """`probs` in position order, read with a content-free state. It *is* the prior."""
        p = np.asarray(probs, dtype=np.float64)
        self._add(kind, p / p.sum())
        self.questions += 1

    def add_permuted(self, kind: str, branches: list[tuple[list[Any], np.ndarray]]) -> None:
        """One question read in several orders: (keys in position order, probs) per branch.

        The mean over orders, taken per option, is the unbiased reading; the excess a
        single order puts on each *position* over that reading is this question's
        contribution to the prior.
        """
        if len(branches) < 2:
            return
        n = len(branches[0][0])
        keys = list(branches[0][0])
        index = {k: i for i, k in enumerate(keys)}
        by_option = np.zeros(n, dtype=np.float64)
        rows = []
        for br_keys, probs in branches:
            p = np.asarray(probs, dtype=np.float64)
            p = p / p.sum()
            rows.append((br_keys, p))
            for k, v in zip(br_keys, p, strict=True):
                by_option[index[k]] += v
        by_option /= len(rows)
        excess = np.zeros(n, dtype=np.float64)
        for br_keys, p in rows:
            excess += p - by_option[[index[k] for k in br_keys]]
        self._add(kind, 1.0 / n + excess / len(rows))
        self.questions += 1

    def finish(self, *, min_questions: int = 20, **meta: Any) -> PositionPrior:
        """Close the accumulator. Cells with too few questions are dropped, not smoothed:
        a noisy prior is worse than backing off to a neighbouring option count."""
        table = {}
        for k, s in self._sum.items():
            if self._count[k] < min_questions:
                continue
            v = np.clip(s / self._count[k], 1e-6, None)
            table[k] = [float(x) for x in v / v.sum()]
        return PositionPrior(table=table, n_fit=self.questions, **meta)
