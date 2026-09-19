"""Turn per-branch label probabilities into typed answers.

Also holds the two post-hoc knobs that make a pretrained LM's readout behave like
a calibrated decision model:
  * temperature scaling per primitive (fit on labelled data, see reflex.train.calibrate)
  * a pluggable `confidence` statistic (TypeSafe: "how peaked the distribution is")
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from reflex.prompt import Branch
from reflex.schema import ChoiceAnswer, NoulAnswer, ScoreAnswer, ScoreQuestion


@dataclass
class Calibration:
    """Temperature applied to the restricted label logits: per primitive, or, when a
    fitted `head` is present, per question (see reflex.calibration_head)."""

    temperature: dict[str, float] = field(
        default_factory=lambda: {"noul": 1.0, "choice": 1.0, "score": 1.0}
    )
    head: list[float] | None = None

    def t(self, kind: str, logits: np.ndarray | None = None, state_tokens: int = 0) -> float:
        if self.head is not None and logits is not None:
            from reflex.calibration_head import features, temperature

            return temperature(np.asarray(self.head), features(kind, logits, state_tokens))
        return float(self.temperature.get(kind, 1.0))

    @classmethod
    def load(cls, path: str | Path | None) -> Calibration:
        if path is None or not Path(path).exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        return cls(temperature=d["temperature"], head=d.get("head"))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature, "head": self.head}, f, indent=2)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / max(temperature, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def confidence(p: np.ndarray) -> float:
    """1 - normalized entropy. 1.0 = all mass on one option, 0.0 = uniform.

    This is the default statistic; callers keep the full `probabilities` and can
    compute their own (margin, max-prob, ...) — "you are never locked into our definition".
    """
    n = len(p)
    if n <= 1:
        return 1.0
    p = np.clip(p, 1e-12, 1.0)
    h = -(p * np.log(p)).sum()
    return float(max(0.0, min(1.0, 1.0 - h / math.log(n))))


def merge_branches(
    kind: str, results: list[tuple[Branch, np.ndarray]], cal: Calibration, state_tokens: int = 0
):
    """Average probability over permuted branches of the same question, keyed by option.

    `results` = [(branch, restricted_logits_in_branch_label_order), ...]
    Returns dict key -> prob (keys in original question order for the first branch).
    """
    acc: dict[Any, float] = {}
    for br, logits in results:
        probs = softmax(logits, cal.t(kind, logits, state_tokens))
        for k, pr in zip(br.keys, probs):
            acc[k] = acc.get(k, 0.0) + float(pr)
    n = len(results)
    total = sum(acc.values())
    return {k: v / n / (total / n) for k, v in acc.items()}  # renormalize


def to_answer(kind: str, key_probs: dict[Any, float], q=None):
    if kind == "noul":
        return NoulAnswer(noul=round(key_probs[True], 6))

    if kind == "choice":
        probs = {str(k): round(v, 6) for k, v in key_probs.items()}
        best = max(probs, key=probs.get)
        return ChoiceAnswer(
            choice=best,
            probabilities=probs,
            confidence=round(confidence(np.array(list(probs.values()))), 6),
        )

    assert kind == "score" and isinstance(q, ScoreQuestion)
    levels = sorted(key_probs.keys())
    p = np.array([key_probs[i] for i in levels])
    score = float((p * np.array(levels)).sum())
    from reflex.prompt import render_text

    return ScoreAnswer(
        score=round(score, 6),
        legend={str(i): render_text(q.criteria[i]) for i in levels},
        probabilities={str(i): round(float(key_probs[i]), 6) for i in levels},
        confidence=round(confidence(p), 6),
    )
