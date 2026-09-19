"""A small calibration head: the temperature depends on the question, not just its type.

A single temperature per primitive cannot express "confident on easy items, hedge on
hard ones". This head predicts log(T) from a handful of cheap features of each branch:

    kind one-hot (noul / choice / score), log(number of options), log(state tokens),
    normalised entropy of the raw restricted softmax, top-two logit margin

It is a linear model with eight weights, fitted on held-out logits by minimising the
proper scoring rule (cross-entropy against hard or soft targets), regularised towards
the per-primitive temperatures. It ships inside calibration.json and is applied by the
readout at serving time; with `head` absent the per-primitive temperatures are used.
"""

from __future__ import annotations

import math

import numpy as np

KINDS = ("noul", "choice", "score")
FEATURES = [
    "bias",
    "noul",
    "choice",
    "score",
    "log_n_options",
    "log_state_tokens",
    "entropy",
    "margin",
]


def features(kind: str, logits: np.ndarray, state_tokens: int) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    p = np.exp(z) / np.exp(z).sum()
    n = len(z)
    ent = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / math.log(max(n, 2)))
    top = np.sort(z)[::-1]
    margin = float(top[0] - top[1]) if n > 1 else 0.0
    return np.array(
        [
            1.0,
            kind == "noul",
            kind == "choice",
            kind == "score",
            math.log(n),
            math.log(max(state_tokens, 1)) / 10.0,
            ent,
            min(margin, 20.0) / 10.0,
        ],
        dtype=np.float64,
    )


def temperature(weights: np.ndarray, f: np.ndarray) -> float:
    return float(np.exp(np.clip(f @ weights, math.log(0.2), math.log(20.0))))


def fit(
    logits: list[np.ndarray],
    targets: list[np.ndarray],
    kinds: list[str],
    state_tokens: list[int],
    base_temps: dict[str, float],
    l2: float = 1e-2,
    steps: int = 300,
) -> np.ndarray:
    """Fit the head with L-BFGS in torch on CPU. Starts from the per-primitive temperatures
    (weights that reproduce them exactly), so it can only do better on the fitting set."""
    import torch

    F = torch.tensor(
        np.stack([features(k, z, s) for k, z, s in zip(kinds, logits, state_tokens, strict=True)])
    )
    w0 = np.zeros(len(FEATURES))
    for i, k in enumerate(KINDS):
        w0[1 + i] = math.log(base_temps.get(k, 1.0))
    w = torch.tensor(w0, requires_grad=True)
    Z = [torch.tensor(np.asarray(z, dtype=np.float64)) for z in logits]
    Q = [torch.tensor(np.asarray(q, dtype=np.float64)) for q in targets]
    w0t = torch.tensor(w0)

    def loss_fn():
        logT = torch.clamp(F @ w, math.log(0.2), math.log(20.0))
        total = 0.0
        for i, (z, q) in enumerate(zip(Z, Q, strict=True)):
            total = total - (q * torch.log_softmax(z / torch.exp(logT[i]), -1)).sum()
        return total / len(Z) + l2 * ((w - w0t) ** 2).sum()

    opt = torch.optim.LBFGS([w], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy()
