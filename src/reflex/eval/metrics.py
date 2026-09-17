"""Calibration metrics for decision models: accuracy, ECE, Brier, NLL, reliability table."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationReport:
    n: int
    accuracy: float
    ece: float
    brier: float
    nll: float
    mean_confidence: float
    bins: list[tuple[float, float, float, int]]  # (bin_lo, mean_conf, accuracy, count)

    def __str__(self) -> str:
        head = (
            f"n={self.n}  acc={self.accuracy:.4f}  ECE={self.ece:.4f}  Brier={self.brier:.4f}  "
            f"NLL={self.nll:.4f}  mean_conf={self.mean_confidence:.4f}"
        )
        lines = [head, "  conf-bin   mean_conf  accuracy  count"]
        for lo, mc, acc, cnt in self.bins:
            if cnt:
                lines.append(
                    f"  [{lo:.2f},{lo + 1 / len(self.bins):.2f})  {mc:8.3f}  {acc:8.3f}  {cnt:5d}"
                )
        return "\n".join(lines)


def report(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> CalibrationReport:
    """probs: [N, K] distributions, labels: [N] int index of the correct option."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    N = len(labels)
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = (pred == labels).astype(np.float64)
    onehot = np.zeros_like(probs)
    onehot[np.arange(N), labels] = 1
    brier = float(((probs - onehot) ** 2).sum(1).mean())
    nll = float(-np.log(np.clip(probs[np.arange(N), labels], 1e-12, 1)).mean())

    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1] if b < n_bins - 1 else conf <= 1.0)
        cnt = int(m.sum())
        if cnt:
            mc, acc = float(conf[m].mean()), float(correct[m].mean())
            ece += cnt / N * abs(acc - mc)
        else:
            mc, acc = 0.0, 0.0
        bins.append((float(edges[b]), mc, acc, cnt))
    return CalibrationReport(
        n=N,
        accuracy=float(correct.mean()),
        ece=float(ece),
        brier=brier,
        nll=nll,
        mean_confidence=float(conf.mean()),
        bins=bins,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """1-D temperature scaling by NLL minimisation (golden-section on log T)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)

    def nll(logT):
        z = logits / np.exp(logT)
        z = z - z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -logp[np.arange(len(labels)), labels].mean()

    lo, hi = np.log(0.05), np.log(20.0)
    gr = (np.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    for _ in range(60):
        if nll(c) < nll(d):
            hi = d
        else:
            lo = c
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    return float(np.exp((lo + hi) / 2))
