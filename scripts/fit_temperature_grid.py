"""Per-primitive temperature refit from saved label logits, cross-validated two ways.

    uv run python scripts/collect_logits.py --model <checkpoint> --out ext.npz
    uv run python scripts/fit_temperature_grid.py --nvfp4 ext.npz \
        --bf16 runs/teacher_27b_external.npz --out calibration/<name>.json

Prints, for the four external sets: the uncalibrated report; leave-one-set-out CV (a
transfer test - each set is a different distribution); five-fold CV inside each set (a
sample-size test - same distribution, unseen items); the fit on everything; and, with
--bf16, what temperatures fitted on another checkpoint's logits do to these ones.
`--out` writes the all-sets fit as a calibration file. See docs/results/nvfp4-27b.md."""

import argparse
import json

import numpy as np

from reflex.eval.metrics import fit_temperature, report

ap = argparse.ArgumentParser()
ap.add_argument("--nvfp4", required=True)
ap.add_argument("--bf16", default=None, help="npz with bf16 logits (fit_calibration dump format)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

d = np.load(a.nvfp4, allow_pickle=True)
logits, order, targets = d["logits"], d["order"], d["targets"]
kinds, sources, rows = d["kinds"], d["sources"], d["rows"]
K = logits.shape[1]
labels = targets.argmax(1)


def branch_targets():
    """Target distribution per branch, in that branch's option order."""
    t = np.zeros_like(logits)
    for gi in range(len(rows)):
        for b in rows[gi]:
            o = order[b]
            m = o >= 0
            t[b, : m.sum()] = targets[gi, o[m]]
    return t


BT = branch_targets()


def merged_probs(temps):
    """Question-level probabilities in canonical option order, averaged over permutations."""
    z = np.empty_like(logits)
    for kind in ("noul", "choice", "score"):
        sel = np.isin(np.arange(len(logits)), rows[kinds == kind].ravel())
        z[sel] = logits[sel] / max(temps[kind], 1e-6)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(1, keepdims=True)
    out = np.zeros_like(targets)
    for gi in range(len(rows)):
        acc = np.zeros(K)
        for b in rows[gi]:
            o = order[b]
            m = o >= 0
            acc[o[m]] += p[b, : m.sum()]
        out[gi] = acc / len(rows[gi])
    return out / out.sum(1, keepdims=True)


def fit(mask_groups):
    """Per-primitive temperatures on the branches of the selected questions."""
    temps = {}
    for kind in ("noul", "choice", "score"):
        sel = mask_groups & (kinds == kind)
        idx = rows[sel].ravel()
        temps[kind] = float(fit_temperature(logits[idx], BT[idx])) if sel.sum() >= 20 else 1.0
    return temps


def line(p, m):
    r = report(p[m], labels[m])
    return f"acc={r.accuracy:.4f} ECE={r.ece:.4f} Brier={r.brier:.4f} NLL={r.nll:.4f} conf={r.mean_confidence:.4f}"


srcs = sorted(set(sources.tolist()))
ones = {"noul": 1.0, "choice": 1.0, "score": 1.0}
p0 = merged_probs(ones)
print("== NVFP4, uncalibrated ==")
for s in srcs + ["POOLED"]:
    m = np.ones(len(labels), bool) if s == "POOLED" else sources == s
    print(f"  {s:<22} n={m.sum():<5} {line(p0, m)}")

print("\n== leave-one-set-out CV ==")
cv = {}
for s in srcs:
    held = sources == s
    t = fit(~held)
    p = merged_probs(t)
    cv[s] = t
    print(f"  {s:<22} T={ {k: round(v, 3) for k, v in t.items()} }")
    print(f"    before  {line(p0, held)}")
    print(f"    after   {line(p, held)}")

print("\n== 5-fold CV inside each set (same distribution, unseen items) ==")
rng = np.random.default_rng(0)
for s_ in srcs:
    idx = np.where(sources == s_)[0]
    fold = rng.permutation(len(idx)) % 5
    p_cv = p0.copy()
    for f in range(5):
        train = np.zeros(len(labels), bool)
        train[idx[fold != f]] = True
        t = fit(train)
        pf_ = merged_probs(t)
        p_cv[idx[fold == f]] = pf_[idx[fold == f]]
    m = sources == s_
    print(f"  {s_:<22} before {line(p0, m)}")
    print(f"  {'':<22} after  {line(p_cv, m)}")

print("\n== fit on all four sets ==")
full = fit(np.ones(len(labels), bool))
pf = merged_probs(full)
print(f"  T={ {k: round(v, 4) for k, v in full.items()} }")
for s in srcs + ["POOLED"]:
    m = np.ones(len(labels), bool) if s == "POOLED" else sources == s
    print(f"  {s:<22} before {line(p0, m)}")
    print(f"  {'':<22} after  {line(pf, m)}")

if a.bf16:
    b = np.load(a.bf16, allow_pickle=True)
    blg, blb, bkd = b["logits"], b["labels"], b["kinds"]
    bt = np.zeros_like(blg)
    bt[np.arange(len(blb)), blb] = 1.0
    bt[blg <= -1e8] = 0.0
    tb = {}
    for kind in ("noul", "choice", "score"):
        sel = bkd == kind
        tb[kind] = float(fit_temperature(blg[sel], bt[sel])) if sel.sum() >= 20 else 1.0
    print("\n== temperatures fitted on bf16, applied to NVFP4 (non-transfer) ==")
    print(f"  bf16 T={ {k: round(v, 4) for k, v in tb.items()} }")

    def sm(rowz):
        z = rowz - rowz.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    scaled = np.stack([r / max(tb[k], 1e-6) for r, k in zip(blg, bkd)])
    r0, r1 = report(sm(blg), blb), report(sm(scaled), blb)
    print(
        f"  bf16 on its own       acc={r0.accuracy:.4f} ECE={r0.ece:.4f} -> "
        f"acc={r1.accuracy:.4f} ECE={r1.ece:.4f}"
    )
    pt = merged_probs(tb)
    for s in srcs + ["POOLED"]:
        m = np.ones(len(labels), bool) if s == "POOLED" else sources == s
        print(f"  {s:<22} NVFP4 with bf16 T  {line(pt, m)}")

if a.out:
    from reflex.readout import Calibration

    Calibration(temperature={k: round(v, 4) for k, v in full.items()}).save(a.out)
    print("\nwrote", a.out, json.dumps(full))
