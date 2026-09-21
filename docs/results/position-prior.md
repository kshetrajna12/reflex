# Position-prior debiasing: one pass cannot buy what two orders buy

**Verdict: rejected.** Measuring the 4B's preference over option letters and dividing it
out of a single reading does not reproduce the two-order result. On the external sets it
costs accuracy (0.808 → 0.803) and on the public items it is worse than the single-order
baseline it was meant to improve (hard 0.658 → 0.649, hard ECE 0.086 → 0.124). The
`stable` configuration is unchanged. The code, the flag and the fitted prior are kept so
the measurement can be redone on a model that actually has the bias this assumes.

## The idea

Two distinct option orders per question is the current `stable` readout
([order-averaging.md](order-averaging.md)). The reason it works is supposed to be letter
bias: whatever the model likes about the first slot in a list, it likes about a *different*
option the second time round, so averaging keeps only what the options say. The cost is
that every question runs twice.

If that preference is a property of the model rather than of the question, it can be
measured once, offline, and divided out of a single reading. That is PriDe (Zheng et al.
2023, "Large language models are not robust multiple choice selectors") and, in its
content-free form, "Calibrate before use" (Zhao et al. 2021).

## Method

`reflex.prior` fits one probability vector per (primitive, option count) cell — `noul:2`,
`choice:3`, `score:5` and so on — for a model and prompt. Cells are separate because a
two-way yes/no pair and a twelve-way intent list are not biased alike. An option count
that was never fitted borrows the same primitive's nearest fitted count, truncated or
extended and renormalised.

Two label-free estimators:

* **permuted** (PriDe). Ask a few hundred questions in four orders. The mean over orders,
  taken per *option*, is treated as the unbiased reading; the excess a single order puts
  on each *position* over that reading, averaged over questions, is the prior.
* **content-free** (Zhao et al.). Ask the same questions with the state replaced by `N/A`.
  Whatever mass the model still puts on a letter cannot be evidence.

Two ways to remove it from a reading, both renormalised afterwards:

    div:  p'(i) ∝ p(i) / (n · prior(i)) ** strength      (log-space, Zhao et al.)
    sub:  p'(i) ∝ p(i) - strength · (prior(i) - 1/n)     (PriDe)

A uniform prior is an exact no-op in both. The correction lands in `merge_branches`, per
branch, before the branches are averaged, so it composes with order averaging instead of
replacing it.

    reflex-calibrate fit-prior --data runs/distill_questions.jsonl --n 200 \
        --method permuted --permutations 4 --out prior/qwen3.5-4b.json
    reflex-calibrate eval --val runs/external_eval.jsonl --permutations 1 \
        --prior prior/qwen3.5-4b.json
    reflex-serve --model Qwen/Qwen3.5-4B --permutations 1 --prior prior/qwen3.5-4b.json

The prior is fitted on 200 states of the distillation corpus (1051 questions), which is
unlabelled and disjoint from both gates — the fit needs no labels, only the same question
asked several ways. That is a stricter hold-out than leave-one-set-out on the external
sets, where each set is the only source of its (primitive, option count) cell; the
leave-one-set-out run is reported below anyway, and it is a no-op.

## What the 4B's letter bias actually looks like

`prior/qwen3.5-4b.json`, permuted, mode `div`, 1051 questions:

| cell | prior over positions A, B, C, … |
|---|---|
| `noul:2` | 0.587  0.413 |
| `choice:3` | 0.368  0.302  0.330 |
| `choice:4` | 0.262  0.254  0.248  0.236 |
| `choice:5` | 0.241  0.202  0.194  0.182  0.181 |
| `score:4` | 0.301  0.245  0.264  0.190 |
| `score:5` | 0.257  0.222  0.185  0.189  0.146 |

This is the first result, and it explains the rest. Outside the yes/no pair the bias is
small: five options are read within four points of uniform. Fitting the same estimator on
the external sets, where the support-intent questions carry twelve options, gives a
`choice:12` row of 0.087 0.085 0.086 0.088 0.082 0.083 0.079 0.080 0.080 0.081 0.084
0.086 — every position within half a point of uniform, with no front-of-list slope at
all. The one real effect is `noul:2` — the lettered
yes/no pair leans "yes" by 17 points — and, as the tables show, removing it is where the
accuracy goes.

The content-free prior, fitted on the same questions with the state replaced by `N/A`:

| cell | content-free prior |
|---|---|
| `noul:2` | 0.618  0.382 |
| `choice:3` | 0.310  0.516  0.175 |
| `choice:4` | 0.426  0.237  0.122  0.215 |
| `choice:5` | 0.229  0.184  0.112  0.053  0.422 |
| `score:4` | 0.393  0.114  0.400  0.092 |
| `score:5` | 0.478  0.131  0.138  0.144  0.109 |

It agrees with the permuted fit only on `noul:2`. Everywhere else it is large and
ragged — 0.053 on the fourth of five options, 0.422 on the fifth. With no evidence the
model is not falling back on a letter habit, it is reading the option *texts*, and those
differ per question; averaging that over questions produces a number that means nothing
about position. That is a real difference from the multiple-choice setting Zhao et al.
measured, where the options are the answer.

## External sets, 4B frozen, `reflex-calibrate eval`

Accuracy per set, then pooled over all 1200 questions.

| readout | support | MNLI | toxic-chat | Yelp | mean | pooled acc | pooled ECE |
|---|---|---|---|---|---|---|---|
| 1 order, no prior | 0.907 | 0.840 | 0.843 | 0.640 | 0.808 | 0.8075 | 0.0279 |
| 2 orders (`stable`) | 0.917 | 0.837 | 0.833 | 0.653 | 0.810 | 0.8100 | 0.0317 |
| 1 order + permuted, `div` s=1.0 | 0.910 | 0.850 | 0.803 | 0.650 | 0.803 | 0.8033 | 0.0321 |
| 1 order + permuted, `sub` s=0.5 | 0.907 | 0.843 | 0.813 | 0.643 | 0.802 | 0.8017 | **0.0208** |
| 1 order + content-free, `div` s=1.0 | 0.880 | 0.823 | 0.793 | 0.663 | 0.790 | 0.7900 | 0.0447 |

The debias never gains accuracy. The one number that improves is the pooled ECE of the
subtractive form at half strength (0.021 against 0.028 unaided), and its strength was
chosen on these very sets, so it is a selected number, not a held-out one.

Where the accuracy goes is visible in the flips. Taking the default `div` s=1.0 prior and
counting the questions whose top option changes:

| set | flips | newly right | newly wrong |
|---|---|---|---|
| support intents | 3 | 2 | 1 |
| MNLI mismatched | 7 | 5 | 2 |
| toxic-chat | 17 | 3 | **14** |
| Yelp stars | 17 | 8 | 5 |

Three of the four sets come out even or ahead. Toxic-chat pays for all of it, and
toxic-chat is the `noul:2` set — the one cell where the prior is large. The model's lean
toward "yes" on a lettered yes/no pair is not a letter habit to be removed; on this data
it is closer to a correct base rate, and dividing it out moves borderline items to the
wrong side.

### Leave one set out

Fitting the prior on three external sets and applying it to the fourth, pooled over the
four held-out runs. Because each set is the only source of its cell, every one of these
priors reaches the held-out set through the option-count backoff.

| readout | pooled acc | pooled ECE |
|---|---|---|
| 1 order, no prior (same harness) | 0.808 | 0.0287 |
| leave-one-set-out, `div` s=1.0 | 0.808 | 0.0317 |
| leave-one-set-out, `sub` s=1.0 | 0.807 | 0.0279 |
| leave-one-set-out, `sub` s=0.5 | 0.807 | 0.0289 |

A borrowed prior is a no-op to three decimal places. The correction does not transfer
across option counts, which is the honest reading of a bias this small.

### Strength sweep

Permuted prior, `div`, one order, pooled over the external sets:

| strength | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|
| accuracy | 0.802 | 0.801 | 0.800 | 0.805 | 0.802 | 0.797 |
| pooled ECE | 0.0303 | 0.0267 | 0.0225 | 0.0341 | 0.0298 | 0.0415 |

No strength recovers the single-order accuracy of 0.808. The ECE wanders without a trend.

## Public items, 4B frozen, one order + prior

| public items | 1 order | 2 orders (`stable`) | + permuted `div` s=1.0 | + permuted `sub` s=0.5 |
|---|---|---|---|---|
| easy | 1.000 | 1.000 | 1.000 | 1.000 |
| standard | 0.917 | 0.917 | 0.917 | 0.917 |
| hard | 0.658 | **0.685** | 0.649 | 0.649 |
| hard ECE | 0.086 | **0.081** | 0.124 | 0.112 |
| probability fidelity | 0.708 | 0.740 | 0.693 | 0.702 |
| calibration axis | 76.8 | **78.9** | 72.3 | 73.9 |
| hard-item p50 latency | 0.18 s | 0.19 s | 0.18 s | 0.18 s |

This is the clearest result in the document. Debiasing leaves the easy and standard tiers
untouched, loses one hard item, and makes the hard tier's calibration error half again
worse than doing nothing. Hard-tier families move both ways and cancel: under `div`,
`long_policy` falls 0.58 → 0.53 while `temporal_numeric` rises 0.40 → 0.47; under `sub`
the pair swaps back to 0.58 and 0.40. Nothing accumulates. The latency saving over two
orders is real, but there is nothing left to pay for it with.

## Why it does not work here

Two orders and a fitted prior are not two routes to the same correction. Order averaging
removes *this question's* position effect, whatever it is, using this question's own two
readings. A fitted prior removes the *average* position effect over a corpus. Those agree
only when the effect is roughly constant across questions, and the fits above say it is
not: the permuted and content-free estimators, run on the same 1051 questions, disagree
about everything except `noul:2`.

That also reframes what order averaging is doing for reflex. If most of its gain came from
a fixed letter habit, a 17-point `noul:2` prior and a near-uniform `choice:12` prior would
have captured it. They do not, which suggests the second order is mostly buying an
*ensemble* — two correlated but not identical readings of the same question, averaged —
and that is not something a table of six vectors can stand in for.

## A note on the single-order baseline

Reconciling against [order-averaging.md](order-averaging.md) turned up one discrepancy
worth recording. That document reports a pooled external-set ECE of about 0.055 for one
order against 0.032 for two. On today's code the single-order pooled ECE measures 0.0279,
through both the plain `reflex-calibrate eval` path and the merging path, which agree to
four decimal places; two orders measures 0.0317, matching the 0.032 on record. So the
0.032 stands and the 0.055 does not reproduce.

On the external sets, then, two orders is worth about two points of accuracy on support
intents and Yelp and nothing in pooled calibration. Its calibration case rests on the
public items, where hard ECE 0.086 → 0.081 and the calibration axis 76.8 → 78.9 were
measured directly and are unaffected by this. Nothing here argues against `stable`; the
external-set calibration line in the earlier document is simply overstated.

## What shipped

Nothing on the request path. `serving/stable.json` is unchanged and `--prior` is off by
default.

* `reflex.prior` — `PositionPrior` (fit, save, load, back off, debias) and `PriorFitter`.
* `reflex-calibrate fit-prior` — both estimators, `--source` / `--exclude-source` for
  hold-outs.
* `--prior` on `reflex-serve` and on `reflex-calibrate eval`. A prior may also ride inside
  a `calibration.json`, since it is one more post-hoc readout knob.
* `prior/qwen3.5-4b.json` — the fitted 4B prior, kept as the measurement's artefact.

## Not done

The 27B was not measured. The plan was to confirm on worker2 only if the 4B came out
positive, and a correction that loses accuracy on both gates for the 4B does not earn an
85 GB run. If it is ever worth revisiting, the thing to check first is whether a larger
model has a *larger* position bias than the 4B's near-uniform tables; the premise fails
here for want of a bias to remove, not for want of a method.

The code (`reflex.prior`, `reflex-calibrate fit-prior`, `--prior`) is kept on the unmerged branch
`prior-debias`; nothing that lost its gate ships on main.
