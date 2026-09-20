# Option-order averaging: the cheapest calibration gain we found

Ask each choice or score question in two distinct option orders (and each yes/no question
both ways round), average the two distributions. Same forward pass, no training, no fitted
parameters. Selected on the external sets, confirmed on the public benchmark items; now the
`stable` configuration (`permutations: 2` in `serving/stable.json`).

Two earlier attempts at this looked negative and were wrong: the permutation code drew
random shuffles that could repeat the identity order and never swapped yes/no questions
(Phase 1 of [MODEL_QUALITY_PLAN.md](../MODEL_QUALITY_PLAN.md)). Every number below is on
the corrected code (`distinct_orders` in `prompt.py`, commit db224f0).

## 4B, frozen, default prompt

| public items | 1 order (old stable) | **2 distinct orders** |
|---|---|---|
| easy / standard | 1.000 / 0.917 | 1.000 / 0.917 |
| hard | 0.658 | **0.685** (76/111) |
| hard ECE | 0.086 | **0.081** |
| probability fidelity | 0.708 | **0.740** |
| calibration axis | 76.8 | **78.9** |
| hard-item p50 latency | 0.18 s | 0.19 s |

Hard-tier families: long_policy 0.58 → 0.68, probability 0.50 → 0.60, temporal 0.40 → 0.47,
ambiguous 0.71 → 0.57 (four items).

| external sets (never trained on) | 1 order | 2 distinct orders |
|---|---|---|
| support intents / MNLI / toxic-chat / Yelp | 0.907 / 0.840 / 0.843 / 0.640 | 0.917 / 0.837 / 0.833 / 0.653 |
| accuracy, 4-set mean | 0.808 | 0.810 |
| pooled top-label ECE | ~0.055 | **0.032** |

Accuracy is unchanged within noise; calibration error roughly halves. The gateway serves a
three-question request in ~280 ms warm (was ~180 ms) because the branch count doubles inside
one pass; the state cache and prefill are shared.

## 27B (Qwen3.8-27B), frozen, default prompt

| public items | 1 order | **2 distinct orders** | Jev (official) |
|---|---|---|---|
| easy / standard | 1.000 / 0.917 | 1.000 / **0.958** | 1.000 / 0.986 |
| hard | 0.703 | **0.766** (85/111) | 0.730 |
| hard ECE | 0.088 | **0.061** | 0.031 |
| probability fidelity | 0.831 | 0.827 | |
| calibration axis | 82.7 | **85.2** | |
| hard-item p50 latency | 0.88 s | 0.96 s | |

Families: judge_hard 0.65 → 0.82, multi_hop 0.67 → 0.72, long_policy 0.68 → 0.74,
probability 0.7 → 0.8, ambiguous 1.0; temporal/numeric stays at 0.40.

On these 111 public hard items the 27B now scores above Jev's official 0.730. Two caveats
stand: we have consulted these items throughout development, so they are a development
suite, not an independent test; and Jev's calibration error (0.031) is still half of ours.
An official run at this configuration is the only claim that counts. Its external-set
confirmation is recorded below when available.

## What did not help

* **Four orders** (27B): hard 0.712 but ECE 0.095 on the buggy code; not rerun, two is enough.
* **Wording ensembles** (four wordings × two orders, `prompts/ensemble-v1.json`): worse for
  both models on the public items (4B hard 0.649 / ECE 0.097) and on the external sets
  (27B 0.786 / ECE 0.083). Two of the four wordings use the Yes/No-token readout, which
  carries a "say yes" prior; averaging pulls the answer toward it.
* **A letters-only two-wording ensemble** (`ensemble-v2.json`, corrected orders): 4B hard
  0.667 / ECE 0.080, calibration axis 79.2. Same calibration as orders alone, two items
  lower, standard-tier ECE worse. Wordings add nothing beyond a disagreement signal.
* **Reasoning on every branch of a wording ensemble**: perfect standard tier (72/72) but 25
  to 120 s per item; the hard tier timed out on 58 of 111 items. Not a serving mode.

## The disagreement signal

Across wordings and orders, how much the readings scatter (mean total-variation distance to
their mean) predicts whether the merged answer is right, on the external sets, 4B, two orders:

| disagreement | questions | accuracy | mean confidence |
|---|---|---|---|
| < 0.05 | 652 | 0.908 | 0.904 |
| 0.05 – 0.15 | 430 | 0.702 | 0.683 |
| 0.15 – 0.30 | 110 | 0.691 | 0.585 |
| > 0.30 | 8 | 0.250 | 0.489 |

That is what `--think-if-disagree` keys on: the fast path answers everything, and only the
questions whose readings scatter are re-answered with the thinking readout. Its public-item
result is recorded below when available.

## Selective reasoning on the 4B (letters-only ensemble × two orders, think 768 tokens when disagreement > 0.15)

| public items | 2 orders | + selective reasoning |
|---|---|---|
| standard | 0.917 | **0.958** (69/72) |
| hard | 0.685 | 0.658 |
| hard ECE | 0.081 | 0.082 |
| calibration axis | 78.9 | 79.1 |
| p50 / p95 latency, hard items | 0.19 s / 0.8 s | 0.25 s / 42 s |

34 escalations across the 231 items, none on the easy tier. Reasoning fixes the standard-tier
adequacy and policy misses (the same ones the always-think run fixes) but on the hard tier it
trades items: multi-hop drops to 0.50 while tradeoff rises to 0.83. The disagreement gate works
as a gate; what it routes to is not yet a better judge on hard items, and the p95 makes it a
batch-mode option only.

## 27B, two distinct orders, external sets

| | 1 order | **2 distinct orders** | 4B, 2 orders |
|---|---|---|---|
| support intents | 0.927 | 0.937 | 0.917 |
| MNLI mismatched | 0.850 | 0.850 | 0.837 |
| toxic-chat | 0.740 | **0.813** | 0.833 |
| Yelp stars | 0.683 | 0.683 | 0.653 |
| 4-set mean | 0.800 | **0.821** | 0.810 |

Reading yes/no questions both ways round removes most of the 27B's over-flagging on
toxic-chat (its one external weakness, ECE there 0.148 → 0.072). With that, the 27B at two
orders is the strongest configuration we have on every gate.
