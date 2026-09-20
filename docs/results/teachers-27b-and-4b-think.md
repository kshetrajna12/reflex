# Choosing a teacher: Qwen3.8-27B, and the 4B that thinks

Before distilling ([../DISTILLATION.md](../DISTILLATION.md)) the candidate teachers had to
show they are better judges than the frozen 4B student. Two candidates, three gates.

## 1. External sets: the 27B is *not* clearly better there

Accuracy on the four never-trained external sets (300 items each), same prompt as the student:

| | support intents | MNLI mismatched | toxic-chat | Yelp stars |
|---|---|---|---|---|
| frozen 4B (student) | 0.907 | 0.840 | **0.843** | 0.640 |
| Qwen3.8-27B, no thinking | **0.927** | **0.850** | 0.740 | **0.683** |
| Qwen3.8-27B, old prompt wording | 0.927 | 0.850 | 0.610 | 0.683 |

Three sets are at their label ceiling, where more parameters cannot buy anything. The
fourth is a policy disagreement: the 27B flags jailbreak-shaped prompts that the
toxic-chat annotators label benign, and it does so more with the older wording. These
sets were built to catch adapters *breaking* general behaviour, not to rank models near
the top, so they are the wrong gate for a teacher.

## 2. Public benchmark items: both candidates beat the student on the hard tier

JevBench public items, one request per item, T = 1, no calibration file:

| | easy (48) | standard (72) | hard (111) | hard ECE | prob. fidelity | calib. axis | p50 latency |
|---|---|---|---|---|---|---|---|
| frozen 4B (student) | 1.000 | 0.917 | 0.658 | **0.086** | 0.708 | 76.8 | 0.12 s |
| Qwen3.8-27B, no thinking | 1.000 | 0.917 | **0.703** | 0.088 | **0.831** | **82.7** | 0.60 s |
| Qwen3.5-4B, thinking (768 tokens) | 1.000 | **1.000** | **0.703** | 0.110 | 0.745 | 76.3 | 20-40 s |

Hard-tier families, where the two teachers differ:

| family | 4B | 27B | 4B thinking |
|---|---|---|---|
| ambiguous | 0.71 | **1.00** | **1.00** |
| judge_hard | 0.65 | 0.65 | **0.94** |
| long_policy | 0.58 | **0.68** | 0.53 |
| multi_hop | 0.67 | 0.67 | 0.61 |
| probability | 0.50 | **0.70** | 0.60 |
| temporal_numeric | 0.40 | 0.40 | 0.33 |
| tradeoff | 0.67 | 0.50 | 0.67 |

Two different kinds of "better". The 27B reads long policies and states probabilities more
faithfully: more knowledge and a wider context window used well. The 4B that is allowed to
reason before the readout becomes near-perfect on judgement-shaped items (judge_hard,
adequacy, policy on the standard tier: 72/72) but gains nothing on long documents or
arithmetic within a 768-token budget. Neither moves temporal/numeric, which at 4B scale
seems to need either tools or a much larger model.

## What was chosen

* **Teacher for the corpus: the 27B without thinking.** Its labels cost 2 s per state
  (≈ 7 questions × 2 option orders) and carry the long-document and probability skill the
  student lacks most. Hard-tier 0.703 is within 3 points of Jev itself (0.730).
* **The thinking readout** (`reflex.think`, `--think N` on `reflex-serve`,
  `reflex-calibrate eval` and `reflex-distill label`) is kept as a second, slower teacher
  for judgement-heavy question families, and as a demonstration that the same 4B holds a
  much better judge inside it than System One exposes. At 20-40 s per item it is not a
  serving mode.

The distilled student is evaluated in [lora-distill-qwen3.5-4b.md](lora-distill-qwen3.5-4b.md).
