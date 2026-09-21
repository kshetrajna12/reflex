# Self-distilling the two-order readout into one pass

The `stable` configuration asks every question in two distinct option orders and averages
the two distributions ([order-averaging.md](order-averaging.md)). It buys its gain by
cancelling the model's option-position prior, and it pays two branches per question. The
question here is whether that cancellation can be moved into the weights: train a small
LoRA whose targets are the frozen model's *own* two-order distributions, and serve one
pass.

This is the one fine-tune in the repo with no foreign labels in it. Every earlier adapter
learned somebody else's dataset rule and applied it, over-confidently, to neighbouring
tasks ([frozen-vs-trained.md](frozen-vs-trained.md)). Here the target is what the model
already says when you ask it twice, so there is no rule to overfit; the only risk is
drift, and the gates below are what measure it.

**Verdict: half of it transfers.** The adapter reproduces the two-order distributions
from a single pass and carries the calibration half of the gain, at the best public-item
calibration axis we have measured on the 4B. It does not carry the accuracy half: the
hard tier lands at 0.640, below both two orders (0.685) and one order (0.658). `stable`
does not move.

## Method

    reflex-distill label --model Qwen/Qwen3.5-4B --permutations 2 \
        --in runs/distill_questions.jsonl --out runs/selfdistill_main.jsonl
    reflex-distill label --model Qwen/Qwen3.5-4B --permutations 2 \
        --in runs/distill_anchor_questions.jsonl --out runs/selfdistill_anchor.jsonl
    cat runs/selfdistill_main.jsonl runs/selfdistill_anchor.jsonl > runs/selfdistill_targets.jsonl
    reflex-distill mix --teacher runs/selfdistill_targets.jsonl --eval-frac 0.06 \
        --out runs/selfdistill_train.jsonl --eval-out runs/selfdistill_eval.jsonl
    reflex-calibrate train --model Qwen/Qwen3.5-4B --permutations 1 --epochs 1 --lr 5e-5 \
        --data runs/selfdistill_train.jsonl --val runs/selfdistill_eval.jsonl \
        --out runs/lora-selfdistill

The corpus is the one built for [DISTILLATION.md](../DISTILLATION.md): 4,500 states from
ten public domains, 23,420 typed questions, a fixed bank plus two LLM question writers.
Both slices were relabelled rather than reused. The label files from the 27B run were
written before the distinct-order fix (commit db224f0), so their "two orders" were random
shuffles that could repeat the identity order and never swapped yes/no questions. They
are not the readout this experiment is about.

No anchor slice and no KL pull: with the model's own answers as the target, the whole
corpus is already an anchor. `--permutations 1` at training time is the point of the
experiment. The student sees one order per question and is asked for the average of two.
Attention-only LoRA, r=16, one epoch, 4,207 training states, 293 held out, 7,415 steps,
3 h 25 min on a GB10, peak footprint 34 GB.

## It learned the thing it was shown

Held-out 6 % of the corpus, 1,502 questions, scored against the frozen model's own
two-order distributions:

| | frozen, 1 order | **adapter, 1 order** |
|---|---|---|
| top answer agrees with the two-order readout | 0.833 | **0.942** |
| distribution fidelity (1 − TV) | 0.894 | **0.953** |

Fitted temperatures come out at 0.98 / 1.02 / 0.98 (noul / choice / score): the student is
already calibrated against its target and wants no temperature at all. The 27B-distilled
student needed 0.35 / 0.66 / 0.45. That is what "the target is my own average" buys.

## External sets: accuracy unchanged, calibration slightly worse

Four held-out sets of 300, none of them trained on, all four configurations run with an
identity calibration file (temperatures 1.0, no head) so no fitted calibration is in play:

| | frozen, 1 order | frozen, 2 orders | **adapter, 1 order** | adapter, 2 orders |
|---|---|---|---|---|
| support intents | 0.907 | 0.917 | 0.910 | 0.910 |
| MNLI mismatched | 0.840 | 0.837 | 0.847 | 0.840 |
| toxic-chat | 0.843 | 0.833 | 0.817 | 0.837 |
| Yelp stars | 0.640 | 0.653 | 0.653 | 0.650 |
| 4-set mean | 0.808 | 0.810 | 0.807 | 0.809 |
| pooled top-label ECE | 0.028 | 0.032 | 0.038 | 0.041 |

The adapter at one pass has the two-order *accuracy profile* without the second branch:
MNLI and Yelp move up, toxic-chat down, the mean is flat. Nothing collapsed, which is the
first thing to check on any adapter here. Pooled calibration on these sets is a point
worse than either frozen configuration, and adding orders back on top makes it worse
still, not better.

## Public items: the calibration transfers, the accuracy does not

No adapter calibration file; the served calibration is the identity in all four columns.

| public items | frozen, 1 order | frozen, 2 orders | **adapter, 1 order** | adapter, 2 orders |
|---|---|---|---|---|
| easy (48) | 1.000 | 1.000 | 1.000 | 1.000 |
| standard (72) | 0.917 | 0.917 | **0.931** | 0.931 |
| hard (111) | 0.658 | **0.685** | 0.640 | 0.649 |
| hard ECE | 0.086 | 0.081 | **0.056** | 0.082 |
| probability fidelity | 0.708 | 0.740 | 0.712 | **0.747** |
| calibration axis | 76.8 | 78.9 | **80.0** | 79.2 |
| hard-item p50 latency | 0.18 s | 0.19 s | 0.18 s | 0.19 s |

The adapter at one order is the best calibration axis measured on this model: hard-tier
ECE falls by a third against two orders, at half the branch cost, and the standard tier
gains an item. It is also the worst hard-tier accuracy of the four.

Two orders on top of the adapter is the control for "does this stack": it does not. One
item of accuracy back, ECE back up to 0.082, the axis down to 79.2. The adapter has
already spent the position prior; averaging a second order over a model that no longer
has much of one only adds noise.

## Item by item on the hard tier, against two orders

Nine items lost, four gained:

| family | items | frozen, 2 orders | adapter, 1 order |
|---|---|---|---|
| long_policy | 19 | 0.68 | 0.47 |
| probability | 10 | 0.60 | 0.40 |
| multi_hop | 18 | 0.67 | 0.61 |
| temporal_numeric | 15 | 0.47 | 0.40 |
| ambiguous | 7 | 0.57 | 0.71 |
| judge_hard | 17 | 0.65 | 0.71 |
| tradeoff | 6 | 0.67 | 0.83 |
| adversarial / routing_hard / trap | 19 | 1.00 | 1.00 |

The shape is the one the 27B distillation produced
([lora-distill-qwen3.5-4b.md](lora-distill-qwen3.5-4b.md)): judgement-shaped families gain,
long documents and arithmetic over the state lose. That it appears again with targets the
model wrote itself is the useful part of this run. It is not a learned wrong rule, because
there is no rule here to learn. It is what a LoRA on 4.5k short states does to a model's
handling of long, multi-step inputs, and it will show up under any target.

## Honest reading

* **Does one pass with the adapter match two orders?** On the distributions, yes: 0.94
  agreement, 0.95 fidelity, up from 0.83 and 0.89. On calibration, better than two orders.
  On hard-tier accuracy, no, and the gap is 5 items.
* Some of the calibration gain is the adapter hedging rather than judging better. Its mean
  confidence on the external sets drops from 0.808 to 0.785 at unchanged accuracy, and on
  the hard tier it is answering fewer items correctly while being better calibrated about
  it. The axis rewards that; a user who wants the right answer does not.
* The corpus is the limit. 4.5k states is what made the 27B distillation drift, and it is
  what made this one drift, on exactly the same families. The experiment worth running
  next is this one at ten times the corpus with hard-tier-shaped states, which is the same
  thing [DISTILLATION.md](../DISTILLATION.md) already concluded.

Kept as a run. The adapter is not published and `serving/stable.json` is unchanged: the
frozen model at two distinct orders remains what we serve.
