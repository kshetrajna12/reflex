# Distilled student (LoRA from Qwen3.8-27B targets): closer to the teacher, not a better judge

The run described in [../DISTILLATION.md](../DISTILLATION.md): 3,625 raw states from ten
public domains, 18,945 typed questions (a fixed bank plus two LLM question writers),
answered by Qwen3.8-27B through reflex's own prompt (two option orders averaged), plus 875
anchor states answered by the frozen 4B itself. One epoch of attention-only LoRA (r=16,
lr 5e-5) on 4,231 states, 3 h 30 min on a GB10. Teacher selection is in
[teachers-27b-and-4b-think.md](teachers-27b-and-4b-think.md).

## Fidelity to the teacher (held-out, 1,225 questions): it learned what it was shown

| | before | after (T = 1) | after + fitted T |
|---|---|---|---|
| top-answer agreement with the 27B | 0.655 | **0.849** | 0.849 |
| distribution fidelity (1 − TV) | 0.743 | **0.880** | 0.837 |
| ECE against the teacher's top label | 0.091 | 0.127 | **0.018** |
| fitted temperatures (noul / choice / score) | | | 0.35 / 0.66 / 0.45 |

Every domain improved by 12 to 30 points. The fitted temperatures are *below* 1: the
student hedges more than its teacher, the opposite of every dataset-trained adapter, which
is what soft targets from a calibrated teacher should produce.

## External sets: moved toward the teacher on all four

| | frozen 4B | **student** | 27B teacher |
|---|---|---|---|
| support intents | 0.907 | 0.927 | 0.927 |
| MNLI mismatched | 0.840 | 0.833 | 0.850 |
| toxic-chat | 0.843 | 0.727 | 0.740 |
| Yelp stars | 0.640 | 0.670 | 0.683 |

The first adapter that did not break anything: no collapse, and each number lands
between the student's and the teacher's. It also inherited the teacher's disagreement with
the toxic-chat annotators (it now flags jailbreak-shaped prompts), which is a policy one
would choose deliberately rather than get for free.

## Public benchmark items: no gain on the hard tier

| | frozen 4B | student, fitted T | student, T = 1 | 27B teacher |
|---|---|---|---|---|
| easy | 1.000 | 1.000 | 1.000 | 1.000 |
| standard | 0.917 | 0.917 | 0.917 | 0.917 |
| hard | **0.658** | 0.613 | 0.613 | 0.703 |
| hard ECE | 0.086 | 0.098 | 0.086 | 0.088 |
| probability fidelity | 0.708 | 0.717 | 0.716 | 0.831 |
| calibration axis | 76.8 | 76.1 | 77.2 | 82.7 |

Per family the student gained on judge_hard (0.65 → 0.76) and lost on long_policy
(0.58 → 0.53), multi_hop (0.67 → 0.56), probability (0.50 → 0.40) and temporal/numeric
(0.40 → 0.27): the families where the teacher is strongest are the ones that did not
transfer. Item by item against the frozen model: 8 items gained (7 of them ones the
teacher gets right), 13 lost (9 of them ones the teacher gets right). Calibration is not
the cause; the T = 1 run is identical in accuracy.

## What this says

Distillation did what the dataset adapters could not: it moved the student toward a
better model *without* the collapse on neighbouring tasks, and it left calibration honest.
What it did not do is make the student a better judge on the benchmark's hard items. The
losses are the signature of general drift from a small corpus (3.4k states) rather than
of learned wrong rules: the adapter nudges behaviour everywhere, and on items the frozen
model already had right that nudge is as likely to hurt as help. The teacher's edge on
long documents and stated probabilities looks like knowledge and context handling that a
LoRA on 3k states cannot instil.

Kept as a run; the frozen model with the default prompt remains `stable`. Two things would
be worth trying before concluding distillation cannot help at 4B scale: ten times the
corpus with hard-tier-shaped questions (multi-hop, arithmetic over the state, long
policies with interacting rules), and targets from the thinking 4B for judgement-shaped
families, where it beats the 27B.
