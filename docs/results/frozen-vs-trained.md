# Frozen model vs trained adapters: what training actually bought

Two control experiments run on 2026-09-19, after three LoRA runs, that every later
decision in this repo rests on.

## 1. The untrained model with reflex's prompt

Same base model (Qwen3.5-4B), same prompt, **no adapter**, per-primitive temperatures off.

| public decision items (external, never trained on) | raw model | mix1 | mix2 | mix3 (published) |
|---|---|---|---|---|
| easy (48) | 1.000 | 1.000 | 1.000 | 1.000 |
| standard (72) | 0.931 | 0.931 | 0.972 | 0.944 |
| hard (111) | **0.640** | 0.595 | 0.541 | 0.604 |

| held-out external sets (300 each, never trained on) | raw model | mix3 |
|---|---|---|
| support intents, different vendor | 0.910 | **0.937** |
| MNLI, mismatched genres | **0.840** | 0.807 |
| toxicity of chatbot prompts | **0.787** | 0.710 |
| Yelp stars (5 levels) | **0.650** | 0.640 |

The frozen model is the best system we have on the hardest external items and on three
of four external domains. LoRA on short classification data bought accuracy on data
that looks like the training data (vendor intents, the standard tier's adequacy and
routing items) and cost general judgement everywhere else. Calibration is where
training helped unambiguously: raw ECE on the external sets is 0.04-0.11 before any
temperature; a fitted temperature or head gets the frozen model most of the way there
without touching a weight.

## 2. Prompt structure is high-leverage and high-variance

SemIf, a frozen Qwen3.5-4B with a compact JSON prompt and letters for every option,
scores 0.986 on the standard tier. A reflex prompt style with the same *structure* in our
own words ("compact") scored **0.847 / 0.559** on standard / hard, against 0.931 / 0.640
for reflex's markdown prompt on the same frozen model. So it is not the JSON structure;
it is the wording, and wording is worth ±8 points on this model.

## What follows from this

* Prefer prompt optimisation on a frozen model over fine-tuning for general use.
  `reflex-optimize` runs GEPA over the prompt's named components against your own
  labelled data; it cannot erode capabilities the way weight updates did.
* Fine-tune only for a specific deployment with in-domain labels, keep it light (one
  epoch, attention-only), and always measure on held-out sets from *other* domains.
* Report external-set numbers first. Held-out splits of the training sources looked
  great for mix2 (82.7 % accuracy) while it was the worst adapter on hard external items.
