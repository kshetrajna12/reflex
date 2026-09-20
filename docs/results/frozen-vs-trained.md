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

## 3. Prompt optimisation with GEPA: a clean negative result

`reflex-optimize` ran GEPA (gepa 0.1.4) over the prompt's named components on the frozen
model: 480 training and 240 validation rows drawn from our own held-out mix across all
twelve sources, a local reflection LM with a general-purpose instruction, dataset names
stripped from the feedback, and a hard rejection of any candidate that names a dataset.
Twenty-seven candidates were accepted; validation score rose from 0.628 to 0.649. The
winning wording was sensible and general ("read the full State, weigh all options,
distribute probability when the State is ambiguous").

It did not transfer:

| | default prompt | GEPA prompt |
|---|---|---|
| support intents (ext) | 0.910 | 0.897 |
| MNLI mismatched (ext) | 0.840 | 0.807 |
| toxic-chat (ext) | 0.787 | 0.793 |
| Yelp stars (ext) | 0.650 | 0.640 |
| public standard tier | 0.931 | 0.889 |
| public hard tier | 0.640 | 0.586 |

A two-point gain on rows from the training distribution became a two-to-five-point loss
everywhere else. This is the mild form of the same lesson as fine-tuning: any search
driven by our mix fits our mix. The default prompt is already close to a local optimum
for this model, and wording changes large enough to move the validation score move the
model's behaviour in ways the external sets punish. The optimised prompt is kept as
`prompts/gepa-v1.rejected.json` for reference.

An earlier GEPA run without the leakage guard produced a system prompt that began
"You are an expert evaluator for the HelpSteer2 dataset", which is the obvious version of
the failure; the guard and the general-purpose reflection template were added in response.

## 4. Post-hoc calibration transfers no better than weights do

`reflex-calibrate refit --val runs/mix3_eval.jsonl --out runs/raw-calibration` fits
per-primitive temperatures (noul 3.19, choice 1.44, score 2.48) and the calibration head
for the frozen model on our held-out mix. On that mix, ECE falls from 0.108 to 0.028.

| | raw, T = 1 | raw + fitted calibration |
|---|---|---|
| external: support intents ECE | 0.035 | 0.085 |
| external: MNLI mismatched ECE | 0.054 | 0.123 |
| external: toxic-chat ECE | 0.104 | 0.090 |
| external: Yelp stars ECE | 0.111 | 0.189 |
| public standard tier ECE | 0.055 | 0.214 |
| public hard tier ECE | 0.143 | **0.102** |
| public hard tier fidelity | 0.726 | **0.730** |

The fitted temperatures are right for inputs that look like our mix (hard, ambiguous
items: the benchmark's hard tier improves, and that is the tier its calibration axis is
computed on) and wrong for easier inputs, which they over-soften. A temperature is a
property of a distribution, not of a model. The recommendation that follows is the one
the README already makes: fit `calibration.json` on a few hundred labelled examples from
the workload you will run, and expect calibration measured elsewhere to be optimistic.

## 5. Two prompt ablations that do transfer

Prompted by how SemIf frames its requests, two one-line changes to reflex's default
prompt were tested on the frozen model, external sets first (`prompts/ablation-*.json`):

* yes/no questions presented as a lettered pair ("A. yes: …", "B. no: …") and read from
  the letter logits, instead of reading the Yes/No tokens;
* headings "# Evidence" / "# Criterion" instead of "# State" / "# Question".

| | default | lettered yes/no | Evidence/Criterion | both |
|---|---|---|---|---|
| toxic-chat (ext, yes/no) | 0.787 / ECE 0.104 | 0.797 / 0.083 | 0.823 / 0.059 | **0.843 / 0.046** |
| support intents (ext) | 0.910 | 0.910 | 0.907 | 0.907 |
| MNLI mismatched (ext) | 0.840 | 0.840 | 0.840 | 0.840 |
| Yelp stars (ext) | 0.650 | 0.650 | 0.650 | 0.640 |
| public standard | 0.931 | | | 0.917 |
| public hard | 0.640 | | | **0.658** |
| public hard ECE (T = 1) | 0.143 | | | **0.086** |

Unlike the GEPA result, these gains appear on data the change was not selected on. The
yes/no token readout carried a "say Yes" prior that the lettered pair removes, and the
verification framing suits questions that ask whether a condition holds. The cost is one
standard-tier item. Both changes are in the default prompt from this commit on.

With the new default prompt, calibration fitted on our mix is *harmful* everywhere: the
fitted noul temperature (4.0) over-softens yes/no questions that the lettered readout
already answers honestly, external ECE rises on all four sets, and public hard-tier ECE
goes from 0.086 (T = 1) to 0.135. The frozen model with this prompt and no calibration
file is the configuration we stand behind for general use; fit a temperature only on
your own workload's labels, and only if its raw ECE there says you need one.
