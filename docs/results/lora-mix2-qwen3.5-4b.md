# LoRA on the broader mix (mix2), Qwen3.5-4B

> **Status: rejected on transfer.** In-distribution numbers; see
> [frozen-vs-trained.md](frozen-vs-trained.md) for what happened off-distribution. Not
> published, not served.

Second training run. Same recipe as the first, plus four new sources aimed at
instruction adequacy, rule-based routing, reading probabilities off stated base rates,
and rater disagreement (HelpSteer2 per-rater scores as soft labels); two epochs;
LoRA on attention and MLP projections; gradient checkpointing with 2,048-token batches
(peak 18 GB on a shared GB10). 8,413 training examples, 2,079 held-out.

    reflex-data mix --sources banking77=500,clinc=500,mmlu_pro=600,civil_comments=1000,halueval=500,msmarco=500,helpsteer2=300,helpsteer2_soft=1000,codereview=700,adequacy_synth=1000,rule_routing_synth=500,base_rates_synth=800 \
        --out runs/mix2_train.jsonl --eval-out runs/mix2_eval.jsonl
    reflex-data check-overlap --train runs/mix2_train.jsonl --against <benchmark items>   # 0 rows
    reflex-calibrate train --data runs/mix2_train.jsonl --val runs/mix2_eval.jsonl --out runs/lora-mix2 \
        --epochs 2 --lr 1e-4 --lora-mlp --grad-checkpoint --max-pack-tokens 2048
    reflex-calibrate refit --adapter runs/lora-mix2 --val runs/mix2_eval.jsonl              # per-primitive T

| source | acc before → after | ECE before → after → per-primitive T |
|---|---|---|
| all | 62.4 → **82.7** | 0.104 → 0.085 → 0.070 |
| adequacy_synth | 66.4 → 100.0 | 0.194 → 0.001 → 0.004 |
| rule_routing_synth | 95.2 → 100.0 | 0.092 → 0.000 → 0.002 |
| base_rates_synth (soft) | 88.5 → 99.5 | fidelity 0.87 after (top-label ECE is meaningless here, see below) |
| civil_comments (soft) | 46.4 → 97.2 | 0.242 → 0.045 → 0.094 |
| halueval | 76.8 → 100.0 | 0.058 → 0.001 → 0.003 |
| codereview | 48.8 → 81.4 | 0.155 → 0.137 → 0.055 |
| helpsteer2 | 30.7 → 56.0 | 0.223 → 0.133 → 0.121 |
| helpsteer2_soft (soft) | 32.4 → 38.4 | fidelity 0.59 |
| mmlu_pro | 46.0 → 60.7 | 0.196 → **0.300** → 0.166 |
| msmarco | 65.6 → 67.2 | 0.233 → 0.228 → 0.152 |
| banking77 / clinc | 91 → 94 / 94 → 97 | ≈ 0.04 / 0.03 |

Temperatures fitted per primitive on the held-out set: noul 1.50, choice 1.94, score 1.68.

What we learned:

* The targeted synthetic sources are learned completely (they are correct by
  construction), and the skills transfer: on an external set of decision items the
  "does the response satisfy the request" family went from 0.75 to 0.83 and rule-based
  routing from 0.83 to 1.00, and distribution fidelity on probability items from 0.66 to 0.80.
* The price was over-confidence on hard, long, exam-like inputs: MMLU-Pro ECE rose from
  0.20 to 0.30 before temperature, and accuracy on long multi-condition policy documents
  in the external set fell. Two epochs of short, deterministic items make the model more
  decisive exactly where it should hedge. A single global temperature over-softens the
  sources that were already honest; per-primitive temperatures are strictly better.
* Top-label ECE is the wrong metric for soft-label sources. A model that answers "60 %"
  on an item whose true frequency is 60 % is right every time and reads ECE 0.4. The
  evaluator now reports distribution fidelity (1 - mean total variation) for them.
* Next: one epoch, attention-only LoRA, synthetic sources at a third of the weight
  (mix3), to keep the standard-tier gains without the hard-tier regression.
