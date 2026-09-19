# LoRA mix3, Qwen3.5-4B (the adapter published on the Hub)

Third run: the first run's recipe (attention-only LoRA r=16, one epoch) plus the four
targeted sources at about a third of their mix2 weight. 7,850 training examples,
2,082 held-out; gradient checkpointing with 2,048-token batches (peak 18 GB); 43 minutes
of training on a shared GB10. Temperatures fitted per primitive on the held-out set:
noul 1.63, choice 1.54, score 1.89.

    reflex-data mix --sources banking77=800,clinc=800,mmlu_pro=800,civil_comments=1000,halueval=800,msmarco=800,helpsteer2=500,helpsteer2_soft=500,codereview=800,adequacy_synth=400,rule_routing_synth=250,base_rates_synth=400 \
        --out runs/mix3_train.jsonl --eval-out runs/mix3_eval.jsonl
    reflex-data check-overlap --train runs/mix3_train.jsonl --against <benchmark items>    # 0 rows
    reflex-calibrate train --data runs/mix3_train.jsonl --val runs/mix3_eval.jsonl --out runs/lora-mix3 \
        --epochs 1 --lr 1e-4 --grad-checkpoint --max-pack-tokens 2048

| source | acc before → after | ECE after → per-primitive T | fidelity |
|---|---|---|---|
| all | 63.6 → **79.6** | 0.066 → **0.048** | 0.75 |
| adequacy_synth | 63.0 → 94.0 | 0.057 → 0.049 | |
| rule_routing_synth | 90.3 → 100.0 | 0.001 → 0.013 | |
| base_rates_synth (soft) | 93.0 → 98.0 | (top-label ECE not meaningful) | 0.88 |
| civil_comments (soft) | 46.4 → 95.6 | 0.044 → 0.102 | 0.87 |
| halueval | 78.5 → 98.5 | 0.015 → 0.015 | |
| codereview | 49.7 → 77.5 | 0.193 → 0.121 | |
| helpsteer2 | 32.8 → 44.0 | 0.161 → 0.065 | |
| helpsteer2_soft (soft) | 33.6 → 36.8 | 0.106 → 0.061 | 0.58 |
| mmlu_pro | 47.5 → 57.0 | 0.143 → 0.101 | |
| msmarco | 66.5 → 65.5 | 0.158 → 0.091 | |
| banking77 / clinc | 92.0 → 93.5 / 93.0 → 93.5 | 0.037 / 0.033 | |

Compared with the two earlier runs on an external set of decision items (not used in
training): mix3 has the best accuracy on the hardest items and the best calibration of
the three, while keeping most of mix2's gain on instruction-adequacy and rule-based
routing. mix2's two epochs over the short synthetic sources bought accuracy on those
sources at the cost of over-confidence on long, ambiguous inputs; a third of the weight
and one epoch keeps the skill without the cost.
