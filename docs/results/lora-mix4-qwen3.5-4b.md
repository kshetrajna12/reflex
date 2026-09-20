# LoRA mix4, Qwen3.5-4B (soft-label heavy, rejected)

Fourth run, and the first designed around the finding in
[frozen-vs-trained.md](frozen-vs-trained.md): the earlier adapters bought accuracy on
data that looks like their training mix and paid for it with judgement on long,
ambiguous items. mix4 tried to fix that with data rather than with less training:

* more **soft-label** sources (civil_comments toxicity fractions, HelpSteer2 per-rater
  score distributions, GoEmotions rater agreement, stated base rates), so the target is
  a distribution rather than a single label;
* two **long-document** sources (QuALITY reading comprehension, synthetic policy documents)
  at full weight, as the closest in-distribution proxies for hard-tier items;
* **option-permutation augmentation** (`--permutations 2`): every choice/score question
  is seen twice with its options shuffled, so the adapter cannot learn a position prior;
* the same attention-only LoRA (r=16, q/k/v/o) and one epoch as mix3.

6,800 training rows (about 12.9k branches with permutations), 1,701 held-out rows (3,164
branches), trained on a dedicated GB10 with 4,096-token batches and no gradient
checkpointing (3 h 3 min). It was trained and evaluated at commit `cdf83ec`, i.e. with the
*old* default prompt (`prompts/legacy-v0.json`); all comparisons below use that prompt for
the frozen model too.

    reflex-data mix --sources banking77=500,clinc=500,mmlu_pro=600,halueval=500,msmarco=500,codereview=600,helpsteer2=300,adequacy_synth=300,rule_routing_synth=200,base_rates_synth=400,policy_synth=400,quality=600,civil_soft=500,helpsteer2_multi=500,go_emotions_soft=400 \
        --per-source 800 --eval-per-source 200 --out runs/mix4_train.jsonl --eval-out runs/mix4_eval.jsonl
    reflex-data check-overlap --train runs/mix4_train.jsonl --against <benchmark items>    # 0 rows
    reflex-calibrate train --data runs/mix4_train.jsonl --val runs/mix4_eval.jsonl --out runs/lora-mix4 \
        --epochs 1 --lr 1e-4 --permutations 2 --max-pack-tokens 4096

## Held-out (in-distribution): the training worked

| | before | after (T = 1) | after + per-primitive T |
|---|---|---|---|
| accuracy (3,164 branches) | 0.580 | **0.821** | 0.821 |
| top-label ECE | 0.157 | 0.029 | **0.024** |
| fidelity, soft-label sources (1 − TV) | 0.612 | **0.773** | 0.770 |
| fitted temperatures (noul / choice / score) | | | 1.03 / 1.48 / 1.14 |

The fitted temperatures are close to 1 (mix3 needed 1.6–1.9; the frozen model 3–4 on
yes/no), so the soft targets and permutations did make the raw logits honest on this
distribution. The long-document sources moved the most: QuALITY 0.20 → 0.74, synthetic
policy 0.54 → 0.91. MMLU-Pro (0.57) and MS MARCO (0.62) barely moved, as in every run.

## External sets (never trained on): it did not transfer

Accuracy on 300 items each; the frozen model uses the same prompt wording mix4 was trained with.

| | mix4 | frozen (same prompt) | mix3 |
|---|---|---|---|
| support intents (bitext, 12-way choice) | **0.930** | 0.910 | 0.937 |
| MNLI mismatched (3-way choice) | 0.807 | **0.840** | 0.807 |
| toxic-chat (yes/no) | **0.533** / ECE 0.358 | 0.787 / 0.104 | 0.710 |
| Yelp stars (5-level score) | **0.667** | 0.650 | 0.640 |

toxic-chat is a collapse, not noise: mix4 says "yes, toxic" on 290 of 300 messages with
mean confidence 0.88, catching all 150 toxic ones and missing 140 of the 150 benign ones.
civil_comments toxicity is the largest single source in the mix, and its notion of
"toxic" (insults in news comments) does not match a chat prompt such as
"python script for a dos attack", which the toxic-chat annotators label benign. The
adapter learned a broad "flag anything edgy" prior for that question wording and applied
it with high confidence outside its domain. This is the same failure as mix1–3, in a
different place: the sharper the adapter gets on its sources, the more confidently it
misreads a neighbouring task.

## JevBench public items

| tier | items | mix4 | frozen (old prompt) | frozen + new prompt (`dae6799`) | mix3 |
|---|---|---|---|---|---|
| easy | 48 | 1.000 | 1.000 | 1.000 | 1.000 |
| standard | 72 | **0.958** | 0.931 | 0.917 | 0.944 |
| hard | 111 | 0.595 | 0.640 | **0.658** | 0.604 |
| hard top-label ECE | | 0.189 | 0.143 | **0.086** | 0.117 |
| hard probability fidelity | | 0.692 | | 0.708 | |
| calibration axis (public) | | 65.7 | | **76.8** | 72.7 |

Per family on the hard tier, mix4 gains on multi-hop (0.72 vs 0.67) and loses on
ambiguous (0.43 vs 0.71), judge-hard (0.53 vs 0.65), temporal/numeric (0.27 vs 0.40) and
probability (0.40 vs 0.50). Standard-tier accuracy is the best of any configuration,
which is the pattern of every adapter so far: better on well-specified classification,
worse on the items that need judgement, and over-confident where it is wrong (hard ECE
0.189 despite near-unity temperatures on the held-out set).

## Verdict

Rejected. Soft labels and permutation augmentation fix the *in-distribution* calibration
problem cleanly, and that is worth keeping for anyone fine-tuning reflex on their own
workload. They do not fix the transfer problem: a 4B model's general judgement is a
shared resource, and one epoch of LoRA on classification-shaped data spends it. The
frozen model with the current default prompt (commit `dae6799`) remains the recommended
configuration and the one requested on the benchmark. The adapter is kept locally under
`runs/lora-mix4` and was not published.
