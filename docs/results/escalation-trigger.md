# The escalation trigger: a better gate onto a slow path that is not yet better

> **Historical experiment.** reflex serves one fast forward pass and no reasoning: see [VISION.md](../VISION.md). The results below are kept as evidence, not as a description of the serving path.

A small classifier decides which questions the fast path should not be trusted on, and
sends only those to the thinking readout. It is fitted on our own labelled sets, never on
a benchmark item, and it is budgeted.

The short version: the trigger is a clearly better gate than cross-order disagreement, on
every split we measured, and it picks out exactly the reasoning-shaped families we
predicted. On the public items it does not buy hard-tier accuracy, because reasoning
itself helps on temporal/numeric items and actively hurts on long-policy ones. Best
hard-tier calibration we have recorded on the 4B; accuracy unchanged.

## Method

`reflex.escalate` scores each question with a logistic regression over 23 features, in
two groups.

**Known before answering** — the shape of the problem: which primitive, how many options,
the state's length in tokens, characters, sentences and JSON leaf fields, the instruction's
length, whether the state and the question carry dates, currency or percentages, how many
numbers each contains, and how many "rule" cues each contains (before, after, until,
deadline, expires, within, per, total, net, exceeds, unless, except, eligible, threshold,
neither, only if, and similar).

**Produced by the fast path itself** — its merged top probability, its normalised entropy,
its top-two margin, and its cross-order disagreement.

The model is fitted by full-batch gradient descent on standardised features with balanced
class weights, numpy only, and saves as JSON. It ships next to a calibration file; a
threshold is stored with it, chosen as the quantile that escalates the intended budget
fraction of the fitting distribution.

    reflex-escalate features --val runs/external_eval.jsonl --out runs/esc_features_ext.jsonl
    reflex-escalate fit --features runs/esc_features_ext.jsonl runs/esc_features_mix3.jsonl \
        --out runs/escalator.json --budget 0.12
    reflex-serve --permutations 2 --think 768 --escalate runs/escalator.json

The label is "the fast path's top answer is wrong", where the fast path is the frozen
Qwen3.5-4B, default prompt, two distinct option orders, temperature 1 — the `stable`
configuration. Training rows come from two pools, 3282 labelled questions in all:

| pool | questions | fast-path errors |
|---|---|---|
| four external sets (support intents, MNLI mismatched, toxic-chat, Yelp stars) | 1200 | 228 (19.0 %) |
| the mix3 evaluation split (12 sources: MMLU-Pro, MS MARCO, HaluEval, code review, civil comments, HelpSteer2, banking77, CLINC, and our synthetic adequacy / base-rate / rule-routing sets) | 2082 | 822 (39.5 %) |

No JevBench item, public or otherwise, is in either pool.

## Does it catch errors better than disagreement alone?

Recall of errors is the number that matters: an error the trigger does not select is one
the slow path never gets a chance to fix. Precision says how much reasoning time is spent
on questions that were already right.

**Leave-one-set-out over the four external sets**, pooled. Each set is scored by a model
that never saw it, then all four held-out slices are pooled and ranked together. 1200
questions, 228 errors.

| budget | trigger recall | trigger precision | disagreement recall | disagreement precision |
|---|---|---|---|---|
| 5 % | **0.114** | 0.433 | 0.092 | 0.350 |
| 10 % | **0.232** | 0.442 | 0.180 | 0.342 |
| 12 % | **0.285** | 0.451 | 0.224 | 0.354 |
| 15 % | **0.355** | 0.450 | 0.272 | 0.344 |
| 25 % | **0.561** | 0.427 | 0.452 | 0.343 |

**Random 70/30 split** over the whole pool, mean of ten seeds. 985 questions held out,
32.0 % errors.

| budget | trigger recall | trigger precision | disagreement recall | disagreement precision |
|---|---|---|---|---|
| 5 % | **0.110** | 0.700 | 0.090 | 0.569 |
| 10 % | **0.216** | 0.686 | 0.170 | 0.540 |
| 12 % | **0.259** | 0.682 | 0.205 | 0.540 |
| 15 % | **0.315** | 0.662 | 0.249 | 0.523 |
| 25 % | **0.501** | 0.633 | 0.405 | 0.513 |

The trigger beats the disagreement baseline at every budget on both splits, by roughly a
quarter to a third in relative recall, and by more in precision. It is a real gain and a
modest one. Note what the same tables say about the ceiling: at a 12 % budget the trigger
reaches about 28 % of the fast path's errors. Most errors stay unreachable inside any
budget worth serving, because most of them are not distinguishable from correct answers by
anything the fast path knows about itself.

Transfer holds but varies a lot by source, entirely with how separable that source's
errors are. Recall of errors at a 12 % budget, each source held out of fitting entirely:

| held-out source | questions | errors | trigger | disagreement |
|---|---|---|---|---|
| ext_bitext_support | 300 | 25 | **0.600** | 0.400 |
| ext_mnli_mismatched | 300 | 49 | **0.245** | 0.224 |
| ext_toxic_chat | 300 | 50 | **0.280** | 0.220 |
| ext_yelp_stars | 300 | 104 | 0.135 | **0.154** |
| banking77 | 200 | 13 | **0.692** | 0.615 |
| clinc | 200 | 18 | **0.556** | 0.444 |
| mmlu_pro | 200 | 95 | **0.211** | 0.137 |
| base_rates_synth | 100 | 11 | **0.273** | 0.000 |
| codereview | 320 | 171 | 0.140 | **0.175** |
| civil_comments | 250 | 189 | 0.095 | **0.111** |
| msmarco | 200 | 73 | 0.110 | 0.110 |

The pattern is consistent: where errors are rare (intent classification, 6-9 % error) the
trigger finds most of them; where errors are the common case (civil comments 76 %, code
review 53 %, Yelp 35 %) nothing ranks them, including the trigger. Sources on which the
model is already bad are not sources on which it knows it is bad.

The fitted weights put most mass on the fast path's own confidence (`max_prob` -0.40,
`entropy` +0.28, `margin` -0.27, `disagreement` +0.18) and then on shape (`log_question_cues`
+0.32, `is_noul` +0.24, `is_choice` -0.23, `log_state_sentences` +0.20). The shape features
earn their place, which the disagreement baseline is the direct test of, but confidence
does the heavier lifting.

## On the public benchmark items

`PERMS=2 THINK=768 ESCALATE=runs/escalator.json`, threshold 0.7990, Qwen3.5-4B frozen.
The stored threshold escalated 37 of the 231 items, 16.0 %.

| public items | 2 orders (baseline) | selective by disagreement | **+ the trigger** |
|---|---|---|---|
| easy | 1.000 | 1.000 | 1.000 (48/48) |
| standard | 0.917 | **0.958** | 0.944 (68/72) |
| hard | **0.685** | 0.658 | 0.676 (75/111) |
| hard ECE | 0.081 | 0.082 | **0.061** |
| probability fidelity | **0.740** | | 0.718 |
| calibration axis | 78.9 | 79.1 | **79.8** |
| escalations | 0 | 34 | 37 (16.0 %) |
| p50 / p95, hard items | 0.19 s / 0.8 s | 0.25 s / 42 s | 0.30 s / 42.4 s |

Against the baseline the trigger fixes 2 standard-tier items (one policy, one adequacy)
and breaks none. On the hard tier it fixes 4 and breaks 5.

Where it sent the reasoning, and how that turned out:

| tier | family | items | escalated | of those correct |
|---|---|---|---|---|
| hard | long_policy | 19 | 11 | 3 |
| hard | temporal_numeric | 15 | 9 | 6 |
| hard | multi_hop | 18 | 5 | 1 |
| hard | judge_hard | 17 | 4 | 3 |
| hard | probability | 10 | 2 | 2 |
| hard | tradeoff | 6 | 1 | 0 |
| hard | trap | 8 | 1 | 1 |
| standard | adequacy | 12 | 3 | 3 |
| standard | policy | 12 | 1 | 1 |

This is the result worth keeping. The targeting is right: the trigger concentrated on
long_policy (58 % of that family) and temporal_numeric (60 %), the two families VISION
named, and left the easy tier entirely alone. The item-level diff says reasoning then
repaid that on temporal/numeric, where three of the four hard-tier fixes live, and lost it
on long policy, where three of the five breaks live, plus one multi-hop.

So the gate improved and the thing behind the gate did not. A 768-token reasoning block
over a 3-4k-token policy document is worse than reading it once, reliably enough to cancel
the gains elsewhere. That is the same conclusion the disagreement gate reached in
[order-averaging.md](order-averaging.md), now with a sharper instrument and a named cause.

## Verdict

> **What happened next.** The cascade was removed the same day, and with it
> `reflex.escalate`: the serving path has one readout and no slow path for a gate to open
> ([../VISION.md](../VISION.md)). The verdict below is kept as it was written, about a
> configuration that no longer exists in the code.

* **Ship the trigger as the gate.** It dominates the disagreement threshold on both splits
  at every budget, and on the public items it is the best of the three configurations on
  hard-tier ECE (0.061, against 0.081 frozen and 0.082 by disagreement) and on the
  calibration axis (79.8). It keeps two thirds of the standard-tier gain the disagreement
  gate found while giving back one hard item instead of three.
* **Do not claim a hard-tier accuracy gain.** 0.676 against 0.685 is one item on 111 and is
  noise. VISION's hard ≥ 0.70 target is not met and this route does not reach it.
* **The next change is to the slow path, not the gate.** Two candidates the family table
  points at directly: make escalation conditional on family shape, never escalating long
  documents to a short reasoning budget; or give long-policy items a different slow path
  (a larger reasoning budget, or retrieval over the document) instead of the same 768
  tokens every other item gets.
* **The p95 stays 42 s**, so this remains a batch-mode configuration, not a serving one.
  A 16 % budget on 231 items cost about 25 minutes of wall clock.

Standing caveat, unchanged: we have consulted the public items throughout development, so
they are a development suite and not an independent test. The trigger itself was fitted and
selected only on the external and mix3 sets.
