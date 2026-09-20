# Reflex model quality across weight classes

Build the strongest validated Reflex configuration in each model weight class. Optimize
decision accuracy and probability quality for each size, and measure the memory, latency,
and compute required to achieve them. Jev is a reference system for the same decision
tasks; matching it is an experimental goal, not an established result.

This plan is based on the repository review on 2026-09-20: `main` at `21c95df` and
`calibration` at `0109bc3`. The experiments below are proposed work. The recorded results
are identified separately.

**Scope and comparison rules**

Start with five dense Qwen tiers. These are initial candidates, not proven winners within
their parameter classes. The shared harness should eventually allow other backbones.

| Weight class | Initial checkpoint | First training hypothesis to test |
|---|---|---|
| 0.8B | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Decision-focused distillation; compare full fine-tuning with LoRA on the same data. |
| 2B | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Broader verified decision supervision, counterfactual examples, and soft probability targets; compare full fine-tuning with LoRA. |
| 4B | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Improve the readout, then distill reasoning capabilities and test adapters covering the hybrid attention layers. |
| 9B | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Establish the missing baseline and reasoning headroom before choosing an adaptation recipe. |
| 27B | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Optimize a shipping Reflex configuration (direct readout) as well as an offline teacher, where reasoning budgets apply. |

Weight class means parameter count. Record precision and peak memory separately. For a
future MoE comparison, record both total and active parameters: a
[35B-A3B model](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) has a different weight-memory
requirement from a dense 3–4B model. Differences between model generations also prevent
treating this comparison as a controlled study of parameter count alone.

Maintain two evaluation tracks for every size:

- **Direct readout:** typed probabilities without autoregressive reasoning. Record the
  number of prompt variants and option orders, since those also consume compute.
- **Additional inference compute (offline only):** the best validated quality within
  explicit reasoning and ensemble budgets, with the same typed output contract. This track
  is a headroom measurement and a source of teacher labels. Reasoning never reaches the
  serving path, which is one fast forward pass ([VISION.md](VISION.md)); the tooling for it
  (`reflex.think`) is reachable only from `reflex-distill label` and `reflex-calibrate eval`.

Report a quality-versus-compute curve for each track. Larger models may supply offline
training labels without changing the student's inference weight class.

**What the existing experiments establish**

The repository has substantial comparative evidence for 4B and 27B, but it does not yet
establish the best configuration at every size.

| Recorded configuration | Public standard accuracy | Public hard accuracy |
|---|---:|---:|
| Frozen 4B, current default prompt | 91.7% | 65.8% |
| Distilled 4B, 27B targets | 91.7% | 61.3% |
| 27B, direct readout | 91.7% | 70.3% |
| 4B, 768-token reasoning budget | 100% | 70.3% |
| Jev 1.13.0, same public items | 98.6% | 73.0% |

Sources: [frozen-model comparison](results/jevbench-public.md),
[teacher experiments](https://github.com/kshetrajna12/reflex/blob/0109bc3/docs/results/teachers-27b-and-4b-think.md),
[distilled student](https://github.com/kshetrajna12/reflex/blob/0109bc3/docs/results/lora-distill-qwen3.5-4b.md),
and [JevBench per-item outcomes](https://github.com/fstandhartinger/jevbench/blob/main/results/v1.2/jevbench-v1.2-per-task.json).
These cover 72 public standard and 111 public hard items. Reflex has repeatedly consulted
these items during development; they are not an independent final test.

Reasoning exposes useful capabilities already present in the 4B: hard judging improves
from about 65% to 94%, while other families do not improve. The 27B has different
strengths, including probability fidelity and long-policy decisions. Select teachers by
verified task performance rather than size alone.

The failed adapters show that the recipes tried so far have not transferred reliably.
They do not establish that fine-tuning cannot help other sizes or better training
distributions. Keep a frozen baseline for every class throughout the program.

**Phase 1 — Make evaluation and serving agree**

Resolve these findings from the reviewed `calibration` snapshot before ranking new
configurations:

| Finding | Required change | Acceptance check |
|---|---|---|
| Selective reasoning invokes reasoning before checking disagreement. | Resolved by removal: the serving path has no reasoning mode at all. | No code path from a request reaches `reflex.think`. |
| Ensemble evaluation omits state length when applying the calibration head. | Share calibration inputs and probability merging between evaluation and serving. | Identical logits, state lengths, and configurations produce matching probabilities in both paths. |
| Scalar temperature fitting converts soft targets to argmax labels. | Fit scalar temperatures against full target distributions. Retain the head's existing soft-target objective. | A prediction already matching a soft target is not sharpened toward a one-hot target. |
| Binary `noul` questions are not permuted, and random permutations can duplicate an order. | Add balanced binary swaps and distinct option orders, preserving mappings back to semantic labels. | Order coverage is verified; adding an unrelated question does not change another question's sampled orders. |

Relevant code:
[engine readout](../src/reflex/engine.py),
[ensemble evaluation and fitting](https://github.com/kshetrajna12/reflex/blob/0109bc3/src/reflex/train/calibrate.py),
[temperature objective](../src/reflex/eval/metrics.py), and
[branch construction](../src/reflex/prompt.py).

Extend evaluation artifacts to retain stable example IDs, full targets, option mappings,
raw branch logits, merged probabilities, state lengths, and execution configuration.
Record generated token counts and whether reasoning completed within its budget.

Validate model-specific chat formatting and label tokenization. For each supported
backbone, compare cached/batched inference with a naive full-prompt forward. Test image
handling separately for configurations advertised as multimodal.

Deliverable: regression checks for the findings above and one reproducible evaluation
path that measures the behavior clients actually receive.

**Phase 2 — Establish quality and headroom at every size**

Build a configuration-driven runner across all five checkpoints. Give 9B early priority
because the gap between the existing 4B and 27B measurements is currently unmeasured.

For every size, evaluate four stages:

| Stage | Experiment | Purpose |
|---|---|---|
| Frozen baseline | Direct readout with no adapter and temperature 1. | Establish the starting point. |
| Optimized readout | Cross one/two wordings with one/two distinct option orders. | Measure gains available without weight updates. |
| Reasoning reference (offline) | Compare direct readout with bounded reasoning, initially 256, 768, and 2,048 generated tokens, through `reflex-calibrate eval --think N`. | Estimate useful headroom and which task families benefit, and pick teachers. Not a serving configuration. |
| Trained model | Evaluate the selected adaptation recipes against the frozen and optimized baselines. | Measure how much improvement survives training and transfers. |

The reasoning budgets are an initial sweep, not a claim about an optimal budget or a
model's maximum capability. Validate thinking-mode formatting and track incomplete or
looping generations. Do not assume more reasoning improves every task.

Select prompt and readout settings separately for each model on development data.
Fit calibration for the actual final inference pipeline. A calibration fitted for one
model, prompt, ensemble, reasoning mode, or precision is not automatically transferable.

Measure accuracy by primitive and task family; log loss and Brier score against the
available hard or soft targets; and distribution fidelity where reference distributions
exist. Keep top-label ECE as a diagnostic alongside these metrics. Measure error rate
versus the fraction of decisions accepted automatically.

Record memory, hardware, cold/warm state latency, and p50/p95 latency at specified loads.
Include different state lengths and question counts. Treat these measurements as the
cost of a quality result rather than silently mixing inference budgets.

Deliverable: per-size baseline and headroom tables, including negative results and
per-family changes. No adapter should be trained solely because the next size exists.

**Phase 3 — Build a shared, verified decision curriculum**

Create a reusable core dataset with additional slices targeting each student's measured
weaknesses. Increase corpus size only after a pilot shows useful transfer.

| Capability | Training examples to add |
|---|---|
| Instruction precision | Format requirements, exact constraints, routing rules, and minimally different compliant/noncompliant responses. |
| Multi-hop decisions | Questions requiring facts from several parts of the state, with plausible distractors and missing-evidence cases. |
| Policy reasoning | Interacting conditions, exceptions, precedence, and evidence placed throughout long documents. |
| Probability estimation | Verified frequency targets, explicit uncertainty, and annotator distributions where those match the intended task. |
| Judging | Good and subtly flawed candidate responses evaluated against explicit criteria. |
| Generalization | Diverse wording, option orders, document structures, and counterfactual pairs where one changed fact changes the answer. |

Long text alone is insufficient supervision for long-document reasoning. The fixed
long-document bank currently emphasizes audience and readability; add questions that
require the specific evidence interactions being targeted.

Use executable ground truth for tasks whose answers can be checked. Elsewhere, compare
candidate teachers against trusted labels and independently check uncertain examples.
Treat teacher probabilities as supervision to validate, not as ground truth by default.
Make the intended rubric explicit so a teacher's policy preferences do not silently
redefine a task.

Use the reasoning 4B and 27B as initial teacher candidates, selecting or combining them
only where their performance supports it. Preserve full soft distributions and label
provenance. Retain anchor examples from each student's own frozen model to measure and
limit loss of existing behavior.

Separate training, model-selection, calibration, and final-test data by source document
and task template. Keep related paraphrases and counterfactual variants together during
splitting. Exclude benchmark items and their derivatives from training. Treat repeatedly
consulted public benchmarks and external sets as development/regression suites.

Deliverable: a versioned curriculum with verified targets, difficulty slices, provenance,
and a fresh final test whose labels are not consulted during model selection.

**Phase 4 — Find an adaptation recipe for each weight class**

Start training pilots with 2B and 4B, where the comparison tests both a smaller student
and the model with the most existing evidence. Then expand to 0.8B, 9B, and 27B using the
measured headroom and failure families for each.

- **0.8B and 2B:** compare full fine-tuning with LoRA on the same verified curriculum.
  Measure basic decision competence, wording robustness, and retention of the frozen
  model's strengths. Test whether the extra trainable capacity helps before scaling data.
- **4B:** compare the current adapter targets with targets covering hybrid attention.
  Distill task families where verified teacher answers improve on the frozen student;
  explicitly measure gains and newly introduced errors.
- **9B:** let the baseline and reasoning comparison determine whether readout changes,
  distillation, or targeted adapters offer the most promising improvement.
- **27B:** optimize the direct readout as a deployable configuration, and reasoning budgets
  for its offline teacher role only. Test adaptation where independently verified labels
  expose a correctable weakness; its role as a teacher does not establish that its own
  decisions are optimal.

For the hybrid-layer ablation, the current LoRA target list contains `q_proj`, `k_proj`,
`v_proj`, and `o_proj`. Qwen3.5's linear-attention projections use names including
`in_proj_qkv` and `out_proj`, so those projections are not covered by the current list.
Verify actual module names for every checkpoint and record the adapted layers and
trainable parameter count. This is an experiment, not a confirmed explanation for the
earlier transfer failures. Sources: [trainer](../src/reflex/train/calibrate.py) and
[Qwen3.5 implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).

Use proper scoring losses against full targets. Compare dataset sizes and preservation
strengths before undertaking a large run. Repeat promising recipes across training seeds;
do not select from training loss or teacher agreement alone. Evaluate quantized deployment
artifacts separately and refit calibration if the actual inference distribution changes.

Deliverable: a winning recipe, or a documented frozen-model winner, for each class and
inference track. Preserve losing runs as evidence for future decisions.

**Phase 5 — Promote independently validated configurations**

Pre-register the primary metrics, practical improvement threshold, and acceptable
regressions before candidate selection. Use paired comparisons with uncertainty estimates,
grouped by source document or template where examples are related. Inspect critical
families so an aggregate gain cannot conceal a substantial regression.

A candidate is eligible for promotion when it improves the chosen quality objective over
its class baseline on fresh evaluation, meets the predeclared regression limits, preserves
the typed API contract, and has reproducible resource measurements. Report tradeoffs
between accuracy and probability quality explicitly. A few extra correct public items
alone do not establish general parity with Jev.

Evaluate the selected candidate on the reserved final test. If that test is subsequently
used to guide revisions, treat it as development data and obtain fresh final validation.
Keep Jev comparisons tied to the same task set and scoring rules, with the inference mode
and probability metrics stated alongside accuracy.

Add per-class release manifests under a proposed `serving/profiles/` directory, with
separate direct and additional-compute profiles where warranted. Each manifest should
pin the checkpoint revision, adapter, prompt, ensemble and order settings, reasoning
budget, precision, calibration, Reflex commit, and evaluation artifact versions. Extend
the existing [serving manifest mechanism](../src/reflex/serving.py) to load them.
These profiles and their loader are planned additions; the existing
[stable manifest](../serving/stable.json) remains the current recorded default.

**Initial implementation order**

1. Correct routing, calibration, and permutation behavior; add focused regression checks.
2. Implement the common runner and complete probability/target artifacts.
3. Establish all five frozen baselines, prioritizing the missing 9B comparison.
4. Run readout and reasoning sweeps on development data for each size.
5. Build and verify the targeted curriculum and its independent splits.
6. Pilot 2B and 4B training; expand the successful methods to the other tiers.
7. Validate deployment precision and promote eligible per-class profiles with their
   reproducible results.
