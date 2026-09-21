# Results: everything we tried, in the order we tried it

Every experiment in this repo has a record here, including the ones that lost. The
numbers in these documents are kept as written on the day of the run; where a
configuration has since been superseded the document says so at the top, and this index
gives the verdict in one line.

Two standing caveats apply to every entry. The JevBench public items have been consulted
throughout development, so they are a **development suite and not an independent test**;
official runs of both reflex entries are filed and queued (issues
[#3](https://github.com/fstandhartinger/jevbench/issues/3) for the 4B and
[#5](https://github.com/fstandhartinger/jevbench/issues/5) for the 27B). And the four
external sets (support intents, MNLI mismatched, toxic-chat, Yelp stars) were never
trained on, which is why they, not the benchmark, decide what ships.

**Where the answers ended up:** the served configuration is the frozen Qwen3.5-4B with
the default prompt at two option orders ([order-averaging](order-averaging.md),
[weight-classes](weight-classes.md)); no adapter has ever beaten it on transfer
([frozen-vs-trained](frozen-vs-trained.md)); and nothing on the request path reasons or
escalates ([../VISION.md](../VISION.md)).

## Chronological

| date | document | what was tried | verdict |
|---|---|---|---|
| 09-18 | [lora-mix-qwen3.5-4b](lora-mix-qwen3.5-4b.md) | First LoRA: one epoch on a mix of eight public datasets, proper scoring rule. | In-distribution accuracy 62.7 → 76.8 %, ECE 0.120 → 0.024. Looked like a win; the transfer controls later said otherwise. |
| 09-19 | [jevbench-public](jevbench-public.md) | First self-run of the JevBench public items (231 of 534), against every published system on the same items. | The frozen 4B reached hard 0.658 and the mix3 adapter 0.604, against Jev's 0.730. Misses are instruction precision on the standard tier and arithmetic on the hard tier. **Superseded** as a configuration record: the served readout now averages two orders. |
| 09-19 | [lora-mix2-qwen3.5-4b](lora-mix2-qwen3.5-4b.md) | Broader mix, two epochs, LoRA on MLP as well as attention. | 82.7 % in distribution, but over-confident on long exam-like inputs (MMLU-Pro ECE 0.20 → 0.30). Not shipped. |
| 09-19 | [lora-mix3-qwen3.5-4b](lora-mix3-qwen3.5-4b.md) | One epoch, attention-only, synthetic sources at a third of mix2's weight. | Best of the three adapters; published to the Hub as `kshetrajna12/reflex-qwen3.5-4b-lora`. **Superseded**: the model card marks it so. |
| 09-19 | [jevbench-request](jevbench-request.md) | Bench request filed for the mix3 adapter (issue #3). | Filed, then amended two days later to the frozen model. **Historical.** |
| 09-19 | [frozen-vs-trained](frozen-vs-trained.md) | The control that every later decision rests on: frozen model, same prompt, no adapter. | The frozen model wins on the hard tier and on three of four external sets. Two prompt changes (lettered yes/no readout, Evidence/Criterion framing) transferred; GEPA prompt optimisation did not. |
| 09-19 | [jevbench-amendment](jevbench-amendment.md) | Amendment to issue #3: evaluate the frozen model, not the adapter. | Accepted as the pending entry. **Historical** in its numbers: the entry now reads two option orders. |
| 09-19 | [lora-mix4-qwen3.5-4b](lora-mix4-qwen3.5-4b.md) | Soft-label-heavy mix, long documents, permutation augmentation, aimed directly at the transfer failure. | Rejected. Fixes in-distribution calibration, does not fix transfer; hard-tier ECE 0.189. |
| 09-20 | [teachers-27b-and-4b-think](teachers-27b-and-4b-think.md) | Two teacher candidates gated for distillation: Qwen3.8-27B, and the 4B allowed to reason. | The 27B was chosen as teacher (hard 0.703). The thinking 4B is near-perfect on judgement items and useless on long documents, at 20–40 s per item. **Offline only**: no serving path reaches it. |
| 09-20 | [lora-distill-qwen3.5-4b](lora-distill-qwen3.5-4b.md) | Distil the 27B's distributions into the 4B over 3.4k in-the-wild states, with an anchor slice. | Agreement with the teacher 0.65 → 0.85 with no collapse, the first adapter to move without breaking things. Hard tier still did not improve (0.613 vs 0.658). Rejected as a release. |
| 09-20 | [order-averaging](order-averaging.md) | Ask each question in two *distinct* option orders and average; yes/no questions both ways round. | **The win, and the current `stable`.** Hard 0.658 → 0.685, hard ECE 0.086 → 0.081, pooled external ECE 0.055 → 0.032, 1.1x latency. Wording ensembles, four orders and disagreement-gated reasoning all failed in the same document. |
| 09-20 | [jevbench-request-27b](jevbench-request-27b.md) | Second bench request (issue #5): the frozen Qwen3.8-27B at two orders. | Filed and queued. Standard 0.958, hard 0.766, hard ECE 0.061. |
| 09-20 | [weight-classes](weight-classes.md) | Frozen baselines at 0.8B / 2B / 4B / 9B / 27B, at one and two orders (Phase 2 of the plan). | Below 4B the frozen readout is not a usable judge, which is why the browser demo's 0.8B is a mechanism demo. The 9B is not a middle ground. The 27B at two orders is the quality configuration. |
| 09-20 | [escalation-trigger](escalation-trigger.md) | A fitted trigger deciding which questions the fast path should not be trusted on, routing those to a thinking readout. | The gate beat cross-order disagreement everywhere; what it routed to was not a better judge, and p95 was 42 s. **Historical**: the cascade and the `reflex.escalate` module were removed. |
| 09-20 | [serving-engines-research](serving-engines-research.md) | Prior art: has anyone run this readout on vLLM or SGLang? | Yes on SGLang (`ekzhang/openjev-sglang`); not on vLLM, which has no exact per-token-id log-probabilities and a prefix-cache cliff on hybrids. |
| 09-20 | [self-distill-orders](self-distill-orders.md) | Train a LoRA whose targets are the frozen model's *own* two-order distributions, so one pass learns order-invariance and nothing else. | Half transfers. Agreement with the two-order readout 0.83 → 0.94 and the best public-item calibration axis we have on the 4B (80.0, hard ECE 0.056) from a single pass; hard-tier accuracy falls to 0.640. Rejected as a release. |
| 09-20 | [sglang-backend](sglang-backend.md) | `--backend sglang`: the same prompt and readout computed by an SGLang server. | Equivalent probabilities (median difference 0.004), about 2.2x slower than the in-process engine at 4B, and the way to serve the 27B: NVFP4 on SGLang is ~208 ms warm but costs calibration (ECE 0.061 → 0.087). |

## Related documents outside this directory

* [../MODEL_QUALITY_PLAN.md](../MODEL_QUALITY_PLAN.md) — the program these experiments run inside, with a status line per phase.
* [../VISION.md](../VISION.md) — why the serving path is one fast pass and nothing else.
* [../DISTILLATION.md](../DISTILLATION.md) — the distillation pipeline, now a tool for your own workload.
* [../SERVING.md](../SERVING.md) — how to reproduce a configuration, and what the `stable` tag means.
