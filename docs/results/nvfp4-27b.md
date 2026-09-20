# The NVFP4 27B: a full latency grid, and a calibration refit (2026-09-20)

`docs/results/sglang-backend.md` left two things open about the NVFP4 checkpoint. Its
latency was quoted from a single microbenchmark (three questions, one client), and its
calibration was measured but not repaired: pooled external ECE 0.067 against bf16's 0.044,
public hard-tier ECE 0.087 against 0.061, with accuracy essentially unchanged.

This note closes both. First a latency table wide enough to serve a capacity question -
six client counts, three question counts, warm and cold states, measured over HTTP the way
a caller sees it. Then a per-primitive temperature refitted on the NVFP4 checkpoint's own
outputs, cross-validated across the four external sets, and re-measured on the public
benchmark items.

## Setup

| | |
|---|---|
| server box | second DGX Spark (GB10, 121 GB unified memory), shared with an unrelated voice stack (~31 GB) |
| client box | the primary DGX Spark, over the LAN |
| image | `lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b` (`0.0.0.dev1+g273d978be`) |
| checkpoint | `RadixArk/Qwen3.8-27B-NVFP4` (`modelopt_mixed`, fp8 KV cache) |
| reflex | `--backend sglang`, `--permutations 2` unless a row says otherwise |

SGLang:

    docker run -d --name reflex-nvfp4 --gpus all --ipc=host -p 30000:30000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b \
        python3 -m sglang.launch_server --model-path RadixArk/Qwen3.8-27B-NVFP4 \
        --host 0.0.0.0 --port 30000 --mem-fraction-static 0.45 \
        --mamba-radix-cache-strategy extra_buffer --context-length 16384 --trust-remote-code

reflex, on the same box:

    reflex-serve --backend sglang --sglang-url http://127.0.0.1:30000 \
        --model RadixArk/Qwen3.8-27B-NVFP4 --served-name reflex \
        --permutations 2 --host 0.0.0.0 --port 8010

and the measurement, from the client box:

    uv run python scripts/bench_latency.py --url http://<server>:8010 \
        --concurrency 1,2,4,8,16,32 --questions 1,3,10 --requests 48 \
        --state-tokens 150 --out grid.json

`scripts/bench_latency.py` builds a realistic request: a support ticket as the state and
`n` questions cycling noul / choice / score. **Warm** repeats one state, so SGLang's radix
cache holds its prefix; **cold** sends a fresh state per request. Every cell is 48
requests (at least 2 per client), preceded by two untimed warm-up requests. p50 and p95
are of the per-request wall time the client measures, HTTP included; requests/s and
questions/s are over the cell's own wall clock.

## Task 1: the latency grid

27B NVFP4 on SGLang, served through reflex at `--permutations 2`, a ~150-token state,
questions cycling noul / choice / score. Every number is measured over HTTP from the other
box.

### Warm: one state, repeated

**p50 latency (ms)**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 192 | 232 | 718 |
| 2 | 215 | 462 | 1263 |
| 4 | 267 | 767 | 2368 |
| 8 | 549 | 1311 | 7588 |
| 16 | 821 | 2620 | 15263 |
| 32 | 2637 | 8086 | 29919 |

**p95 latency (ms)**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 196 | 240 | 721 |
| 2 | 221 | 516 | 1832 |
| 4 | 915 | 790 | 2378 |
| 8 | 1063 | 1574 | 7791 |
| 16 | 1094 | 2892 | 15526 |
| 32 | 3083 | 9250 | 30823 |

**requests/s**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 5.20 | 4.24 | 1.39 |
| 2 | 9.29 | 4.62 | 1.50 |
| 4 | 12.03 | 5.61 | 1.71 |
| 8 | 13.71 | 5.72 | 1.04 |
| 16 | 16.16 | 5.72 | 1.04 |
| 32 | 10.06 | 3.49 | 1.04 |

**questions/s**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 5.20 | 12.73 | 13.92 |
| 2 | 9.29 | 13.85 | 15.01 |
| 4 | 12.03 | 16.83 | 17.06 |
| 8 | 13.71 | 17.17 | 10.44 |
| 16 | 16.16 | 17.15 | 10.42 |
| 32 | 10.06 | 10.46 | 10.41 |


### Cold: a fresh state every request

**p50 latency (ms)**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 423 | 600 | 1333 |
| 2 | 507 | 909 | 2239 |
| 4 | 729 | 1799 | 4508 |
| 8 | 1649 | 3800 | 9857 |
| 16 | 3242 | 8685 | 19506 |
| 32 | 8005 | 18414 | 48674 |

**p95 latency (ms)**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 426 | 605 | 1343 |
| 2 | 518 | 928 | 2302 |
| 4 | 736 | 11536 | 6575 |
| 8 | 2256 | 4680 | 13371 |
| 16 | 4199 | 9858 | 26301 |
| 32 | 8951 | 20725 | 57751 |

**requests/s**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 2.36 | 1.67 | 0.75 |
| 2 | 3.92 | 2.18 | 0.89 |
| 4 | 5.49 | 1.48 | 0.88 |
| 8 | 5.08 | 2.08 | 0.79 |
| 16 | 4.53 | 1.76 | 0.76 |
| 32 | 3.76 | 1.61 | 0.60 |

**questions/s**

| clients | 1 question | 3 questions | 10 questions |
|---|---|---|---|
| 1 | 2.36 | 5.00 | 7.49 |
| 2 | 3.92 | 6.55 | 8.90 |
| 4 | 5.49 | 4.45 | 8.80 |
| 8 | 5.08 | 6.23 | 7.85 |
| 16 | 4.53 | 5.29 | 7.60 |
| 32 | 3.76 | 4.82 | 5.99 |

### What the shape of that grid means

A reflex request is not one call to SGLang. It is one prefix call to warm the radix cache
(skipped when reflex has seen the state before) plus one `/generate` per branch, where
branches = questions x permutations. Ten questions at two orders is **21 HTTP calls**, and
they all go out at once.

SGLang on this configuration reports `max_running_requests=9` - the hybrid backbone's
mamba state pool, not the KV cache, is what bounds concurrency at
`--mem-fraction-static 0.45`. So:

* **A single 10-question request already saturates the server.** Its 21 branches queue
  behind 9 slots, which is why warm 10-question throughput sits at 10-17 questions/s from
  one client to thirty-two, and why p50 grows almost exactly linearly with client count
  above that point. Latency degrades; work done per second does not improve.
* **Throughput peaks around 17 questions/s warm** and is reached at 4 clients with 3
  questions each. Everything past that is queueing. Scale the *question count per request*
  to fill the server, not the client count.
* **Cold states cost roughly 2x warm** at one client (600 ms against 232 ms for three
  questions) and the gap widens under load, because every branch of a fresh state pays its
  own prefill rather than reading the radix cache.
* **p95 tracks p50 closely when the server is not saturated** and separates once it is.
  The one exception in the grid, 4 clients / 3 questions / cold at p95 11.5 s against a
  p50 of 1.8 s, is a single outlier request; the neighbouring cells do not show it.

The practical reading for a caller: **ask for a p50 under half a second and you get it
with up to 4 concurrent clients asking 1-3 questions each on a state the server has
seen.** Everything beyond that trades latency for nothing.

### A 2,000-token state

Three questions, `--permutations 2`, state rendered to ~1,970 tokens instead of ~160.

| clients | warm p50 | warm p95 | warm req/s | cold p50 | cold p95 | cold req/s |
|---|---|---|---|---|---|---|
| 1 | 257 | 262 | 3.89 | 1356* | 6298* | 0.34* |
| 2 | 518 | 611 | 4.62 | 2583 | 2598 | 0.77 |
| 4 | 905 | 919 | 4.85 | 5453 | 8071 | 0.72 |
| 8 | 1517 | 1825 | 4.94 | 11726 | 16824 | 0.64 |
| 16 | 4705 | 5185 | 3.18 | 49509 | 58171 | 0.30 |
| 32 | 9743 | 10221 | 3.16 | 113600 | 146183 | 0.24 |

**Warm, a 12x longer state costs almost nothing**: 257 ms against 232 ms at one client.
The radix cache means the long prefix is prefilled once and the seven branches reuse it.

**Cold, it costs everything**: 0.34 req/s against 1.67, and at 32 clients a cold
2,000-token state takes nearly two minutes at p50. Seven branches x 2,000 tokens of
prefill per request is the whole bill. If states are long and never repeat, this
deployment is prefill-bound and the question count is nearly free by comparison.

*The one-client cold cell is unstable. The sweep measured p50 6172 ms / 0.16 req/s; a
re-run of the same cell measured 1356 ms / 0.34 req/s with a p95 of 6298 ms, so individual
requests differ by 4x depending on whether the prefix survived in the radix cache. Both
runs are reported rather than the prettier one.

### The cost of the second option order

`--permutations 2` asks every question twice with the options in different orders and
averages (`docs/results/order-averaging.md`). It doubles the branch count, so it should
roughly halve throughput.

| | perm 1 p50 | perm 2 p50 | perm 1 req/s | perm 2 req/s |
|---|---|---|---|---|
| 1 client, 1 question, warm | 179 | 192 | 5.60 | 5.20 |
| 1 client, 3 questions, warm | 200 | 232 | 4.99 | 4.24 |
| 1 client, 10 questions, warm | 439 | 718 | 2.28 | 1.39 |
| 8 clients, 1 question, warm | 257 | 549 | 31.06 | 13.71 |
| 8 clients, 3 questions, warm | 782 | 1311 | 11.49 | 5.72 |
| 8 clients, 10 questions, warm | 2342 | 7588 | 3.41 | 1.04 |
| 1 client, 3 questions, cold | 475 | 600 | 2.11 | 1.67 |
| 8 clients, 3 questions, cold | 2121 | 3800 | 3.59 | 2.08 |

At one client with few questions the second order is nearly free (179 -> 192 ms), because
the branches run concurrently and the server is not busy. Under load it costs exactly what
it should: **8 clients, 3 questions goes from 11.5 to 5.7 requests/s, a 2.0x**, and the
10-question cell degrades worse than 2x (3.41 -> 1.04) because it pushes the server from
"busy" into "saturated". Order averaging is cheap when you have headroom and the first
thing to drop when you do not.

### The 4B reference

`reflex-serve --stable` on the primary box: `Qwen/Qwen3.5-4B`, the transformers backend,
`--permutations 2`, measured with the same script over the loopback.

| | 4B stable (transformers) | 27B NVFP4 (SGLang) |
|---|---|---|
| 1 client, 3 questions, warm p50 | **154 ms** | 232 ms |
| 1 client, 3 questions, cold p50 | **222 ms** | 600 ms |
| 8 clients, 3 questions, warm p50 | 1226 ms | **1311 ms** |
| 8 clients, 3 questions, warm req/s | **6.51** | 5.72 |
| 8 clients, 3 questions, cold req/s | **4.50** | 2.08 |

The 4B in-process engine is still the faster server at this shape: 1.5x on warm single
requests, 2.7x on cold, and it holds more throughput at eight clients. Its throughput is
also flat from 1 to 8 clients (6.51 against 6.51) because the engine serialises behind a
lock - the concurrency buys nothing but it costs nothing either. The 27B NVFP4 buys
accuracy, not speed, and the earlier "NVFP4 beats the transformers engine on every axis"
was a comparison against the **27B bf16** transformers engine, not against the 4B.

### The reflex HTTP hop is free

The same four cells measured in-process on the server box, calling `SGLangBackend.answer`
directly with no reflex server in the path:

| | in-process p50 | over HTTP p50 |
|---|---|---|
| 1 client, 3 questions, warm | 230 ms | 232 ms |
| 1 client, 3 questions, cold | 599 ms | 600 ms |
| 8 clients, 3 questions, warm | 1311 ms | 1311 ms |
| 8 clients, 3 questions, cold | 3790 ms | 3800 ms |

Within a millisecond or two everywhere. uvicorn, the pydantic validation and a LAN round
trip together cost less than 1 % of a request; all of the time is SGLang's.

## Task 2: refitting the calibration on the NVFP4 checkpoint

`docs/results/sglang-backend.md` measured NVFP4 as accurate but over-confident, and
recommended refitting a temperature on the NVFP4 checkpoint itself. This is that refit,
and the recommendation does not survive it.

### Method

The 1200 external items (`runs/external_eval.jsonl`) were answered through the NVFP4
SGLang backend at `--permutations 2`, keeping the **restricted label logits of every
branch** - 2,400 rows - rather than the merged probabilities, because a temperature
applies per branch before the orders are averaged. Targets are the question's labels
mapped onto each branch's own option order. One scalar per primitive, fitted by
`reflex.eval.metrics.fit_temperature` (soft-target NLL, golden section on log T), nothing
fancier. Scoring is `reflex.eval.metrics.report` on the merged, order-averaged
distributions, so the numbers are what a caller would see.

Uncalibrated, this run:

| source | n | acc | ECE | Brier | NLL | mean conf |
|---|---|---|---|---|---|---|
| bitext support intents (choice) | 300 | 0.9367 | 0.0295 | 0.0894 | 0.1590 | 0.9318 |
| MNLI mismatched (choice) | 300 | 0.8467 | 0.0697 | 0.2370 | 0.4711 | 0.8894 |
| toxic-chat (noul) | 300 | 0.8067 | 0.0787 | 0.2879 | 0.4626 | 0.8598 |
| Yelp stars (score) | 300 | 0.6733 | 0.1351 | 0.4594 | 0.8791 | 0.8075 |
| **pooled** | 1200 | **0.8158** | **0.0621** | 0.2684 | 0.4930 | 0.8721 |

Pooled ECE 0.062 against the 0.067 recorded yesterday for the same configuration, and
accuracy 0.8158 against 0.8167. SGLang is not batch-invariant, so a re-run moves the third
decimal; the reference bf16 figure remains 0.044.

### Leave-one-set-out says nothing useful, and that is itself a finding

The four external sets do not overlap in primitive: bitext and MNLI are both `choice`,
toxic-chat is the only `noul` source and Yelp stars the only `score` source. So holding a
set out usually removes **all** the training data for the primitive it uses, and the fit
falls back to T = 1.

| held-out set | temperatures fitted on the other three | ECE before | ECE after |
|---|---|---|---|
| bitext support | noul 2.17, choice **1.49**, score 1.76 | 0.0295 | **0.0623** |
| MNLI mismatched | noul 2.17, choice **1.00**, score 1.76 | 0.0697 | 0.0695 |
| toxic-chat | noul **1.00** (no other noul source) | 0.0787 | 0.0787 |
| Yelp stars | score **1.00** (no other score source) | 0.1351 | 0.1351 |

Only the two choice sources test transfer at all, and they fail it: the temperature MNLI
wants (1.49) more than doubles bitext's ECE, and the one bitext wants (1.00) leaves MNLI
where it started. **A temperature fitted on one distribution does not carry to another**,
which is the same lesson every earlier fitted calibration in this project taught.

### Within a distribution, the refit is real

Five-fold cross-validation inside each set - fit on 80 % of that set's questions, score the
held-out 20 % - separates "the temperature is genuine" from "300 items were memorised":

| source | ECE before | ECE after | Brier before | Brier after | NLL before | NLL after |
|---|---|---|---|---|---|---|
| bitext support | 0.0295 | **0.0270** | 0.0894 | 0.0896 | 0.1590 | 0.1601 |
| MNLI mismatched | 0.0697 | **0.0500** | 0.2370 | 0.2391 | 0.4711 | **0.4505** |
| toxic-chat | 0.0787 | **0.0356** | 0.2879 | **0.2719** | 0.4626 | **0.4240** |
| Yelp stars | 0.1351 | **0.0594** | 0.4594 | **0.4332** | 0.8791 | **0.7971** |

Accuracy is unchanged everywhere (Yelp moves by one item). So the over-confidence is a
real, fittable property of this checkpoint **on these distributions**: unseen Yelp items
go from 0.135 to 0.059 ECE, unseen toxic-chat items from 0.079 to 0.036.

Fitted on all four sets at once:

    calibration/nvfp4-27b.json   {"noul": 2.1712, "choice": 1.2712, "score": 1.7623}

| source | ECE before | ECE after |
|---|---|---|
| bitext support | 0.0295 | 0.0361 |
| MNLI mismatched | 0.0697 | 0.0551 |
| toxic-chat | 0.0787 | 0.0335 |
| Yelp stars | 0.1351 | 0.0546 |
| **pooled** | **0.0621** | **0.0258** |

Pooled ECE more than halves, and lands below the bf16 reference of 0.044. Brier 0.2684 ->
0.2575, NLL 0.4930 -> 0.4596, accuracy 0.8158 -> 0.8167. In-sample on three of four sets,
but the five-fold table above says most of that would survive on fresh items of the same
kind.

### Are these temperatures NVFP4-specific?

Mostly not, and this is where the earlier recommendation starts to break. Temperatures
fitted the same way on **bf16 27B** logits over the same 1200 items (from the teacher run,
`runs/teacher_27b_external.npz`: transformers backend, one option order) come out at
`{noul 2.43, choice 1.10, score 1.22}` and fix bf16's own ECE (0.0703 -> 0.0268). Applied
to the NVFP4 outputs:

| source | NVFP4 uncal | NVFP4 + bf16 T | NVFP4 + own T |
|---|---|---|---|
| bitext support | 0.0295 | 0.0323 | 0.0361 |
| MNLI mismatched | 0.0697 | 0.0513 | 0.0551 |
| toxic-chat | 0.0787 | 0.0506 | 0.0335 |
| Yelp stars | 0.1351 | **0.1024** | **0.0546** |
| **pooled** | 0.0621 | **0.0231** | **0.0258** |

The bf16 temperatures repair most of NVFP4's pooled over-confidence too. The honest
reading is not "bf16 temperatures do not transfer" but something narrower: **most of the
over-confidence is the readout's, not the quantization's.** Where NVFP4 genuinely differs
is `score`, the primitive that degraded most under quantization: Yelp's ECE needs T = 1.76
and only reaches 0.055 with it, while bf16's T = 1.22 leaves it at 0.102. The caveat on
this comparison is real: the bf16 logits come from a one-order transformers run with
earlier prompt wording, not from a matched permutations-2 SGLang run, which no longer
fits on the box.

### The public items reject the fit

`calibration/nvfp4-27b.json` was then served (`--calibration`) against the JevBench public
items, with an uncalibrated run of the same server on the same day as the control.

| tier | n | uncal acc | cal acc | uncal ECE | cal ECE |
|---|---|---|---|---|---|
| easy | 48 | 1.000 | 1.000 | **0.004** | 0.028 |
| standard | 72 | 0.944 | 0.944 | **0.064** | 0.106 |
| hard | 111 | 0.721 | 0.730 | **0.091** | 0.106 |

Schema validity 1.00 on both. Hard-tier probability fidelity improves, 0.808 -> 0.817, and
the composite calibration axis falls, 81.3 -> 80.2.

**Calibration error gets worse on every tier.** The mechanism is visible in the table: the
public items are much easier for this model than the external sets (accuracy 1.00 and 0.94
against 0.67-0.94), so a temperature fitted where the model is right 82 % of the time
makes it under-confident where it is right 97 % of the time. The one thing that improves
is the hard tier's *distributional* fidelity, because softened distributions sit closer to
the gold probabilities on the `probability` family - the same softening that hurts
top-label ECE.

For reference, the uncalibrated control also differs slightly from yesterday's recorded
run (standard 0.944 against 0.958, hard 0.721 against 0.730, hard ECE 0.091 against
0.087), which is one or two items and the SGLang non-determinism already noted.

### Verdict

**The refit works, and it is not shippable.** Those are both true and they are not in
tension:

1. NVFP4's over-confidence is real and fittable. One scalar per primitive halves pooled
   external ECE, and five-fold cross-validation inside each set says that holds on unseen
   items of the same distribution.
2. It does not transfer. Leave-one-set-out fails between the two `choice` sources, and the
   public benchmark items - a genuine out-of-distribution test with the file fixed in
   advance - come out **worse on all three tiers**.
3. Most of what the fit corrects is not quantization. bf16 temperatures recover nearly the
   same pooled improvement on NVFP4 outputs; only `score` shows a clearly NVFP4-specific
   gap.

So the recommendation in `docs/results/sglang-backend.md` - "serve NVFP4 only behind a
temperature refitted on the NVFP4 checkpoint" - is **too optimistic as written**. Refitting
on a distribution that is not your traffic buys nothing and can cost you. The file is
committed as `calibration/nvfp4-27b.json` so the fit is reproducible and so anyone whose
traffic looks like the external sets can use it, and `docs/SERVING.md` says how to pass
it, but nothing in this note supports serving it by default.

What remains true from the earlier note: NVFP4 is over-confident relative to bf16, the
argmax is unaffected, and a deployment that thresholds on probabilities should fit a
temperature **on its own labelled traffic** and re-measure. What this note adds is that
somebody else's labelled traffic will not do.

## Caveats

* One box, shared with an unrelated voice stack holding ~31 GB. SGLang ran at
  `--mem-fraction-static 0.45`, which is what gave `max_running_requests=9`; a dedicated
  box would give a larger state pool and push the saturation point out. The *shape* of the
  grid would not change, its ceiling would.
* 48 requests per cell. Enough to rank configurations, not enough to quote a service level,
  and visibly not enough to pin a p95 on the cells that show an outlier.
* The bf16 comparison uses previously recorded logits (one option order, transformers
  backend, earlier prompt wording). The 27B bf16 checkpoint needs 85 GB and was not run.
* The public-item control was re-run today, so the calibrated / uncalibrated comparison is
  same-session; the *external-set* bf16 reference (ECE 0.044) is not.
