# reflex: architecture and internals

A **System One decision model** in the style of TypeSafe's [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
rebuilt on top of a stock pretrained model (Qwen3.5-4B by default; Qwen3 and Qwen3-VL also work).

Give it a block of *state* (text, JSON, **images**) and a set of typed *questions*. It
evaluates every question in one parallel pass and returns **probability distributions
over the answers you supplied**. It never generates text, so it cannot produce a type
error or hallucinate an option you didn't list.

```json
POST /v1/systemone
{
  "state": {"ticket": "My payouts have failed three times this week..."},
  "questions": {
    "queue":    {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"payments": "payouts, refunds", "account": "login, 2FA", "other": null}},
    "escalate": {"type": "noul",   "instructions": "Should this be escalated to a manager?"},
    "urgency":  {"type": "score",  "instructions": "How urgent is this?",
                 "criteria": ["can wait a week", "handle today", "blocked right now"]}
  }
}
```
```json
{
  "answers": {
    "queue":    {"type": "choice", "choice": "payments", "probabilities": {"payments": 0.91, "account": 0.06, "other": 0.03}, "confidence": 0.78},
    "escalate": {"type": "noul",   "noul": 0.42},
    "urgency":  {"type": "score",  "score": 1.64, "probabilities": {"0": 0.08, "1": 0.20, "2": 0.72}, "legend": {...}, "confidence": 0.35}
  },
  "usage": {"input_tokens": 412, "state_tokens": 180, "question_tokens": 232, "state_cache_hit": false}
}
```

The request/response contract mirrors TypeSafe's `POST /v1/systemone` (three primitives:
`noul`, `choice`, `score`), so client code written for Jev works against a local reflex server.

## How it works

This is a reconstruction of the design inferred in
["Jev's architecture, unmasked"](https://archerhume.com/posts/jevs-architecture-unmasked/):
*shared state, isolated questions, direct probability readout*. Everything is prefill;
there is no decoding loop anywhere.

```
                       ┌──────────── one forward pass ────────────┐
 state ──► KV cache ──►│ q1 order A │ q1 order B │ q2 order A │ … │
   (encoded once,      │ ▲ a branch sees the state and itself only│
    LRU by hash)       └────────────┴────────────┴────────────┴───┘
                              │            │            │
                         logits@last  logits@last  logits@last
                              │            │            │
               restrict to the letter tokens of that branch
                              │            │            │
               temperature ─► softmax ─► average over the orders
```

One request is one forward pass. There is no decoding loop, no reasoning and no
escalation anywhere on this path, and `reflex-serve` has no flag that can make a request
slower than a forward pass ([VISION.md](VISION.md)).

1. **Shared state encoding** (`Engine.encode_state`). The state prefix (text, plus images
   through the vision encoder on VL models) is tokenised and run once; its cache is kept
   (LRU keyed by content hash). Repeat requests over the same state only pay for the
   questions ("prefix sharing").
2. **Isolated question branches** in one pass. Each branch is, to the model, exactly
   "state + this question" and never sees another branch. Two strategies, picked from
   the model config:
   * `packed` (attention-only backbones: Qwen3, Qwen3-VL). All branches are concatenated
     into a single sequence and run with a custom 4D attention mask (`engine.build_pack`):
     a branch token may attend to the state and to earlier tokens of *its own* branch.
     Position ids restart at `len(state)` per branch. No cache copies at all.
   * `batched` (hybrid backbones with linear-attention/recurrent layers: Qwen3.5). A mask
     cannot isolate branches inside a recurrent scan, so branches run as a right-padded
     batch against a batch-expanded copy of the state cache.

   `tests/test_engine_gpu.py` and `tests/test_engine_vl_gpu.py` check both strategies are
   numerically identical to running each question on its own (text and image state).
3. **Direct probability readout** (`reflex.readout`). At the last token of each branch we
   take the next-token logits, restrict them to that branch's label tokens,
   temperature-scale and softmax. That distribution is the answer. `score` reports the
   probability-weighted level; `confidence` is `1 - normalised entropy` (pluggable; the
   full distribution is always returned).

   The labels are **letters** (`A/B/C…`) for all three primitives. A yes/no question is
   presented as a lettered pair rather than read off the `Yes`/`No` tokens
   (`noul_readout: "letters"` in `prompt.DEFAULT_TEXTS`), because an answer word carries a
   stylistic pull that a letter does not. That change and the `# Evidence` / `# Criterion`
   / `# Options` headings are the two prompt edits that transferred on held-out external
   data ([results/frozen-vs-trained.md](results/frozen-vs-trained.md)); every piece of
   wording the model sees is a named component in `prompt.DEFAULT_TEXTS`, which is what
   `reflex-optimize` and the ensemble variants edit.
4. **Two option orders, averaged** (`prompt.distinct_orders`, `readout.merge_branches`).
   A choice or score question is rendered in `permutations` *distinct* option orders: the
   identity order first, then distinct shuffles. A yes/no question's second order is
   always the swap, so binary coverage is balanced. Each order is another branch in the
   same forward pass; the distributions are mapped back to semantic keys through
   `Branch.keys` and averaged. Orders come from a generator seeded by `(seed, qid)`, so
   adding an unrelated question cannot change another question's orders.

   `permutations: 2` is the `stable` configuration. It costs about 1.1x the latency and
   buys hard-tier accuracy 0.658 → 0.685 and roughly half the pooled external calibration
   error ([results/order-averaging.md](results/order-averaging.md)). A request may set
   `permutations` itself; `--permutations N` sets the default for requests that do not.
5. **Prompt ensembles and disagreement** (`reflex.ensemble`, optional, off by default).
   `--ensemble prompts/ensemble-v1.json` adds question-side wording variants, each an
   extra branch in the *same* pass. A variant may change the headings, the ask lines and
   the readout style but never the system prompt or the state heading, which is checked,
   so all variants share one state cache. `ensemble.disagreement` is the mean
   total-variation distance between a question's branches and their mean: 0 when every
   wording and order agrees, 1 when they scatter. Wording ensembles did not pay for
   themselves on quality ([results/order-averaging.md](results/order-averaging.md), "What
   did not help"); the machinery stays because the disagreement statistic is useful
   offline, and because option-order averaging runs through the same merge.
6. **Calibration** (`reflex.train.calibrate`, "RLCD-lite"). Jev is trained with
   *Reinforcement Learning for Calibrated Decisions*: reward = a proper scoring rule on
   the emitted distribution. When the model's output *is* the distribution, the expected
   reward is differentiable in closed form, so RL collapses to supervised minimisation of
   the scoring rule (NLL or Brier). We train LoRA adapters on the label-restricted logits
   through the same packed forward the server uses, then fit a post-hoc temperature.

What it is **not** (yet): a dedicated classification head (we read out label tokens, so
choice cardinality is capped at 26), an MoE backbone, or a batching server that packs
branches from *different* requests. See "Limits".

What the trainer is **for**, today: steps 1–5 describe a frozen model, and that is what
ships. Every LoRA mix trained here, and the distillation from a 27B teacher, bought
accuracy on data shaped like their training data and lost general judgement on long,
ambiguous inputs ([results/frozen-vs-trained.md](results/frozen-vs-trained.md),
[results/lora-distill-qwen3.5-4b.md](results/lora-distill-qwen3.5-4b.md)). So
`reflex-calibrate train` and `reflex-distill` are tools for fine-tuning on **your own
workload's labels**, where in-distribution is the distribution you care about, not a step
in the recommended setup. The post-hoc temperature is the part of the training stack that
helps everyone.

## What is served: the `stable` manifest

`serving/stable.json` is the configuration reflex recommends, as data: the base model, an
adapter (a hub id, or `null`), a calibration file, the prompt style, prompt-text overrides
and `permutations`, plus the gate numbers it was selected on. The git tag `stable` points
at the commit that file describes. `reflex-serve --stable` applies it and explicit flags
still win; `reflex.serving.load_stable()` returns it to other entrypoints.

Today that is the frozen `Qwen/Qwen3.5-4B`, no adapter, no calibration file, the default
markdown prompt and `permutations: 2`. [SERVING.md](SERVING.md) has the release gate a run
must pass before the tag moves.

## Backends

| backend | what runs the forward pass | when |
|---|---|---|
| `transformers`, strategy `packed` | this process, one sequence, 4D mask | attention-only backbones (Qwen3, Qwen3-VL) |
| `transformers`, strategy `batched` | this process, right-padded batch over a batch-expanded state cache | hybrid backbones with recurrent layers (Qwen3.5) |
| `sglang` | an SGLang server over HTTP | a deployment that already runs SGLang, and the 27B |

The strategy is picked from the model config; `--device cuda|mps|cpu` chooses where the
weights live, and Apple Silicon works through PyTorch MPS (`reflex.mps`).

`--backend sglang --sglang-url ...` (`reflex.backends.sglang`) keeps the whole prompt and
readout stack here and hands only the forward pass over: the prefix is sent once with
`max_new_tokens=1` to warm SGLang's radix cache, then one `/generate` per branch fires
concurrently with `return_logprob`, `logprob_start_len=-1` and `token_ids_logprob` set to
that branch's label ids. Isolation between questions is structural, one request each,
rather than a mask. The probabilities match the in-process engine to 0.004 at the median.
Images, adapters, ensembles and `--device` need the weights here and are rejected rather
than ignored. At 4B the in-process engine is about 2.2x faster; at 27B SGLang wins, and
the NVFP4 checkpoint wins everything except calibration
([results/sglang-backend.md](results/sglang-backend.md)).

## Reasoning is an offline tool, not a mode

`reflex.think` generates a bounded `<think>` block per branch and then reads the same
label logits after it. Nothing on the request path imports it, and `reflex-serve` has no
flag for it. It exists for two offline jobs: `reflex-distill label --think N`, where a
thinking teacher labels a corpus, and `reflex-calibrate eval --think N`, which measures
what reasoning would buy. A thinking readout costs 20–40 s per item and its p95 is
nothing like its p50, which is why it is not a serving mode
([VISION.md](VISION.md), [results/teachers-27b-and-4b-think.md](results/teachers-27b-and-4b-think.md)).
An escalation cascade in front of it was built, measured and removed
([results/escalation-trigger.md](results/escalation-trigger.md)).

## Quickstart

Hardware: any CUDA GPU, or an Apple Silicon GPU with `--device mps`. Qwen3.5-4B in bf16
needs ~9 GB; Qwen3.5-0.8B / Qwen3-0.6B run anywhere.

For MPS, use `main`: the current `stable` tag predates the server's `--device` option. The
[Apple Silicon setup in the README](../README.md#on-a-mac-apple-silicon) also installs the
PyTorch build that includes MPS support.

```bash
uv sync --extra dev
uv run pytest tests                       # mask unit tests + GPU equivalence tests (small models)

uv run python examples/support_ticket.py  # in-process demo, default Qwen/Qwen3.5-4B (see ../README.md for a gentler intro)
uv run python examples/image_triage.py photo.jpg --caption "a tabby cat on grass"
uv run reflex-serve --stable --port 8008     # what serving/stable.json recommends
uv run python examples/support_ticket.py --http

uv run reflex-serve --model Qwen/Qwen3.5-4B --permutations 2 --port 8008   # the same, spelled out
uv run reflex-serve --model Qwen/Qwen3.5-2B --device mps --dtype float16   # Apple Silicon
uv run reflex-serve --backend sglang --sglang-url http://127.0.0.1:30000 --model Qwen/Qwen3.5-4B
```

```python
from reflex import Engine, SystemOneRequest
engine = Engine.load("Qwen/Qwen3.5-4B")
resp = engine.answer(SystemOneRequest(state=..., questions={...}))
resp.answers["queue"].probabilities
```

## Images in the state

Any object of the form `{"type": "image", "source": ...}` anywhere in the state is sent to
the vision encoder; `source` is a file path, an `http(s)` URL or a base64 `data:` URI.
Images are part of the state prefix, so they are encoded once and cached like text, and
every question branch sees them. Questions stay typed and text-only.

```json
{
  "state": {"photo": {"type": "image", "source": "https://example.com/p.jpg"},
            "caption": "a tabby cat on grass"},
  "questions": {
    "subject":         {"type": "choice", "instructions": "Main subject?", "criteria": {"person": null, "animal": null, "food": null}},
    "matches_caption": {"type": "noul",   "instructions": "Does the `caption` accurately describe the photo?"},
    "quality":         {"type": "score",  "instructions": "Technical image quality?", "criteria": ["unusable", "poor", "acceptable", "good"]}
  }
}
```
Requires a vision-language checkpoint (Qwen3.5-*, Qwen3-VL-*); text-only checkpoints
return 422 for image state. `--max-image-pixels` bounds the token cost per image
(default 1 MP ≈ 1k tokens).

## Measure calibration (the MMLU experiment from the write-up)

```bash
uv run reflex-eval-mmlu --model Qwen/Qwen3.5-4B --n 1200
uv run reflex-eval-mmlu --model Qwen/Qwen3.5-4B --n 1200 --fit-temperature runs/calibration.json
uv run reflex-serve --calibration runs/calibration.json
```
Reports accuracy, ECE (15 bins), Brier, NLL and a reliability table; with
`--fit-temperature` fits T on half the items and reports the held-out half before/after.

Results so far (1200 random MMLU test items, seed 0, raw letter readout):

| model | acc | ECE raw | ECE after temperature (held-out half) | NLL raw → scaled |
|---|---|---|---|---|
| Qwen3.5-4B (no-think, batched) | 72.0% | 0.090 (mean conf 0.81) | **0.039** (T≈1.7) | 0.72 → 0.67 |
| Qwen3-8B (no-think, packed) | 70.8% | 0.264 (mean conf 0.97) | 0.061 (T≈9.7) | 3.64 → 0.75 |

Jev reports ECE 0.031 on 1200 MMLU items. Qwen3.5-4B's raw readout is already close to
honest and one temperature brings it within noise of Jev's number; Qwen3-8B is grossly
over-confident and needs a large temperature. The RLCD-lite trainer below is there to
close the rest with labelled data from your own domain.

Latency on a GB10 (DGX Spark), Qwen3.5-4B bf16, state cached: ~100 ms for 4 text
questions, ~140 ms for 5 questions over a 1 MP image. First call pays ~20 s of Triton
kernel compilation.

## Train for calibrated decisions (RLCD-lite)

Nothing below is part of the recommended setup: the frozen model is what ships, and
every adapter trained here was rejected on transfer
([results/README.md](results/README.md)). This is the path for fine-tuning on your own
workload's labels.

```bash
uv run reflex-calibrate make-mmlu --out runs/mmlu_val.jsonl --split validation
uv run reflex-calibrate make-mmlu --out runs/mmlu_dev.jsonl --split dev
uv run reflex-calibrate train --model Qwen/Qwen3.5-4B --data runs/mmlu_val.jsonl \
    --val runs/mmlu_dev.jsonl --out runs/lora-mmlu --loss nll --permutations 2
uv run reflex-serve --model Qwen/Qwen3.5-4B --adapter runs/lora-mmlu \
    --calibration runs/lora-mmlu/calibration.json
```
Training data is JSONL of `{state, questions, labels, source}` (see `reflex/train/data.py`);
labels may be hard or soft distributions, and the loss is the cross-entropy / Brier
score against that distribution. `reflex/train/recipes.py` maps eight public datasets
(banking77, clinc, MMLU-Pro, civil_comments with soft toxicity labels, HaluEval,
MS MARCO, HelpSteer2, github-codereview) onto the three primitives; `reflex-data mix`
samples them into train/eval files and the trainer reports per source. `--permutations N`
shuffles option order as augmentation, which also attacks letter-position bias. `--full`
trains all weights instead of LoRA adapters.

## Browser version (`docs/`)

**What it demonstrates.** The browser page runs Qwen3.5-0.8B, which is the mechanism, not
the served quality: frozen below 4B the readout is not a usable judge (the 0.8B is near
chance on every external set, [results/weight-classes.md](results/weight-classes.md)).
Read it as a working model of the design you can watch run on your own GPU.

`docs/index.html` + `docs/app.js` + `docs/reflex.js` (+ `docs/pr.js`, the PR triage
port of `examples/pr_review.py`) run the same design on
[transformers.js](https://github.com/huggingface/transformers.js) with WebGPU and
`onnx-community/Qwen3.5-0.8B-ONNX-OPT` (q4f16 decoder, fp16 vision encoder, ~650 MB;
`?dtype=q4` / `?dtype=fp16` and `?model=` URL overrides exist for A/B tests).
`reflex.js` mirrors `prompt.py` + `readout.py`: same ChatML prefix, same option labels,
same restricted-softmax readout via `model.forward` (no `generate`). All questions of a
request run in one of two ways, picked automatically. **Cold state**: one left-padded
batched forward where every row re-reads the state, `num_logits_to_keep = 1` so only each
row's last position goes through the 248k-vocab output projection; then the state prefix
(text and image) is encoded in the background and its decoder cache kept in a small LRU.
**Warm state**: each question runs alone as a short continuation of that cache (only the
branch tokens). The ONNX `GroupQueryAttention` op refuses a batched multi-token
continuation from a cache (shared check in `group_query_attention_helper.h`, used by the
WebGPU kernel too), so the Python engine's batch-expanded cache is impossible here; a
per-question continuation only wins once the state is cached, hence the hybrid.
Measured on one laptop WebGPU, ticket preset (4 text questions, 840 tokens re-read):
q4 848 ms → q4f16 604 ms → + single-position logits 381 ms (cold path); warm path 260
tokens, 300 ms. Answers from both paths match the naive per-question forward to four
decimals. `answer(req, { share: true|false, batch: false })` forces a path. The batched
cold pass does about twice the token work of the shared path (every row re-reads the
state), so on slow or thermally limited GPUs the page's "low power" policy uses the
shared path for cold states too; it switches on automatically when a batched pass
measures under 1,200 tokens/s (`engine.setPolicy({ lowPower })`). Differences from the Python engine:
no state-cache sharing across requests, one image per request, no calibration file (a
temperature slider instead). `answer(req, { batch: false })` keeps the one-forward-per-
question path for comparison. Served by GitHub Pages from `docs/`; the model weights are
fetched from the Hugging Face hub and cached by the browser.

## Request extensions

* `permutations: N` (1–8): average the readout over N *distinct* option orders, yes/no
  questions included. Costs N branches per question, all in the same pass; halves the
  position bias of the letter readout. The server default is set by `--permutations` (2
  under `--stable`), and the request wins when it sets the field.

## Limits

| | Jev | reflex (today) |
|---|---|---|
| choice cardinality | 255 | 26 (single-token letter labels) |
| score levels | 2–10 | 2–10 |
| backbone | undisclosed (~10B active, MoE suspected) | any HF causal / image-text LM; Qwen3.5-4B default (hybrid linear attention, native vision) |
| images | ? | yes, in state, cached with the prefix (VL checkpoints) |
| questions per request | thousands | limited by `--max-pack-tokens` (packed: dense mask is O(T²); batched: cache copy × branches) |
| cross-request batching | yes | no (one request per forward, state cache shared) |

## Layout

```
src/reflex/schema.py        request/response contract (pydantic)
src/reflex/prompt.py        state prefix + question branch rendering, label tokens
src/reflex/engine.py        packed / batched strategies, state cache, readout forward
src/reflex/images.py        image refs in state -> placeholders + pixels
src/reflex/readout.py       temperature, confidence, typed answers, order merging
src/reflex/ensemble.py      prompt-variant branches, disagreement statistic
src/reflex/serving.py       serving/stable.json -> Engine.load kwargs
src/reflex/server.py        FastAPI  POST /v1/systemone, the CLI flags
src/reflex/backends/        sglang.py: the same readout off an SGLang server
src/reflex/mps.py           Apple Silicon helpers for --device mps
src/reflex/client.py        tiny httpx client
src/reflex/think.py         offline thinking readout (never reachable from the server)
src/reflex/optimize_prompt.py  GEPA prompt-text search (rejected; prompts/gepa-v1.rejected.json)
src/reflex/publish.py       upload an adapter + calibration + card to the Hub
src/reflex/eval/            calibration metrics, MMLU experiment
src/reflex/train/           JSONL data, LoRA proper-scoring-rule trainer
src/reflex/distill/         corpus, questions, teacher labelling, mixing
serving/stable.json         the recommended configuration, as data
prompts/                    prompt-text overrides and ensemble variant sets
examples/support_ticket.py  the canonical demo
examples/image_triage.py    typed judgments about a photo
tests/                      mask unit tests, GPU equivalence tests
```

## Where the evidence is

[results/README.md](results/README.md) indexes every experiment in this repo in order,
with its verdict: what the readout work bought, why no adapter ships, why the serving
path has one mode, and what each backend costs.

## References

* TypeSafe, [Introducing System One models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
* TypeSafe docs: [API](https://docs.typesafe.ai/api), [primitives](https://docs.typesafe.ai/primitives), [confidence](https://docs.typesafe.ai/confidence)
* Archer Hume, [Jev's architecture, unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)
