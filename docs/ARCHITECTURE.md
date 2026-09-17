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
 state ──► KV cache ──►│ branch q1 │ branch q2 │ branch q3 │ ...  │
   (encoded once,      │  ▲ attends to state + itself only         │
    LRU by hash)       └──┴────────┴───────────┴───────────────────┘
                              │           │           │
                         logits@last  logits@last  logits@last
                              │           │           │
                      restrict to {A,B,C} {Yes,No}  {A,B,C}  ─► temperature ─► softmax
```

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
   take the next-token logits, restrict them to the label tokens (`A/B/C…` for choice and
   score, `Yes/No` for noul), temperature-scale and softmax. That distribution is the
   answer. `score` reports the probability-weighted level; `confidence` is
   `1 - normalised entropy` (pluggable; the full distribution is always returned).
4. **Calibration** (`reflex.train.calibrate`, "RLCD-lite"). Jev is trained with
   *Reinforcement Learning for Calibrated Decisions*: reward = a proper scoring rule on
   the emitted distribution. When the model's output *is* the distribution, the expected
   reward is differentiable in closed form, so RL collapses to supervised minimisation of
   the scoring rule (NLL or Brier). We train LoRA adapters on the label-restricted logits
   through the same packed forward the server uses, then fit a post-hoc temperature.

What it is **not** (yet): a dedicated classification head (we read out label tokens, so
choice cardinality is capped at 26), an MoE backbone, or a batching server that packs
branches from *different* requests. See "Limits".

## Quickstart

Hardware: any CUDA GPU. Qwen3.5-4B in bf16 needs ~9 GB; Qwen3.5-0.8B / Qwen3-0.6B run anywhere.

```bash
uv sync --extra dev
uv run pytest tests                       # mask unit tests + GPU equivalence tests (small models)

uv run python examples/support_ticket.py  # in-process demo, default Qwen/Qwen3.5-4B (see ../README.md for a gentler intro)
uv run python examples/image_triage.py photo.jpg --caption "a tabby cat on grass"
uv run reflex-serve --model Qwen/Qwen3.5-4B --port 8008
uv run python examples/support_ticket.py --http
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

```bash
uv run reflex-calibrate make-mmlu --out runs/mmlu_val.jsonl --split validation
uv run reflex-calibrate make-mmlu --out runs/mmlu_dev.jsonl --split dev
uv run reflex-calibrate train --model Qwen/Qwen3.5-4B --data runs/mmlu_val.jsonl \
    --val runs/mmlu_dev.jsonl --out runs/lora-mmlu --loss nll --permutations 2
uv run reflex-serve --model Qwen/Qwen3.5-4B --adapter runs/lora-mmlu \
    --calibration runs/lora-mmlu/calibration.json
```
Training data is JSONL of `{state, questions, labels}` (see `reflex/train/data.py`), so any
labelled routing / moderation / scoring dataset plugs in the same way. `--permutations N`
shuffles option order as augmentation, which also attacks letter-position bias.

## Browser version (`docs/`)

`docs/index.html` + `docs/app.js` + `docs/reflex.js` run the same design on
[transformers.js](https://github.com/huggingface/transformers.js) with WebGPU and
`onnx-community/Qwen3.5-0.8B-ONNX-OPT` (q4 decoder, fp16 vision encoder, ~650 MB).
`reflex.js` mirrors `prompt.py` + `readout.py`: same ChatML prefix, same option labels,
same restricted-softmax readout via a single `model.forward` (no `generate`). Differences
from the Python engine: one forward per question (no branch packing or cache sharing, the
ONNX graph builds its own causal mask), one image per request, no calibration file (a
temperature slider instead). Served by GitHub Pages from `docs/`; the model weights are
fetched from the Hugging Face hub and cached by the browser.

## Request extensions

* `permutations: N` (1–8): average a choice/score readout over N shuffled option orders.
  Costs N branches per question; reduces position bias of the letter readout.

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
src/reflex/readout.py       temperature, confidence, typed answers
src/reflex/server.py        FastAPI  POST /v1/systemone
src/reflex/client.py        tiny httpx client
src/reflex/eval/            calibration metrics, MMLU experiment
src/reflex/train/           JSONL data, LoRA proper-scoring-rule trainer
examples/support_ticket.py  the canonical demo
examples/image_triage.py    typed judgments about a photo
tests/                      mask unit tests, GPU equivalence tests
```

## References

* TypeSafe, [Introducing System One models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
* TypeSafe docs: [API](https://docs.typesafe.ai/api), [primitives](https://docs.typesafe.ai/primitives), [confidence](https://docs.typesafe.ai/confidence)
* Archer Hume, [Jev's architecture, unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/)
