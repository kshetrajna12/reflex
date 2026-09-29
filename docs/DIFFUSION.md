# Experimental DiffusionGemma backend

`reflex-serve --backend diffusion` exposes the same `POST /v1/systemone` contract
through a DiffusionGemma server. Reflex sends structured, read-only canvas requests
and converts selected-label log probabilities into `noul`, `choice`, and `score`
answers. This backend has its own prompt and identity calibration; the Qwen
adapter, prompt, and calibration flags do not apply.

The upstream server must implement the `diffusion_*` fields in `vllm_xargs`,
`logprob_token_ids`, and `return_tokens_as_token_ids`. Constrained readout also
requires upstream support for `diffusion_constrained`. Configure its model context
and selected-logprob limit to cover the corresponding Reflex limits. For example,
with a compatible server already listening on port 8000:

```sh
uv run reflex-serve --backend diffusion \
  --model google/diffusiongemma-26B-A4B-it \
  --diffusion-url http://127.0.0.1:8000 \
  --diffusion-mode joint --permutations 2 \
  --diffusion-steps 1 --diffusion-constrained \
  --diffusion-prompt-layout user_state_first \
  --diffusion-canvas 128 --max-pack-tokens 16384 \
  --diffusion-max-selected-label-ids 256 \
  --diffusion-split-overflow
```

The example uses two **separate** option-order reads, one noise initialization per
read, and a numbered answer canvas. `user_state_first` places the state before the
question schema in one user message. The default `system_questions_first` retains
the earlier system-question/user-state layout. Prompt layout can change model
answers; choose it by evaluating your own cases. Both layouts report actual
upstream prompt tokens and check the final tokenized prompt against the context
limit before sending any reads.

`--diffusion-request-concurrency` bounds complete requests admitted for
compilation and execution (default 8). It is separate from
`--diffusion-concurrency`, which bounds simultaneous upstream calls.

The API schema accepts up to 255 choice options, though a diffusion request can
still be refused if its tokenizer cannot produce enough distinct one-token answer
codes, the canvas or context is too small, or the selected-label limit is lower
than the choice width. Up to 64 questions are accepted, and states must be text or
JSON without image references. `--diffusion-split-overflow` splits a joint read
that exceeds the canvas, context, or combined selected-label limit, while keeping
every question and option order; it cannot rescue a single question that exceeds
a limit. Splitting changes which questions share a canvas, so it may change their
answers. The Qwen and SGLang backends continue to refuse choices above 26 options.

The diffusion probabilities are uncalibrated. The CPU tests verify prompt layout,
capacity refusals, token selection, response parsing, and typed answers against a
fake upstream server; they do not establish model quality or serving latency.
The state-first message transformation was also checked offline with the pinned
tokenizer on 524 local requests, including exact answer slots and seed canvases.
The public backend now uses a newer JSON question encoding than the earlier GPU
quality validation, so those quality results do not transfer to this prompt.
