# One batched `/generate` per request (2026-09-20)

The SGLang backend used to send one `/generate` per branch. A three-question request at
two orders was six calls plus a prefix warm-up; a ten-question one was twenty-one. The
latency grid in `docs/results/nvfp4-27b.md` grows steeply with question count - 192 ms for
one question, 232 for three, 718 for ten on the 27B NVFP4 - while the hosted Jev is flat at
115 / 143 / 129 ms. The obvious suspect was the fan-out: twenty-one HTTP round trips and
twenty-one scheduler admissions where a fleet needs one call.

SGLang's native `/generate` takes a batch, and it takes every field reflex needs per item:
`input_ids` as a list of lists, with `sampling_params`, `return_logprob`,
`logprob_start_len`, `top_logprobs_num`, `token_ids_logprob` and `rid` as lists of the same
length. `token_ids_logprob` really is per item - `io_struct.py` hands `token_ids_logprob[i]`
to item *i* - so each branch asks for its own label ids and no union is needed. The backend
now sends one call per request and the per-branch path is gone.

**The round trip was never the cost, and the change is still worth about 2x on the 27B.**
Those two sentences are not in tension, and the difference between them is the fan-out
bound. Matched call for call, batched and per-branch are the same speed everywhere. But
`--sglang-concurrency 8` used to cap *branches*, which throttled any request wider than
eight branches below what the server would have run; it now caps *calls*, which is one per
request, so the same default stops throttling. On the 27B, where SGLang runs 22 requests at
a time, that is 658 ms to 329 ms for ten questions and 5.85 to 9.33 requests/s at eight
clients. The batching itself contributes nothing; it makes the bound mean the right thing.

## What was measured

| | |
|---|---|
| box 1 | primary DGX Spark (GB10, 121 GB unified memory), `Qwen/Qwen3.5-4B`, `--mem-fraction-static 0.15`, `--mamba-ssm-dtype bfloat16`; SGLang reports `max_running_requests=3` |
| box 2 | second DGX Spark, `RadixArk/Qwen3.8-27B-NVFP4`, `--mem-fraction-static 0.45`, `--mamba-ssm-dtype bfloat16`, `--context-length 16384`; `max_running_requests=22`; the unrelated voice stack (~30 GB) resident |
| image | `lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b` |
| reflex | `reflex-serve --backend sglang --permutations 2`, default `--sglang-concurrency 8` |
| client | `scripts/bench_latency.py`, 24 requests per cell after two untimed warm-ups; box 2 measured over the LAN from box 1, box 1 over the loopback |

"per-branch" is `origin/main` (d603dea), "batched" is this branch. Each pair ran against the
same SGLang process, back to back, with the same seed, so the cold states are the same
states. The hosted-Jev column is `docs/results/jev-latency-probe.md`, a fleet over the
public internet, for scale rather than for comparison.

## The 27B NVFP4, a ~160-token state

**p50 latency**

| cell | per-branch | batched | Jev (hosted) |
|---|---|---|---|
| 1 client, 1 question, warm | 187 ms | 187 ms | 115 ms |
| 1 client, 1 question, cold | 409 ms | 408 ms | 108 ms |
| 1 client, 3 questions, warm | 213 ms | 213 ms | 143 ms |
| 1 client, 3 questions, cold | 563 ms | 562 ms | 111 ms |
| 1 client, 10 questions, warm | 658 ms | **329 ms** | 129 ms |
| 1 client, 10 questions, cold | 1180 ms | 1251 ms | 142 ms |
| 8 clients, 3 questions, warm | 1366 ms | **695 ms** | - |
| 8 clients, 3 questions, cold | 3134 ms | 3097 ms | 149 ms |

**requests/s**

| cell | per-branch | batched | Jev (hosted) |
|---|---|---|---|
| 1 client, 10 questions, warm | 1.49 | **3.03** | - |
| 8 clients, 3 questions, warm | 5.85 | **9.33** | - |
| 8 clients, 3 questions, cold | 2.54 | 2.55 | 43.7 |

Every narrow cell is a tie to within a millisecond. The two that move are the two where a
request wants more than eight branches at once: ten questions at two orders is twenty, and
eight clients asking three questions each is forty-eight.

## The 2,000-token cold state

Three questions, two orders, a fresh ~1,620-token state per request.

| | per-branch | batched |
|---|---|---|
| 1 client, p50 | 1089 / 1090 ms | 1086 / 1088 ms |
| 8 clients, p50 | 9244 ms | 8535 ms |
| 8 clients, req/s | 0.85 | 0.90 |

A tie. The one-client row is two clean re-runs each, because the first sweep produced
1081 ms for per-branch and 5694 ms for batched, and that gap is not the code. It is the
bimodal prefix-reuse collapse `docs/results/nvfp4-27b.md` diagnoses: eight cold requests in
a row through the batched path measured 1000, 5852, 1068 ms, and SGLang's log says which
mode each landed in.

    seq=1 new=1617 cached=0       <- the prefix warm-up
    seq=6 new=544  cached=9600    <- fast: six branches reusing ~1,600 tokens each
    seq=6 new=8192 cached=384     <- slow: six branches re-prefilling the whole state

With distinct fresh states and no collapse, both paths sit at 1085 ± 5 ms over eight
repetitions. The cliff belongs to the mamba state cache and both paths fall off it equally.

## The 4B, a ~160-token state

Box 1 is a deliberately starved server: `--mem-fraction-static 0.15` leaves 18 mamba slots
and `max_running_requests=3`.

| cell | per-branch | batched |
|---|---|---|
| 1 client, 1 question, warm | 85 ms | 85 ms |
| 1 client, 1 question, cold | 203 ms | 197 ms |
| 1 client, 3 questions, warm | 175 ms | 174 ms |
| 1 client, 3 questions, cold | 318 ms | 322 ms |
| 1 client, 10 questions, warm | 674 ms | 694 ms |
| 1 client, 10 questions, cold | 806 ms | 836 ms |
| 8 clients, 3 questions, warm | 1402 ms / 5.66 req/s | 1405 ms / 5.67 req/s |
| 8 clients, 3 questions, cold | 2126 ms / 3.58 req/s | 3745 ms / 2.10 req/s |

Ties except the last row, which is the same bound story with the sign flipped. Three
running slots cannot use forty-eight admitted branches, so the extra queue depth buys
nothing and costs prefix reuse on cold states, exactly as at 2,000 tokens. Lowering the
bound to match recovers it:

| 8 clients, 3 questions, cold | p50 | req/s |
|---|---|---|
| per-branch, bound 8 (8 branches in flight) | 2126 ms | 3.58 |
| batched, bound 8 (8 requests = 48 branches) | 3745 ms | 2.10 |
| batched, bound 2 (2 requests = 12 branches) | 2260 ms | 3.25 |

So the default of 8 is right for a server sized like box 2 and too generous for one sized
like box 1. It is one flag, and it now means something a caller can reason about: how many
requests reflex lets through at once.

## It is the bound, not the batch

Two checks, both on the 27B, warm, ~160-token state.

**Unbind the old path and it catches up.** `--sglang-concurrency 64` on `origin/main`:

| cell | per-branch, bound 8 | per-branch, bound 64 | batched, bound 8 |
|---|---|---|---|
| 1 client, 10 questions | 658 ms / 1.49 req/s | 328 ms / 2.99 req/s | 329 ms / 3.03 req/s |
| 8 clients, 3 questions | 1366 ms / 5.85 req/s | 904 ms / 7.76 req/s | 695 ms / 9.33 req/s |

The ten-question row is an exact tie once the branches are allowed out, which is the whole
claim. The eight-client row still favours batching by about 20 %, the one place in this
note where one envelope beats many; with forty-eight branches arriving as eight batched
calls rather than forty-eight connections there is less for the tokenizer process to do.

**Strip reflex out entirely.** Raw `/generate` against the 4B with a warm prefix, one
client, nothing else on the server, thirty repetitions: the same *n* branch prompts sent as
one batched call, or as *n* concurrent calls.

| items | batched p50 | one call each p50 |
|---|---|---|
| 1 | 91.3 ms | 90.3 ms |
| 6 | 176.6 ms | 177.2 ms |
| 20 | 614.1 ms | 616.4 ms |

Identical at every width. Twenty branches cost 614 ms because twenty branches are twenty
forward passes on a server that runs three at a time, not because they arrived in twenty
envelopes.

## The prefix warm-up earns its round trip, and only on long states

Before the branches go out, reflex sends the prefix alone with `max_new_tokens=1` to put it
in the radix cache, and skips that call on a state it has seen. A batch whose items share a
prefix might make it redundant, so measure it. The 4B, in-process, cold states only (a warm
state skips the warm-up either way):

| cold cell, p50 | with warm-up | without |
|---|---|---|
| 160-token state, 1 client, 3 questions | 325 ms | **257 ms** |
| 160-token state, 1 client, 10 questions | 824 ms | **773 ms** |
| 160-token state, 8 clients, 3 questions | 2194 ms | **2060 ms** |
| 160-token state, 8 clients, 10 questions | 6002 ms | **5671 ms** |
| 2,000-token state, 1 client, 3 questions | **510 ms** | 1529 ms |
| 2,000-token state, 8 clients, 3 questions | **11262 ms** | 12223 ms |

The items of a batch are admitted together, so nothing serialises them into "the first
prefills, the rest extend": on a cold state every branch prefills the whole prefix unless
something already put it in the cache. At 160 tokens that duplication is cheaper than an
extra round trip and dropping the warm-up wins about 20 %. At 2,000 tokens it is six full
prefills against one, and dropping it costs 3x. Long states are the case the warm-up exists
for, so it stays.

## Equivalence

52 items spanning all four external sets (`runs/external_eval.jsonl`, every 23rd row),
`--permutations 2`, answered through the 4B on the same server by both code paths:

| | |
|---|---|
| max abs. probability difference | **0.00036** |
| p50 / p95 difference | 0.000000 / 0.000000 |
| top-label agreement | 39/39 |
| answers over the 0.01 tolerance | 0 |

Fifty-one of the fifty-two are bit-identical and the fifty-second moves by three
ten-thousandths. SGLang is not batch-invariant (`docs/results/sglang-backend.md`), so an
exact tie was not owed to us; the batch shapes simply come out the same when the same
branches run together either way. `tests/test_sglang_backend.py` also passes unchanged
against the transformers reference, including the question-isolation test.

## Verdict

Ship it. One reflex request is one `/generate` call.

* **Fewer things to lose.** One connection instead of twenty-one. The dropped connection
  that killed one request in a hundred thousand in `docs/results/nvfp4-27b.md` had twenty
  siblings to be dropped from; now it has none.
* **`--sglang-concurrency` means requests.** That is what makes the 27B numbers move, and
  it is also why box 1 wants a lower value than box 2. Set it from the server's
  `max_running_requests` divided by the branches a typical request carries.
* **Batching itself is free, not fast.** Every matched-bound cell in this note is a tie,
  and the raw `/generate` microbenchmark is a tie at 1, 6 and 20 items.
* **The question-count curve is compute.** Ten questions at two orders is twenty forward
  passes on one GB10. Jev's flat 129 ms is a fleet with headroom, and no amount of request
  packing turns one box into one. To go faster, cut branches (`--permutations 1`) or raise
  `max_running_requests` (`--mamba-ssm-dtype bfloat16`, a larger `--mem-fraction-static`),
  both measured in `docs/results/nvfp4-27b.md`.
* **The prefix warm-up stays**, at about 65 ms on a cold short state and a second saved on
  a cold long one.

## Caveats

* Box 2 shares a GB10 with a voice stack, and `--mem-fraction-static 0.45` is what fits
  beside it. A server with the whole box would run more than 22 requests at a time and the
  bound would matter differently.
* Cells are 24 requests, not 48 as in the `nvfp4-27b.md` grid, so p95 is a coarser estimate
  here. The p50s of the unchanged cells agree with that grid's bf16-SSM rows to a few
  percent, which is the cross-check that matters.
* The 2,000-token cold cell is bimodal on both paths. Nothing here fixes that, and a single
  measurement of it means little.

## Reproducing

    # server
    docker run -d --name reflex-sglang --gpus all --ipc=host -p 30000:30000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b \
        python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-4B --host 0.0.0.0 \
        --port 30000 --mem-fraction-static 0.15 --mamba-ssm-dtype bfloat16

    reflex-serve --backend sglang --sglang-url http://127.0.0.1:30000 \
        --model Qwen/Qwen3.5-4B --permutations 2 --host 0.0.0.0 --port 8010

    # client
    uv run python scripts/bench_latency.py --url http://<host>:8010 \
        --concurrency 1 --questions 1,3,10 --requests 24 --state-tokens 150 \
        --permutations 2 --out grid.json

`tests/test_sglang_fanout.py` holds the wire shape without a GPU, weights or a network: one
POST per reflex request, every per-item field the right length, the bound respected and
reached, items matched by `rid` rather than by position, and a short batch or a missing
label id raised rather than answered from the wrong row.
