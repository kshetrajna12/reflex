# One batched `/generate` per request (2026-09-20)

The SGLang backend used to send one `/generate` per branch. A three-question request at
two orders was six calls plus a prefix warm-up; a ten-question one was twenty-one. The
latency grid in `docs/results/nvfp4-27b.md` grows steeply with question count - 192 ms for
one question, 232 for three, 718 for ten on the 27B NVFP4 - while the hosted Jev is flat
at 115 / 143 / 129 ms. The obvious suspect was the fan-out itself: twenty-one HTTP round
trips and twenty-one scheduler admissions where a fleet needs one call.

SGLang's native `/generate` takes a batch, and it takes every field reflex needs per item:
`input_ids` as a list of lists, with `sampling_params`, `return_logprob`,
`logprob_start_len`, `top_logprobs_num`, `token_ids_logprob` and `rid` as lists of the same
length. `token_ids_logprob` really is per item (`io_struct.py` hands `token_ids_logprob[i]`
to item *i*), so each branch asks for its own label ids and no union is needed. The backend
now sends one call per request.

**It buys nothing.** Not on the 4B, not on the 27B NVFP4, not warm, not cold, not at one
client and not at eight. Every table below is a tie inside run-to-run noise. The HTTP round
trip and the scheduler admission were never the cost: the branches already ran concurrently
and the time is the forward passes. The change ships anyway, because one call per request
is simpler than a bounded fan-out over many, and because it removes a whole class of
failure - the dropped connection that killed one request in a hundred thousand in
`docs/results/nvfp4-27b.md` had twenty other connections to be dropped from.

## What was measured

| | |
|---|---|
| box 1 | primary DGX Spark (GB10, 121 GB unified memory), `Qwen/Qwen3.5-4B`, `--mem-fraction-static 0.15`, `--mamba-ssm-dtype bfloat16`; `max_running_requests=3` |
| box 2 | second DGX Spark, `RadixArk/Qwen3.8-27B-NVFP4`, `--mem-fraction-static 0.45`, `--mamba-ssm-dtype bfloat16`, `--context-length 16384`; `max_running_requests=22`; the voice stack (~30 GB) resident |
| image | `lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b` |
| reflex | `reflex-serve --backend sglang --permutations 2`, default `--sglang-concurrency 8` |
| client | `scripts/bench_latency.py`, 24 requests per cell after two untimed warm-ups; box 2 measured over the LAN from box 1 |

"per-branch" is `origin/main` (d603dea), "batched" is this branch. Both were run against the
same server process, back to back, with the same seed, so the cold states are the same
states.

<!-- TABLES -->

## The prefix warm-up earns its round trip, but only on long states

Before the branches go out, reflex sends the prefix alone with `max_new_tokens=1` to put it
in the radix cache, and skips that call when it has seen the state before. A batch whose
items share a prefix might make that redundant - so measure it. The 4B, in-process, one
client and eight, three and ten questions, cold states only (a warm state skips the
warm-up either way):

| cold cell, p50 | with warm-up | without |
|---|---|---|
| 160-token state, 1 client, 3 questions | 325 ms | **257 ms** |
| 160-token state, 1 client, 10 questions | 824 ms | **773 ms** |
| 160-token state, 8 clients, 3 questions | 2194 ms | **2060 ms** |
| 160-token state, 8 clients, 10 questions | 6002 ms | **5671 ms** |
| 2,000-token state, 1 client, 3 questions | **510 ms** | 1529 ms |
| 2,000-token state, 8 clients, 3 questions | **11262 ms** | 12223 ms |

The items of a batch are admitted together, so nothing serialises them into "first item
prefills, the rest extend": on a cold state every branch prefills the whole prefix unless
something already put it in the cache. On a 160-token state that duplication is cheaper
than an extra round trip, and dropping the warm-up wins about 20 %. On a 2,000-token state
it is six full prefills against one, and dropping the warm-up costs 3x. Long states are
exactly the case the warm-up exists for, so it stays.

## The round trip was never the cost

Raw `/generate` against the 4B with a warm prefix, one client, nothing else on the server,
thirty repetitions: the same *n* branch prompts sent as one batched call, or as *n*
concurrent calls.

| items | batched p50 | one call each p50 |
|---|---|---|
| 1 | 91.3 ms | 90.3 ms |
| 6 | 176.6 ms | 177.2 ms |
| 20 | 614.1 ms | 616.4 ms |

Identical at every width, and both grow with the item count the same way. That is the
whole result in one table: twenty branches cost 614 ms because twenty branches are twenty
forward passes on a server that runs three at a time, not because they arrived in twenty
envelopes. Batching changes the envelope.

## Equivalence

52 items spanning all four external sets (`runs/external_eval.jsonl`, every 23rd row),
`--permutations 2`, answered through the 4B on the same server by both code paths:

| | |
|---|---|
| max abs. probability difference | **0.00036** |
| p50 / p95 difference | 0.000000 / 0.000000 |
| top-label agreement | 39/39 |
| answers over the 0.01 tolerance | 0 |

Fifty-one of the fifty-two items are bit-identical and the fifty-second moves by three
ten-thousandths. SGLang is not batch-invariant (`docs/results/sglang-backend.md`), so an
exact tie was not guaranteed; the batch shapes simply happen to come out the same when the
same branches run together either way.

## What `--sglang-concurrency` now means, and why the default still fits

The bound used to count branch calls; it now counts `/generate` calls, which is one per
request. At the same numeric value the queue is therefore `questions x permutations` times
deeper. That is measurable: 8 clients, 3 questions, cold, on the 4B.

| | p50 | req/s |
|---|---|---|
| per-branch, bound 8 (8 branches in flight) | 2126 ms | 3.58 |
| batched, bound 8 (8 requests = 48 branches in flight) | 3745 ms | 2.10 |
| batched, bound 2 (2 requests = 12 branches in flight) | 2260 ms | 3.25 |

The middle row is the long-cold-state failure of `docs/results/nvfp4-27b.md` in miniature:
48 branches of distinct states against 3 running slots, and prefix reuse collapses. Matched
in branch terms the two paths agree again. The default of 8 is kept, because on a server
sized like box 2 (`max_running_requests=22`) it is not a queue at all; on a server as
starved as box 1 it should be lowered, and the flag is there for that.

## Verdict

Ship it, and do not expect anything from it.

* **One reflex request is one `/generate` call.** Simpler on the wire, one connection to
  lose instead of twenty-one, and the fan-out bound is now a plain request limit.
* **It is not faster.** Every cell in this note is a tie. The per-branch path is deleted
  because the repo keeps one way of doing things, not because it lost a race.
* **The question-count curve is compute.** Ten questions at two orders is twenty forward
  passes on one GB10. Jev's flat 129 ms is a fleet with headroom, and no amount of
  request packing turns one box into one. To go faster, cut branches (`--permutations 1`)
  or raise `max_running_requests` (`--mamba-ssm-dtype bfloat16`, a bigger
  `--mem-fraction-static`), both measured in `docs/results/nvfp4-27b.md`.
* **The prefix warm-up stays.** It costs about 65 ms on a cold short state and saves a
  second on a cold long one.

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

`tests/test_sglang_fanout.py` holds the wire shape without a GPU: one POST per reflex
request, every per-item field the right length, the bound respected and reached, items
matched by `rid` rather than by position, and a short batch or a missing label id raised
rather than answered from the wrong row.
