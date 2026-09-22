# Decision Index: wiring reflex into the harness (2026-09-22)

The [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) is a
leaderboard for typed decision engines: exactly our interface, a `state` plus a set of
`choice` questions, one answer each with a full distribution over the supplied options. The
reproduction kit is [apolinario/decision-index](https://github.com/apolinario/decision-index)
at commit `52a6989`. Not affiliated with TypeSafe AI, and not JevBench: where JevBench scores
534 decisions on four axes, this one is 132,422 requests over 37 benchmarks, of which 19 form
the scored panel, each with its native metric, averaged into five equal-weight areas. The
index is the mean of those five areas on a 0-100 scale.

**Status: we cannot run it.** The frozen suite is not obtainable (below). Two capacity gaps
this exercise found are fixed and on `main`; the 4B run is written, staged on worker2 and
waiting on data and on large-choice support. Nothing was uploaded and no pull request was
opened.

For scale, the leaderboard's 31 entries top out at `Jevfire` on 55.74, and `Kev 4B`, the
closest published analogue to ours, sits at 47.43. Coverage there is worth reading before
ours: engines that answer every request report 0.76, and `system-one-gemma`, which refuses
48,200 requests on a capacity limit, reports 0.449 and an index of 17.09. Refusals are
allowed and honest, and they are also most of what separates the bottom of that table from
the top.

## The rules, and how the harness talks to us

Six rules, enforced in the runner rather than left to the reader. **No truncation**: an engine
that cannot fit a request raises `Unsupported` and the row is scored wrong; nothing is ever
cut to fit. **No option filtering**: every option in `criteria` is scored or the response is
rejected. **No prompt tuning**: one fixed rendering for every benchmark, no per-benchmark
prompts or few-shot examples. **Unanswered is wrong**: unsupported, errored, abstained and
pending requests all score zero against the full frozen denominator. **Linked cases stay
whole**: a multi-request case counts only when every one of its requests succeeded. And the
442 excluded questions are dropped for every engine alike.

Their `http` engine is a client for our endpoint, near enough to Jev's that no adapter is
needed. It posts `{"model", "state", "questions"}` to `{base_url}/v1/systemone` with a bearer
token from `DECISION_INDEX_API_KEY`, and uses the response body as-is, minus
`evaluation_trace`. Our `confidence` field and our `usage` block are simply ignored; the
leaderboard's own calibration axis is computed from the probability we place on the chosen
option, not from anything we label as confidence. Validation is strict and we pass it: every
question answered, the type matching, the chosen key among the options, and a finite
probability in [0, 1] for every option summing to 1 within 0.01.

The one thing that carries real weight is how a refusal is worded. HTTP 400, 413 or 422 whose
body contains a known capacity phrase becomes `unsupported`; anything else propagates as
`error`. That distinction is not cosmetic. Errors are retried on every resume, and five errors
before the first success abort the run outright.

## Two capacity gaps, both now fixed

**The 26-option cap was reported as a server fault.** A `choice` in the suite carries 2 to 255
options; we accept 26, for the single-letter readout. That limit is legitimate and the rules
say so explicitly, but our 422 read `choice supports at most 26 options`, which matches none of
their capacity markers, so every refusal was recorded as a runtime error. In the smoke sample
below that turned 17 honest refusals into 17 errors, and on a CLINC-heavy opening stretch it
would have aborted the run before the first success. Commit `6dce470` words the three schema
refusals the way Jev words them, which is the phrasing their marker list was built from.
Verified against their `CAPACITY_MARKERS` directly: 151 options, one option and 12 score levels
now all classify as `unsupported`.

**The one-question token ceiling was not reachable from the command line.** Nothing refuses a
long *state*: we answered a 192,182-token state in 98 s, and the window is the model's, 262,144
positions on Qwen3.5-4B. `--max-pack-tokens` is a batching budget and refuses nothing. The only
size limit that refuses is `max_branch_tokens`, default 4096, covering one question's
instructions and options together, and it was settable only through `Engine.load`. That matters
because ToolRet and BRIGHT put an entire candidate document inside each question's
instructions, up to about 31k tokens per question and 60k per request, with two options apiece.
At the default, all 9,416 requests across those two panel benchmarks are refused, and since a
linked case needs every chunk, one refusal zeroes the whole query group. Commit `e8af022` adds
`--max-branch-tokens` and leaves the default alone.

Measured with the ceiling raised, on the primary box:

| retrieval-shaped request | input tokens | wall |
|---|---:|---:|
| 1 candidate | 20,069 | 14.3 s |
| 4 candidates | 80,027 | 50.5 s |
| 8 candidates | 159,971 | 101.3 s |

## The smoke run

Six benchmarks rebuilt from their pinned public sources (34,399 real frozen rows: MMLU,
ARC-Easy, ARC-Challenge, WinoGrande, HellaSwag, CLINC150+OOS), sampled to 100 requests,
against `stable` on the primary box. Real data in the frozen format, but not representative:
no long contexts, no tool use, no multi-question rows.

| benchmark | ok | accuracy | median ms |
|---|---:|---:|---:|
| MMLU | 17 | 0.882 | 82.4 |
| ARC-Easy | 17 | 0.882 | 74.9 |
| ARC-Challenge | 17 | 0.941 | 75.3 |
| WinoGrande | 16 | 0.813 | 68.9 |
| HellaSwag | 16 | 0.875 | 115.6 |
| CLINC150+OOS | 0 | refused, 151 options | |

83 ok and 17 refused, with zero errors after `6dce470`. Overall median 75.3 ms, p95 123.4 ms.
`score` itself cannot be run on a partial suite: an area with no scored benchmark reaches
`statistics.mean` on an empty sequence, so the index step raises. That is a property of
scoring a subset, not of our responses.

## Throughput, and what a full run would cost

**The runner is sequential and there is no concurrency flag.** We checked whether asking for
one would be worth it, and it would not: against the in-process transformers backend,
throughput is flat from 1 to 16 concurrent clients while latency grows linearly, because the
GPU work serializes behind the single model.

| concurrency | req/s | median ms |
|---:|---:|---:|
| 1 | 12.24 | 74.1 |
| 4 | 12.33 | 326.9 |
| 16 | 12.04 | 1330.5 |

Cold prefill is what a full run actually pays for, and it is where the two boxes differ. A
repeated identical state is nearly free, 144 ms for 12k tokens, because the state cache hits;
the candidate documents in retrieval questions are branch tokens and are never cached. Note
also that `stable` averages two option orders, so every branch is computed twice.

| cold prefill | primary | worker2 |
|---|---:|---:|
| 12k tokens | 2,889 ms | 2,580 ms |
| 48k tokens | 19,586 ms | 11,253 ms |

worker2's advantage is `causal-conv1d`. `--require-fast-kernels` refused to start on both
boxes, correctly: the convolution ops were falling back to reference PyTorch, which the kernel
report calls out and which costs an order of magnitude on long sequences. There is no wheel
for this platform, so it was built from source against the CUDA 13.0 toolkit at
`/usr/local/cuda` (`CAUSAL_CONV1D_FORCE_BUILD=TRUE`, no build isolation). `causal-conv1d
1.7.0` now imports there and the server starts clean with all four kernels fast. It buys
nothing on short rows, where the 100-row sample takes the same 6.9 s on both boxes, and about
1.7x at 48k tokens.

That leaves a real choice, because the two configurations differ by more than an order of
magnitude:

* **Default 4096 ceiling.** ToolRet and BRIGHT come back `unsupported`, which breaks no rule.
  The remaining ~123,000 requests run at about 12 req/s, and the full run lands near 16 hours.
  Two panel benchmarks score zero.
* **Raised ceiling.** Those families are answered and they dominate everything: roughly 35
  hours on worker2 for the retrieval families alone, for a full run of 1.5 to 3 days.

## Why there is no run yet

`multimodalart/decision-index-suite`, the dataset holding the 132,422 frozen rows, **returns
404**. Confirmed from two machines and with an authenticated account; a Hub search finds no
dataset of that name under any owner, while the leaderboard space is live. `suite download`,
and therefore `pipeline`, fails before an engine is ever loaded.

The kit offers `suite rebuild` from pinned public sources, and it is the only path, but it
cannot produce a submittable run. We drove it far enough on worker2 to know exactly why, with
each source isolated so one failure did not abort the rest. 36 of 37 sources acquired cleanly
in about six minutes and 3.2 GB. The blockers are:

* **BANKING77 cannot be rebuilt.** At the pinned revision the Hub repo is a loading script
  with no data files, so the kit's snapshot fetches nothing. The two files it wants can be
  reconstructed from where the script points, GitHub's `test.csv` and the 77 label names in
  the repo's own `dataset_infos.json`, but the option *ordering* is then a reconstruction and
  cannot be checked against a frozen file we do not have.
* **HLE is gated.** `cais/hle` returns 401 without accepted terms and a logged-in token.
* **Three harness bugs.** `acquire` writes its `hf:` and `http:` targets under the work root
  while the normalizers read them from `artifacts/benchmark-suite/raw`, so most sources land
  in the wrong place (one symlink works around it). GPQA cannot be rebuilt alone because its
  builder is fused with Humicroedit. And `score` crashes on any partial suite.

A rebuilt file that is missing a benchmark is not byte-identical to the pinned
`750d353a…`, so `scores.json` would never say `"complete": true` and the run would be
re-scored and rejected on review. The rebuild was stopped by decision, not by failure.

For the record, submission is two steps once a run exists: upload the run directory to a Hub
dataset containing `runs/<name>/` with `scores.json`, `benchmark-summary.json`, `index.json`,
`environment.json`, `status.json` and `results.jsonl.gz`; then open a pull request adding a
line to the kit's `submissions/README.md` with the model name, the results dataset link, the
engine and commit, and the hardware. Declared capacity limits should be stated, and are fine,
as long as nothing was truncated.

## What is staged, and what it is waiting for

worker2 carries the harness at `~/scratch/di/decision-index`, `serve_4b.sh` (which now passes
`--require-fast-kernels --max-branch-tokens 65536 --max-pack-tokens 65536`) and `run_4b.sh`,
which runs `pipeline` against `127.0.0.1:8020` with checkpoint and resume into
`~/scratch/di/runs/reflex-4b`. The run is held for two things: the frozen suite, and support
for choices wider than 26 options. The second is worth its own number. CLINC150 at 151 options
is 5,500 requests and BANKING77 at 77 options is 3,080; beyond those, ChessBench, ToolRet,
BRIGHT, RouterBench, SGD, BFCL and API-Bank all build options from variable-length candidate
lists that the kit permits up to 255. The exact census needs the frozen rows.
