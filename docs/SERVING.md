# Serving reflex for evaluation

Everything an evaluator (for example [JevBench](https://github.com/fstandhartinger/jevbench),
whose `typesafe` adapter speaks reflex's wire format unchanged) needs to reproduce a
configuration on a rented GPU.

## The configuration

| | |
|---|---|
| base model | `Qwen/Qwen3.5-4B` (bf16) |
| adapter | none. The frozen model with reflex's default prompt scores best on held-out external data. `kshetrajna12/reflex-qwen3.5-4b-lora` is the earlier LoRA, superseded (its model card says so) and still servable with `--adapter` |
| calibration | none; temperature 1 |
| option orders | two distinct orders per question, averaged (`permutations: 2`) |
| choice cardinality | 2 to 256 options. Past 26 the options are shown as pages of 26 in the same pass and read as one distribution ([results/large-choice.md](results/large-choice.md)); `score` takes 2 to 10 levels |
| reflex commit | whatever the `stable` tag points at; `serving/stable.json` there is the configuration |
| endpoint | `POST /v1/systemone`, TypeSafe-compatible; `model` may be omitted |
| GPU | any CUDA GPU with ≥ 16 GB (RTX PRO 4500 32 GB, A40, L4 all fine); ~9 GB weights |
| readout | the fast single pass, always; the server has no reasoning or escalation mode, so p95 is a small multiple of p50 |

## One command, two ways

With uv (Python 3.12):

    git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout stable
    uv sync
    REFLEX_API_KEY=<secret> uv run reflex-serve --stable --served-name reflex --host 0.0.0.0 --port 8000
    # --stable reads serving/stable.json; spelled out that is
    #   --model Qwen/Qwen3.5-4B --permutations 2, no adapter and no calibration file.
    # Pin a specific experiment with `git checkout <sha>` and explicit flags instead;
    # explicit flags win over the manifest.

With Docker:

    docker build -t reflex-server -f docker/Dockerfile .
    docker run --gpus all -p 8000:8000 -e REFLEX_API_KEY=<secret> \
        -v ~/.cache/huggingface:/root/.cache/huggingface reflex-server

The first request after start compiles Triton kernels (~20 s). Send one warm-up request
before timing, as you would for any server. `GET /health` reports the loaded adapter and
temperatures.

## Timing it fairly

Three things separate a 0.2 s number from a 1.8 s one, and only the first is about us.

**Check the kernel path.** Qwen3.5 is a hybrid backbone: its gated delta rule wants
`flash-linear-attention`'s Triton kernels and its short convolution wants
`causal-conv1d`'s compiled CUDA kernel. When either is missing transformers substitutes
a readable PyTorch reference, logs one warning, and serves correct answers slowly. At
startup reflex now prints which one it got:

    reflex.server kernels: qwen3_5 | attn=sdpa | bfloat16 on cuda:0 |
      fast kernels: torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule |
      REFERENCE FALLBACK: causal_conv1d_fn, causal_conv1d_update |
      installed: flash-linear-attention==0.5.2

Pass `--require-fast-kernels` and the server refuses to start on any fallback instead of
posting a slow number. Use it for every benchmark run.

`flash-linear-attention` is a hard dependency, so the delta rule is always fast and that
is the one that matters: it is the whole recurrent backbone. `causal-conv1d` is not a
dependency, because it has no wheels on PyPI at all -- every release is an sdist that
compiles against your exact torch and CUDA, ~6 minutes with the arch list pinned to one
GPU and far longer without. It is worth almost nothing here. Measured on a GB10
(Qwen3.5-4B, bf16, three questions, two option orders):

| | p50, reference conv | p50, `causal-conv1d` |
|---|---|---|
| warm, new state | 185 ms | 183 ms |
| warm, cached state | 123 ms | 124 ms |

The convolution itself is 1.6x faster with the kernel, on 24 layers of an 8192-wide
conv over a ~220-token sequence: 2.0 ms per request against 1.2 ms, under 1% of the
request. The gap only opens on long sequences (3x, ~15 ms per request, at 2048 tokens),
and reflex never reaches the per-token `causal_conv1d_update` path at all because the
readout is prefill-only. Install it if your states are long:

    # wheels keyed to your torch and CUDA, from the project's GitHub releases
    uv pip install causal_conv1d@https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
    # or build it, which needs nvcc and several minutes
    CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=9.0 \
        uv pip install --no-build-isolation causal-conv1d

**Expose a direct TCP port.** Put the harness on `http://<host>:8000` straight at
uvicorn. A tunnel, an ingress proxy or a serverless front door in between adds its own
connection setup and buffering to every request, and on a rented pod that can be most of
the number.

**Read `x-reflex-latency-ms`.** Every `/v1/systemone` response carries it. It is the
wall time of `engine.answer` inside the handler -- prompt building, prefill, readout --
and nothing else: no request parsing, no response serialization, no network. Your
client's wall time minus that header is everything outside reflex. If the two are far
apart, the model is not what is slow.

## Run the harness against it

    export REFLEX_KEY=<secret>
    python -m jevbench.cli run --tasks datasets/public/hard.jsonl --adapter typesafe \
        --endpoint http://<host>:8000 --model reflex --key-env REFLEX_KEY \
        --price-in-per-m 0 --price-out-per-m 0 --results out/hard.jsonl --raw-dir out/raw --ledger out/ledger.jsonl --cap-usd 1

Our own numbers on the public items are in `docs/results/jevbench-public.md`, so a
re-run can be checked against them.

## Publishing an adapter

    reflex-publish-adapter runs/<adapter dir> --repo <you>/reflex-qwen3.5-4b-lora

uploads the adapter weights, `calibration.json`, the training-time evaluation report and
a model card. Anyone can then serve it with `--adapter <you>/reflex-qwen3.5-4b-lora`.

## Releases: the `stable` tag and `serving/stable.json`

Experiments land on `main`, including the ones that lose. The configuration we
actually recommend is recorded in two places that move together:

* the git tag **`stable`**, which points at the commit to deploy;
* **`serving/stable.json`** at that commit, which names the base model, the adapter
  (a hub id, or `null` for the frozen model), the calibration file, the prompt style and
  prompt-text overrides, plus the gate numbers it was selected on.

`reflex-serve --stable` reads the file and applies it, and `reflex.serving.load_stable()`
returns it for other entrypoints, so a deployer that tracks the tag never has to change a
flag when a better configuration is found. A run only gets to move the tag after it beats
the current `stable` on the never-trained external sets *and* on the public benchmark
items (`docs/results/frozen-vs-trained.md` has the method); the release commit updates the
manifest, then:

    git tag -f stable && git push -f origin stable

The sparkstation deployment follows the tag: it redeploys from whatever `stable` points
at and reads `serving/stable.json` for its flags, so moving the tag moves the deployment
and no flag is set by hand anywhere. Anything else that tracks the tag gets the same
configuration for free; that is the whole point of the manifest.

The current `stable` is the frozen model with the default prompt, no calibration file, and
`permutations: 2`: every choice or score question is asked in two distinct option orders (yes/no
questions in both orders) and the two distributions are averaged. Same forward pass, about
1.1x the latency, hard-tier accuracy 0.658 -> 0.685 and calibration error 0.086 -> 0.081 on the
public benchmark items, pooled external-set calibration error 0.055 -> 0.032.

## Serving the readout on SGLang

`--backend sglang` keeps the same prompt, the same labels and the same readout but lets
an [SGLang](https://github.com/sgl-project/sglang) server own the GPU, which is what a
production deployment usually already runs. The wire format does not change: the same
`POST /v1/systemone` request gets the same response.

Start SGLang on the checkpoint you want to serve, then point reflex at it:

    docker run -d --name reflex-sglang --gpus all --ipc=host -p 30000:30000 \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b \
        python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-4B \
        --host 0.0.0.0 --port 30000 --mem-fraction-static 0.16 \
        --mamba-radix-cache-strategy extra_buffer

    uv run reflex-serve --backend sglang --sglang-url http://127.0.0.1:30000 \
        --model Qwen/Qwen3.5-4B --permutations 2 --port 8008

**About that image tag.** Qwen3.5's hybrid attention needs a recent SGLang, and the
nightly we measured on, `lmsysorg/sglang:nightly-dev-cu13-20260813-273d978b`, has since
been removed from Docker Hub. Nightly tags are deleted on a rolling basis, so pin the
image by **digest** (`lmsysorg/sglang@sha256:…`) and keep a copy in your own registry once
you have one that works. A newer nightly or a release that supports the checkpoint is
fine; check that it launches the model before relying on it, and re-measure, because
neither latency nor calibration carries over between engine versions for free.

**Two flags to set before you benchmark it.** SGLang caps concurrency by the mamba state
cache, not the KV pool: at `--mem-fraction-static 0.45` the 27B gets 51 state slots and ten
running requests, and since one reflex request fans out to `questions x permutations`
branch calls, that ceiling is reached by a single ten-question request. Halving the state
size with `--mamba-ssm-dtype bfloat16` doubles both (111 slots, 22 running requests) at the
same memory, roughly doubles throughput wherever the server was saturated, and leaves
pooled external accuracy and ECE unchanged - though individual probabilities move as much
as quantization moves them, so re-measure if you threshold on them.
`reflex-serve --sglang-concurrency N` bounds how many `/generate` calls reflex keeps in
flight, and one reflex request is one call whatever its question count, so N is really "how
many requests at once" (default 8). Set it from the server's own `max_running_requests`
divided by the branches a typical request carries: on the 27B above that is 22 / 6, so 8 is
about right and lets a wide request use the whole server; on a starved server (the 4B at
`--mem-fraction-static 0.15` runs three at a time) 8 requests is 48 branches of queue, which
costs prefix reuse on cold states, and 2 measures better. `docs/results/nvfp4-27b.md` has
the mamba measurements and the long-cold-state cliff; `docs/results/sglang-batched.md` has
the bound.

reflex still renders the prefix and the branches itself, so `--permutations`,
`--prompt-style`, `--prompt-texts` and `--calibration` all behave as they do on the
transformers backend. `--model` names the tokenizer and must be the checkpoint SGLang is
serving. The reflex process holds no weights, so it needs no GPU of its own.

How it works: the shared prefix is sent once with `max_new_tokens=1` to warm SGLang's
radix cache, then every branch goes out in a single batched `/generate` - `input_ids` as a
list of lists, with `return_logprob`, `logprob_start_len=-1` and `token_ids_logprob` as
per-item lists, so each branch asks for its own label token ids. Each item is still a
separate scheduler request, so isolation between questions is structural rather than a
mask. The recipe follows
[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang);
`docs/results/sglang-backend.md` has the equivalence, latency and accuracy numbers and
`docs/results/sglang-batched.md` has what batching the branches was and was not worth.

**What the SGLang backend does not do.** Images in the state, LoRA adapters
(`--adapter`), prompt ensembles (`--ensemble`) and `--device` all need the weights in
reflex's own process; the flags are rejected rather than ignored. Serve the default
transformers backend for those. The device SGLang runs on is set when you launch its
server, not here.

### Which backend, and at what size

Measured on GB10 (`docs/results/sglang-backend.md`), three questions, warm state:

| | warm p50 | 8 clients | memory |
|---|---|---|---|
| Qwen3.5-4B, transformers | 82 ms | 6.92 req/s | ~9 GB |
| Qwen3.5-4B, SGLang | 182 ms | 4.87 req/s | ~20 GB |
| Qwen3.8-27B bf16, transformers | 364 ms | 1.52 req/s | ~52 GB |
| Qwen3.8-27B bf16, SGLang | 507 ms | 2.22 req/s | ~85 GB |
| Qwen3.8-27B NVFP4, SGLang | 208 ms | 4.52 req/s | ~54 GB |

At 4B the in-process engine wins; serve `--backend transformers`. At 27B SGLang wins
throughput, and the NVFP4 checkpoint (`RadixArk/Qwen3.8-27B-NVFP4`, which the stock
nightly image loads with no extra flags) wins everything.

**One caveat before you serve NVFP4.** Its accuracy matches bf16 but its calibration does
not: pooled external-set ECE rises from 0.044 to 0.067, and hard-tier ECE on the public
items from 0.061 to 0.087. Quantization moves the probability distribution about ten times
more than the choice of backend does. If you rely on the numbers rather than the argmax,
fit a temperature **on your own labelled traffic** and re-measure before thresholding on
it.

Refitting on somebody else's traffic does not work, and `docs/results/nvfp4-27b.md` is the
measurement: temperatures fitted on the four external sets halve pooled external ECE
(0.062 -> 0.026, and five-fold cross-validation inside each set says that holds on unseen
items of the same kind), yet the same temperatures make calibration error **worse on all
three public-benchmark tiers**. The fitted values are recorded there for reproducibility;
no calibration file ships, because nothing that lost its gate does. Fit on your own traffic
with `reflex-calibrate refit` and measure before thresholding.
