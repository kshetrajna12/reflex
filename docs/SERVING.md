# Serving reflex for evaluation

Everything an evaluator (for example [JevBench](https://github.com/fstandhartinger/jevbench),
whose `typesafe` adapter speaks reflex's wire format unchanged) needs to reproduce a
configuration on a rented GPU.

## The configuration

| | |
|---|---|
| base model | `Qwen/Qwen3.5-4B` (bf16) |
| adapter | none for the general-purpose configuration (the frozen model with reflex's default prompt scores best on held-out external data); `kshetrajna12/reflex-qwen3.5-4b-lora` is the earlier LoRA, still servable with `--adapter` |
| reflex commit | `dae6799` |
| endpoint | `POST /v1/systemone`, TypeSafe-compatible; `model` may be omitted |
| GPU | any CUDA GPU with ≥ 16 GB (RTX PRO 4500 32 GB, A40, L4 all fine); ~9 GB weights |
| readout | the fast single pass, always; the server has no reasoning or escalation mode, so p95 is a small multiple of p50 |

## One command, two ways

With uv (Python 3.12):

    git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout <sha>
    uv sync
    REFLEX_API_KEY=<secret> uv run reflex-serve --model Qwen/Qwen3.5-4B --served-name reflex --host 0.0.0.0 --port 8000
    # add --adapter kshetrajna12/reflex-qwen3.5-4b-lora to serve the earlier fine-tuned variant instead

With Docker:

    docker build -t reflex-server -f docker/Dockerfile .
    docker run --gpus all -p 8000:8000 -e REFLEX_API_KEY=<secret> \
        -v ~/.cache/huggingface:/root/.cache/huggingface reflex-server

The first request after start compiles Triton kernels (~20 s). Send one warm-up request
before timing, as you would for any server. `GET /health` reports the loaded adapter and
temperatures.

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

reflex still renders the prefix and the branches itself, so `--permutations`,
`--prompt-style`, `--prompt-texts` and `--calibration` all behave as they do on the
transformers backend. `--model` names the tokenizer and must be the checkpoint SGLang is
serving. The reflex process holds no weights, so it needs no GPU of its own.

How it works: the shared prefix is sent once with `max_new_tokens=1` to warm SGLang's
radix cache, then one `/generate` per branch goes out concurrently with
`return_logprob`, `logprob_start_len=-1` and `token_ids_logprob` set to that branch's
label token ids. Each branch is a separate request, so isolation between questions is
structural rather than a mask. The recipe follows
[ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang);
`docs/results/sglang-backend.md` has the equivalence, latency and accuracy numbers.

**What the SGLang backend does not do.** Images in the state, the thinking readout
(`--think`), the escalation cascade (`--escalate`) and LoRA adapters (`--adapter`) all
need the weights in reflex's own process; the flags are rejected rather than ignored.
Serve the default transformers backend for those.

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
fit a temperature on the NVFP4 checkpoint itself (`reflex-calibrate`) and re-measure before
thresholding on it. A calibration file fitted on bf16 does not transfer.
