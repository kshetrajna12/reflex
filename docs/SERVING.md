# Serving reflex for evaluation

Everything an evaluator (for example [JevBench](https://github.com/fstandhartinger/jevbench),
whose `typesafe` adapter speaks reflex's wire format unchanged) needs to reproduce a
configuration on a rented GPU.

## The configuration

| | |
|---|---|
| base model | `Qwen/Qwen3.5-4B` (bf16) |
| adapter | `kshetrajna12/reflex-qwen3.5-4b-lora` (LoRA from `reflex-calibrate`; `calibration.json` inside it is applied automatically) |
| reflex commit | `1add693` |
| endpoint | `POST /v1/systemone`, TypeSafe-compatible; `model` may be omitted |
| GPU | any CUDA GPU with ≥ 16 GB (RTX PRO 4500 32 GB, A40, L4 all fine); ~9 GB weights |

## One command, two ways

With uv (Python 3.12):

    git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout <sha>
    uv sync
    REFLEX_API_KEY=<secret> uv run reflex-serve --model Qwen/Qwen3.5-4B \
        --adapter kshetrajna12/reflex-qwen3.5-4b-lora --served-name reflex --host 0.0.0.0 --port 8000

With Docker:

    docker build -t reflex-server -f docker/Dockerfile .
    docker run --gpus all -p 8000:8000 -e REFLEX_ADAPTER=kshetrajna12/reflex-qwen3.5-4b-lora -e REFLEX_API_KEY=<secret> \
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
