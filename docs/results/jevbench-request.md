# Bench request filed on fstandhartinger/jevbench (2026-09-19)

> **Historical.** This filing named the mix3 LoRA adapter. It was amended the next day to
> the frozen model ([jevbench-amendment.md](jevbench-amendment.md)) and the entry now reads
> two option orders. The adapter is superseded.

Title: [bench request]: Add reflex (kshetrajna12/reflex-qwen3.5-4b-lora, TypeSafe wire format)

Request to add [reflex](https://github.com/kshetrajna12/reflex) (Qwen3.5-4B + a LoRA adapter,
[kshetrajna12/reflex-qwen3.5-4b-lora](https://huggingface.co/kshetrajna12/reflex-qwen3.5-4b-lora);
MIT code and adapter, Apache-2.0 base weights) to the ranked systems.

**What it is.** An open re-creation of the Jev design: the state is encoded once and cached,
every question runs as an isolated branch in one pass, and the answer is read directly from the
label logits (no decoding). Trained with a proper scoring rule on public datasets and synthetic
correct-by-construction data ("RLCD-lite"); temperatures fitted per primitive on our own held-out
set. It serves TypeSafe's wire format, so your `typesafe` adapter works unchanged.

```sh
git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout 1add693
uv sync
REFLEX_API_KEY=any uv run reflex-serve --model Qwen/Qwen3.5-4B \
    --adapter kshetrajna12/reflex-qwen3.5-4b-lora --served-name reflex --host 0.0.0.0 --port 8000
# adapter + calibration.json are pulled from the Hub; ~9 GB bf16, any 16 GB CUDA GPU;
# the first request compiles Triton kernels (~20 s), so send one warm-up request before timing.

export REFLEX_KEY=any
python -m jevbench.cli run --tasks datasets/public/original.jsonl \
  --adapter typesafe --endpoint http://127.0.0.1:8000 --key-env REFLEX_KEY --model reflex \
  --cost-basis self_hosted_gpu --reserve-usd 0 ...
```
A Dockerfile (`docker/Dockerfile`) and the full recipe are in `docs/SERVING.md`.

**Our own numbers on the public items** (your adapter's requests, one at a time, argmax over the
exact label set):

| tier | items | reflex |
|---|---|---|
| easy | 48 | 1.000 |
| standard (original) | 72 | 0.944 |
| hard | 111 | 0.604 |

Hard-tier top-label ECE 0.117; distribution fidelity on the 10 public probability items 0.689.
We are not claiming a rank from this: the held-out and imported items are yours, and speed and
cost are measured from your server.

**Disclosure.** The training recipes were designed after looking at which public families the
raw model failed (answer adequacy, rule-based routing, base-rate probabilities; a long-policy
recipe exists in the repo but this adapter was not trained on it). No benchmark item or paraphrase was used in training; `reflex-data check-overlap` refuses to
train if any training row shares an 8-word sequence with a public item. The public 231 were used
as a development gate four times (three adapters, one re-calibration), so expect the held-out hard items to score somewhat below the
public ones.

Happy to answer questions here or rerun anything.
