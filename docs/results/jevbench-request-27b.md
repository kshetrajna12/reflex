# Bench request filed on fstandhartinger/jevbench (2026-09-20): reflex-27b

Issue: https://github.com/fstandhartinger/jevbench/issues/5 (the 4B entry is #3).

Title: [bench request]: Add reflex-27b (Qwen3.8-27B through reflex, frozen, two option orders; TypeSafe wire format)

A second reflex entry, please, alongside the pending `reflex` (4B) request in #3: the same open re-creation of the Jev design ([kshetrajna12/reflex](https://github.com/kshetrajna12/reflex), MIT) running the frozen **Qwen3.8-27B** (Apache-2.0) with no adapter, no calibration file, and each question read in two distinct option orders and averaged.

**What it is.** The state is encoded once and cached; every question runs as an isolated branch in one forward pass; the answer distribution is read from the option-label logits, no decoding. For each choice/score question the options are presented in two distinct orders (yes/no questions both ways round) inside the same pass and the two distributions are averaged. No weights are trained and nothing is fitted: temperature 1. It serves TypeSafe's wire format, so your `typesafe` adapter runs unchanged.

Configuration (commit `b77f03b`, branch `main`):

```sh
git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout b77f03b
uv sync
REFLEX_API_KEY=any uv run reflex-serve --model Qwen/Qwen3.8-27B --permutations 2 \
    --served-name reflex-27b --host 0.0.0.0 --port 8000
# bf16, ~54 GB of weights: an 80 GB GPU (A100/H100) or a 64 GB+ unified-memory box.
# The first request compiles Triton kernels (~20 s); send one warm-up request before timing.
```

The 4B entry in #3 is unchanged except that it too now reads two orders (`git checkout stable`, `reflex-serve --stable`); happy to have that picked up in the same round or left as filed.

**Our own numbers on the public items** (your adapter's requests, one at a time, argmax over the exact label set; served on a DGX Spark GB10):

| tier | items | reflex-27b | reflex (4B, 2 orders) |
|---|---|---|---|
| easy | 48 | 1.000 | 1.000 |
| standard (original) | 72 | 0.958 | 0.917 |
| hard | 111 | 0.766 | 0.685 |

Hard-tier top-label ECE 0.061, distribution fidelity on the 10 public probability items 0.827; p50 latency 0.6 s on easy/standard items and 1.0 s on hard ones on our box. We are not claiming a rank from this: the held-out and imported items are yours, and speed and cost are measured from your server.

**Disclosure.** No training and no benchmark item or paraphrase is involved in this configuration. The prompt wording and the two-order readout were selected on our own external sets (support intents, MNLI-mismatched, toxic-chat, Yelp; details in the repo's `docs/results/order-averaging.md`) and then checked on the public 231, which we have consulted repeatedly during development, so treat the public numbers as development-suite numbers and expect the held-out hard items to come in below them. Full method and every negative result are in the repo.

If a 60 GB GPU is an obstacle for the run, say so and we can host the endpoint for you for the duration.
