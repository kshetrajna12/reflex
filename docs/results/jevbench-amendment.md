# Amendment posted on fstandhartinger/jevbench issue #3 (2026-09-19)

Update to this request: please evaluate the **frozen base model with reflex's default
prompt and no adapter** instead of the LoRA adapter named above. Our controls since
filing showed the adapter traded accuracy on long, ambiguous items for accuracy on items
that resemble its training data, and that two prompt-level changes (a lettered yes/no
readout and "Evidence / Criterion" framing), selected on held-out external sets, did
better without touching a weight. Details and every number are in
`docs/results/frozen-vs-trained.md` in the repo.

Configuration:

```sh
git clone https://github.com/kshetrajna12/reflex && cd reflex && git checkout dae6799
uv sync
REFLEX_API_KEY=any uv run reflex-serve --model Qwen/Qwen3.5-4B --served-name reflex --host 0.0.0.0 --port 8000
```

No adapter, no calibration file: `Qwen/Qwen3.5-4B` in bf16, reflex's default prompt at
that commit, probabilities read from the option-label logits with temperature 1. Same
`/v1/systemone` wire format as before; your `typesafe` adapter runs unchanged. Any 16 GB
CUDA GPU; the first request compiles Triton kernels (~20 s), so send one warm-up request.

Our own numbers on the public items (your adapter's requests, one at a time):

| tier | items | frozen + reflex prompt | previously filed adapter |
|---|---|---|---|
| easy | 48 | 1.000 | 1.000 |
| standard (original) | 72 | 0.917 | 0.944 |
| hard | 111 | 0.658 | 0.604 |

Hard-tier top-label ECE 0.086, fidelity on the 10 public probability items 0.708. Not
claiming a rank from this. Disclosure as before: the public items were used as a
development gate (now nine self-runs across adapters, prompts and calibration variants);
prompt changes were selected on four external labelled sets first and confirmed on the
public items second; no benchmark item was used for training or for fitting anything.

If you would rather keep the adapter row as filed, that is fine too; the frozen
configuration is the one we will stand behind going forward.
