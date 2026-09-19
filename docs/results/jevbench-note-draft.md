# Draft note to the JevBench author (NOT SENT; for review)

Subject: reflex, an open Jev-style decision model on Qwen3.5-4B, for a future JevBench round

Hi Florian,

reflex (https://github.com/kshetrajna12/reflex, MIT) is an open re-creation of the Jev
design: shared state prefix, isolated question branches, direct probability readout,
trained with a proper scoring rule ("RLCD-lite"). It serves TypeSafe's `/v1/systemone`
wire format, so your `typesafe` adapter runs it unchanged.

Configuration to evaluate:
- base Qwen/Qwen3.5-4B, LoRA adapter `kshetrajna12/reflex-qwen3.5-4b-lora` (calibration.json inside), reflex commit `<sha>`
- serving recipe (uv or Docker, one command): docs/SERVING.md
- any 16 GB+ CUDA GPU; ~9 GB weights; first request compiles Triton kernels (~20 s)

Our self-run on the 231 public items (docs/results/jevbench-public.md): easy 48/48,
standard <x>/72, hard <y>/111; schema validity 100 %. The adapter was trained on public
datasets and synthetic data only; `reflex-data check-overlap` verifies no 8-gram overlap
with the public items, and no JevBench item or paraphrase was used.

Happy to adjust anything about the configuration.
