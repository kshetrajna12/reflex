# Decision Index reproduction kit

This directory records the exact client harness and H100 serving configuration used for
the full Reflex 27B Decision Index run reported in
[`docs/results/decision-index.md`](../../docs/results/decision-index.md). The small scoring
artifacts are committed here. The 256 MB `results.jsonl` is identified by SHA-256 and is
uploaded separately for leaderboard review.

## What ran

The endpoint served `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` through SGLang and Reflex with two option
orders. The full sweep used eight one-H100 replicas and 32 deterministic HTTP workers.
No benchmark-specific prompt, truncation, option filtering, or retry-to-a-different-model
policy was used. Choices wider than Reflex's 26-label alphabet were returned as declared
capacity refusals and scored as wrong.

`manifest.json` pins every source, image, model and result hash. In particular:

- Reflex source: `bdfc9c80f0c5ac402b0c09c73539a7d507ee4287`
- Decision Index kit: `52a698928a9ae5bdf16b75687c903871db29c6e5`
- result rows: `46195fbb8c676ef2c09740ff55fb8416bf32145ee7fc4cff30cd17f4b71dbf7d`
- frozen suite, uncompressed: `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99`

## Files

- `decision-index-52a6989.patch` is the exact patch applied to the reproduction kit. It
  adds the Reflex capacity adapter, deterministic parallel runner, `--workers` CLI path,
  tests, and the public-source acquisition fixes needed to rebuild the byte-identical
  frozen suite.
- `run_reflex_http.sh` is the exact launch script. It belongs at the root of a patched
  Decision Index checkout; it reads the bearer token from a file and never prints it.
- `serve_h100.py` is the exact process supervisor embedded in each GPU replica.
- `selective-admission.patch` is the exact Reflex serving patch. It changes request
  admission only; request contents, prompts, probabilities and scoring are untouched.
- `results/reflex-27b-p2-full-r8-p32/` contains the untouched score, index, benchmark,
  environment and final-status JSON files.

## Reproduce the harness

Start with the pinned public kit, apply the patch, and copy the runner into its root:

```bash
git clone https://github.com/apolinario/decision-index.git
cd decision-index
git checkout 52a698928a9ae5bdf16b75687c903871db29c6e5
git apply /path/to/reflex/scripts/decision_index/decision-index-52a6989.patch
cp /path/to/reflex/scripts/decision_index/run_reflex_http.sh .
uv sync
```

Against a running endpoint and a local token file:

```bash
./run_reflex_http.sh \
  https://your-endpoint.example \
  /path/to/endpoint.token \
  runs/reflex-27b-p2-full-r8-p32 \
  full reflex-27b 32
```

The H100 process used the pinned SGLang image in `manifest.json`. Inside that image,
`serve_h100.py` launched the model server and Reflex frontend with the recorded flags. The
selective-admission patch is included because it was part of the measured deployment; it
routes short cold work directly when the replica is not under long or wide request pressure.

The supervisor's `SOURCE` constant is an older informational log label. The embedded build
manifest and source snapshot are authoritative: they pin Reflex at `bdfc9c80...`, apply the
recorded patch, and hash the resulting snapshot as `dada7d1e...`. The supervisor is kept
byte-for-byte unchanged here so its recorded SHA-256 remains the code that actually ran.

The local gzip stream had a different compressed hash from the published suite because it
was rebuilt and recompressed. Its uncompressed bytes and the exclusions file match the
official pinned hashes exactly; validation checks those content hashes before accepting the
run.
