# Decision Index: full Reflex 27B run (2026-09-22)

The full [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index)
run is complete. Reflex 27B scores **56.23**, above Jevfire, the kit's reference run, at
**55.74**, and behind Jev itself, which the leaderboard lists at **59.51**. On the live
leaderboard 56.23 is first among the open reproductions.
All 132,422 frozen requests were accounted for, with 117,922 successful responses, 14,500
declared capacity refusals and no errors.

This is a different benchmark from JevBench. Decision Index covers 37 benchmark families;
19 enter the headline panel through five equally weighted areas. Every engine receives the
same `state` and typed `questions`. Unsupported, errored and unanswered work scores zero.
The reproduction kit is `apolinario/decision-index` at
`52a698928a9ae5bdf16b75687c903871db29c6e5`.

## Headline result

| metric | Reflex 27B | Jevfire (kit reference) | delta |
|---|---:|---:|---:|
| **Decision Index** | **56.23** | 55.74 | **+0.49** |
| balanced skill | **42.15** | 40.86 | **+1.29** |
| breadth skill | **40.88** | 39.45 | **+1.43** |

The Jevfire figures are the reproduction kit's own `jevfire-uncapped` reference; scoring that
run locally reproduces its published 55.74 / 40.86 / 39.45 entry. The comparison therefore
uses the same suite, exclusions, metrics and formulas.

| area | raw | skill | coverage |
|---|---:|---:|---:|
| Knowledge & Reasoning | 55.58 | 39.59 | 88.63% |
| Language Understanding | 62.60 | 47.86 | 100% |
| Retrieval & Classification | 36.13 | 28.29 | 100% |
| Tools & Automation | 73.46 | 62.06 | 100% |
| Arts & Human Judgment | 53.37 | 32.96 | 80% |

The weighted index coverage is **93.73%**. Coverage is below 100% solely because Reflex's
letter readout supports at most 26 choice options. The exact refusal census is:

| benchmark | unsupported requests |
|---|---:|
| API-Bank | 508 |
| BANKING77 | 3,080 |
| CLINC150+OOS | 5,500 |
| POP909-CL | 2,000 |
| ChessBench | 3,412 |
| **total** | **14,500** |

Every refused row actually exceeded 26 options. No input was truncated, no option was
removed, and no request was silently replaced with another model.

## The run

| setting | value |
|---|---|
| model | `Qwen/Qwen3.8-27B-FP8` |
| model revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Reflex source | `bdfc9c80f0c5ac402b0c09c73539a7d507ee4287` plus the recorded admission patch |
| backend | SGLang, pinned container digest |
| readout | two option orders, averaged |
| hardware | 8 x NVIDIA H100, one independent replica per GPU |
| client load | 32 deterministic HTTP workers |
| successful / unsupported / error | 117,922 / 14,500 / 0 |
| wall time | 1 h 23 m 4 s |

This was the full cold sweep, not a repeated-state latency grid. Request states and shapes
come from the benchmark and vary from tiny classifications to long retrieval candidates.
The request timer covers the `/v1/systemone` HTTP wall time, including prompt construction
and inference but excluding server startup.

| loaded successful-request statistic | result |
|---|---:|
| median | 421.9 ms |
| mean | 1,158.0 ms |
| p95 | 4,233.6 ms |
| supported throughput | 23.66 requests/s |

At an assumed **$2 per H100-hour**, keeping all eight GPUs allocated for the 4,984-second
sweep costs **$22.15**. That estimate covers benchmark runtime only; it excludes model
startup and idle time before or after the run.

## Serving configuration

Each H100 ran one SGLang server and one Reflex frontend. The model server used a 65,536-token
context, bf16 recurrent state, 320 Mamba cache slots, FlashAttention 3 and an 8,192-token
chunked-prefill size. The container image, full flags and hashes live in
[`scripts/decision_index/manifest.json`](../../scripts/decision_index/manifest.json).

The deployment also carried the recorded selective-admission patch. Under ordinary short
cold load it skips the separate prefix preparation pass. When long or wide requests are
pending it falls back to the shared-prefix path so concurrent branches do not duplicate a
large prefill. The patch changes scheduling only: prompts, option orders, model logits,
probabilities and benchmark scoring are untouched.

## Integrity

The final validator checked every row id and payload hash against the frozen suite, required
the exact expected unsupported set, verified the exclusions, checked two permutations and
the 32-worker environment, and rejected duplicates or incomplete lines. It passed with:

- 132,422 distinct expected and observed rows;
- 117,922 `ok`, 14,500 `unsupported`, zero other statuses;
- frozen suite uncompressed SHA-256
  `288d37207a9581187bdf83eada1983aa63de6fc50b0108e2badb229547a57f99`;
- exclusions SHA-256
  `331df32d4b719c7db43214d0e5d85859d39c3b2eb7d0b3812214cce150155e81`;
- result rows SHA-256
  `46195fbb8c676ef2c09740ff55fb8416bf32145ee7fc4cff30cd17f4b71dbf7d`.

The suite was rebuilt from the pinned public sources because its prebuilt Hub dataset was
unavailable during the first attempt. HLE access and the missing BANKING77 source files were
then resolved. Rebuilding produced different gzip container bytes, as expected, but the
uncompressed suite and exclusions match the official pinned hashes exactly.

## Reproduction and submission

[`scripts/decision_index/`](../../scripts/decision_index/) contains the exact kit patch,
HTTP runner, H100 supervisor, scheduling patch, manifest and untouched score artifacts.
The raw `results.jsonl` is 256 MB and is kept out of git; its hash above binds the result.

The leaderboard submission remains review-gated. Its documented flow is to upload the full
run directory to a public Hugging Face dataset and open a pull request against the Decision
Index repository. Reviewers re-score `results.jsonl`; the score in this document is the
locally verified candidate result until that review merges.
