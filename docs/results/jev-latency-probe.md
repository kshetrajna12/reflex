# Hosted Jev latency, measured from our box (2026-09-20)

`scripts/jev_latency_probe.py` against `jev-1.13.0` through the official SDK, from the DGX
Spark over the public internet: a ~160-token ticket state, 1 / 3 / 10 questions per call,
"warm" = the same state repeated, "cold" = a fresh ticket id in the state every call; 12
timed calls per cell after one warm-up; then 24 cold three-question calls from 8 threads.

| questions per call | warm p50 / p95 | cold p50 / p95 |
|---|---|---|
| 1 | 115 / 139 ms | 108 / 161 ms |
| 3 | 143 / 202 ms | 111 / 162 ms |
| 10 | 129 / 186 ms | 142 / 216 ms |

8 concurrent clients, 3 questions, cold: p50 149 ms, p95 238 ms, **43.7 req/s (131 q/s)**.

Two things stand out. Latency is flat in the number of questions (ten cost the same as one)
and flat between warm and cold at this state size: the branches run fully in parallel on a
fleet with headroom, and a 160-token prefill is noise. And throughput scales with clients,
which is a fleet, not a box. Usage reports `output_tokens` per call (78 for three questions),
so the answers are metered even though nothing is decoded.

## Against reflex on one GB10

| single client, 3 questions | warm p50 | cold p50 |
|---|---|---|
| Jev (hosted) | 143 ms | 111 ms |
| reflex 4B `stable`, transformers, in-process, 1 order | 82 ms | 144 ms |
| reflex 4B `stable` through the sparkstation gateway, 2 orders | ~280 ms | |
| reflex 27B NVFP4 on SGLang, 2 orders, over HTTP | 232 ms | 600 ms |
| reflex 27B bf16, transformers, in-process, 1 order | 364 ms | 671 ms |

| 8 clients, 3 questions, cold | req/s | questions/s |
|---|---|---|
| Jev (hosted) | 43.7 | 131 |
| reflex 27B NVFP4 on SGLang (one GB10, voice stack co-resident) | 2.1 | 6.2 |
| reflex 27B NVFP4 on SGLang, warm | 5.7 | 17 |
| reflex 4B, transformers (one GB10) | 6.9 | 21 |

Read it as: on a single request the 4B is level with Jev and the 27B is 2x slower warm and
5x slower cold; on ten questions per call the 27B is 5x slower (718 ms warm) because one
GB10 is compute-bound where Jev's fleet is not; and on throughput a hosted fleet is a
different product from one box. The 27B's accuracy edge on the hard tier (0.766 vs 0.730)
is bought at that latency. Numbers for reflex are from docs/results/sglang-backend.md and
the NVFP4 grid in docs/results/nvfp4-27b.md.
