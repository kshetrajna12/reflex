# JevBench official results (v1.2, scored 2026-09-21)

Both reflex entries were run by the benchmark's author on a rented H100, one request at a
time, over the full frozen v1.2 set of 534 decisions including the held-out items, with the
unchanged `typesafe` adapter. Source: comments on issues
[#3](https://github.com/fstandhartinger/jevbench/issues/3) and
[#5](https://github.com/fstandhartinger/jevbench/issues/5); leaderboard at
[benchmarkheaven.com/jev-models](https://benchmarkheaven.com/jev-models); data and method at
[jevbench v1.2.8](https://github.com/fstandhartinger/jevbench/tree/v1.2.8). The JevBench
Score is a geometric mean of four equally weighted axes: Intelligence, Calibration, Speed, Cost.

| rank | system | score | intelligence | calibration | speed | cost | easy | standard | judge | hard |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Jev 1.13.0 | 75.4 | 90.4 | 82.7 | 83.3 | 52.0 | 100 | 99.0 | 94.5 | 74.1 |
| 2 | SemIf (Qwen3.5-4B) | 74.7 | 85.9 | 72.6 | 83.7 | 59.5 | 100 | 97.9 | 95.2 | 59.5 |
| 3 | djev (Maisa) | 74.3 | 88.4 | 65.4 | 91.4 | 57.6 | 100 | 97.9 | 93.2 | 69.5 |
| **5** | **reflex (4B)** | **71.7** | 86.7 | 75.2 | 68.0 | 59.7 | 100 | 94.8 | 97.3 | 63.2 |
| 8 | jqv (Qwen3-32B) | 70.1 | 86.1 | 79.0 | 74.6 | 47.5 | 100 | 95.8 | 92.5 | 64.5 |
| 17 | openjev-sglang | 66.3 | 88.9 | 77.4 | 77.1 | 36.5 | | | | |
| **22** | **reflex-27b** | **64.2** | **90.5** | **86.2** | 67.5 | 32.3 | 100 | 95.8 | 95.9 | **75.9** |

36 ranked systems (40 tested). "reflex" is "the best-placed newcomer this round" in the
author's words.

## What was actually run

* **reflex (#3)** ran commit `1add693`: Qwen3.5-4B **with the mix3 LoRA adapter and its
  calibration file**, one option order. That is the configuration originally filed, not the
  frozen model requested in the amendment and not the current `stable` (frozen, two orders).
  On our public-item development suite that adapter scored hard 0.604; the full frozen set
  gave 63.2 %, so the held-out items were kinder to it than the public ones.
* **reflex-27b (#5)** ran commit `b77f03b`: Qwen3.8-27B frozen, two distinct option orders,
  no adapter, no calibration file, through the transformers engine.

Latency for both: p50 about 1.8-1.9 s raw including the trip from the author's server in
Germany to a pod in Canada, doubled plus 0.15 s by the benchmark's self-host adjustment.
Cost: $0.022 per 1,000 decisions for the 4B, $0.181 for the 27B.

## What it says

* **The 27B is the most intelligent and the best-calibrated system on the board**,
  Intelligence 90.5 against Jev's 90.4 and Calibration 86.2 against Jev's 82.7, and it beats
  Jev on the hard tier, 75.9 % against 74.1 %, on 534 items the author holds. It ranks 22nd
  because the score's geometric mean punishes its speed (67.5) and above all its cost (32.3,
  the price of an H100 per decision at one request at a time).
* **The 4B at #5** sits between SemIf and Jev on Intelligence, ahead of SemIf on
  Calibration, and has the highest judge-tier accuracy of the top eight (97.3 %). Its
  hard-tier 63.2 % is where a 4B lands frozen or trained; speed is what separates it from
  the top three.
* **The public items were a fair development suite.** Every ordering they predicted holds
  on the frozen set.

## What would move the placings

* **reflex (4B)**: the configuration that ran is a superseded one. The current `stable`
  (frozen, two orders) scores higher on our development suite (hard 0.685 vs 0.604, ECE
  0.081 vs 0.117) at about 1.1x the latency; an updated run at commit `19586a1` is the
  obvious request for the next round. Speed is the other lever: 1.8 s p50 on an H100 is far
  above the ~150 ms this engine does on a Spark for the same shape, so most of that is the
  cross-Atlantic hop and the one-at-a-time protocol, which every self-hosted entry pays.
* **reflex-27b**: intelligence and calibration are already at the top; cost and speed are
  the whole gap. The SGLang backend with the NVFP4 checkpoint (docs/results/nvfp4-27b.md,
  sglang-batched.md) is 1.75x faster and 3x the throughput on a Spark, and on an H100 it
  should cut both axes substantially, at a measured calibration cost (hard ECE 0.087 vs
  0.061 on our public items) that the Calibration axis would feel. That trade is the
  decision for a second 27B entry.
