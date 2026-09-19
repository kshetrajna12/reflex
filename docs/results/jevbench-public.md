# reflex on JevBench v1.2 (public items only, unofficial self-run; official run requested 2026-09-19)

[JevBench](https://github.com/fstandhartinger/jevbench) is Benchmark Heaven's benchmark for
Jev-class decision models. Its `typesafe` adapter speaks the same `/v1/systemone` wire
format reflex implements, so the harness runs against a reflex server unchanged. This is
a self-run on the **public** items (231 of 534; the 109 held-out hard items and the 146
imported judge items are not public), served through sparkstation on a GB10:
Qwen3.5-4B + the reflex LoRA (public 8-dataset mix) + its fitted temperature. No
benchmark-specific tuning of any kind.

Reproduce:

    git clone https://github.com/fstandhartinger/jevbench && cd jevbench
    for t in easy original hard; do
      python -m jevbench.cli run --tasks datasets/public/$t.jsonl --adapter typesafe \
        --endpoint http://127.0.0.1:8000 --model reflex --key-env SPARK_KEY \
        --price-in-per-m 0 --price-out-per-m 0 --results ../jbrun/$t.results.jsonl \
        --raw-dir ../jbrun/raw-$t --ledger ../jbrun/$t.ledger.jsonl --cap-usd 1
      python -m jevbench.cli summarize --tasks datasets/public/$t.jsonl --results ../jbrun/$t.results.jsonl
    done

## Accuracy on the same public items (published systems from the v1.2 per-task artifact)

| system | easy (48) | standard (72) | hard (111) |
|---|---|---|---|
| GPT-5.6 Luna / DeepSeek V4.1 Flash (chat models, verbalized probabilities) | 1.000 | 0.972 / 0.986 | 0.964 |
| Jev 1.13.0 (TypeSafe, proprietary) | 1.000 | 0.986 | 0.730 |
| SemIf (Qwen3.5-4B) | 1.000 | 0.986 | 0.613 |
| **reflex mix3 (Qwen3.5-4B + LoRA, the published adapter)** | **1.000** | **0.944** | **0.604** |
| reflex mix2 | 1.000 | 0.972 | 0.541 |
| reflex mix1 | 1.000 | 0.931 | 0.595 |
| open-alternative-jev (Qwen3.5-4B) | 1.000 | 0.833 | 0.568 |
| system-one-open (Gemma 4 E2B LoRA) | 1.000 | 0.931 | 0.486 |
| decider-2b (Qwen3.5-2B, self-reported in its bench request) | 1.000 | 0.847 | 0.459 |

reflex mix3, other measurements: schema validity 100 %, 0 failed requests; ECE 0.025 (easy),
0.033 (standard), 0.117 (hard); distribution fidelity on the 10 public probability items
0.689; calibration axis on public hard items ≈ 72.7 (mix1 70.6, mix2 68.2). Latency p50
0.12 s / p95 0.78 s on localhost (not comparable to the published Speed axis, which is
measured from Germany and adjusted ×2 + 0.15 s for self-hosted endpoints).

The three adapters and the public 231 were used as a development gate four times (mix1,
mix2, mix2 with refit temperatures, mix3); recipes were designed after seeing which public
families the raw model failed. No benchmark item was used in training (8-gram overlap
check, 0 rows). Expect held-out hard items to score somewhat below the public ones.

## What the misses are

* Standard tier, 5 of 72: three "adequacy" items where the response subtly fails the
  request (`{"value":3}` is not a JSON array; "All done now" is three words, not two) and
  two routing items where "read the contract and list dates" was routed to `coding_agent`
  instead of `document`. All five are instruction-precision errors, not format errors.
* Hard tier, 45 of 111, by family: temporal/numeric 11 of 15 (date arithmetic and
  money sums: a 4B model without generation cannot compute), long policy 9 of 19
  (2-3k-token rule documents), multi-hop 7 of 18, probability 5 of 10, judge 5 of 17,
  ambiguous 4 of 7, trade-off 3 of 6.
* The probability items show the remaining calibration gap: the model is often right
  about the direction but far too peaked (e.g. gold 0.65/0.20/0.15, reflex 0.07/0.88/0.05
  on an incident-cause item).

## What this says

The LoRA'd 4B sits between the two other Qwen3.5-4B rebuilds on the hard tier, with
zero benchmark-directed training. The standard-tier gap to Jev and SemIf (0.931 vs
0.986) is five items and is the cheapest thing to close: they are exactly the
"adequacy" and "routing with a rule" shapes that a few hundred labelled examples of
would fix. The hard-tier numeric family is a model-size problem, not a reflex problem.
