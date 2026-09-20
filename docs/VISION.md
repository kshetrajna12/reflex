# Where reflex is going: a cascade, not a single pass

Target: a 4B decision model that is as close to Jev as its weight class allows, on any 16 GB
GPU, with a ~200 ms fast path. The evidence so far (docs/results/) says the 4B cannot get
there in one forward pass: its hard-tier misses are reasoning-shaped (temporal/numeric,
multi-hop, long policy), no training route has moved them without losing elsewhere, and no
readout trick touches them. But the same 4B, allowed to reason, scores at the 27B's level on
the hard tier and above Jev on the standard tier. The judge is in there.

So the system is a cascade:

1. **Fast path.** Frozen 4B, default prompt, two distinct option orders averaged (`stable`).
   Answers everything in one pass, ~200 ms. Its cross-order disagreement is a calibrated
   uncertainty signal for free.
2. **Escalation trigger.** A small, separately trained classifier that predicts "the fast path
   is likely wrong here" from what is visible before answering: the fast path's own
   distribution and disagreement, the question type and option count, the state length, and
   cues for reasoning-shaped items (numbers, dates, money, multi-part questions, long
   documents). Trained and selected on our own external sets, never on benchmark items.
   Budgeted: it may escalate roughly 10-15 % of questions.
3. **Slow path.** The thinking readout (`reflex.think`) on escalated questions only, same
   label logits after a bounded reasoning block. Tens of seconds per escalated question.
4. **Honest probabilities.** The fast slice keeps its disagreement-calibrated distribution;
   the slow slice reports the post-reasoning distribution; the response says which path
   answered and why, so callers can set thresholds per path.

What "close to Jev" means for this system, on the public items: hard ≥ 0.70, standard ≥ 0.95,
hard-tier ECE ≤ 0.05, p50 ≤ 0.25 s, p95 bounded by the escalation budget. The 27B at two
orders remains the quality configuration when a 60 GB GPU is available (hard 0.766), and the
same cascade applies to it.

Longer-term levers, in the order we expect them to pay: the escalation trigger; a curriculum of
hard-shaped, verified decision data (MODEL_QUALITY_PLAN.md, Phase 3) for training pilots on
the 2B and 4B; calibration fitted to the cascade as deployed rather than to a single readout.
