# Where reflex is going: one fast pass, made better

reflex is a System One model. Every request is answered by a single forward pass over the
state and the question branches, with no decoding and no reasoning, and the answer is the
label distribution read off the next-token logits. That constraint is the product, not an
implementation detail. Everything below is about making that one pass better, not about
adding a second one.

Target: a 4B decision model that is as close to Jev as its weight class allows, on any 16 GB
GPU, answering in ~200 ms. On the public items the frozen 4B at two orders is at hard 0.685,
standard 0.917, hard ECE 0.081; Jev's official numbers are 0.730 / 0.986 / 0.031. Closing
that gap inside one pass is the whole job.

## The four levers

1. **The readout.** The cheapest gains we have found are here. Asking each question in two
   distinct option orders and averaging the distributions costs one wider pass, no training
   and no fitted parameters, and roughly halves calibration error
   ([order-averaging.md](results/order-averaging.md)). It is the `stable` configuration.
   Wording ensembles did not pay ([order-averaging.md](results/order-averaging.md), "What
   did not help"); more orders than two did not either. Further readout work is welcome as
   long as it stays inside the single pass.
2. **Model size.** The 4B at two orders is the serving default at ~200 ms. The 27B at two
   orders is the quality configuration when a 60 GB GPU is there: hard 0.766, hard ECE
   0.061, about a second per request ([weight-classes.md](results/weight-classes.md)). The
   9B is not a middle ground; it costs twice the 4B's latency for the same hard-tier
   accuracy and worse calibration.
3. **Data-driven calibration.** Per-primitive temperatures and the calibration head are
   fitted, not guessed, and they are what make the probabilities usable as probabilities.
   The remaining distance to Jev is as much calibration as accuracy: 0.081 against 0.031 on
   the 4B. Fitting the calibration to the configuration actually deployed, on our own
   external sets, is the standing lever here.
4. **Training on a verified curriculum.** The frozen 4B still beats every LoRA run we have
   done on the hardest external items ([frozen-vs-trained.md](results/frozen-vs-trained.md)),
   because short classification data is the wrong supervision for judgement. The open bet is
   a curriculum of hard-shaped, verified decision data (MODEL_QUALITY_PLAN.md, Phase 3),
   with a strong offline teacher supplying the soft targets. That is where a trained 4B
   could plausibly beat the frozen one on the hard tier.

## Out of scope for serving: reasoning and cascades

We have measured both. A 4B allowed to reason scores at the 27B's level on the hard tier
([teachers-27b-and-4b-think.md](results/teachers-27b-and-4b-think.md)), and a fitted
escalation trigger routes the right questions to it
([escalation-trigger.md](results/escalation-trigger.md)). Neither ships.

The reason is what reflex is for. A System One model is called inside other software, often
several times per user action, often in a loop, and its value is that a caller can afford to
ask. A readout that answers in 200 ms and a readout that answers in 30 s are different
products with different call sites, and a cascade between them has the worse property of
both: the caller must budget for the tail, so it is a 30 s service that is usually fast. Any
p95 that is not a small multiple of p50 breaks the contract that makes the model worth
calling. When a caller genuinely wants reasoning, it should call a reasoning model directly,
with its own budget and its own timeout, and use reflex's probability to decide when to do
that.

So the serving path has one readout, and `reflex-serve` has no flag that can make it slower
than a forward pass. Reasoning survives in this repo as an offline tool only: `reflex.think`
labels a distillation corpus with a stronger teacher (`reflex-distill label --think N`) and
answers benchmark questions in experiments (`reflex-calibrate eval --think N`). Nothing on
the request path imports it.

## What "close to Jev" means for this system

On the public items: hard >= 0.70, standard >= 0.95, hard-tier ECE <= 0.05, p50 <= 0.25 s,
and p95 within a small multiple of p50 because nothing branches into a slow path. The public
items have been consulted throughout development, so they are a development suite and not an
independent test; an official run is the only claim that counts.
