# Frozen baselines across weight classes (Phase 2 of MODEL_QUALITY_PLAN.md)

Direct readout, default prompt, no adapter, temperature 1, at one and at two distinct option
orders. All Qwen3.5 dense checkpoints except the 27B, which is Qwen3.8. bf16 throughout.

## External sets (300 items each, never trained on; accuracy)

| model | orders | support intents | MNLI mismatched | toxic-chat | Yelp stars | mean |
|---|---|---|---|---|---|---|
| 0.8B | 1 | 0.477 | 0.470 | 0.500 | 0.197 | 0.411 |
| 0.8B | 2 | 0.543 | 0.433 | 0.543 | 0.253 | 0.443 |
| 2B | 1 | 0.760 | 0.490 | 0.717 | 0.440 | 0.602 |
| 2B | 2 | 0.820 | 0.583 | 0.660 | 0.450 | 0.628 |
| 4B (stable) | 1 | 0.907 | 0.840 | 0.843 | 0.640 | 0.808 |
| 4B (stable) | 2 | 0.917 | 0.837 | 0.833 | 0.653 | 0.810 |
| 9B | 1 | 0.950 | 0.860 | 0.773 | 0.600 | 0.796 |
| 9B | 2 | 0.950 | 0.857 | 0.810 | 0.617 | 0.808 |
| 27B | 1 | 0.927 | 0.850 | 0.740 | 0.683 | 0.800 |
| 27B | 2 | 0.937 | 0.850 | 0.813 | 0.683 | 0.821 |

Top-label ECE per set at two orders: 0.8B 0.08 / 0.16 / 0.03 / 0.31; 2B 0.08 / 0.13 / 0.10 / 0.08;
4B 0.05 / 0.07 / 0.04 / 0.06; 9B 0.03 / 0.07 / 0.05 / 0.15; 27B 0.03 / 0.05 / 0.07 / 0.11.

What the table says:

* **Below 4B the frozen readout is not a usable judge.** The 0.8B is near chance on every set,
  and its jump under two orders shows how much of its answer was the position of the option
  rather than its content. The 2B is at 0.60. Both classes need training to be worth serving;
  the browser demo, which runs the 0.8B, is a demonstration of the mechanism, not of the
  quality.
* **4B to 27B is a plateau on these sets**, which sit at their label ceiling for models this
  size; the differences are inside the noise of 300 items except toxic-chat, where the larger
  models flag more jailbreak-shaped prompts than the annotators do.
* **Two orders help every size**, most of all the small ones, and mainly through calibration
  at 4B and above.

## Public benchmark items

*(pending: 0.8B, 2B, 9B at one and two orders; 4B and 27B are in
[order-averaging.md](order-averaging.md))*
