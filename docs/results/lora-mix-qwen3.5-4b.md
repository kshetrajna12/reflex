# LoRA on the public mix, Qwen3.5-4B

> **Status: rejected on transfer.** The gains below are in-distribution. The control that
> followed showed the frozen model is the better judge on the hard tier and on held-out
> external data ([frozen-vs-trained.md](frozen-vs-trained.md)). Kept as a record, and as
> the recipe to copy if you are training on your own workload's labels.

Command:

    reflex-data mix --out runs/mix_train.jsonl --eval-out runs/mix_eval.jsonl --per-source 800
    reflex-calibrate train --model Qwen/Qwen3.5-4B --data runs/mix_train.jsonl \
        --val runs/mix_eval.jsonl --out runs/lora-mix --epochs 1 --lr 1e-4

6,400 training examples (800 per source), 1,720 held-out questions (200 items per source;
code-review items carry two questions). One epoch, LoRA r=16 on the attention
projections (3.1M trainable parameters, 0.07 %), 43 minutes on one GB10.

| source | task | acc before → after | ECE before → after → +T |
|---|---|---|---|
| all | | 62.7 → **76.8** | 0.120 → 0.051 → **0.024** |
| banking77 | 12-way intent | 92.0 → 94.0 | 0.052 → 0.040 → 0.047 |
| clinc | 11-way intent + none | 92.5 → 95.5 | 0.047 → 0.030 → 0.042 |
| mmlu_pro | 10-way exam | 47.5 → 59.0 | 0.189 → 0.093 → 0.086 |
| civil_comments | toxic? (soft) | 50.0 → 94.0 | 0.222 → 0.044 → 0.060 |
| halueval | hallucinated? | 77.5 → 99.0 | 0.037 → 0.010 → 0.005 |
| msmarco | passage relevant? | 66.5 → 64.0 | 0.220 → 0.076 → 0.056 |
| helpsteer2 | helpfulness 0-4 | 33.0 → 37.0 | 0.252 → 0.136 → 0.122 |
| codereview | needs comment? + type | 50.0 → 73.8 | 0.136 → 0.128 → 0.082 |

T = 1.234 fitted on the same held-out set after training (so the last column is
slightly optimistic; fit it on a separate split for a real deployment).

Reading it:

* One cheap epoch lifts overall accuracy 14 points and cuts calibration error by
  more than half; with a temperature on top the whole mix is at ECE 0.024, better than
  the 0.031 Jev reports on MMLU.
* The biggest wins are where the raw model had the *format* wrong rather than the
  knowledge: toxicity (it said "toxic" too often), hallucination checks, code review.
* MS MARCO accuracy dipped 2.5 points while its calibration improved a lot; the model
  learned to hedge on passages that partially answer the query. Whether that is a win
  depends on how you use the number.
* Scoring (HelpSteer2, 5 levels) stays hard: 37 % exact-level accuracy. Exact level is a
  harsh metric for a Score; mean absolute level error or Brier is what to track there.
* Nothing here is tuned. Same recipe, more data, two epochs, or `--lora-mlp` are the
  obvious next knobs; the trainer prints this table so you can see what each buys.

Raw report:

```
### all
n=1720  acc=0.7680  ECE=0.0510  Brier=0.2979  NLL=0.5837  mean_conf=0.8177
  conf-bin   mean_conf  accuracy  count
  [0.13,0.20)     0.180     0.333      3
  [0.20,0.27)     0.247     0.182     22
  [0.27,0.33)     0.311     0.333     33
  [0.33,0.40)     0.368     0.273     44
  [0.40,0.47)     0.434     0.375     48
  [0.47,0.53)     0.507     0.352    108
  [0.53,0.60)     0.570     0.449     89
  [0.60,0.67)     0.635     0.569     72
  [0.67,0.73)     0.706     0.690    100
  [0.73,0.80)     0.775     0.620    129
  [0.80,0.87)     0.833     0.755    110
  [0.87,0.93)     0.904     0.877    122
  [0.93,1.00)     0.987     0.973    840

### banking77
n=200  acc=0.9400  ECE=0.0398  Brier=0.0952  NLL=0.2193  mean_conf=0.9516
  conf-bin   mean_conf  accuracy  count
  [0.40,0.47)     0.427     0.333      3
  [0.47,0.53)     0.522     0.000      2
  [0.53,0.60)     0.583     0.667      3
  [0.60,0.67)     0.653     1.000      3
  [0.67,0.73)     0.700     1.000      2
  [0.73,0.80)     0.768     0.500      4
  [0.80,0.87)     0.844     1.000      6
  [0.87,0.93)     0.899     0.625      8
  [0.93,1.00)     0.991     0.988    169

### civil_comments
n=200  acc=0.9400  ECE=0.0436  Brier=0.0854  NLL=0.1555  mean_conf=0.9162
  conf-bin   mean_conf  accuracy  count
  [0.47,0.53)     0.531     0.500      4
  [0.53,0.60)     0.585     0.250      4
  [0.60,0.67)     0.632     1.000      3
  [0.67,0.73)     0.706     0.667      6
  [0.73,0.80)     0.785     0.750      8
  [0.80,0.87)     0.837     0.889      9
  [0.87,0.93)     0.912     0.950     40
  [0.93,1.00)     0.971     1.000    126

### clinc
n=200  acc=0.9550  ECE=0.0303  Brier=0.0762  NLL=0.1837  mean_conf=0.9536
  conf-bin   mean_conf  accuracy  count
  [0.47,0.53)     0.511     1.000      2
  [0.53,0.60)     0.553     1.000      1
  [0.60,0.67)     0.629     0.250      4
  [0.67,0.73)     0.692     0.667      3
  [0.73,0.80)     0.768     0.833      6
  [0.80,0.87)     0.836     0.833      6
  [0.87,0.93)     0.903     1.000     14
  [0.93,1.00)     0.990     0.982    164

### codereview
n=320  acc=0.7375  ECE=0.1281  Brier=0.4135  NLL=0.8334  mean_conf=0.8597
  conf-bin   mean_conf  accuracy  count
  [0.47,0.53)     0.516     0.750      4
  [0.53,0.60)     0.580     0.286      7
  [0.60,0.67)     0.644     0.333      6
  [0.67,0.73)     0.707     0.667     33
  [0.73,0.80)     0.775     0.603     63
  [0.80,0.87)     0.830     0.679     56
  [0.87,0.93)     0.897     0.731     26
  [0.93,1.00)     0.986     0.896    125

### halueval
n=200  acc=0.9900  ECE=0.0104  Brier=0.0168  NLL=0.0426  mean_conf=0.9914
  conf-bin   mean_conf  accuracy  count
  [0.53,0.60)     0.562     1.000      1
  [0.60,0.67)     0.622     0.000      1
  [0.67,0.73)     0.706     1.000      1
  [0.87,0.93)     0.919     1.000      2
  [0.93,1.00)     0.998     0.995    195

### helpsteer2
n=200  acc=0.3700  ECE=0.1359  Brier=0.6759  NLL=1.2576  mean_conf=0.4668
  conf-bin   mean_conf  accuracy  count
  [0.20,0.27)     0.254     0.111      9
  [0.27,0.33)     0.313     0.500     16
  [0.33,0.40)     0.371     0.346     26
  [0.40,0.47)     0.436     0.378     37
  [0.47,0.53)     0.502     0.277     65
  [0.53,0.60)     0.561     0.433     30
  [0.60,0.67)     0.623     0.571     14
  [0.67,0.73)     0.690     1.000      3

### mmlu_pro
n=200  acc=0.5900  ECE=0.0928  Brier=0.5184  NLL=1.2097  mean_conf=0.6652
  conf-bin   mean_conf  accuracy  count
  [0.13,0.20)     0.180     0.333      3
  [0.20,0.27)     0.241     0.231     13
  [0.27,0.33)     0.310     0.176     17
  [0.33,0.40)     0.364     0.167     18
  [0.40,0.47)     0.430     0.375      8
  [0.47,0.53)     0.485     0.333      9
  [0.53,0.60)     0.563     0.471     17
  [0.60,0.67)     0.631     0.444      9
  [0.67,0.73)     0.700     0.818     11
  [0.73,0.80)     0.765     0.583     12
  [0.80,0.87)     0.835     0.733     15
  [0.87,0.93)     0.899     0.889     18
  [0.93,1.00)     0.978     0.940     50

### msmarco
n=200  acc=0.6400  ECE=0.0757  Brier=0.4328  NLL=0.6181  mean_conf=0.7116
  conf-bin   mean_conf  accuracy  count
  [0.47,0.53)     0.521     0.455     22
  [0.53,0.60)     0.580     0.462     26
  [0.60,0.67)     0.640     0.625     32
  [0.67,0.73)     0.709     0.634     41
  [0.73,0.80)     0.779     0.611     36
  [0.80,0.87)     0.831     0.833     18
  [0.87,0.93)     0.903     0.929     14
  [0.93,1.00)     0.965     0.909     11

```
