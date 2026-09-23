# Paged choice: more options than there are letters (2026-09-22)

> **Status.** Current, and the reason `MAX_CHOICE_OPTIONS` is 256 rather than 26
> (`src/reflex/schema.py`, [../SERVING.md](../SERVING.md)). Measured on the 4B, against the
> frozen Decision Index rows the 27B run refused.

Question: the readout labels options with single letters, so a choice was capped at 26,
and on the Decision Index that cap refused 14,500 requests outright — every one of them
scored zero. If the options are shown as pages instead, does the readout answer those
questions, does it answer them better than not answering, and does anything change for
the questions that already fitted?

Answer: **it answers all of them, and nothing changes below 27 options.** Every one of
the 16,088 frozen rows across the five refused benchmarks came back answered, none
unsupported and none in error. Questions of 2 to 26 options are bit-for-bit what they
were: across 650 probabilities compared at every size from 2 to 26, the largest absolute
difference from the parent commit is **0**, and no top choice moves. The cost is latency
that grows about linearly with the page count, from 186 ms at one page to 2.3 s at ten.

## What the refused benchmarks score

All rows, both option orders, 4B `stable`, nothing sampled. `raw` is correct over *all*
rows, the same quantity the index uses, so the 27B column is what those benchmarks
contributed to the 56.23.

| benchmark | rows | options | pages | answered | raw, 4B | raw, 27B run | chance |
|---|---|---|---|---|---|---|---|
| API-Bank | 508 | 53 | 3 | 508 | 0.6122 | refused | — |
| BANKING77 | 3080 | 77 | 3 | 3080 | 0.4951 | refused | — |
| CLINC150+OOS | 5500 | 151 | 6 | 5500 | 0.6209 | refused | — |
| POP909-CL | 2000 | 129 | 5 | 2000 | **0.0480** | 0.0000 | 0.0078 |
| ChessBench | 5000 | 2–81 | 1–4 | 5000 | **0.1096** | 0.0880 | 0.0819 |
| | 16,088 | | | **16,088** | | | |

Not one row came back unsupported and not one came back in error. The three benchmarks
outside the index panel are the ones that gain most: API-Bank and CLINC150+OOS go from
zero to above 0.6, and CLINC150+OOS is a 151-way classification read six pages at a time.

The two panel benchmarks gain much less, and it is worth being plain about why. POP909-CL
asks the 4B to name a chord's pitch class from a piece of music rendered as text; at
0.0480 against a chance rate of 0.0078 it is doing something, roughly six times chance,
but it is not doing it well. ChessBench at 0.1096 is barely above its 0.0819 chance rate.
Pages remove the refusal. They do not make a 4B good at music theory or chess.

ChessBench is the one benchmark here that spans the boundary, so it can separate "paged"
from "harder". Its single-page rows are exactly the 1588 the 27B run could already
answer:

| pages | rows | accuracy | chance | lift over chance |
|---|---|---|---|---|
| 1 | 1588 | 0.1820 | 0.1265 | 1.44x |
| 2 | 3372 | 0.0759 | 0.0284 | **2.67x** |
| 3 | 39 | 0.0769 | 0.0178 | **4.32x** |
| 4 | 1 | 0.0000 | 0.0123 | — |

Raw accuracy halves once a question needs a second page, which looks alarming until you
notice that those rows also offer two to three times as many moves to choose between.
Measured against each row's own chance rate the paged rows do **better**, not worse:
2.67x against 1.44x. Whatever the pooled softmax is doing across page boundaries, it is
not throwing away the signal.

## What a page is

There are 26 single-letter labels, so a branch can show at most 26 options. A choice with
more is cut into **pages** of at most 26, and a page is an ordinary branch: the same
prompt structure, the options relettered from A, plus one line saying which slice of the
options this page shows — "Options 27-52 of 151; the other options are shown separately."
All the pages of one option order go into the same forward pass, exactly as separate
questions already did.

Three details carry the design.

**The pages of one order are read as one distribution.** They share a `Branch.group`, and
`readout.merge_branches` takes a single softmax over their pooled labels rather than one
softmax per page. This is the whole point. A per-page softmax would give every page total
probability one, which asserts that the answer is as likely to sit among 26 also-rans as
among the 26 that actually hold it.

**Pooling requires one scale.** A raw logit carries a per-prompt offset, and a pooled
softmax would read that offset as preference for whichever page happened to sit higher.
So `Engine.restrict` hands a paged branch's labels over as full-vocabulary
log-probabilities instead of raw logits. The SGLang backend reports log-probabilities
natively and needed no change.

**Pages are cut after the order is permuted.** `stable` asks every question in two
distinct option orders, so the two orders page the question differently, and an option
that shared a page with its strongest rival in one order generally does not in the other.
Order averaging therefore also averages over page boundaries, which is the behaviour you
want from it.

A question of 2 to 26 options is one page. It renders no page line at all, gets no
normalisation, and its pooled array is that single branch's logits — so its prompt is
byte-for-byte and its answer bit-for-bit what reflex produced before pages existed.

## The regression: nothing below 27 options moved

The claim to check is the narrow one, because it is the one that protects every existing
result. It was checked two ways, both against **6dce470**, the branch's actual parent,
served with identical flags from a second worktree.

The 100-row Decision Index sample first. 83 of its rows are answered by both commits, and
those 83 contain only 2- and 4-option questions:

| | branch 04df994 | parent 6dce470 |
|---|---|---|
| rows answered | 100 | 83 |
| rows refused | 0 | 17 (CLINC150+OOS, 151 options) |
| probabilities compared | 300 | |
| max absolute difference | **0** | |
| changed top choices | **0** | |

That sample cannot reach the boundary, so both servers were then probed directly with a
synthetic choice at every size from 2 to 29, plus 52 and 151, with fixed option texts:

| option count | parent 6dce470 | branch 04df994 | probabilities compared | max difference |
|---|---|---|---|---|
| 2 … 26 | answers | answers | 350 | **0** |
| 27, 28, 29 | 422 `at most 26 options per choice` | answers (2 pages) | | |
| 52 | 422 | answers (2 pages) | | |
| 151 | 422 | answers (6 pages) | | |

650 probabilities compared in total, maximum absolute difference 0, no top choice moved.
The same comparison against the older `smoke-4b` run, taken at e8af022, is also exactly 0,
so there is no unrelated drift to disentangle.

## Latency

Single requests against a warm server, small state, five repetitions, median:

| options | pages | median |
|---|---|---|
| 26 | 1 | 186 ms |
| 52 | 2 | 425 ms |
| 78 | 3 | 624 ms |
| 104 | 4 | 859 ms |
| 130 | 5 | 1129 ms |
| 156 | 6 | 1339 ms |
| 208 | 8 | 1803 ms |
| 256 | 10 | 2261 ms |

Growth is close to linear in the page count, at roughly 230 ms per additional page, which
is what one expects when a page is just another branch in the same batched forward pass.
The first 256-option request cost 8.4 s against 2.26 s for the four that followed; that is
a one-off warm-up of the widest batch shape, not a property of ten pages.

Under the measurement run the medians are much larger — API-Bank 10.8 s, POP909-CL 16.2 s,
CLINC150+OOS 3.9 s, ChessBench 3.1 s, BANKING77 2.0 s — and pages are not why. Two other
things dominate. The first is state prefill: API-Bank carries about 6.5k tokens of tool
definitions and POP909-CL about 2.7k tokens of music, no two rows share a prefix, so every
request pays a full cold prefill. The second is that these ran four at a time against one
server, so each median includes queueing behind three others. Sort the benchmarks by
median and you get the order of their state sizes, not the order of their page counts:
POP909-CL needs five pages and is slower than CLINC150+OOS, which needs six.

The whole 16,088-row run took 20,198 s, about 5 h 37 m, at four concurrent workers.

## What this would be worth to the index

**This is an estimate, and it mixes model scales. It is not a 27B measurement.** The
scores substituted below were measured on the 4B; the rest of the arithmetic is the 27B
run's. Read it as what answering those rows is worth, not as a prediction of what the 27B
would score.

The index is 100 times the mean of five area scores, and an area score is the mean of its
benchmarks' raw scores. Two of the refused benchmarks sit in the panel: POP909-CL in arts,
ChessBench in knowledge. Substituting the 4B's measured raw scores as a floor:

```
arts      = (0.9619 + 0.6301 + 0.0480 + 0.6438 + 0.4326) / 5 = 0.5433   (was 0.5337)
knowledge = (0.8263 + 0.5153 + 0.6535 + 0.6123 + 0.6396 + 0.1096) / 6 = 0.5594   (was 0.5558)

index = 100 x (0.5594 + 0.6260 + 0.3613 + 0.7346 + 0.5433) / 5 = 56.49
```

**56.23 → 56.49, up 0.26.** That is small, and the reason is worth stating: the two panel
benchmarks are the two the 4B is worst at. The three that improve most — API-Bank,
BANKING77, CLINC150+OOS — are outside the panel entirely, so none of their gain reaches
the headline number even though they account for 9,088 of the 14,500 refused requests.

A second, more generous estimate uses only the 27B's own data for ChessBench. It answered
1588 of 5000 rows and scored 0.0880 overall, which is 0.2771 on the rows it answered; if
the 3412 it refused had gone the same way, ChessBench raw would be 0.2771 and the index
**57.05**, up 0.82. The truth for a 27B is likely between the two, and in both cases the
honest summary is that removing 14,500 zeros is worth under a point of index.

The index is not the whole of it. 14,500 refusals were 11 % of the suite, and a refusal is
a different kind of failure from a wrong answer: it is the model declining to have an
opinion. The report now says the 27B answered every question it was asked.

## The frozen suite, and one hash that does not match

These rows were rebuilt on the measurement box rather than downloaded, and the rebuild is
**incomplete**: HLE (catalog 45) is a gated HuggingFace dataset and no credentials were
available, so it contributes no rows. The published suite (`suite download`) needs
credentials for the same reason. The result:

| | rebuilt here | office manifest |
|---|---|---|
| requests | 131,909 | 132,422 |
| benchmarks | 36 | 37 |
| `selected-rows.jsonl` SHA-256 | `3eb7aa1e…` | `288d3720…` |

The entire 513-request deficit is HLE, which is the only one of the 37 benchmarks to fail,
in both acquisition and normalisation. The suite was built twice, independently: once by
the tolerant per-benchmark driver that records a bad source instead of raising, and once
by calling the five normalizers and `freeze` directly. Both produced `selected-rows.jsonl`
with the same SHA-256, `3eb7aa1e…`, which is worth more than either build alone.

**For the five benchmarks measured here the selection is provably the office one**,
because `freeze` chooses rows by a SHA-256 priority over a fixed seed, per benchmark, so
an absent benchmark cannot perturb another.
The row counts come out exactly as the 27B run records them — API-Bank 508, BANKING77
3080, CLINC150+OOS 5500, POP909-CL 2000, ChessBench 5000 — and ChessBench is the
independent check: 1588 of its 5000 rows fit in a single page, and the 27B run reports
ChessBench coverage of 0.3176, which is 1588/5000. The rows scored below are the rows that
run refused.

## Reproducing

The branch predates e8af022, so `--max-branch-tokens` does not exist on it; `serve_4b.sh`
from the Decision Index kit will not start here. `uv sync` also does not install
causal-conv1d, which `--require-fast-kernels` insists on.

```bash
# worktrees: the branch, and its parent as the regression baseline
git worktree add ~/src/github.com/reflex-lc   origin/large-choice
git worktree add ~/src/github.com/reflex-base 6dce470
for w in reflex-lc reflex-base; do (cd ~/src/github.com/$w && uv sync); done
# the fast kernels --require-fast-kernels wants, from the wheel uv already built
VIRTUAL_ENV=~/src/github.com/reflex-lc/.venv uv pip install --no-deps causal-conv1d==1.7.0

# serve the branch (8021) and, separately, the parent (8022) — one at a time
REFLEX_API_KEY=di uv run reflex-serve --stable --served-name reflex \
  --host 127.0.0.1 --port 8021 --require-fast-kernels --max-pack-tokens 65536

# the frozen rows. The whole suite, with a bad source recorded rather than raised, since
# the stock `suite rebuild` aborts on the first one and HLE is gated:
python rebuild_driver.py ~/scratch/di/work      # writes rebuild-report.json with the hashes

# or, for these five benchmarks only, the normalizers and freeze called directly —
# same selected-rows.jsonl, same SHA-256:
python -c 'from decision_index.suite.build import rebuild; from decision_index.suite.build.layout import Layout; from pathlib import Path; l=Layout(Path.home()/"scratch/di/work"); [rebuild.BUILDERS[n](l) for n in (3,4,5,22,31)]'
python -c 'from decision_index.suite.build import freeze; from decision_index.suite.build.layout import Layout; from pathlib import Path; l=Layout(Path.home()/"scratch/di/work"); print(freeze.freeze(l, l.suite/"release-v1-rebuilt", log=lambda *a: None)["selected_rows_sha256"])'

# the regression, against each server in turn
DECISION_INDEX_API_KEY=di python -m decision_index run \
  --engine decision_index.engines.reflex_http:ReflexHttpSystemOne \
  --option base_url=http://127.0.0.1:8021 --option model=reflex \
  --rows ~/scratch/di/sample-100.jsonl.gz --out ~/scratch/di/runs/lc-sample100 --fresh

# the measurement: all rows of the five benchmarks the 27B run refused
DECISION_INDEX_API_KEY=di python -m decision_index run \
  --engine decision_index.engines.reflex_http:ReflexHttpSystemOne \
  --option base_url=http://127.0.0.1:8021 --option model=reflex \
  --rows ~/scratch/di/rows-paged.jsonl.gz --out ~/scratch/di/runs/lc-paged-5bench --workers 4
```

`--stable` already sets two option orders (`serving/stable.json`, `permutations: 2`) and
the HTTP engine sends no `permutations` of its own, so the runs above are two-order runs,
the same shape as the 27B run.
