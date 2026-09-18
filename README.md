# reflex

**A tiny "decision model" you can run on your own GPU.**

You give it some information (a support ticket, a document, a photo) and a list of
questions with fixed answer options. It answers *all* the questions at once and tells you
**how sure it is about each option**, as percentages. It never writes free text, so it can
never make up an answer that isn't on your list.

It is an open re-creation of [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev),
the "System One" model TypeSafe released in September 2026, built on top of a normal
open-weights model ([Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by default).

```
state:     "My payouts have failed three times this week and nobody replied to my emails."

question   which team?        ->  payments 99.9%   account 0.04%   other 0.04%
question   escalate?          ->  yes 90%
question   how urgent (0-2)?  ->  1.9   (low 4%  medium 1%  high 95%)
question   refund requested?  ->  yes 2%
```

That whole answer comes back in about 100 ms once the state is cached. Your code then
decides what to do with the numbers ("auto-route if above 90 %, otherwise ask a human").

## Try it in your browser first

**https://kshetrajna12.github.io/reflex/** runs the whole thing on your own GPU through
WebGPU (Chrome, Edge, or Safari 18+) with a 650 MB Qwen3.5-0.8B model. Pick a preset,
load the model once, drop in a photo, press Run. Nothing is uploaded anywhere. It is the
same request format and the same readout as the Python version below, just smaller and
less calibrated. The page lives in [`docs/`](docs/): `reflex.js` is the inference module,
`app.js` the UI.

## Why would I want this?

Large chat models are great at reasoning but slow and expensive when all you need is a
quick, structured judgment: *which queue, is this spam, how angry is this customer, does
this photo match its caption*. A decision model:

- **returns numbers, not prose** – nothing to parse, no JSON that fails to validate;
- **answers many questions in one pass** – ask 20 questions about one document for
  roughly the cost of one;
- **tells you its confidence, honestly** – after calibration, "85 % sure" really means
  right about 85 % of the time, so you can set thresholds;
- **works on images too** – put a photo in the state and ask typed questions about it.

## Get started in five minutes

You need a Linux machine with an NVIDIA GPU (8 GB of memory is enough for the default
model), Python 3.12, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kshetrajna12/reflex
cd reflex
uv sync                                   # installs PyTorch (CUDA 13), transformers, etc.
uv run python examples/support_ticket.py  # downloads Qwen3.5-4B (~8 GB) the first time
```

You should see the ticket example above printed as JSON, plus timings. The very first
call takes ~20 seconds while GPU kernels compile; after that it is fast.

Try it on a photo:

```bash
uv run python examples/image_triage.py path/to/photo.jpg --caption "two cats on a couch"
```

### Ask your own questions

```python
from reflex import Engine, SystemOneRequest

engine = Engine.load("Qwen/Qwen3.5-4B")

resp = engine.answer(SystemOneRequest(
    state={"email": "Hi, I was charged twice for my subscription this month, please fix."},
    questions={
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "charges, refunds, invoices",
                         "technical": "bugs, outages",
                         "sales": "pricing, upgrades"},
        },
        "angry": {"type": "noul", "instructions": "Is the customer angry?"},
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["can wait a week", "should be handled today", "blocked right now"],
        },
    },
))

print(resp.answers["team"].choice, resp.answers["team"].probabilities)
print(resp.answers["angry"].noul)          # probability of "yes"
print(resp.answers["urgency"].score)       # 0.0 .. 2.0, probability-weighted
```

### The three question types

| type | asks | you get back |
|---|---|---|
| `noul` | "is this true?" | `noul`: probability of yes |
| `choice` | "which one of these?" | `choice` (best option), `probabilities` for every option, `confidence` |
| `score` | "how much, on this scale?" | `score` (weighted position), `probabilities` per level, `legend`, `confidence` |

- `instructions` is the question. `criteria` are the options (choice), the ordered levels
  from low to high (score), or optional descriptions of what "yes" and "no" mean (noul).
- `state` can be a string or any JSON. Put an image anywhere in it as
  `{"type": "image", "source": "<file path, URL, or data: URI>"}`.
- All questions in one request are answered independently and in parallel.

### Run it as a server

```bash
uv run reflex-serve --model Qwen/Qwen3.5-4B --port 8008
```

```bash
curl -s localhost:8008/v1/systemone -H 'content-type: application/json' -d '{
  "state": "The export button crashes in Safari but works in Chrome.",
  "questions": {
    "browser_specific": {"type": "noul", "instructions": "Is the bug browser-specific?"},
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["cosmetic", "degraded but there is a workaround", "blocking"]}
  }
}'
```

The request and response shapes are the same as TypeSafe's hosted API, so client code
written for Jev can point at `http://localhost:8008` instead.

## Make the percentages honest (calibration)

Out of the box the numbers are *roughly* right. To make them trustworthy for
thresholds, run the calibration check. It answers 1,200 exam questions and measures how
well confidence matches accuracy:

```bash
uv run reflex-eval-mmlu --n 1200 --fit-temperature runs/calibration.json
uv run reflex-serve --calibration runs/calibration.json
```

What we measured (lower ECE = more honest; Jev reports 0.031):

| model | accuracy | honesty (ECE) before | after |
|---|---|---|---|
| Qwen3.5-4B | 72 % | 0.090 | **0.039** |
| Qwen3-8B | 71 % | 0.264 | 0.061 |

If you have your own labelled examples (say, 1,000 old tickets with the right queue),
you can go further and fine-tune a small adapter so the model is calibrated *on your
data*:

```bash
uv run reflex-calibrate train --data my_tickets.jsonl --val my_tickets_val.jsonl --out runs/lora
uv run reflex-serve --adapter runs/lora --calibration runs/lora/calibration.json
```

The data format is one JSON object per line: `{"state": ..., "questions": {...}, "labels": {question_id: answer}}`.

## Example: triaging a pull request

`examples/pr_review.py` is a small AI PR-review triage built on this: a PR-level state
(title, description, files) answers *what kind of change, how risky, breaking, needs a
migration, does the description match*, and every diff hunk answers *sensitive area,
weakens error handling, debug leftovers, public API change, behaviour change, and how
much a senior reviewer would want to look*. Code aggregates the numbers and prints the
hunks to hand to a reasoning model or a human.

```bash
uv run python examples/pr_review.py --repo pydantic/pydantic --pr 13824
```
```
kind            feature    feature   98%  bugfix    1%  chore    0%
risk            1.96 / 3   (max hunk needs-eyes 2.02 / 3)
breaking          29%      needs migration   27%      description matches   82%
hunks           40 reviewed, 0 skipped   logic hunks 20   test files changed 7
escalate to a reasoning model / human (P(needs a careful read) >= 50%):
    84%  pydantic-core/src/validators/counter.rs        @@ -0,0 +1,182 @@   +182/-0
    82%  pydantic-core/src/input/input_python.rs        @@ -487,6 +488,28 @@ +22/-0
    81%  pydantic-core/src/serializers/type_serializers/counter.rs  ...    +164/-0
1.2s PR level, 13.3s for 40 hunks
```
The raw model already orders things sensibly; the point of the design is that the
outputs are numbers, so thresholds are yours, and with your own history of reverted or
hotfixed PRs the calibrator can be trained so "80 %" means 80 % on your codebase.

## How it works, in one paragraph

The state is run through the model once and its internal cache is kept. Every question is
then run as a separate branch that can see the state but not the other questions, all in
the same forward pass. Instead of letting the model write an answer, we look at what it
*would* say next, keep only the answer labels (A/B/C or Yes/No), and turn those scores
into percentages. A single "temperature" number, fitted on labelled data, makes the
percentages honest. The details, the design trade-offs, and the mapping to the Jev
write-ups are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Good to know

- Works with Qwen3.5 (default, handles images), Qwen3, and Qwen3-VL checkpoints. Any
  size that fits your GPU; `--model Qwen/Qwen3.5-0.8B` runs on very small cards.
- The browser demo runs all questions of a request in one batched forward pass, but
  re-reads the state for every question and does not cache it between requests, so it
  is fine for a handful of questions, not hundreds.
- A `choice` question can have up to 26 options; `score` can have 2 to 10 levels.
- The model is not magic: check its answers on a handful of your own examples before
  trusting it, and use the confidence numbers to route uncertain cases to a person.
- Tests: `uv sync --extra dev && uv run pytest` (the GPU tests download small models).

## Credits

Built after reading TypeSafe's [Jev announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
and [docs](https://docs.typesafe.ai/), and Archer Hume's
[Jev's architecture, unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/).
Models by [Qwen](https://huggingface.co/Qwen). Not affiliated with TypeSafe.

## License

[MIT](LICENSE). Do whatever you like with it. The Qwen model weights carry their own
(Apache-2.0) license.
