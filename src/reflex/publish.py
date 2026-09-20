"""Publish a trained adapter (plus its calibration.json) to the Hugging Face hub, so anyone
can serve it with `reflex-serve --adapter <repo>`.

    reflex-publish-adapter runs/lora-mix2 --repo you/reflex-qwen3.5-4b-lora [--private]

Needs a hub token with write access (`hf auth login`, or env HF_TOKEN).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def model_card(base: str, src: Path, repo: str) -> str:
    cal = None
    if (src / "calibration.json").exists():
        with open(src / "calibration.json") as f:
            cal = json.load(f)["temperature"]
    report = (src / "eval_report.txt").read_text() if (src / "eval_report.txt").exists() else ""
    temp = f"{cal['choice']:.3f}" if cal else "1.0"
    return f"""---
base_model: {base}
library_name: peft
license: mit
pipeline_tag: text-classification
tags: [reflex, decision-model, system-one, jev, lora, calibration, routing, moderation]
---

# reflex: a System One decision model (LoRA for {base})

**Give it some state and a list of questions with fixed answers. It answers all of them in
one pass and returns a calibrated probability for every option.** It never writes free
text, so it cannot invent an answer that is not on your list, and there is nothing to parse.

This repo holds a LoRA adapter (plus the fitted calibration temperature) for
[reflex](https://github.com/kshetrajna12/reflex), an open re-creation of TypeSafe's
[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). The adapter was trained
with a proper scoring rule on public datasets and synthetic data.

> **Status (2026-09-19): superseded for general use.** Controls run after this adapter was
> published showed that the *frozen* `{base}` with reflex's current default prompt and no
> calibration file scores better on data neither was trained on, and better on the public
> [JevBench](https://github.com/fstandhartinger/jevbench) items (hard tier 0.658 vs 0.604,
> calibration error 0.086 vs 0.117). The adapter is sharper on inputs that resemble its
> training mix and worse on long, ambiguous ones. Full numbers and the reasoning:
> [docs/results/frozen-vs-trained.md](https://github.com/kshetrajna12/reflex/blob/main/docs/results/frozen-vs-trained.md).
>
> **Recommended:** `git checkout stable && uv run reflex-serve --stable` in the reflex repo
> (no adapter). This adapter stays published for reproducibility and as a worked example
> of the fine-tuning recipe, which *is* the right tool when you have labels from your own
> workload.

## Quick start (this adapter)

```bash
pip install uv && git clone https://github.com/kshetrajna12/reflex && cd reflex && uv sync
uv run reflex-serve --model {base} --adapter {repo} --host 0.0.0.0 --port 8000
```

The adapter and its `calibration.json` are pulled from this repo automatically. Any CUDA GPU
with 16 GB or more works; the first request compiles kernels for about 20 s.

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{{
  "state": {{"ticket": "My payouts have failed three times this week and nobody replied."}},
  "questions": {{
    "queue":    {{"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {{"payments": "payouts, refunds", "account": "login, 2FA", "other": null}}}},
    "escalate": {{"type": "noul",   "instructions": "Should this be escalated to a manager?"}},
    "urgency":  {{"type": "score",  "instructions": "How urgent is this?",
                 "criteria": ["can wait a week", "handle today", "blocked right now"]}}
  }}
}}'
```

```json
{{
  "answers": {{
    "queue":    {{"type": "choice", "choice": "payments",
                 "probabilities": {{"payments": 0.97, "account": 0.02, "other": 0.01}}, "confidence": 0.91}},
    "escalate": {{"type": "noul", "noul": 0.71}},
    "urgency":  {{"type": "score", "score": 1.77, "probabilities": {{"0": 0.05, "1": 0.13, "2": 0.82}},
                 "legend": {{"0": "can wait a week", "1": "handle today", "2": "blocked right now"}}, "confidence": 0.62}}
  }},
  "usage": {{"input_tokens": 258, "state_tokens": 79, "question_tokens": 179, "state_cache_hit": false}}
}}
```

In Python, without a server:

```python
from reflex import Engine, SystemOneRequest

engine = Engine.load("{base}", adapter_path="{repo}")   # calibration picked up from the repo
resp = engine.answer(SystemOneRequest(
    state={{"email": "I was charged twice this month, please fix."}},
    questions={{
        "team":  {{"type": "choice", "instructions": "Which team?", "criteria": {{"billing": None, "technical": None, "sales": None}}}},
        "angry": {{"type": "noul", "instructions": "Is the customer angry?"}},
    }},
))
resp.answers["team"].probabilities, resp.answers["angry"].noul
```

The request and response shapes are the same as TypeSafe's hosted `/v1/systemone`, so
Jev client code works against a reflex server unchanged.

## The three question types

| type | asks | you get back |
|---|---|---|
| `noul` | "is this true?" | `noul`: probability of yes |
| `choice` | "which one of these?" (up to 26 options) | `choice`, `probabilities` per option, `confidence` |
| `score` | "how much, on this ordered scale?" (2 to 10 levels) | `score` (probability-weighted level), `probabilities`, `legend`, `confidence` |

`state` can be a string or any JSON, and can carry images as
`{{"type": "image", "source": "<path, URL or data: URI>"}}` (the base model is natively
multimodal). All questions in a request are answered independently, in parallel, over one
encoding of the state, so asking twenty questions about one document costs about the
same as asking one.

## What it is good for

- **Routing and triage**: support tickets, incidents, pull-request hunks ("does a senior
  reviewer need to read this?"), with thresholds you choose in code.
- **Guardrails on LLM output**: is this answer grounded in the source, does it contain
  PII, is it in scope, as numbers you can threshold, at a small fraction of the cost of
  the model that wrote the answer.
- **Bulk labelling and reranking**: score millions of rows overnight; the probabilities
  become features.
- Not for: generating text, arithmetic, or anything that needs a chain of reasoning.

## How it was trained

`reflex-calibrate` attaches LoRA adapters and minimises a proper scoring rule
(cross-entropy against a hard or soft target) on the model's label-restricted next-token
logits, the supervised form of Jev's "RLCD". Data came from `reflex-data`: public
datasets (banking77, CLINC, MMLU-Pro, civil_comments with soft toxicity labels, HaluEval,
MS MARCO, HelpSteer2 with per-rater score distributions, a GitHub code-review set) plus
synthetic, correct-by-construction data for instruction adequacy, rule-based routing and
reading probabilities off stated base rates. No benchmark items were used; an 8-gram
overlap check against JevBench's public items is part of the pipeline.

Fitted calibration temperature: **{temp}** (applied automatically when served through reflex).

## Held-out evaluation at training time

```
{report}
```

## Limits and honesty

- A 4B model: it judges text well and reads rules reasonably, but it cannot do date or
  money arithmetic and it degrades on multi-thousand-token policy documents.
- Calibration was measured on the sources above. Check a few hundred of your own
  labelled examples before trusting a threshold, and route low-confidence cases to a
  person or a reasoning model.
- Base model weights: Qwen, Apache-2.0. Adapter and code: MIT.
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter_dir")
    ap.add_argument("--repo", required=True, help="hub repo id, e.g. you/reflex-qwen3.5-4b-lora")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--private", action="store_true")
    ap.add_argument(
        "--card-only", action="store_true", help="upload only README.md (refresh the model card)"
    )
    args = ap.parse_args(argv)

    from huggingface_hub import HfApi

    src = Path(args.adapter_dir)
    if not (src / "adapter_config.json").exists():
        raise SystemExit(f"{src} is not an adapter dir (no adapter_config.json)")
    (src / "README.md").write_text(model_card(args.base, src, args.repo))
    api = HfApi()
    api.create_repo(args.repo, private=args.private, exist_ok=True)
    if args.card_only:
        api.upload_file(
            path_or_fileobj=str(src / "README.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            commit_message="model card: superseded for general use by the frozen model + default prompt",
        )
        print(f"updated model card at https://huggingface.co/{args.repo}")
        return
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "calibration.json",
            "eval_report.txt",
            "README.md",
        ],
        commit_message="reflex adapter",
    )
    print(f"published https://huggingface.co/{args.repo}")
    print(f"serve with: reflex-serve --model {args.base} --adapter {args.repo}")


if __name__ == "__main__":
    main()
