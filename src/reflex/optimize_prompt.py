"""Optimise the prompt wording with GEPA, keeping the model frozen.

Why: on a frozen model the instruction wording swings accuracy by many points, and unlike
fine-tuning it cannot cost general judgement. GEPA (Agrawal et al., ICLR 2026) evolves the
named text components of the prompt: a reflection LM reads the worst-scoring examples,
proposes new wording, and a Pareto front over per-example scores keeps diverse winners.

    reflex-optimize --train runs/gepa_train.jsonl --val runs/gepa_val.jsonl \\
        --out runs/prompt_gepa.json --max-metric-calls 8000
    reflex-serve --prompt-texts runs/prompt_gepa.json

Score per example = probability mass the model puts on the target distribution
(1 - total variation), so hard and soft labels score on the same 0-1 scale. Use your
own labelled data for --train/--val, never a benchmark's items.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from reflex.prompt import DEFAULT_TEXTS, PromptFormat
from reflex.readout import softmax
from reflex.train.data import examples_from_rows, read_jsonl

log = logging.getLogger("reflex.optimize")

COMPONENTS = list(DEFAULT_TEXTS)


def score_example(probs: np.ndarray, target: np.ndarray) -> float:
    return float(1.0 - 0.5 * np.abs(probs - target).sum())


class ReflexAdapter:
    """GEPAAdapter for reflex: a candidate is a dict of prompt components."""

    propose_new_texts = None  # None = use gepa's built-in reflective proposer

    def __init__(self, engine, components: list[str]):
        self.engine = engine
        self.base_fmt = engine.fmt
        self.components = components

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        fmt = PromptFormat.with_texts(self.base_fmt, {k: candidate[k] for k in candidate})
        exs = examples_from_rows(batch, fmt)
        self.engine.fmt = fmt
        try:
            rows = self.engine.label_logits_batch([(e.state, e.branch) for e in exs])
        finally:
            self.engine.fmt = self.base_fmt
        outputs, scores, traces = [], [], []
        for e, z in zip(exs, rows, strict=True):
            p = softmax(z)
            sc = score_example(p, e.target)
            dist = {str(k): round(float(v), 3) for k, v in zip(e.branch.keys, p, strict=True)}
            want = {
                str(k): round(float(v), 3)
                for k, v in zip(e.branch.keys, e.target, strict=True)
                if v > 0
            }
            out = {"kind": e.branch.kind, "predicted": dist, "target": want, "score": round(sc, 3)}
            outputs.append(out)
            scores.append(sc)
            if capture_traces:
                st = json.dumps(e.state, ensure_ascii=False)
                traces.append(
                    {
                        **out,
                        "source": e.source,
                        "question": e.branch.text.split("\n")[1][:300],
                        "state_excerpt": st[:600],
                    }
                )
        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=traces if capture_traces else None
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        items = list(zip(eval_batch.trajectories or [], eval_batch.scores, strict=True))
        items.sort(key=lambda x: x[1])
        chosen = items[:8] + items[-2:]  # mostly failures, a couple of successes for contrast
        data = {}
        for comp in components_to_update:
            rows = []
            for tr, sc in chosen:
                top = max(tr["predicted"], key=tr["predicted"].get)
                fb = f"score {sc:.2f}. "
                if sc < 0.5:
                    fb += f"Wrong or too uncertain: put {tr['predicted'][top]:.2f} on '{top}', target was {tr['target']}."
                elif sc < 0.9:
                    fb += f"Right direction but not confident enough: {tr['predicted']} vs target {tr['target']}."
                else:
                    fb += "Good."
                if comp.startswith("noul") and tr["kind"] != "noul":
                    continue
                if comp.startswith("score") and tr["kind"] != "score":
                    continue
                if comp.startswith("choice") and tr["kind"] != "choice":
                    continue
                rows.append(
                    {
                        "Inputs": {
                            "question_type": tr["kind"],
                            "source": tr["source"],
                            "question": tr["question"],
                            "state_excerpt": tr["state_excerpt"],
                        },
                        "Generated Outputs": tr["predicted"],
                        "Feedback": fb,
                    }
                )
            data[comp] = rows or [
                {
                    "Inputs": {},
                    "Generated Outputs": {},
                    "Feedback": "No examples of this question type in the batch; keep this component unchanged.",
                }
            ]
        return data


def gateway_reflection_lm(base_url: str, key: str, model: str, max_tokens: int = 4096):
    """A reflection LM callable backed by an OpenAI-compatible endpoint (thinking off)."""
    import httpx

    def call(prompt: str) -> str:
        for attempt in range(3):
            try:
                r = httpx.post(
                    f"{base_url}/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=600,
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception as e:  # noqa: BLE001 - retry transient gateway errors
                log.warning("reflection LM attempt %d failed: %s", attempt + 1, e)
                time.sleep(5)
        return ""

    return call


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--components", default=",".join(COMPONENTS))
    ap.add_argument("--max-metric-calls", type=int, default=8000)
    ap.add_argument("--minibatch", type=int, default=6)
    ap.add_argument("--gateway", default="http://127.0.0.1:8000")
    ap.add_argument("--gateway-key-env", default="SPARK_KEY")
    ap.add_argument("--reflection-model", default="default")
    ap.add_argument("--max-pack-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    import gepa

    from reflex.engine import Engine

    engine = Engine.load(
        args.model, adapter_path=args.adapter, max_pack_tokens=args.max_pack_tokens
    )
    comps = [c.strip() for c in args.components.split(",") if c.strip()]
    seed = {c: DEFAULT_TEXTS[c] for c in comps}
    train = list(read_jsonl(args.train))
    val = list(read_jsonl(args.val))
    adapter = ReflexAdapter(engine, comps)
    base = adapter.evaluate(val, seed)
    log.info("seed prompt: mean score %.4f on %d val rows", float(np.mean(base.scores)), len(val))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = gepa.optimize(
        seed_candidate=seed,
        trainset=train,
        valset=val,
        adapter=adapter,
        reflection_lm=gateway_reflection_lm(
            args.gateway, os.environ.get(args.gateway_key_env, ""), args.reflection_model
        ),
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.minibatch,
        run_dir=str(out.parent / (out.stem + "_gepa")),
        seed=args.seed,
        display_progress_bar=False,
    )
    best = result.best_candidate
    final = adapter.evaluate(val, best)
    log.info(
        "best prompt: mean score %.4f on val (seed %.4f)",
        float(np.mean(final.scores)),
        float(np.mean(base.scores)),
    )
    with open(out, "w") as f:
        json.dump(
            {
                "texts": best,
                "seed_val_score": float(np.mean(base.scores)),
                "best_val_score": float(np.mean(final.scores)),
                "val_scores": getattr(result, "val_aggregate_scores", None),
                "model": args.model,
                "adapter": args.adapter,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"wrote {out}")
    for k, v in best.items():
        if v != seed[k]:
            print(f"\n[{k}]\n  was: {seed[k]}\n  now: {v}")


if __name__ == "__main__":
    main()
