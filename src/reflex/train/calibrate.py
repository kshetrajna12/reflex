"""RLCD-lite: fine-tune the readout to be *calibrated*, with a proper scoring rule.

TypeSafe trains Jev with "Reinforcement Learning for Calibrated Decisions": the policy
emits a probability distribution and is rewarded with a proper scoring rule. For a model
whose output *is* the distribution (no sampling), the expected reward is differentiable
in closed form, so the RL objective collapses to plain minimisation of the scoring rule:

    log score   ->  NLL       -log p(y)
    Brier score ->  ||p - onehot(y)||^2

Both are proper: the unique minimiser is the true conditional distribution, which is
exactly what "epistemically honest probabilities" means. We train LoRA adapters on the
label-restricted next-token logits through the very same packed forward the server uses.

    reflex-calibrate make-mmlu --out runs/mmlu_val.jsonl --split validation
    reflex-calibrate train --data runs/mmlu_val.jsonl --val runs/mmlu_dev.jsonl \
        --model Qwen/Qwen3-8B --out runs/lora-mmlu --epochs 1
    reflex-serve --model Qwen/Qwen3-8B --adapter runs/lora-mmlu --calibration runs/lora-mmlu/calibration.json
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reflex.eval.metrics import fit_temperature, report
from reflex.readout import Calibration, softmax
from reflex.train.data import examples, mmlu_to_jsonl

log = logging.getLogger("reflex.train")


def scoring_loss(logits: torch.Tensor, target: int, kind: str) -> torch.Tensor:
    if kind == "nll":
        return -F.log_softmax(logits, -1)[target]
    if kind == "brier":
        p = F.softmax(logits, -1)
        onehot = torch.zeros_like(p)
        onehot[target] = 1
        return ((p - onehot) ** 2).sum()
    raise ValueError(kind)


def evaluate(engine, val, permutations=1):
    rows = engine.label_logits_batch([(s, b) for s, b, _ in val])
    labels = np.array([t for _, _, t in val])
    # pad to common width for the report (questions may have different K)
    K = max(len(r) for r in rows)
    logits = np.full((len(rows), K), -1e9)
    for i, r in enumerate(rows):
        logits[i, : len(r)] = r
    return logits, labels


def train(args):
    from peft import LoraConfig, get_peft_model

    from reflex.engine import Engine

    engine = Engine.load(args.model, max_pack_tokens=args.max_pack_tokens)
    fmt = engine.fmt
    train_ex = examples(args.data, fmt, permutations=args.permutations, seed=args.seed)
    val_ex = examples(args.val, fmt) if args.val else []
    log.info("%d training examples, %d val examples", len(train_ex), len(val_ex))

    if val_ex:
        logits, labels = evaluate(engine, val_ex)
        print("== before ==\n" + str(report(np.stack([softmax(l) for l in logits]), labels)))

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        + (["gate_proj", "up_proj", "down_proj"] if args.lora_mlp else []),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(engine.model, lcfg)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    engine.model = model
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    n_packs = sum(1 for _ in engine.iter_batches([(s, b) for s, b, _ in train_ex]))
    total_steps = math.ceil(n_packs * args.epochs / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * max(0.0, 1 - s / max(1, total_steps)),
    )

    step, t0 = 0, time.perf_counter()
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(train_ex)
        items = [(s, b) for s, b, _ in train_ex]
        model.train()
        for micro, (idx, batch) in enumerate(engine.iter_batches(items)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = engine.forward_batch(batch)
            losses = []
            for i, row in zip(idx, logits):
                z = engine.restrict(row.float(), train_ex[i][1])
                losses.append(scoring_loss(z, train_ex[i][2], args.loss))
            loss = torch.stack(losses).mean() / args.accum
            loss.backward()
            if (micro + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    log.info(
                        "epoch %d step %d/%d loss %.4f lr %.2e (%.0fs)",
                        epoch,
                        step,
                        total_steps,
                        loss.item() * args.accum,
                        sched.get_last_lr()[0],
                        time.perf_counter() - t0,
                    )
        model.eval()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    log.info("saved adapter to %s", out)

    if val_ex:
        logits, labels = evaluate(engine, val_ex)
        print("== after ==\n" + str(report(np.stack([softmax(l) for l in logits]), labels)))
        T = fit_temperature(logits, labels)
        print(
            f"== after + temperature T={T:.3f} ==\n"
            + str(report(np.stack([softmax(l, T) for l in logits]), labels))
        )
        Calibration(temperature={"noul": T, "choice": T, "score": T}).save(out / "calibration.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-mmlu", help="write MMLU as reflex training JSONL")
    m.add_argument("--out", required=True)
    m.add_argument("--split", default="validation")
    m.add_argument("--n", type=int, default=None)
    m.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("train")
    t.add_argument("--model", default="Qwen/Qwen3.5-4B")
    t.add_argument("--data", required=True)
    t.add_argument("--val", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--loss", default="nll", choices=["nll", "brier"])
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=10)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--lora-r", type=int, default=16)
    t.add_argument("--lora-mlp", action="store_true")
    t.add_argument("--grad-checkpoint", action="store_true")
    t.add_argument("--permutations", type=int, default=1, help="option-order augmentation")
    t.add_argument("--max-pack-tokens", type=int, default=4096)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--log-every", type=int, default=10)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.cmd == "make-mmlu":
        n = mmlu_to_jsonl(args.out, args.split, args.n, args.seed)
        print(f"wrote {n} items to {args.out}")
    else:
        train(args)


if __name__ == "__main__":
    main()
