"""Fine-tune the readout to be *calibrated*, with a proper scoring rule ("RLCD-lite").

TypeSafe trains Jev with "Reinforcement Learning for Calibrated Decisions": the policy
emits a probability distribution and is rewarded with a proper scoring rule. For a model
whose output *is* the distribution (no sampling), the expected reward is differentiable
in closed form, so the RL objective collapses to plain minimisation of the scoring rule:

    log score   ->  NLL     -sum_k q_k log p_k          (cross-entropy against the target)
    Brier score ->  ||p - q||^2

Both are proper: the unique minimiser is the true conditional distribution q, which is
exactly what "epistemically honest probabilities" means. q may be a one-hot label or a
soft label (e.g. the fraction of annotators who said "toxic"). We train on the
label-restricted next-token logits through the same forward the server uses.

Two ways to update the weights:
  * LoRA (default): small adapters on the attention/MLP projections. Minutes on one GPU,
    keeps the base model intact, and is all a few thousand calibration examples need.
  * --full: update every weight. Fits a 4B model on an 80-128 GB GPU. Rarely worth it
    for calibration; try it when you have tens of thousands of in-domain labels and LoRA
    has plateaued.

    reflex-data mix --out runs/mix_train.jsonl --eval-out runs/mix_eval.jsonl
    reflex-calibrate train --data runs/mix_train.jsonl --val runs/mix_eval.jsonl \
        --model Qwen/Qwen3.5-4B --out runs/lora-mix
    reflex-serve --model Qwen/Qwen3.5-4B --adapter runs/lora-mix \
        --calibration runs/lora-mix/calibration.json
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reflex.eval.metrics import fit_temperature, report
from reflex.readout import Calibration, softmax
from reflex.train.data import Example, examples, mmlu_to_jsonl

log = logging.getLogger("reflex.train")


# ------------------------------------------------------------------------------- loss


def scoring_loss(logits: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Proper scoring rule between the model's distribution and the target distribution."""
    if kind == "nll":
        return -(target * F.log_softmax(logits, -1)).sum()
    if kind == "brier":
        return ((F.softmax(logits, -1) - target) ** 2).sum()
    raise ValueError(kind)


# ------------------------------------------------------------------------------- eval


def fidelity(probs: np.ndarray, targets: np.ndarray) -> float:
    """1 - mean total-variation distance between predicted and target distributions.
    The right metric for soft-label sources, where top-label ECE is misleading: a model
    that honestly answers "60 %" on an item whose true frequency is 60 % is *right*
    every time, so its ECE reads 0.4 by construction."""
    return float(1.0 - 0.5 * np.abs(probs - targets).sum(1).mean())


def evaluate(engine, exs: list[Example], temperature: float = 1.0, logits=None):
    """Calibration report overall and per source. Returns ({name: text}, logits, labels).
    Pass `logits` to re-score without a forward pass (e.g. at another temperature)."""
    if logits is None:
        log.info("evaluating %d examples", len(exs))
        rows = engine.label_logits_batch([(e.state, e.branch) for e in exs])
        K = max(len(r) for r in rows)
        logits = np.full((len(rows), K), -1e9)
        for i, r in enumerate(rows):
            logits[i, : len(r)] = r
    probs = np.stack([softmax(r, temperature) for r in logits])
    labels = np.array([e.hard_label for e in exs])
    return evaluate_probs(exs, probs, logits), logits, labels


def evaluate_probs(exs: list[Example], probs: np.ndarray, logits: np.ndarray) -> dict[str, str]:
    """Score given probability rows (any temperature scheme) overall and per source."""
    K = logits.shape[1]
    labels = np.array([e.hard_label for e in exs])
    targets = np.zeros((len(exs), K))
    for i, e in enumerate(exs):
        targets[i, : len(e.target)] = e.target
    soft = np.array([e.target.max() < 0.999 for e in exs])

    def line(idx):
        text = str(report(probs[idx], labels[idx]))
        if soft[idx].any():  # add fidelity for sources with soft labels
            fid = fidelity(probs[idx][soft[idx]], targets[idx][soft[idx]])
            head, *rest = text.splitlines()
            text = "\n".join([f"{head}  fidelity={fid:.4f}", *rest])
        return text

    out = {"all": line(np.arange(len(exs)))}
    by_src = defaultdict(list)
    for i, e in enumerate(exs):
        by_src[e.source or "?"].append(i)
    for src, idx in sorted(by_src.items()):
        out[src] = line(np.array(idx))
    return out


def print_reports(title: str, reports: dict[str, str]) -> None:
    lines = [f"== {title} =="]
    for name, text in reports.items():
        head = text.splitlines()[0]  # one summary line per source keeps the log readable
        lines.append(f"  {name:<16} {head}")
    print("\n" + "\n".join(lines))
    log.info("\n".join(lines))


# ------------------------------------------------------------------------------- train


def train(args):
    from reflex.engine import Engine

    # Always keep a plain log file next to the output: `tail -f runs/<out>/train.log`
    # shows progress regardless of how stdout is piped.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(fh)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "fla.utils"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log.info("args: %s", vars(args))

    engine = Engine.load(args.model, max_pack_tokens=args.max_pack_tokens)
    train_ex = examples(args.data, engine.fmt, permutations=args.permutations, seed=args.seed)
    val_ex = examples(args.val, engine.fmt) if args.val else []
    log.info("%d training examples, %d val examples", len(train_ex), len(val_ex))

    if val_ex:
        reports, _, _ = evaluate(engine, val_ex)
        print_reports("before training", reports)

    # ---- which weights to train
    if args.full:
        # fp32 master weights: bf16 cannot represent the tiny updates a 1e-5 step makes.
        # A 4B model then needs ~64 GB (weights + grads + Adam) plus activations.
        model = engine.model.float()
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            engine.model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=2 * args.lora_r,
                lora_dropout=0.0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                + (["gate_proj", "up_proj", "down_proj"] if args.lora_mlp else []),
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
        engine.model = model
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    if not args.full:  # LoRA weights in fp32 for a stable optimiser; the base stays bf16
        for p in params:
            p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    # Batch boundaries depend on example order, so fix the order of every epoch up front
    # and count the batches exactly; otherwise the LR schedule can end early or late.
    rng = random.Random(args.seed)
    epochs: list[list[Example]] = []
    for _ in range(args.epochs):
        rng.shuffle(train_ex)
        epochs.append(list(train_ex))
    n_batches = sum(
        sum(1 for _ in engine.iter_batches([(e.state, e.branch) for e in ep])) for ep in epochs
    )
    total_steps = math.ceil(n_batches / args.accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup)) * max(0.0, 1 - s / max(1, total_steps)),
    )

    step, t0 = 0, time.perf_counter()
    for epoch, train_ex in enumerate(epochs):
        model.train()
        running = []
        for micro, (idx, batch) in enumerate(
            engine.iter_batches([(e.state, e.branch) for e in train_ex])
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = engine.forward_batch(batch)
            losses = []
            for i, row in zip(idx, logits):
                e = train_ex[i]
                z = engine.restrict(row.float(), e.branch)
                q = torch.tensor(e.target, dtype=torch.float32, device=z.device)
                losses.append(scoring_loss(z, q, args.loss))
            loss = torch.stack(losses).mean() / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
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
                        float(np.mean(running[-args.log_every * args.accum :])),
                        sched.get_last_lr()[0],
                        time.perf_counter() - t0,
                    )
        model.eval()
        if args.epochs > 1 and epoch < args.epochs - 1:
            model.save_pretrained(out_dir / f"epoch{epoch + 1}")  # keep earlier epochs to compare

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    if args.full:
        engine.tok.save_pretrained(out)
    log.info("saved %s to %s", "full model" if args.full else "adapter", out)

    if val_ex:
        reports, logits, labels = evaluate(engine, val_ex)
        print_reports("after training", reports)
        fit_calibration(engine, val_ex, logits, labels, out)
        with open(out / "eval_report.txt", "w") as f:
            f.writelines(f"### {name}\n{text}\n\n" for name, text in reports.items())


def item_meta(engine, exs: list[Example]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """kinds, n_options, state_tokens per example (cheap: tokenises the state prefix)."""
    kinds = np.array([e.branch.kind for e in exs])
    n_opts = np.array([len(e.branch.keys) for e in exs])
    cache: dict[str, int] = {}
    st = []
    for e in exs:
        text = engine.fmt.prefix(e.state)
        if text not in cache:
            cache[text] = len(engine._encode(text))
        st.append(cache[text])
    return kinds, n_opts, np.array(st)


def fit_calibration(engine, exs: list[Example], logits: np.ndarray, labels: np.ndarray, out: Path):
    """Per-primitive temperatures, then the calibration head on top; prints both, saves both."""
    from reflex.calibration_head import features, temperature
    from reflex.calibration_head import fit as fit_head

    kinds, n_opts, st = item_meta(engine, exs)
    temps = {}
    for kind in ("noul", "choice", "score"):
        m = kinds == kind
        temps[kind] = fit_temperature(logits[m], labels[m]) if m.sum() >= 20 else 1.0
    probs_t = np.stack([softmax(r, temps[k]) for r, k in zip(logits, kinds, strict=True)])
    print_reports(
        "per-primitive temperature " + ", ".join(f"{k}={v:.2f}" for k, v in temps.items()),
        evaluate_probs(exs, probs_t, logits),
    )
    zs = [logits[i, : n_opts[i]] for i in range(len(exs))]
    head = fit_head(zs, [e.target for e in exs], list(kinds), list(st), temps)
    probs_h = np.stack(
        [
            softmax(logits[i], temperature(head, features(kinds[i], zs[i], int(st[i]))))
            for i in range(len(exs))
        ]
    )
    print_reports(
        "calibration head (per-question temperature)", evaluate_probs(exs, probs_h, logits)
    )
    np.savez(
        out / "eval_logits.npz",
        logits=logits,
        labels=labels,
        kinds=kinds,
        n_options=n_opts,
        state_tokens=st,
    )
    Calibration(temperature=temps, head=[float(w) for w in head]).save(out / "calibration.json")
    log.info(
        "wrote %s (temps %s, %d head weights)",
        out / "calibration.json",
        {k: round(v, 3) for k, v in temps.items()},
        len(head),
    )


def evaluate_adapter(args):
    """Evaluate an adapter (with its shipped calibration) on a labelled set. No writes."""
    from reflex.engine import Engine

    engine = Engine.load(
        args.model,
        adapter_path=args.adapter,
        calibration_path=args.calibration,
        max_pack_tokens=args.max_pack_tokens,
    )
    exs = examples(args.val, engine.fmt)
    reports, logits, labels = evaluate(engine, exs)
    print_reports("raw (T=1)", reports)
    kinds, n_opts, st = item_meta(engine, exs)
    probs = np.stack(
        [
            softmax(logits[i], engine.cal.t(kinds[i], logits[i, : n_opts[i]], int(st[i])))
            for i in range(len(exs))
        ]
    )
    how = "head" if engine.cal.head else "per-primitive temperatures"
    print_reports(f"with the adapter's calibration ({how})", evaluate_probs(exs, probs, logits))
    if args.dump:
        np.savez(
            args.dump, logits=logits, labels=labels, kinds=kinds, n_options=n_opts, state_tokens=st
        )


def refit(args):
    """Re-fit the calibration (per-primitive temperatures + head) for an existing adapter on a
    labelled set, without training."""
    from reflex.engine import Engine

    engine = Engine.load(
        args.model, adapter_path=args.adapter, max_pack_tokens=args.max_pack_tokens
    )
    val_ex = examples(args.val, engine.fmt)
    reports, logits, labels = evaluate(engine, val_ex)
    print_reports("uncalibrated", reports)
    out = Path(args.out or args.adapter)
    out.mkdir(parents=True, exist_ok=True)
    fit_calibration(engine, val_ex, logits, labels, out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser(
        "eval", help="evaluate an adapter with its calibration on a labelled set (no writes)"
    )
    ev.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--calibration", default=None)
    ev.add_argument("--val", required=True)
    ev.add_argument("--dump", default=None, help="save per-item logits to this .npz")
    ev.add_argument("--max-pack-tokens", type=int, default=4096)

    r = sub.add_parser(
        "refit",
        help="re-fit the calibration (temperatures + head) for an adapter on a labelled set",
    )
    r.add_argument("--model", default="Qwen/Qwen3.5-4B")
    r.add_argument("--adapter", required=True)
    r.add_argument("--val", required=True)
    r.add_argument(
        "--out", default=None, help="where to write calibration.json (default: the adapter dir)"
    )
    r.add_argument("--max-pack-tokens", type=int, default=4096)

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
    t.add_argument(
        "--full", action="store_true", help="update all weights instead of LoRA adapters"
    )
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
    elif args.cmd == "refit":
        refit(args)
    elif args.cmd == "eval":
        evaluate_adapter(args)
    else:
        if args.full and args.lr >= 1e-4:
            log.warning("--full with lr %.0e is aggressive; 1e-5 is a safer start", args.lr)
        train(args)


if __name__ == "__main__":
    main()
