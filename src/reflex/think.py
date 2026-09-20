"""System Two readout: let the model think first, then read the same label logits.

**This is an offline tool, never a serving mode.** reflex is a System One model: the
server answers every request with one fast forward pass and no reasoning. Nothing here is
reachable from `reflex-serve`; it exists for two offline jobs only, which construct the
readout explicitly:

  * `reflex-distill label --think N` - a thinking *teacher* labels a corpus. Speed does
    not matter there, and a stronger teacher makes better soft targets.
  * `reflex-calibrate eval --think N` - an experiment, to measure what reasoning would
    buy on a benchmark.

Each branch generates a bounded `<think>` block, the block is closed, and the label
logits are read from the position right after it. The output has exactly the same shape
as the fast path, so a thinking teacher's answers are valid soft targets for a
non-thinking student (System 2 -> System 1 distillation).

No state cache is used (every branch re-reads its state), and generation is batched with
left padding. Works for any chat model whose template thinks in `<think>...</think>`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from reflex.engine import right_pad
from reflex.prompt import Branch, build_branches
from reflex.readout import merge_branches, to_answer
from reflex.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger("reflex.think")

NO_THINK_TAIL = "<think>\n\n</think>\n\n"
CLOSE = "</think>"


@torch.inference_mode()
def think_logits(
    engine, items: Sequence[tuple[Any, Branch]], max_new_tokens: int, batch_size: int = 16
) -> list[np.ndarray]:
    """Restricted label logits for (state, branch) pairs, each after its own reasoning."""
    tok, pad, dev = engine.tok, engine.pad_id, engine.device
    seqs: list[list[int]] = []
    for state, br in items:
        _, sids, _, _ = engine.state_inputs(state)
        text = br.text.removesuffix(NO_THINK_TAIL)
        seqs.append(sids + engine._encode(text))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    results: list[np.ndarray | None] = [None] * len(items)
    t0, n_closed, n_tokens = time.time(), 0, 0
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = [seqs[i] for i in idx]
        B, L = len(batch), max(len(s) for s in batch)
        ids = torch.full((B, L), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((B, L), dtype=torch.long, device=dev)
        for r, s in enumerate(batch):
            ids[r, L - len(s) :] = torch.tensor(s, device=dev)
            mask[r, L - len(s) :] = 1
        out = engine.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_strings=[CLOSE],
            tokenizer=tok,
            pad_token_id=pad,
        )
        finals = []
        for s, g in zip(batch, out[:, L:].tolist()):
            g = [t for t in g if t != pad]
            n_tokens += len(g)
            text = tok.decode(g)
            if CLOSE in text:
                n_closed += 1
                text = text.split(CLOSE)[0] + CLOSE + "\n\n"
            else:  # out of budget: close the block and answer with what it has
                text = text + "\n" + CLOSE + "\n\n"
            finals.append(s + engine._encode(text))
        ids2, mask2, last = right_pad(finals, pad, dev)
        hidden = (
            engine._causal_lm()
            .model(input_ids=ids2, attention_mask=mask2, use_cache=False)
            .last_hidden_state
        )
        h = hidden[torch.arange(B, device=dev), last]
        logits = engine._causal_lm().lm_head(h).float()
        for i, row in zip(idx, logits):
            results[i] = engine.restrict(row, items[i][1]).cpu().numpy()
        done = start + B
        if done % (batch_size * 4) == 0 or done == len(order):
            log.info(
                "think: %d/%d branches, %.1f s/branch, %.0f think tokens avg, %.0f%% closed",
                done,
                len(order),
                (time.time() - t0) / done,
                n_tokens / done,
                100 * n_closed / done,
            )
    return results  # type: ignore[return-value]


def think_readout(engine, max_new_tokens: int, batch_size: int = 16):
    """A readout callable `(items) -> [logits]` that reasons before reading.

    Offline only. Hand it to the evaluator in place of `engine.label_logits_batch` to
    score a benchmark with reasoning; the serving path never sees it.
    """

    def readout(items: Sequence[tuple[Any, Branch]]) -> list[np.ndarray]:
        return think_logits(engine, items, max_new_tokens, batch_size)

    return readout


def think_answer(
    engine, req: SystemOneRequest, max_new_tokens: int, batch_size: int = 16
) -> SystemOneResponse:
    """Answer a request the slow way: every branch reasons, then the labels are read.

    Offline only (the thinking teacher in `reflex-distill label --think N`). The response
    has the same shape as the fast path's, so the labelling code does not care which
    readout produced it. Tens of seconds per question - never call this from a server.
    """
    branches: list[Branch] = []
    for qid, q in req.questions.items():
        for fmt in engine.variants:
            branches.extend(
                build_branches(qid, q, fmt, req.permutations or engine.default_permutations)
            )
    _, sids, _, n_images = engine.state_inputs(req.state)
    rows = think_logits(engine, [(req.state, b) for b in branches], max_new_tokens, batch_size)
    per_q: dict[str, list[tuple[Branch, np.ndarray]]] = {}
    for b, row in zip(branches, rows):
        per_q.setdefault(b.qid, []).append((b, row))
    answers = {
        qid: to_answer(
            q.type,
            merge_branches(q.type, per_q[qid], engine.cal, state_tokens=len(sids)),
            q,
        )
        for qid, q in req.questions.items()
    }
    q_tokens = sum(len(engine._encode(b.text)) for b in branches)
    usage = Usage(
        input_tokens=len(sids) + q_tokens,
        state_tokens=len(sids),
        question_tokens=q_tokens,
        images=n_images,
    )
    return SystemOneResponse(model=engine.model_name, answers=answers, usage=usage)
