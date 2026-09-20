"""System Two readout: let the model think first, then read the same label logits.

reflex normally suppresses reasoning (an empty `<think>` block) and reads the answer
distribution from the next-token logits, which is what makes it fast. For an *offline
teacher* speed does not matter: here each branch generates a bounded `<think>` block,
the block is closed, and the label logits are read from the position right after it. The
output has exactly the same shape as the fast path, so a thinking teacher's answers are
valid soft targets for a non-thinking student (System 2 -> System 1 distillation).

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
from reflex.prompt import Branch

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
