"""Prompt-ensemble readout: ask each question several ways, average the distributions.

Wording is high-variance for a 4B model (docs/results/frozen-vs-trained.md, section 2). An
ensemble turns that into a signal: variants that agree give a confident answer, variants
that scatter give a spread one, and the average is better calibrated than any single
reading without any fitted parameters. Variants may only change the *question side* of
the prompt (headings after the state, readout style, ask lines), never the state heading
or the system prompt, so every variant shares one state cache and all branches run in the
same forward pass as before.

    prompts/ensemble-v1.json:  {"variants": [{}, {"question_heading": "# Question", ...}, ...]}
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.readout import Calibration, group_results, merge_branches, softmax

PREFIX_KEYS = {"system_prompt", "state_heading"}


def load_ensemble(path_or_list: str | list[dict]) -> list[dict]:
    if isinstance(path_or_list, str):
        with open(path_or_list) as f:
            data = json.load(f)
        variants = data["variants"] if isinstance(data, dict) else data
    else:
        variants = path_or_list
    for v in variants:
        bad = PREFIX_KEYS & set(v)
        if bad:
            raise ValueError(f"ensemble variants may not change the state prefix: {sorted(bad)}")
    return list(variants)


def variant_formats(base: PromptFormat, variants: list[dict]) -> list[PromptFormat]:
    """One PromptFormat per variant, overrides layered on the base's texts. The state
    prefix is checked to be byte-identical so the cache is shared."""
    out = []
    probe = {"x": "probe"}
    for v in variants:
        fmt = PromptFormat.with_texts(base, {**base.texts, **v})
        fmt.system_prompt = base.system_prompt
        if fmt.prefix(probe) != base.prefix(probe):
            raise ValueError(f"variant changes the state prefix: {v}")
        out.append(fmt)
    return out


def disagreement(
    results: list[tuple[Branch, np.ndarray]], cal: Calibration, kind: str, state_tokens: int = 0
) -> float:
    """Mean total-variation distance between each reading's distribution and their mean:
    0 = every wording and order agrees, 1 = complete scatter. A reading is one option
    order of one wording, which is one branch unless the choice was split into pages."""
    groups = group_results(results)
    if len(groups) < 2:
        return 0.0
    keys = list(groups[0][0])
    rows = []
    for br_keys, logits in groups:
        p = softmax(logits, cal.t(kind, logits, state_tokens))
        by_key = dict(zip(br_keys, p))
        rows.append(np.array([by_key[k] for k in keys]))
    m = np.stack(rows)
    mean = m.mean(0)
    return float(0.5 * np.abs(m - mean).sum(1).mean())


def group_branches(qid: str, q, fmts: list[PromptFormat], permutations: int, rng) -> list[Branch]:
    """All branches for one question across variants and permutations."""
    out: list[Branch] = []
    for fmt in fmts:
        out.extend(build_branches(qid, q, fmt, permutations, rng))
    return out


def merged_probs_for_keys(
    kind: str,
    results: list[tuple[Branch, np.ndarray]],
    cal: Calibration,
    keys: list[Any],
    state_tokens: int = 0,
) -> np.ndarray:
    kp = merge_branches(kind, results, cal, state_tokens)
    return np.array([kp[k] for k in keys])
