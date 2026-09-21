"""Per-branch restricted label logits for a labelled set, off an SGLang backend.

    uv run python scripts/collect_logits.py --model RadixArk/Qwen3.8-27B-NVFP4 \
        --sglang-url http://127.0.0.1:30000 --permutations 2 --out ext_nvfp4_p2.npz


Saves everything the offline temperature fit needs: the logits, which canonical option
each branch position stands for (permutations reorder them), the target distribution per
question, and the source of each item.
"""

import argparse
import sys

import numpy as np
from transformers import AutoTokenizer

from reflex.backends.sglang import SGLangBackend
from reflex.prompt import PromptFormat, build_branches
from reflex.schema import SystemOneRequest
from reflex.train.data import read_jsonl, target_vector

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="runs/external_eval.jsonl")
ap.add_argument("--model", required=True)
ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
ap.add_argument("--permutations", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--chunk", type=int, default=64)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
template = tok.chat_template or ""
fmt = PromptFormat(chat=bool(template), no_think="enable_thinking" in template)
backend = SGLangBackend(
    a.sglang_url,
    tokenizer=tok,
    fmt=fmt,
    default_permutations=a.permutations,
    max_concurrent_calls=8,
    timeout=600.0,
)

items, groups = [], []
for ri, row in enumerate(read_jsonl(a.data)):
    req = SystemOneRequest(state=row["state"], questions=row["questions"])
    for qid, q in req.questions.items():
        if qid not in row.get("labels", {}):
            continue
        brs = build_branches(qid, q, fmt, a.permutations, seed=a.seed * 100003 + ri)
        canon = list(brs[0].keys)
        g = {
            "kind": q.type,
            "source": row.get("source", ""),
            "target": target_vector(q.type, row["labels"][qid], canon).tolist(),
            "rows": [],
            "orders": [],
        }
        for br in brs:
            g["rows"].append(len(items))
            g["orders"].append([canon.index(k) for k in br.keys])
            items.append((req.state, br))
        groups.append(g)

print(f"{len(groups)} questions, {len(items)} branches", flush=True)
rows = []
for i in range(0, len(items), a.chunk):
    rows.extend(backend.label_logits_batch(items[i : i + a.chunk]))
    print(f"  {len(rows)}/{len(items)}", flush=True, file=sys.stderr)
backend.close()

K = max(len(r) for r in rows)
logits = np.full((len(rows), K), -1e9)
order = np.full((len(rows), K), -1, dtype=np.int64)
for i, r in enumerate(rows):
    logits[i, : len(r)] = r
targets = np.zeros((len(groups), K))
for gi, g in enumerate(groups):
    targets[gi, : len(g["target"])] = g["target"]
    for ri_, o in zip(g["rows"], g["orders"]):
        order[ri_, : len(o)] = o
np.savez(
    a.out,
    logits=logits,
    order=order,
    targets=targets,
    kinds=np.array([g["kind"] for g in groups]),
    sources=np.array([g["source"] for g in groups]),
    rows=np.array([g["rows"] for g in groups]),
    n_options=np.array([len(g["target"]) for g in groups]),
)
print("wrote", a.out)
