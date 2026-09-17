"""The System One engine: shared-state prefill, isolated question branches, direct readout.

Reconstruction of the "shared-state, isolated-questions" design described in
https://archerhume.com/posts/jevs-architecture-unmasked/ , on top of a stock
pretrained model (Qwen3 / Qwen3-VL / Qwen3.5 by default):

  1. **Shared state encoding.** The state prefix (text, and images for VL models) is run
     through the model once and its cache is kept (LRU by content hash), so repeated
     requests over the same state only pay for the questions.
  2. **Parallel, isolated question branches.** Every branch sees exactly "state + this
     question" and never another branch. Two execution strategies, chosen from the model
     config:
       * ``packed`` (attention-only backbones, e.g. Qwen3, Qwen3-VL): all branches are
         concatenated into ONE sequence and run in a single prefill with a custom 4D
         attention mask (branch tokens attend to the state and to earlier tokens of their
         own branch). Position ids restart at ``len(state)`` per branch. No cache copies.
       * ``batched`` (hybrid backbones with recurrent/linear-attention layers, e.g.
         Qwen3.5): a mask cannot isolate branches inside a recurrent scan, so branches
         run as a right-padded batch against a batch-expanded copy of the state cache.
  3. **Direct probability readout.** No decoding. At the last token of each branch we
     take the next-token logits, restrict them to the label tokens (A/B/C…, Yes/No),
     temperature-scale, softmax. That distribution *is* the answer.

Everything is "prefill only" – there is no autoregressive loop anywhere.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import random
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from reflex.images import digest, has_images, split_images, to_pil
from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.readout import Calibration, merge_branches, to_answer
from reflex.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger(__name__)

CACHED = -2  # sentinel segment id for "the state already sitting in the KV cache"


# --------------------------------------------------------------------------------------
# Packed strategy: turn a forest of segments into one sequence + block mask
# --------------------------------------------------------------------------------------


@dataclass
class Segment:
    """A run of tokens. `parent` is the index of the segment it may attend to (a state),
    -1 for a root (state) segment, or CACHED for the state in the KV cache."""

    ids: list[int]
    parent: int = -1


@dataclass
class Pack:
    input_ids: torch.Tensor  # [1, T]
    position_ids: torch.Tensor  # [1, T]
    attention_mask: torch.Tensor  # [1, 1, T, past + T] bool, True = may attend
    last_index: torch.Tensor  # [n_segments] index in T of each segment's last token
    seg_start: list[int]
    seg_len: list[int]


def build_pack(
    segments: Sequence[Segment],
    past_len: int = 0,
    past_seg: int = -1,
    device: torch.device | str = "cpu",
) -> Pack:
    """Concatenate segments; build the mask that isolates them.

    Token j (key) is visible to token i (query) iff
        seg[j] == seg[i] and j <= i                (causal within own segment)
     or seg[j] == parent[seg[i]]                   (whole parent state)
    Keys 0..past_len-1 belong to a cached segment `past_seg` (if any) and are not part of
    the query. Position ids: root segments start at 0; child segments start at
    len(parent) so RoPE sees "state then question" for every branch.
    """
    ids: list[int] = []
    seg_of: list[int] = []
    pos: list[int] = []
    starts, lens = [], []
    seg_lengths = {past_seg: past_len} if past_len else {}
    for s_idx, seg in enumerate(segments):
        base = seg_lengths[seg.parent] if seg.parent != -1 else 0
        starts.append(len(ids))
        lens.append(len(seg.ids))
        ids.extend(seg.ids)
        seg_of.extend([s_idx] * len(seg.ids))
        pos.extend(range(base, base + len(seg.ids)))
        seg_lengths[s_idx] = len(seg.ids)

    T = len(ids)
    q_seg = torch.tensor(seg_of, device=device)  # [T]
    parent = torch.tensor([s.parent for s in segments], device=device)  # [S]
    q_parent = parent[q_seg]  # [T]
    k_seg = torch.cat([torch.full((past_len,), past_seg, device=device), q_seg])  # [P+T]
    q_idx = torch.arange(T, device=device)
    k_idx = torch.arange(past_len + T, device=device) - past_len  # past keys get negative idx
    same = (k_seg[None, :] == q_seg[:, None]) & (k_idx[None, :] <= q_idx[:, None])
    par = k_seg[None, :] == q_parent[:, None]
    mask = (same | par)[None, None]  # [1,1,T,P+T]

    last = torch.tensor([s + n - 1 for s, n in zip(starts, lens)], device=device)
    return Pack(
        input_ids=torch.tensor([ids], device=device),
        position_ids=torch.tensor([pos], device=device),
        attention_mask=mask,
        last_index=last,
        seg_start=starts,
        seg_len=lens,
    )


# --------------------------------------------------------------------------------------
# Batched strategy: right-padded batch of independent sequences
# --------------------------------------------------------------------------------------


def right_pad(
    seqs: Sequence[Sequence[int]], pad_id: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """-> input_ids [B, L], attention_mask [B, L] (1 = real), last index [B]."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    last = torch.tensor([len(s) - 1 for s in seqs])
    return ids.to(device), mask.to(device), last.to(device)


@dataclass
class Batch:
    """Model inputs for a group of independent (state, branch) items."""

    kwargs: dict[str, Any]
    last: torch.Tensor  # last token index per item (in T for packed, in L for batched)
    packed: bool


# --------------------------------------------------------------------------------------
# State cache
# --------------------------------------------------------------------------------------


@dataclass
class StateEntry:
    key: str
    ids: list[int]
    cache: Any  # transformers DynamicCache holding exactly len(ids) tokens
    created: float
    rope_deltas: torch.Tensor | None = None  # VL models: M-RoPE offset after images
    n_images: int = 0


class Engine:
    def __init__(
        self,
        model,
        tokenizer,
        fmt: PromptFormat,
        calibration: Calibration | None = None,
        *,
        max_pack_tokens: int = 8192,
        max_branch_tokens: int = 4096,
        state_cache_entries: int = 8,
        model_name: str = "reflex-latest",
        processor=None,
        max_image_pixels: int = 1024 * 1024,
        strategy: str | None = None,
    ):
        self.model = model
        self.tok = tokenizer
        self.fmt = fmt
        self.cal = calibration or Calibration()
        self.max_pack_tokens = max_pack_tokens
        self.max_branch_tokens = max_branch_tokens
        self.model_name = model_name
        self.device = next(model.parameters()).device
        self.processor = processor  # set for vision-language models; None for text-only
        if processor is not None and hasattr(processor, "image_processor"):
            size = dict(getattr(processor.image_processor, "size", {}) or {})
            size["longest_edge"] = max_image_pixels
            processor.image_processor.size = size
        self.strategy = strategy or ("batched" if self.is_hybrid else "packed")
        if self.strategy not in ("packed", "batched"):
            raise ValueError(f"unknown strategy {self.strategy!r}")
        if self.strategy == "packed" and self.is_hybrid:
            raise ValueError("packed strategy cannot isolate branches in recurrent layers")
        self._states: OrderedDict[str, StateEntry] = OrderedDict()
        self._state_cache_entries = state_cache_entries
        self._lock = threading.Lock()
        self._label_ids: dict[str, int] = {}
        self.model.eval()

    # ----------------------------------------------------------------------------- load
    @classmethod
    def load(
        cls,
        model_id: str = "Qwen/Qwen3.5-4B",
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        calibration_path: str | None = None,
        chat: bool | None = None,
        adapter_path: str | None = None,
        **engine_kwargs,
    ) -> Engine:
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoProcessor,
            AutoTokenizer,
        )

        tok = AutoTokenizer.from_pretrained(model_id)
        cfg = AutoConfig.from_pretrained(model_id)
        multimodal = getattr(cfg, "vision_config", None) is not None
        processor = None
        loader = AutoModelForCausalLM
        if multimodal:
            processor = AutoProcessor.from_pretrained(model_id)
            loader = AutoModelForImageTextToText
        model = loader.from_pretrained(
            model_id, dtype=dtype, device_map=device, attn_implementation=attn_implementation
        )
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
        template = tok.chat_template or ""
        fmt = PromptFormat(
            chat=bool(template) if chat is None else chat,
            no_think="enable_thinking" in template,
        )
        cal = Calibration.load(calibration_path)
        eng = cls(model, tok, fmt, cal, model_name=model_id, processor=processor, **engine_kwargs)
        log.info(
            "loaded %s (chat=%s no_think=%s multimodal=%s strategy=%s) on %s",
            model_id,
            fmt.chat,
            fmt.no_think,
            multimodal,
            eng.strategy,
            eng.device,
        )
        return eng

    # ------------------------------------------------------------------- model plumbing
    @property
    def text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", None) or cfg

    @property
    def is_hybrid(self) -> bool:
        """True if some decoder layers are recurrent (no KV cache to mask)."""
        return any(
            t != "full_attention" for t in getattr(self.text_config, "layer_types", []) or []
        )

    @property
    def multimodal(self) -> bool:
        return self.processor is not None

    def _causal_lm(self):
        """The *ForCausalLM / *ForConditionalGeneration module (unwrapping PEFT)."""
        m = self.model
        return m.get_base_model() if hasattr(m, "get_base_model") else m

    def _inner(self):
        """The base model that owns `rope_deltas` on Qwen-VL style models (or None)."""
        m = getattr(self._causal_lm(), "model", None)
        return m if m is not None and hasattr(m, "rope_deltas") else None

    def _new_cache(self):
        from transformers import DynamicCache

        return DynamicCache(config=self.text_config)

    @property
    def pad_id(self) -> int:
        return self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

    # --------------------------------------------------------------------------- tokens
    def label_id(self, label: str) -> int:
        """Token id of a label ('A', 'Yes', ...). Labels must be single tokens."""
        if label not in self._label_ids:
            ids = self.tok.encode(label, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"label {label!r} is not a single token: {ids}")
            self._label_ids[label] = ids[0]
        return self._label_ids[label]

    def _encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def restrict(self, row: torch.Tensor, br: Branch) -> torch.Tensor:
        """Select the label-token logits of one branch, in label order."""
        lab = torch.tensor([self.label_id(lb) for lb in br.labels], device=row.device)
        return row[lab]

    # ------------------------------------------------------------------------ state cache
    def _state_key(self, prefix_text: str) -> str:
        return hashlib.sha256(prefix_text.encode()).hexdigest()

    def state_inputs(self, state) -> tuple[dict, list[int], str, int]:
        """Tokenise a state (text, or text + images) -> (model kwargs, ids, cache key, n_images)."""
        blobs: list[bytes] = []
        if has_images(state):
            if not self.multimodal:
                raise ValueError("state contains images but the loaded model is text-only")
            state, blobs = split_images(state)
        text = self.fmt.prefix(state)
        key = self._state_key(text + ("|" + digest(blobs) if blobs else ""))
        if blobs:
            inputs = self.processor(
                text=[text], images=[to_pil(b) for b in blobs], return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs.pop("attention_mask", None)  # single sequence, no padding
            ids = inputs["input_ids"][0].tolist()
        else:
            ids = self._encode(text)
            inputs = {"input_ids": torch.tensor([ids], device=self.device)}
        return inputs, ids, key, len(blobs)

    @torch.inference_mode()
    def encode_state(self, state) -> tuple[StateEntry, bool]:
        """Run the state prefix once (text and any images); return (entry, cache_hit)."""
        inputs, ids, key, n_images = self.state_inputs(state)
        if key in self._states:
            self._states.move_to_end(key)
            return self._states[key], True
        cache = self._new_cache()
        inner = self._inner()
        if inner is not None:
            inner.rope_deltas = None  # never let a previous request's offset leak in
        self.model(**inputs, past_key_values=cache, use_cache=True, logits_to_keep=1)
        rope_deltas = (
            inner.rope_deltas.clone()
            if inner is not None and inner.rope_deltas is not None
            else None
        )
        entry = StateEntry(
            key=key,
            ids=ids,
            cache=cache,
            created=time.time(),
            rope_deltas=rope_deltas,
            n_images=n_images,
        )
        self._states[key] = entry
        while len(self._states) > self._state_cache_entries:
            self._states.popitem(last=False)
        return entry, False

    # ---------------------------------------------------------------------- core forward
    @torch.inference_mode()
    def _forward_branches(self, entry: StateEntry, branch_ids: list[list[int]]) -> torch.Tensor:
        """Run all branches against a cached state. Returns last-token logits [B, vocab] (fp32)."""
        out = []
        for chunk in self._chunks(branch_ids):
            if self.strategy == "packed":
                out.append(self._branches_packed(entry, chunk))
            else:
                out.append(self._branches_batched(entry, chunk))
        return torch.cat(out, 0)

    def _branches_packed(self, entry: StateEntry, chunk: list[list[int]]) -> torch.Tensor:
        P = len(entry.ids)
        pack = build_pack(
            [Segment(b, parent=CACHED) for b in chunk],
            past_len=P,
            past_seg=CACHED,
            device=self.device,
        )
        logits = self.model(
            input_ids=pack.input_ids,
            position_ids=self._packed_positions(pack, entry),
            attention_mask=pack.attention_mask,
            past_key_values=entry.cache,
            use_cache=True,
            logits_to_keep=pack.last_index,
        ).logits[0]
        entry.cache.crop(-pack.input_ids.shape[1])  # drop branch tokens, keep state
        return logits.float()

    def _packed_positions(self, pack: Pack, entry: StateEntry) -> torch.Tensor:
        """Text models: [1, T] = len(state) + offset. Qwen-VL models use M-RoPE with 3 rotary
        axes; text after images continues from (len(state) + rope_delta), so we pass
        [4, 1, T]: row 0 the plain text position, rows 1-3 the shifted M-RoPE positions."""
        if not self.multimodal:
            return pack.position_ids
        pos = pack.position_ids
        delta = int(entry.rope_deltas.reshape(-1)[0].item()) if entry.rope_deltas is not None else 0
        shifted = pos + delta
        return torch.stack([pos, shifted, shifted, shifted], 0)

    def _branches_batched(self, entry: StateEntry, chunk: list[list[int]]) -> torch.Tensor:
        P, B = len(entry.ids), len(chunk)
        ids, mask, last = right_pad(chunk, self.pad_id, self.device)
        full_mask = torch.cat([torch.ones((B, P), dtype=mask.dtype, device=self.device), mask], 1)
        cache = copy.deepcopy(entry.cache)
        cache.reorder_cache(torch.zeros(B, dtype=torch.long, device=self.device))  # expand to B
        hidden = (
            self._causal_lm()
            .model(
                input_ids=ids,
                attention_mask=full_mask,
                position_ids=self._batched_positions(entry, B, ids.shape[1]),
                past_key_values=cache,
                use_cache=True,
            )
            .last_hidden_state
        )
        h = hidden[torch.arange(B, device=self.device), last]
        return self._causal_lm().lm_head(h).float()

    def _batched_positions(self, entry: StateEntry, B: int, L: int) -> torch.Tensor:
        """Right-padded branches behind a cached state: text position P + offset. Pads are
        past the last real token so their positions do not matter. VL models get the
        [4, B, L] text + shifted M-RoPE layout (see `_packed_positions`)."""
        P = len(entry.ids)
        pos = (torch.arange(L, device=self.device) + P).expand(B, L)
        if not self.multimodal:
            return pos
        delta = int(entry.rope_deltas.reshape(-1)[0].item()) if entry.rope_deltas is not None else 0
        shifted = pos + delta
        return torch.stack([pos, shifted, shifted, shifted], 0)

    def _chunks(self, branch_ids: list[list[int]]) -> Iterable[list[list[int]]]:
        """Group branches under the token budget (padded size counts for `batched`)."""
        cur, n, longest = [], 0, 0
        for b in branch_ids:
            if len(b) > self.max_branch_tokens:
                raise ValueError(f"question branch too long: {len(b)} > {self.max_branch_tokens}")
            new_longest = max(longest, len(b))
            size = n + len(b) if self.strategy == "packed" else (len(cur) + 1) * new_longest
            if cur and size > self.max_pack_tokens:
                yield cur
                cur, n, longest = [], 0, 0
                new_longest = len(b)
            cur.append(b)
            n += len(b)
            longest = new_longest
        if cur:
            yield cur

    # ---------------------------------------------------------------------------- answer
    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        with self._lock:
            return self._answer(req)

    def _answer(self, req: SystemOneRequest) -> SystemOneResponse:
        rng = random.Random(0)
        branches: list[Branch] = []
        for qid, q in req.questions.items():
            branches.extend(build_branches(qid, q, self.fmt, req.permutations, rng))
        branch_ids = [self._encode(b.text) for b in branches]

        entry, hit = self.encode_state(req.state)
        logits = self._forward_branches(entry, branch_ids)  # [B, vocab]

        per_q: dict[str, list[tuple[Branch, np.ndarray]]] = {}
        for b, row in zip(branches, logits):
            per_q.setdefault(b.qid, []).append((b, self.restrict(row, b).cpu().numpy()))

        answers = {}
        for qid, q in req.questions.items():
            key_probs = merge_branches(q.type, per_q[qid], self.cal)
            answers[qid] = to_answer(q.type, key_probs, q)

        q_tokens = sum(len(b) for b in branch_ids)
        usage = Usage(
            input_tokens=len(entry.ids) + q_tokens,
            state_tokens=len(entry.ids),
            question_tokens=q_tokens,
            state_cache_hit=hit,
            images=entry.n_images,
        )
        return SystemOneResponse(model=self.model_name, answers=answers, usage=usage)

    # ------------------------------------------------------------ batched (eval / train)
    def iter_batches(
        self, items: Sequence[tuple[Any, Branch]]
    ) -> Iterator[tuple[list[int], Batch]]:
        """Group many independent text-only (state, branch) pairs into forwards.

        No cache: each item is "state + branch" from scratch. Yields (item_indices, Batch).
        Shared by the evaluator (inference) and the trainer (with grad) so both see the
        identical computation the server runs.
        """
        seqs = [
            (i, self._encode(self.fmt.prefix(state)), self._encode(br.text))
            for i, (state, br) in enumerate(items)
        ]

        def make(group) -> tuple[list[int], Batch]:
            idx = [g[0] for g in group]
            if self.strategy == "packed":
                segs: list[Segment] = []
                for _, s_ids, b_ids in group:
                    segs.append(Segment(s_ids, -1))
                    segs.append(Segment(b_ids, len(segs) - 1))
                pack = build_pack(segs, device=self.device)
                kwargs = {
                    "input_ids": pack.input_ids,
                    "position_ids": pack.position_ids,
                    "attention_mask": pack.attention_mask,
                }
                return idx, Batch(kwargs, pack.last_index[1::2], packed=True)
            ids, mask, last = right_pad([s + b for _, s, b in group], self.pad_id, self.device)
            return idx, Batch({"input_ids": ids, "attention_mask": mask}, last, packed=False)

        group: list = []
        n, longest = 0, 0
        for item in seqs:
            size = len(item[1]) + len(item[2])
            new_longest = max(longest, size)
            budget = n + size if self.strategy == "packed" else (len(group) + 1) * new_longest
            if group and budget > self.max_pack_tokens:
                yield make(group)
                group, n, longest = [], 0, 0
                new_longest = size
            group.append(item)
            n += size
            longest = new_longest
        if group:
            yield make(group)

    def forward_batch(self, batch: Batch) -> torch.Tensor:
        """Logits [n_items, vocab] at each item's last token. Differentiable."""
        inner = self._inner()
        if inner is not None:
            inner.rope_deltas = None
        if batch.packed:
            return self.model(**batch.kwargs, use_cache=False, logits_to_keep=batch.last).logits[0]
        hidden = self._causal_lm().model(**batch.kwargs, use_cache=False).last_hidden_state
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), batch.last]
        return self._causal_lm().lm_head(h)

    @torch.inference_mode()
    def label_logits_batch(self, items: Sequence[tuple[Any, Branch]]) -> list[np.ndarray]:
        """Restricted last-token logits (fp32 numpy) for many independent (state, branch) pairs."""
        results: list[np.ndarray | None] = [None] * len(items)
        for idx, batch in self.iter_batches(items):
            logits = self.forward_batch(batch).float()
            for i, row in zip(idx, logits):
                results[i] = self.restrict(row, items[i][1]).cpu().numpy()
        return results  # type: ignore[return-value]
