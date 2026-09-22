"""The same readout, served by SGLang instead of transformers.

reflex's own engine owns the GPU: it keeps a KV cache per state and isolates question
branches with an attention mask (packed) or a batch-expanded cache copy (batched). A
production inference engine already does the first part — SGLang's radix cache shares any
token prefix across requests — and gives the second part for free, because two items of a
batch cannot see each other by construction. So the whole design reduces to: render the
prefix once, warm the radix cache with it, then ask for every branch in **one** `/generate`
call and read the label log-probabilities back.

The recipe is the one [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)
demonstrated on Qwen3.6-35B-A3B (see `docs/results/serving-engines-research.md`). What we
reuse from it is the *protocol*, not its code: send `input_ids` (already tokenized, so
the prefix ids are byte-identical to the ones the branches carry), `max_new_tokens=1`,
`return_logprob=true`, `logprob_start_len=-1` so no prompt log-probabilities are
recomputed, and `token_ids_logprob=[label ids]` to get the *exact* log-probability of
every label rather than a truncated top-k list; then read `meta_info.output_token_ids_logprobs`
and softmax over the labels. Its two workarounds are reused as well: the warm-up call
also asks for one unused token id, because SGLang crashes when logprob and plain requests
share a batch (sgl-project/sglang#34719), and every item carries its own `rid` so a
cancelled evaluation can abort its siblings.

SGLang's native `/generate` takes all of those fields per item: `input_ids` as a list of
lists makes it a batch, and `sampling_params`, `return_logprob`, `logprob_start_len`,
`top_logprobs_num`, `token_ids_logprob` and `rid` are then lists of the same length, one
entry per item. So one reflex request is one HTTP call, whatever its question count. The
items still become independent scheduler requests, so nothing about isolation or the radix
cache changes, and `docs/results/sglang-batched.md` measures what it is worth: packing the
branches into one envelope is free rather than fast, and what actually moves is that
`max_concurrent_calls` below now bounds requests instead of branches.

A choice with more than 26 options arrives here as several pages, each an ordinary
branch with its own label ids, so nothing about the protocol changes: the pages are
items of the same batched call and `merge_branches` pools them. SGLang reports
log-probabilities rather than logits, which is exactly the per-page normalisation the
pooled softmax wants, so the backend needs no counterpart to `Engine.restrict`.

Everything above the HTTP call is reflex's: `PromptFormat` renders the prefix and the
branches, `build_branches` produces the permutations, the pages and the lettered pair, and
`readout.merge_branches` does the temperature scaling, the order-averaging and the typed
answer. A label log-probability is the label logit minus a constant that is shared by
every label at that position, so softmax — with or without a temperature — gives exactly
the distribution the transformers engine computes from raw logits.

Out of scope for this backend: images in the state, and anything that needs the weights
in this process (a LoRA adapter, a prompt ensemble across wordings). SGLang owns the GPU
and reflex holds no weights, so those are refused rather than quietly ignored.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import threading
from collections import OrderedDict
from typing import Any
from uuid import uuid4

import httpx
import numpy as np

from reflex.backends import BackendError
from reflex.images import has_images
from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.readout import Calibration, merge_branches, to_answer
from reflex.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger(__name__)

# Most items reflex puts in one `/generate` call. A reflex request is questions x
# permutations items and stays well under this; the eval path (`label_logits_batch`)
# chunks a whole dataset down to it.
MAX_BATCH_ITEMS = 32


class SGLangError(BackendError):
    """The SGLang server refused, timed out or answered something unusable."""


class SGLangBackend:
    """Answers `/v1/systemone` requests against an SGLang server.

    Exposes the attributes `reflex.server` reads off an engine (`model_name`, `cal`,
    `strategy`, `device`) plus `answer(req)`, so it drops into `create_app` unchanged.
    """

    def __init__(
        self,
        url: str = "http://127.0.0.1:30000",
        *,
        model_id: str = "Qwen/Qwen3.5-4B",
        tokenizer=None,
        fmt: PromptFormat | None = None,
        calibration: Calibration | None = None,
        model_name: str | None = None,
        default_permutations: int = 1,
        max_concurrent_calls: int = 8,
        timeout: float = 120.0,
        state_cache_entries: int = 8,
    ):
        from transformers import AutoTokenizer

        self.url = url.rstrip("/")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_id)
        if fmt is None:
            template = self.tok.chat_template or ""
            fmt = PromptFormat(chat=bool(template), no_think="enable_thinking" in template)
        self.fmt = fmt
        self.cal = calibration or Calibration()
        self.model_name = model_name or model_id
        self.default_permutations = default_permutations
        self.timeout = timeout
        # what the /healthz route reports; there is no local device
        self.strategy = "sglang"
        self.device = self.url
        self._label_ids: dict[str, int] = {}
        # Which prefixes we have already warmed. SGLang's radix cache is opportunistic and
        # does not promise a hit, so this reports "we sent this state before", not a
        # guarantee that the KV blocks survived.
        self._seen: OrderedDict[str, int] = OrderedDict()
        self._state_cache_entries = state_cache_entries

        # Bound the concurrent `/generate` calls. One reflex request is one batched call
        # (two on a state we have not seen, counting the prefix warm-up), so this is
        # roughly "reflex requests in flight against SGLang". The queue sits here, where it
        # is visible, rather than as open connections on the inference server.
        self.max_concurrent_calls = max_concurrent_calls
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="sglang-io")
        self._thread.start()
        self._ready.wait()
        self._slots = asyncio.Semaphore(max_concurrent_calls)

    # ------------------------------------------------------------------ event loop thread
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=256)
        self._client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout, limits=limits)
        self._loop.call_soon(self._ready.set)
        self._loop.run_forever()

    def close(self) -> None:
        """Shut the io thread down. Safe to call twice."""
        if not self._loop.is_closed():
            fut = asyncio.run_coroutine_threadsafe(self._client.aclose(), self._loop)
            fut.result(timeout=10)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)

    # ------------------------------------------------------------------------ tokenisation
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

    def state_inputs(self, state) -> tuple[list[int], str]:
        """Prefix token ids and the cache key for a state. Text only."""
        if has_images(state):
            raise NotImplementedError(
                "the SGLang backend does not carry images in the state; "
                "serve the transformers backend for image states"
            )
        text = self.fmt.prefix(state)
        return self._encode(text), hashlib.sha256(text.encode()).hexdigest()

    # -------------------------------------------------------------------------- transport
    async def _generate(
        self, items: list[tuple[list[int], list[int]]]
    ) -> tuple[list[list[float]], list[int]]:
        """One batched `max_new_tokens=1` call.

        `items` is `(input_ids, label_ids)` per item. Returns the label log-probs of every
        item, in that item's label order, and its prompt-token count.
        """
        base = f"reflex-{uuid4().hex}"
        rids = [f"{base}-{i}" for i in range(len(items))]
        payload: dict[str, Any] = {
            "rid": rids,
            "input_ids": [ids for ids, _ in items],
            "sampling_params": [
                {
                    "max_new_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "ignore_eos": True,
                }
                for _ in items
            ],
            "stream": False,
            "return_logprob": [True] * len(items),
            # openjev-sglang's workaround for sgl-project/sglang#34719: a request without
            # selected-token logprobs in the same batch as one with them crashes the
            # scheduler, so the warm-up asks for an unused id too.
            "token_ids_logprob": [list(labels) or [0] for _, labels in items],
            "logprob_start_len": [-1] * len(items),
            "top_logprobs_num": [0] * len(items),
            "return_text_in_logprobs": False,
        }
        async with self._slots:
            try:
                r = await self._client.post("/generate", json=payload)
            except asyncio.CancelledError:
                await self._abort(rids)
                raise
            except httpx.TimeoutException as e:
                await self._abort(rids)
                raise SGLangError("SGLang request timed out") from e
            except httpx.HTTPError as e:
                raise SGLangError(f"cannot reach SGLang at {self.url}: {e}") from e
        if r.status_code != 200:
            raise SGLangError(f"SGLang returned HTTP {r.status_code}: {r.text[:200]}")
        return _parse_batch(r.json(), rids, [labels for _, labels in items])

    async def _abort(self, rids: list[str]) -> None:
        for rid in rids:
            try:
                await self._client.post("/abort_request", json={"rid": rid}, timeout=2)
            except httpx.HTTPError:
                pass

    async def _run(
        self, prefix_ids: list[int], branches: list[Branch], branch_ids: list[list[int]], warm: bool
    ) -> tuple[list[np.ndarray], int]:
        """Warm the radix cache with the prefix, then ask for every branch in one call."""
        prompt_tokens = 0
        if not warm:
            # The branches of one batch are admitted together, so without this they would
            # each prefill the whole state instead of extending it. Measured in
            # docs/results/sglang-batched.md: it is what keeps a cold request at one
            # prefill rather than one per branch.
            _, n = await self._generate([(prefix_ids, [])])
            prompt_tokens += sum(n)
        items = [
            (prefix_ids + ids, [self.label_id(lb) for lb in br.labels])
            for br, ids in zip(branches, branch_ids, strict=True)
        ]
        results = await _gather(
            [self._generate(chunk) for chunk in _chunks(items, MAX_BATCH_ITEMS)]
        )
        rows: list[np.ndarray] = []
        for logps, counts in results:
            rows.extend(np.asarray(v, dtype=np.float64) for v in logps)
            prompt_tokens += sum(counts)
        return rows, prompt_tokens

    # ----------------------------------------------------------------------------- answer
    def answer(self, req: SystemOneRequest) -> SystemOneResponse:
        branches: list[Branch] = []
        for qid, q in req.questions.items():
            branches.extend(
                build_branches(qid, q, self.fmt, req.permutations or self.default_permutations)
            )
        branch_ids = [self._encode(b.text) for b in branches]
        prefix_ids, key = self.state_inputs(req.state)

        hit = key in self._seen
        fut = asyncio.run_coroutine_threadsafe(
            self._run(prefix_ids, branches, branch_ids, hit), self._loop
        )
        rows, _ = fut.result()
        self._seen[key] = len(prefix_ids)
        self._seen.move_to_end(key)
        while len(self._seen) > self._state_cache_entries:
            self._seen.popitem(last=False)

        per_q: dict[str, list[tuple[Branch, np.ndarray]]] = {}
        for b, row in zip(branches, rows, strict=True):
            per_q.setdefault(b.qid, []).append((b, row))

        answers = {}
        for qid, q in req.questions.items():
            key_probs = merge_branches(q.type, per_q[qid], self.cal, state_tokens=len(prefix_ids))
            answers[qid] = to_answer(q.type, key_probs, q)

        q_tokens = sum(len(b) for b in branch_ids)
        usage = Usage(
            input_tokens=len(prefix_ids) + q_tokens,
            state_tokens=len(prefix_ids),
            question_tokens=q_tokens,
            state_cache_hit=hit,
        )
        return SystemOneResponse(model=self.model_name, answers=answers, usage=usage)

    # ------------------------------------------------------- parity with Engine for evals
    def label_logits_batch(self, items) -> list[np.ndarray]:
        """Restricted label log-probs for many independent (state, branch) pairs.

        Same contract as `Engine.label_logits_batch` (up to the per-row constant a softmax
        removes), so the evaluator and the calibration fitter run against SGLang unchanged.
        """
        pairs = [
            (
                self._encode(self.fmt.prefix(state)) + self._encode(br.text),
                [self.label_id(lb) for lb in br.labels],
            )
            for state, br in items
        ]

        async def go():
            return await _gather([self._generate(c) for c in _chunks(pairs, MAX_BATCH_ITEMS)])

        out = asyncio.run_coroutine_threadsafe(go(), self._loop).result()
        return [np.asarray(v, dtype=np.float64) for logps, _ in out for v in logps]


def _chunks(xs: list, n: int) -> list[list]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


async def _gather(coros: list):
    """`asyncio.gather`, but a failure cancels the siblings instead of leaving them running.

    The semaphore holds most of them un-started, so a request that dies takes its own queue
    with it rather than filling the server's.
    """
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _parse_batch(
    data: Any, rids: list[str], label_ids: list[list[int]]
) -> tuple[list[list[float]], list[int]]:
    """Pull the label log-probs out of one batched `/generate` response, in item order.

    SGLang answers a batch in request order, but the reply also carries each item's `rid`,
    so we key on that rather than trusting the order: a reordered, short or duplicated
    reply is a mismatch we raise on instead of reading someone else's row.
    """
    if not isinstance(data, list):
        raise SGLangError(f"expected a list of {len(rids)} results, got {type(data).__name__}")
    by_rid: dict[str, dict] = {}
    for entry in data:
        rid = ((entry or {}).get("meta_info") or {}).get("id")
        if rid in by_rid:
            raise SGLangError(f"SGLang returned two results for {rid!r}")
        by_rid[rid] = entry
    missing = [rid for rid in rids if rid not in by_rid]
    if missing or len(data) != len(rids):
        raise SGLangError(
            f"SGLang returned {len(data)} of {len(rids)} batch items"
            + (f"; missing {missing[:3]}" if missing else "")
        )
    rows, counts = [], []
    for rid, labels in zip(rids, label_ids, strict=True):
        logps, n = _parse(by_rid[rid], list(labels))
        rows.append(logps)
        counts.append(n)
    return rows, counts


def _parse(data: dict, label_ids: list[int]) -> tuple[list[float], int]:
    """Pull the label log-probs out of one item of a `/generate` response, in label order."""
    meta = data["meta_info"]
    if (meta.get("finish_reason") or {}).get("type") == "abort":
        raise SGLangError("generation was aborted")
    if int(meta.get("completion_tokens", 0)) != 1:
        raise SGLangError(f"expected exactly one output token, got {meta.get('completion_tokens')}")
    prompt_tokens = int(meta["prompt_tokens"])
    if not label_ids:
        return [], prompt_tokens
    positions = meta["output_token_ids_logprobs"]
    if len(positions) != 1:
        raise SGLangError("expected one position of selected-token log-probs")
    by_id: dict[int, float] = {}
    for entry in positions[0]:
        value, token_id = entry[0], entry[1]
        # A label the model gives no mass to comes back as None or -inf; floor it below,
        # where softmax turns it into zero. A label that is *absent* is a different thing:
        # the server answered about some other item's labels, and we must not guess.
        by_id[int(token_id)] = float(value) if value is not None else -math.inf
    absent = [i for i in label_ids if i not in by_id]
    if absent:
        raise SGLangError(f"SGLang returned no log-probability for label ids {absent[:5]}")
    finite = [v for v in by_id.values() if math.isfinite(v)]
    if not finite:
        raise SGLangError("SGLang returned no usable label log-probabilities")
    floor = min(finite) - 30.0  # far below every real label: effectively zero mass
    out = [by_id[i] if math.isfinite(by_id[i]) else floor for i in label_ids]
    return out, prompt_tokens
