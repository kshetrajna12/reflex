"""The same readout, served by SGLang instead of transformers.

reflex's own engine owns the GPU: it keeps a KV cache per state and isolates question
branches with an attention mask (packed) or a batch-expanded cache copy (batched). A
production inference engine already does the first part — SGLang's radix cache shares any
token prefix across requests — and gives the second part for free, because two separate
HTTP requests cannot see each other by construction. So the whole design reduces to:
render the prefix once, warm the radix cache with it, then fire one `/generate` per
branch and read the label log-probabilities back.

The recipe is the one [ekzhang/openjev-sglang](https://github.com/ekzhang/openjev-sglang)
demonstrated on Qwen3.6-35B-A3B (see `docs/results/serving-engines-research.md`). What we
reuse from it is the *protocol*, not its code: send `input_ids` (already tokenized, so
the prefix ids are byte-identical to the ones the branches carry), `max_new_tokens=1`,
`return_logprob=true`, `logprob_start_len=-1` so no prompt log-probabilities are
recomputed, and `token_ids_logprob=[label ids]` to get the *exact* log-probability of
every label rather than a truncated top-k list; then read `meta_info.output_token_ids_logprobs`
and softmax over the labels. Its two workarounds are reused as well: the warm-up call
also asks for one unused token id, because SGLang crashes when logprob and plain requests
share a batch (sgl-project/sglang#34719), and every request carries its own `rid` so a
cancelled evaluation can abort its siblings.

Everything above the HTTP call is reflex's: `PromptFormat` renders the prefix and the
branches, `build_branches` produces the permutations and the lettered yes/no pair, and
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

from reflex.images import has_images
from reflex.prompt import Branch, PromptFormat, build_branches
from reflex.readout import Calibration, merge_branches, to_answer
from reflex.schema import SystemOneRequest, SystemOneResponse, Usage

log = logging.getLogger(__name__)


class SGLangError(RuntimeError):
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
        max_concurrent_branches: int = 64,
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

        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="sglang-io")
        self._thread.start()
        self._ready.wait()
        self._slots = asyncio.Semaphore(max_concurrent_branches)

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
        self, input_ids: list[int], label_ids: list[int]
    ) -> tuple[list[float], int]:
        """One `max_new_tokens=1` call. Returns (label log-probs in label order, prompt tokens)."""
        rid = f"reflex-{uuid4().hex}"
        payload: dict[str, Any] = {
            "rid": rid,
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "ignore_eos": True,
            },
            "stream": False,
            "return_logprob": True,
            # openjev-sglang's workaround for sgl-project/sglang#34719: a request without
            # selected-token logprobs in the same batch as one with them crashes the
            # scheduler, so the warm-up asks for an unused id too.
            "token_ids_logprob": list(label_ids) or [0],
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
        }
        async with self._slots:
            try:
                r = await self._client.post("/generate", json=payload)
            except asyncio.CancelledError:
                await self._abort(rid)
                raise
            except httpx.TimeoutException as e:
                await self._abort(rid)
                raise SGLangError("SGLang request timed out") from e
            except httpx.HTTPError as e:
                raise SGLangError(f"cannot reach SGLang at {self.url}: {e}") from e
        if r.status_code != 200:
            raise SGLangError(f"SGLang returned HTTP {r.status_code}: {r.text[:200]}")
        return _parse(r.json(), list(label_ids))

    async def _abort(self, rid: str) -> None:
        try:
            await self._client.post("/abort_request", json={"rid": rid}, timeout=2)
        except httpx.HTTPError:
            pass

    async def _run(
        self, prefix_ids: list[int], branches: list[Branch], branch_ids: list[list[int]], warm: bool
    ) -> tuple[list[np.ndarray], int]:
        """Warm the radix cache with the prefix, then fan the branches out concurrently."""
        prompt_tokens = 0
        if not warm:
            _, n = await self._generate(prefix_ids, [])
            prompt_tokens += n
        tasks = [
            asyncio.ensure_future(
                self._generate(prefix_ids + ids, [self.label_id(lb) for lb in br.labels])
            )
            for br, ids in zip(branches, branch_ids, strict=True)
        ]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        rows = []
        for logps, n in results:
            rows.append(np.asarray(logps, dtype=np.float64))
            prompt_tokens += n
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

        async def go():
            tasks = [
                asyncio.ensure_future(
                    self._generate(
                        self._encode(self.fmt.prefix(state)) + self._encode(br.text),
                        [self.label_id(lb) for lb in br.labels],
                    )
                )
                for state, br in items
            ]
            return await asyncio.gather(*tasks)

        out = asyncio.run_coroutine_threadsafe(go(), self._loop).result()
        return [np.asarray(logps, dtype=np.float64) for logps, _ in out]


def _parse(data: dict, label_ids: list[int]) -> tuple[list[float], int]:
    """Pull the label log-probs out of one `/generate` response, in label order."""
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
        if value is None or not math.isfinite(float(value)):
            continue  # a label the model gives no mass to; softmax handles -inf below
        by_id[int(token_id)] = float(value)
    if not by_id:
        raise SGLangError("SGLang returned no usable label log-probabilities")
    floor = min(by_id.values()) - 30.0  # far below every real label: effectively zero mass
    return [by_id.get(i, floor) for i in label_ids], prompt_tokens
