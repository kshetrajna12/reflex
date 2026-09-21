"""One reflex request must be one `/generate` call, and the calls must stay inside the bound.

The backend asks SGLang for every branch of a request in a single batched call, so a
12-question request at two orders is one POST, not twenty-five. `--sglang-concurrency`
then bounds how many such calls are in flight at once, across all callers. This runs the
backend against a fake server that records the request shapes and the peak concurrency, so
it needs no GPU, no weights and no network.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from reflex.backends.sglang import MAX_BATCH_ITEMS, SGLangBackend, SGLangError
from reflex.prompt import PromptFormat
from reflex.schema import SystemOneRequest


class FakeTokenizer:
    """One id per whitespace-separated word, so every label is a single token."""

    chat_template = ""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        return self.vocab.setdefault(word, 1000 + len(self.vocab))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id(w) for w in (text.split() or [text])]


class FakeServer:
    """Answers a batched `/generate` after a short delay, tracking shapes and concurrency."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.calls = 0
        self.batch_sizes: list[int] = []
        # test hooks: mangle the reply the way a broken server would
        self.shuffle = False
        self.drop_label = False
        self.drop_item = False

    def _item(self, rid: str, ids: list[int], labels: list[int]) -> dict:
        if self.drop_label:
            labels = labels[:-1]
        return {
            "text": "",
            "meta_info": {
                "id": rid,
                "prompt_tokens": len(ids),
                "completion_tokens": 1,
                "output_token_ids_logprobs": [
                    [[-0.5 - 0.1 * i, tid, None] for i, tid in enumerate(labels)]
                ],
            },
        }

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert isinstance(body["input_ids"][0], list), "every call must be a batch"
        n = len(body["input_ids"])
        for field in ("rid", "sampling_params", "return_logprob", "token_ids_logprob"):
            assert len(body[field]) == n, f"{field} must carry one entry per item"
        self.calls += 1
        self.batch_sizes.append(n)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        items = [
            self._item(rid, ids, labels)
            for rid, ids, labels in zip(
                body["rid"], body["input_ids"], body["token_ids_logprob"], strict=True
            )
        ]
        if self.shuffle and len(items) > 1:
            items = items[1:] + items[:1]
        if self.drop_item and len(items) > 1:
            items = items[:-1]
        return httpx.Response(200, json=items)


def _backend(bound: int, server: FakeServer) -> SGLangBackend:
    backend = SGLangBackend(
        "http://fake",
        tokenizer=FakeTokenizer(),
        fmt=PromptFormat(chat=False),
        max_concurrent_calls=bound,
    )
    # Swap the transport for the fake server; the io thread owns the client, and replacing
    # it before any request is in flight is safe.
    old = backend._client
    backend._client = httpx.AsyncClient(
        base_url="http://fake", transport=httpx.MockTransport(server)
    )
    asyncio.run_coroutine_threadsafe(old.aclose(), backend._loop).result(timeout=5)
    return backend


def _request(n_questions: int, permutations: int) -> SystemOneRequest:
    questions = {
        f"q{i}": {
            "type": "choice",
            "instructions": f"pick one for question {i}",
            "criteria": {"alpha": None, "beta": None, "gamma": None},
        }
        for i in range(n_questions)
    }
    return SystemOneRequest(
        state={"ticket": "a customer wrote in about a refund"},
        questions=questions,
        permutations=permutations,
    )


def test_one_request_is_one_batched_call():
    """Three questions at two orders: the prefix warm-up, then all six branches at once."""
    server = FakeServer()
    backend = _backend(8, server)
    try:
        resp = backend.answer(_request(n_questions=3, permutations=2))
        first = list(server.batch_sizes)
        backend.answer(_request(n_questions=3, permutations=2))  # same state, now warm
    finally:
        backend.close()

    assert len(resp.answers) == 3
    assert first == [1, 6], f"expected a 1-item warm-up then a 6-item batch, got {first}"
    assert server.batch_sizes[2:] == [6], "a warm state skips the warm-up entirely"


@pytest.mark.parametrize("bound", [1, 3, 8])
def test_calls_never_exceed_the_bound(bound):
    """The bound is on `/generate` calls, and a big request is only a handful of them."""
    server = FakeServer()
    backend = _backend(bound, server)
    try:
        resp = backend.answer(_request(n_questions=12, permutations=2))
    finally:
        backend.close()

    assert len(resp.answers) == 12
    # 24 branches, chunked at MAX_BATCH_ITEMS, plus one prefix warm-up call
    expected = 1 + -(-24 // MAX_BATCH_ITEMS)
    assert server.calls == expected, f"{server.calls} calls for 24 branches"
    assert max(server.batch_sizes) <= MAX_BATCH_ITEMS
    assert server.peak <= bound, f"{server.peak} calls in flight, bound was {bound}"


def test_bound_is_shared_across_concurrent_requests():
    """Two callers at once still cannot put more than `bound` calls on the server."""
    from concurrent.futures import ThreadPoolExecutor

    server = FakeServer()
    backend = _backend(2, server)
    try:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda _: backend.answer(_request(6, 2)), range(6)))
    finally:
        backend.close()

    assert server.peak <= 2, f"{server.peak} calls in flight across six callers"
    assert server.peak == 2, "the bound should be reached, not just respected"


def test_items_are_matched_by_rid_not_by_position():
    """A reply that comes back rotated must still answer the right question."""
    ordered, rotated = FakeServer(), FakeServer()
    rotated.shuffle = True
    req = _request(3, 2)
    out = []
    for server in (ordered, rotated):
        backend = _backend(8, server)
        try:
            out.append(backend.answer(req))
        finally:
            backend.close()
    for qid in out[0].answers:
        assert out[0].answers[qid].probabilities == out[1].answers[qid].probabilities


def test_a_short_batch_is_an_error():
    """A reply missing an item must fail loudly rather than answer from someone else's row."""
    server = FakeServer()
    server.drop_item = True
    backend = _backend(8, server)
    try:
        with pytest.raises(SGLangError, match="batch items"):
            backend.answer(_request(3, 2))
    finally:
        backend.close()


def test_a_missing_label_id_is_an_error():
    """A label the server never reported is not a zero; it means the rows do not line up."""
    server = FakeServer()
    server.drop_label = True
    backend = _backend(8, server)
    try:
        with pytest.raises(SGLangError, match="no log-probability for label ids"):
            backend.answer(_request(3, 2))
    finally:
        backend.close()
