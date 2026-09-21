"""The branch fan-out must stay inside its bound.

One reflex request becomes `questions x permutations` `/generate` calls, fired together.
SGLang runs only a handful at a time (on a hybrid model its mamba state cache caps
`max_running_requests`), so an unbounded fan-out buys nothing and costs open connections:
a dropped one fails the whole request. This runs the backend against a fake server that
records how many calls are in flight, so it needs no GPU, no weights and no network.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from reflex.backends.sglang import SGLangBackend
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
    """Answers `/generate` after a short delay, tracking peak concurrency."""

    def __init__(self, delay: float = 0.02):
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.calls = 0

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls += 1
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
        finally:
            self.in_flight -= 1
        label_ids = body["token_ids_logprob"]
        return httpx.Response(
            200,
            json={
                "text": "",
                "meta_info": {
                    "prompt_tokens": len(body["input_ids"]),
                    "completion_tokens": 1,
                    "output_token_ids_logprobs": [
                        [[-0.5 - 0.1 * i, tid, None] for i, tid in enumerate(label_ids)]
                    ],
                },
            },
        )


def _backend(bound: int, server: FakeServer) -> SGLangBackend:
    backend = SGLangBackend(
        "http://fake",
        tokenizer=FakeTokenizer(),
        fmt=PromptFormat(chat=False),
        max_concurrent_branches=bound,
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


@pytest.mark.parametrize("bound", [1, 3, 8])
def test_fanout_never_exceeds_the_bound(bound):
    server = FakeServer()
    backend = _backend(bound, server)
    try:
        resp = backend.answer(_request(n_questions=12, permutations=2))
    finally:
        backend.close()

    assert len(resp.answers) == 12
    # 12 questions x 2 orders + one prefix warm-up call
    assert server.calls == 25
    assert server.peak <= bound, f"{server.peak} calls in flight, bound was {bound}"
    assert server.peak == min(bound, 24), "the bound should be reached, not just respected"


def test_bound_is_shared_across_concurrent_requests():
    """Two callers at once still cannot put more than `bound` calls on the server."""
    from concurrent.futures import ThreadPoolExecutor

    server = FakeServer()
    backend = _backend(4, server)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: backend.answer(_request(6, 2)), range(3)))
    finally:
        backend.close()

    assert server.peak <= 4, f"{server.peak} calls in flight across three callers"
