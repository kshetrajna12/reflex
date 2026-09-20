"""MPS memory defaults must stay valid next to whatever the user already set. Runs on CPU."""

import pytest
from fastapi.testclient import TestClient

from reflex.mps import HIGH_WATERMARK, LOW_WATERMARK, is_out_of_memory, memory_watermarks
from reflex.server import create_app


def test_defaults_when_nothing_is_set():
    assert memory_watermarks({}) == {HIGH_WATERMARK: "0.7", LOW_WATERMARK: "0.6"}


def test_a_user_low_above_our_high_raises_the_high_default():
    # torch rejects low=1.0 with high=0.7 ("invalid low watermark ratio 1")
    assert memory_watermarks({LOW_WATERMARK: "1.0"}) == {HIGH_WATERMARK: "1"}
    assert memory_watermarks({LOW_WATERMARK: "0.3"}) == {HIGH_WATERMARK: "0.7"}


def test_a_user_high_below_our_low_lowers_the_low_default():
    assert memory_watermarks({HIGH_WATERMARK: "0.5"}) == {LOW_WATERMARK: "0.5"}
    assert memory_watermarks({HIGH_WATERMARK: "1.4"}) == {LOW_WATERMARK: "0.6"}


def test_no_limit_is_respected():
    assert memory_watermarks({HIGH_WATERMARK: "0.0"}) == {LOW_WATERMARK: "0.6"}
    assert memory_watermarks({HIGH_WATERMARK: "0", LOW_WATERMARK: "1.4"}) == {}


def test_a_valid_user_pair_is_left_alone():
    assert memory_watermarks({HIGH_WATERMARK: "1.0", LOW_WATERMARK: "1.0"}) == {}


@pytest.mark.parametrize(
    "env",
    [
        {HIGH_WATERMARK: "0.5", LOW_WATERMARK: "0.6"},  # low above high
        {HIGH_WATERMARK: "2.5"},  # torch caps high at 2
        {LOW_WATERMARK: "2.5"},  # no valid high exists for it
        {LOW_WATERMARK: "-0.1"},
        {HIGH_WATERMARK: "lots"},
    ],
)
def test_an_invalid_pair_fails_with_a_clear_message(env):
    with pytest.raises(ValueError, match="PYTORCH_MPS"):
        memory_watermarks(env)


MPS_OOM = (
    "MPS backend out of memory (MPS allocated: 9.00 GiB, other allocations: 464.00 KiB, "
    "max allowed: 9.32 GiB). Tried to allocate 1024.00 MiB on private pool."
)
REQ = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}


class FailingEngine:
    model_name = "stub"
    device = "mps"
    strategy = "batched"

    class cal:
        temperature = {"noul": 1.0, "choice": 1.0, "score": 1.0}  # noqa: RUF012

    def __init__(self, error):
        self.error = error

    def answer(self, req):
        raise self.error


def test_recognises_only_the_mps_allocation_failure():
    assert is_out_of_memory(RuntimeError(MPS_OOM))
    assert not is_out_of_memory(RuntimeError("invalid low watermark ratio 1"))
    assert not is_out_of_memory(ValueError(MPS_OOM))


def test_mps_out_of_memory_is_a_529_like_cuda():
    client = TestClient(create_app(FailingEngine(RuntimeError(MPS_OOM))))
    assert client.post("/v1/systemone", json=REQ).status_code == 529


def test_other_runtime_errors_still_propagate():
    app = create_app(FailingEngine(RuntimeError("shape mismatch")))
    assert (
        TestClient(app, raise_server_exceptions=False).post("/v1/systemone", json=REQ).status_code
        == 500
    )
    with pytest.raises(RuntimeError, match="shape mismatch"):
        TestClient(app).post("/v1/systemone", json=REQ)
