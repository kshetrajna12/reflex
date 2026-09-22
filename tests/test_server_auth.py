"""The HTTP layer without a GPU: a stub engine, the real app."""

from fastapi.testclient import TestClient

from reflex.schema import NoulAnswer, SystemOneResponse, Usage
from reflex.server import create_app


class StubEngine:
    model_name = "stub"
    device = "cpu"
    strategy = "packed"

    class cal:
        temperature = {"noul": 1.0, "choice": 1.0, "score": 1.0}  # noqa: RUF012

    def answer(self, req):
        return SystemOneResponse(
            model="stub",
            answers={q: NoulAnswer(noul=0.5) for q in req.questions},
            usage=Usage(input_tokens=1),
        )


REQ = {"state": "x", "questions": {"q": {"type": "noul", "instructions": "?"}}}


def test_open_when_no_key():
    c = TestClient(create_app(StubEngine()))
    assert c.get("/health").json()["status"] == "healthy"
    assert c.post("/v1/systemone", json=REQ).status_code == 200
    assert c.get("/v1/models").json()["data"][0]["id"] == "stub"


def test_bearer_required_when_key_set():
    c = TestClient(create_app(StubEngine(), api_key="s3cret"))
    assert c.get("/health").status_code == 200  # health stays open
    r = c.post("/v1/systemone", json=REQ)
    assert r.status_code == 401 and r.json()["error"]["type"] == "invalid_api_key"
    ok = c.post("/v1/systemone", json=REQ, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["answers"]["q"]["noul"] == 0.5


def test_validation_error_is_422():
    c = TestClient(create_app(StubEngine()))
    bad = {
        "state": "x",
        "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"only": None}}},
    }
    assert c.post("/v1/systemone", json=bad).status_code == 422


def _choice(n):
    return {
        "state": "x",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "which?",
                "criteria": {f"k{i}": None for i in range(n)},
            }
        },
    }


def test_choice_ceiling_is_256_over_the_wire():
    c = TestClient(create_app(StubEngine()))
    assert c.post("/v1/systemone", json=_choice(256)).status_code == 200
    r = c.post("/v1/systemone", json=_choice(257))
    assert r.status_code == 422
    assert "at most 256 options per choice" in r.text
