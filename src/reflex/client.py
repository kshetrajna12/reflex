"""Minimal client, same call shape as the TypeSafe SDK's `client.systemone(...)`."""

from __future__ import annotations

from typing import Any

import httpx


class Reflex:
    def __init__(self, base_url: str = "http://127.0.0.1:8008", timeout: float = 120.0):
        self._http = httpx.Client(base_url=base_url, timeout=timeout)

    def systemone(
        self, state: Any, questions: dict[str, dict], model: str = "reflex-latest", **extra
    ) -> dict:
        r = self._http.post(
            "/v1/systemone", json={"model": model, "state": state, "questions": questions, **extra}
        )
        r.raise_for_status()
        return r.json()
