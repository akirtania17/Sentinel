"""Helpers for driving the TypeSafe adapter through the SDK with recorded HTTP responses."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest

BODIES = Path(__file__).parent / "http_bodies"
# A fake key used to prove the real key never leaks; not a credential.
KEY_MARKER = "apikey_contracttest_KEYMARKER_0f9e8d7c"  # gitleaks:allow


def recorded(name: str) -> dict[str, Any]:
    """A recorded response: ``{"status": ..., "headers": {...}, "body": {...}}``."""
    return json.loads((BODIES / f"{name}.json").read_text())


class Recorder:
    """A mock transport handler that replays responses in order and records requests."""

    def __init__(self, *responses: dict[str, Any] | Callable[[httpx2.Request], httpx2.Response]) -> None:
        self._responses = list(responses)
        self.requests: list[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if callable(response):
            return response(request)
        return httpx2.Response(response["status"], headers=response["headers"], json=response["body"])

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


@pytest.fixture(autouse=True)
def _marker_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every contract test runs with a recognizable fake key so leaks are detectable."""
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY_MARKER)
