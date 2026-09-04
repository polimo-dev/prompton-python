from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from prompton.errors import TransportError
from prompton.http import HttpResponse
from prompton.testing import make_snapshot

CONFORMANCE = Path(__file__).parent / "conformance"


@pytest.fixture(autouse=True)
def isolate_environment(request, monkeypatch):
    """Keep the suite hermetic: no PTN_* variable leaks into a unit test.

    The live integration module is the one place that wants them.
    """
    if request.module.__name__.endswith("test_integration_live"):
        return
    for name in [key for key in os.environ if key.startswith("PTN_")]:
        monkeypatch.delenv(name, raising=False)


def load_conformance(name: str) -> dict[str, Any]:
    return json.loads((CONFORMANCE / f"{name}.json").read_text(encoding="utf-8"))


class FakeTransport:
    """A scripted transport: hand it responses, read back the requests it received."""

    def __init__(self, handler: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.handler = handler
        self.responses: list[Any] = []
        self.lock = threading.Lock()

    def push(self, response: Any) -> None:
        with self.lock:
            self.responses.append(response)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        body: bytes | None = None,
        timeout: float = 5.0,
    ) -> HttpResponse:
        call = {
            "method": method,
            "url": url,
            "headers": dict(headers),
            "body": json.loads(body) if body else None,
            "timeout": timeout,
        }
        with self.lock:
            self.requests.append(call)
            outcome = self.responses.pop(0) if self.responses else None
        if outcome is None and self.handler is not None:
            outcome = self.handler(call)
        if outcome is None:
            raise AssertionError(f"FakeTransport has no response for {method} {url}")
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def snapshot_requests(self) -> list[dict[str, Any]]:
        return [call for call in self.requests if "/snapshot" in call["url"]]

    @property
    def generation_requests(self) -> list[dict[str, Any]]:
        return [call for call in self.requests if "/generations" in call["url"]]


def json_response(status: int, body: Any, **headers: str) -> HttpResponse:
    return HttpResponse(
        status=status,
        body=json.dumps(body).encode("utf-8") if body is not None else b"",
        headers={"content-type": "application/json", **headers},
    )


def transport_error() -> TransportError:
    return TransportError("could not reach PromptOn: connection refused")


@pytest.fixture
def snapshot_document() -> dict[str, Any]:
    return make_snapshot(
        project="sdkfixture",
        environment="production",
        greeting={
            "model": "openai/gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a friendly greeter."},
                {"role": "user", "content": "Say hello to {{ name }}."},
            ],
            "prompts": {"ko": [{"role": "user", "content": "{{ name }}님에게 인사해줘."}]},
            "params": {"temperature": 0.2},
            "default_params": {"max_tokens": 256},
            "provider_options": {"only": ["OpenAI"]},
        },
        summarize={
            "kind": "text",
            "model": "openai/gpt-4o-mini",
            "text": "Summarize:\n{% for item in items %}- {{ item }}\n{% endfor %}",
        },
        embed={"kind": "embedding", "model": "openai/text-embedding-3-small"},
    )


@pytest.fixture
def snapshot_bytes(snapshot_document: dict[str, Any]) -> bytes:
    return json.dumps(snapshot_document).encode("utf-8")
