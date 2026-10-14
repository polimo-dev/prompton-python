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
from prompton.testing import make_use_case_document

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


def single_prompt_document(
    key: str = "greeting", environment: str = "production", **overrides: Any
) -> dict[str, Any]:
    """Convert the readable schema v1 fixture into the schema v1 single-prompt shape."""
    readable = readable_document(
        make_use_case_document(
            project="sdkfixture",
            environment=environment,
            greeting={
                "messages": [
                    {"role": "system", "content": "You are a friendly greeter."},
                    {"role": "user", "content": "Say hello to {{ name }}."},
                ],
                "provider_options": {"only": ["OpenAI"]},
                "variables": [{"name": "name", "type": "string", "required": True}],
            },
            sentiment={
                "kind": "decision",
                "model": "typesafe/jev-1.13",
                "messages": [],
            },
        )
    )
    prompt = readable["prompts"].get(key)
    deployment = readable["deployments"].get(key)
    if prompt is None or deployment is None:
        raise KeyError(key)
    version = readable["prompt_versions"].get(f"{key}@{deployment['prompt_version']}")
    model = readable["models"].get(f"{deployment['provider']}:{deployment['model']}")
    if version is None or model is None:
        raise KeyError(key)
    if key == "sentiment":
        deployment = {**deployment, "api": "decisions", "request_path": "/api/v1/systemone"}
    elif key == "greeting":
        deployment = {
            **deployment,
            "api": "chat_completions",
            "request_path": "/api/v1/chat/completions",
        }
    document = {
        "schema_version": 1,
        "project": readable["project"],
        "environment": environment,
        "prompt": {"key": key, **prompt},
        "deployment": deployment,
        "prompt_version": version,
        "model": model,
    }
    document.update(overrides)
    return document


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
            result = self.responses.pop(0) if self.responses else None
        if result is None and self.handler is not None:
            result = self.handler(call)
        if result is None:
            raise AssertionError(f"FakeTransport has no response for {method} {url}")
        if isinstance(result, BaseException):
            raise result
        return result

    @property
    def snapshot_requests(self) -> list[dict[str, Any]]:
        return [call for call in self.requests if "/prompts" in call["url"]]

    @property
    def generation_requests(self) -> list[dict[str, Any]]:
        return [call for call in self.requests if "/logs" in call["url"]]


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
    return make_use_case_document(
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
            "variables": [{"name": "name", "type": "string", "required": True}],
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


def readable_document(document: dict[str, Any]) -> dict[str, Any]:
    """Convert the test document into the schema v1 aggregate readable-resource wire shape."""
    legacy = json.loads(json.dumps(document))
    models_by_id = legacy["models"]
    versions_by_id = legacy["prompt_versions"]
    deployments: dict[str, Any] = {}
    prompt_versions: dict[str, Any] = {}
    models: dict[str, Any] = {}

    for key, deployment in legacy["deployments"].items():
        model = models_by_id[deployment["model_id"]]
        provider = model["provider"]
        model_name = model["model_id"]
        pins = deployment.get("prompt_pins") or deployment.get("template_pins") or {}
        version = versions_by_id.get(pins.get("default"))
        deployments[key] = {
            "revision": deployment["revision"],
            "provider": provider,
            "model": model_name,
            "prompt_version": version["number"] if version else None,
            "params": deployment["params"],
            "provider_options": deployment["provider_options"],
            "api": deployment.get("api"),
            "request_path": deployment.get("request_path"),
        }
        if version:
            number = version["number"]
            prompt_versions[f"{key}@{number}"] = {
                "prompt": key,
                "number": number,
                "kind": version.get("kind") or legacy["prompts"][key]["kind"],
                "engine": version["engine"],
                "messages": version["messages"],
                "text_template": version["text_template"],
                "decision": version.get("decision"),
                "tools": version.get("tools"),
            }

    for model in models_by_id.values():
        provider = model["provider"]
        model_name = model["model_id"]
        models[f"{provider}:{model_name}"] = {
            "provider": provider,
            "model": model_name,
            "display_name": model["display_name"],
            "metadata": model["metadata"],
            "provider_options": model["provider_options"],
            "capabilities": model["capabilities"],
            "status": model["status"],
        }

    return {
        "schema_version": 1,
        "project": legacy["project"],
        "environment": legacy["environment"],
        "prompts": {
            key: {
                "kind": use_case["kind"],
                "variables": use_case["variables"],
                "default_params": use_case["default_params"],
                "payload_policy": use_case["payload_policy"],
            }
            for key, use_case in legacy["prompts"].items()
        },
        "deployments": deployments,
        "prompt_versions": prompt_versions,
        "models": models,
    }
