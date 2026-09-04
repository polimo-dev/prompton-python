"""The ``POST /resolve`` client: same caching rules, same "serve the last answer" behaviour."""

from __future__ import annotations

import pytest

from prompton.config import Config
from prompton.errors import APIError
from prompton.resolve_client import ResolveClient

from .conftest import FakeTransport, json_response, transport_error

ANSWER = {
    "use_case": "greeting",
    "kind": "chat",
    "deployment": {"id": "d1", "revision": 1},
    "prompt": "default",
    "prompts": ["default", "ko"],
    "model": "openai/gpt-4o-mini",
    "model_id": "m1",
    "provider": "openrouter",
    "effective_params": {"temperature": 0.2},
    "effective_provider_options": {"only": ["OpenAI"]},
    "prompt_version": {"id": "v1", "number": 1},
    "messages": [
        {"role": "system", "content": "You are a greeter."},
        {"role": "user", "content": "Say hello to {{ name }}."},
    ],
    "warnings": [],
    "etag": "sha256-abc",
}


def build(transport, **options) -> ResolveClient:
    settings = {
        "api_key": "ptn_demo_key",
        "host": "http://localhost:4000",
        "disk_cache": False,
        "cache_ttl": 10.0,
    }
    settings.update(options)
    return ResolveClient(Config.build(**settings), transport)


def test_the_raw_answer_is_fetched_once_and_rendered_locally():
    transport = FakeTransport(lambda call: json_response(200, ANSWER))
    client = build(transport)

    first = client.resolve("greeting", variables={"name": "Ada"})
    second = client.resolve("greeting", variables={"name": "Bob"})

    assert len(transport.requests) == 1, "the cached raw answer is rendered locally"
    assert first.messages[1]["content"] == "Say hello to Ada."
    assert second.messages[1]["content"] == "Say hello to Bob."
    assert transport.requests[0]["body"] == {"use_case": "greeting", "environment": "production"}


def test_the_cache_is_keyed_by_use_case_prompt_and_environment():
    transport = FakeTransport(lambda call: json_response(200, ANSWER))
    client = build(transport)
    client.resolve("greeting")
    client.resolve("greeting", prompt="ko")
    client.resolve("greeting", environment="staging")
    client.resolve("greeting")
    assert len(transport.requests) == 3


def test_a_rate_limit_or_outage_serves_the_cached_answer():
    answers = [
        json_response(200, ANSWER),
        json_response(429, {"error": {"code": "rate_limited"}}, **{"retry-after": "60"}),
    ]
    transport = FakeTransport(lambda call: answers.pop(0) if answers else transport_error())
    client = build(transport, cache_ttl=0.0)

    client.resolve("greeting")
    assert (
        client.resolve("greeting", variables={"name": "Ada"}).messages[1]["content"]
        == "Say hello to Ada."
    )
    # the retry-after pause means no further request is attempted
    before = len(transport.requests)
    client.resolve("greeting")
    assert len(transport.requests) == before


def test_without_a_cached_answer_the_error_reaches_the_caller():
    transport = FakeTransport(
        lambda call: json_response(
            404,
            {
                "error": {
                    "code": "not_found",
                    "message": "no live deployment",
                    "details": {"reason": "unresolved"},
                }
            },
        )
    )
    client = build(transport)
    with pytest.raises(APIError) as error:
        client.resolve("greeting")
    assert error.value.status == 404
    assert error.value.details["reason"] == "unresolved"


def test_a_text_use_case_renders_the_text_field():
    answer = dict(ANSWER)
    answer.pop("messages")
    answer["kind"] = "text"
    answer["text"] = "Summarize:\n{% for i in items %}- {{ i }}\n{% endfor %}"
    transport = FakeTransport(lambda call: json_response(200, answer))
    client = build(transport)
    resolved = client.resolve("summarize", variables={"items": ["a", "b"]})
    assert resolved.text == "Summarize:\n- a\n- b\n"
