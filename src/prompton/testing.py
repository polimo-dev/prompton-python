"""Helpers for testing an app that uses PromptOn, without a server and without HTTP.

Run your suite with ``mode="test"``: nothing is sent, and every monitoring log the app produced is
kept on ``client.captured`` for you to assert on. Build a schema v1 prompt document for the prompts
under test with :func:`make_use_case_document` and hand it to ``client.load_use_cases``.

    client = PromptOn(mode="test", api_key=None)
    client.load_use_cases(
        make_use_case_document(greeting={"messages": [...], "model": "openai/gpt-4o-mini"})
    )

    use_case = client.use_case("greeting")
    ...
    assert client.captured[0]["status"] == "ok"
"""

from __future__ import annotations

from typing import Any

__all__ = ["make_use_case_document"]

_DEFAULT_MODEL = "openai/gpt-4o-mini"


def make_use_case_document(
    *,
    project: str = "test",
    environment: str = "production",
    **use_cases: dict[str, Any],
) -> dict[str, Any]:
    """Build a schema v1 prompt document for the given use cases.

    Each keyword is a use case key; its value describes the pin::

        make_use_case_document(
            greeting={
                "kind": "chat",                       # chat (default) | text | embedding
                "model": "openai/gpt-4o-mini",
                "provider": "openrouter",
                "messages": [{"role": "user", "content": "Say hello to {{ name }}."}],
                "prompts": {"terse": [{"role": "user", "content": "Hi {{ name }}."}]},
                "params": {"temperature": 0.2},
                "default_params": {"max_tokens": 256},
                "provider_options": {"only": ["OpenAI"]},
                "payload_policy": {"mode": "full", "sample_rate": 1.0, "max_bytes": 262144},
                "revision": "v2026.09.30-1",
            },
        )
    """
    document: dict[str, Any] = {
        "schema_version": 1,
        "project": project,
        "environment": environment,
        "prompts": {},
        "deployments": {},
        "prompt_versions": {},
        "models": {},
    }

    for _index, (key, spec) in enumerate(use_cases.items(), start=1):
        spec = dict(spec)
        kind = spec.get("kind", "chat")
        provider = spec.get("provider", "openrouter")
        model_name = spec.get("model", _DEFAULT_MODEL)
        model_key = f"{provider}:{model_name}"
        document["models"][model_key] = {
            "provider": provider,
            "model": model_name,
            "display_name": model_name,
            "metadata": {},
            "provider_options": spec.get("model_provider_options", {}),
            "capabilities": [],
            "status": "active",
        }
        document["prompts"][key] = {
            "kind": kind,
            "variables": spec.get("variables", []),
            "default_params": spec.get("default_params", {}),
            "payload_policy": spec.get("payload_policy"),
        }

        pins: dict[str, int] = {}
        if kind != "embedding":
            variants: dict[str, Any] = {}
            if "messages" in spec or "text" in spec:
                variants["default"] = spec.get("messages", spec.get("text"))
            variants.update(spec.get("prompts", {}))
            if not variants:
                variants["default"] = [] if kind == "chat" else ""
            for order, (name, body) in enumerate(variants.items(), start=1):
                version_key = f"{key}@{order}"
                document["prompt_versions"][version_key] = {
                    "prompt": key,
                    "number": order,
                    "kind": kind,
                    "engine": spec.get("engine", "liquid"),
                    "messages": body if kind == "chat" else None,
                    "text_template": body if kind == "text" else None,
                }
                pins[name] = order

        document["deployments"][key] = {
            "revision": spec.get("revision", "v2026.09.30-1"),
            "provider": provider,
            "model": model_name,
            "params": spec.get("params", {}),
            "provider_options": spec.get("provider_options", {}),
            "prompt_pins": pins,
            "prompt_version": pins.get("default"),
        }

    return document
