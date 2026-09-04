"""Helpers for testing an app that uses PromptOn, without a server and without HTTP.

Run your suite with ``mode="test"``: nothing is sent, and every monitoring log the app produced is
kept on ``client.captured`` for you to assert on. Build a snapshot for the use cases under test
with :func:`make_snapshot` and hand it to ``client.load_snapshot``.

    client = PromptOn(mode="test", api_key=None)
    client.load_snapshot(make_snapshot(greeting={"messages": [...], "model": "openai/gpt-4o-mini"}))

    resolution = client.resolve("greeting")
    ...
    assert client.captured[0]["status"] == "ok"
"""

from __future__ import annotations

from typing import Any

from .snapshot_data import SCHEMA_VERSION

__all__ = ["make_snapshot"]

_DEFAULT_MODEL = "openai/gpt-4o-mini"


def make_snapshot(
    *,
    project: str = "test",
    environment: str = "production",
    **use_cases: dict[str, Any],
) -> dict[str, Any]:
    """Build a schema-v3 snapshot document for the given use cases.

    Each keyword is a use case key; its value describes the pin::

        make_snapshot(
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
                "revision": 1,
            },
        )
    """
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project,
        "environment": environment,
        "use_cases": {},
        "deployments": {},
        "prompt_versions": {},
        "models": {},
    }

    for index, (key, spec) in enumerate(use_cases.items(), start=1):
        spec = dict(spec)
        kind = spec.get("kind", "chat")
        model_key = f"model-{index:04d}"
        document["models"][model_key] = {
            "id": model_key,
            "provider": spec.get("provider", "openrouter"),
            "model_id": spec.get("model", _DEFAULT_MODEL),
            "display_name": spec.get("model", _DEFAULT_MODEL),
            "metadata": {},
            "provider_options": spec.get("model_provider_options", {}),
            "capabilities": [],
            "status": "active",
        }
        document["use_cases"][key] = {
            "id": f"use-case-{index:04d}",
            "kind": kind,
            "input_schema": spec.get("input_schema", []),
            "default_params": spec.get("default_params", {}),
            "payload_policy": spec.get("payload_policy"),
        }

        pins: dict[str, str] = {}
        if kind != "embedding":
            variants: dict[str, Any] = {}
            if "messages" in spec or "text" in spec:
                variants["default"] = spec.get("messages", spec.get("text"))
            variants.update(spec.get("prompts", {}))
            if not variants:
                variants["default"] = [] if kind == "chat" else ""
            for order, (name, body) in enumerate(variants.items(), start=1):
                version_id = f"version-{index:04d}-{order:02d}"
                document["prompt_versions"][version_id] = {
                    "id": version_id,
                    "prompt_id": f"prompt-{index:04d}",
                    "number": order,
                    "engine": spec.get("engine", "liquid"),
                    "messages": body if kind == "chat" else None,
                    "text_template": body if kind == "text" else None,
                }
                pins[name] = version_id

        document["deployments"][key] = {
            "id": f"deployment-{index:04d}",
            "revision": spec.get("revision", 1),
            "model_id": model_key,
            "params": spec.get("params", {}),
            "provider_options": spec.get("provider_options", {}),
            "prompt_pins": pins,
        }

    return document
