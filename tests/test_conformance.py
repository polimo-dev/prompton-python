"""The cross-language conformance suite, replayed through this SDK.

``tests/conformance/*.json`` is copied verbatim from the reference implementation. When two SDKs
disagree about how a prompt renders, which model a snapshot resolves to, or how a monitoring log is
truncated, an app that talks to PromptOn from two languages gets two different answers. These cases
are what prevents that, so every one of them runs.
"""

from __future__ import annotations

from typing import Any

import pytest

from prompton import payload, resolver, stop_kind, template
from prompton.errors import (
    MissingVariableError,
    RenderError,
    TemplateSyntaxError,
    UnknownPromptError,
    UnknownUseCaseError,
    UnresolvedError,
)
from prompton.generation import CallMeta, Outcome, build_record
from prompton.snapshot_data import SnapshotData

from .conftest import load_conformance

TEMPLATE = load_conformance("template")
RESOLVE = load_conformance("resolve")
TRUNCATION = load_conformance("truncation")
STOP_KIND = load_conformance("stop_kind")
GENERATION_RECORD = load_conformance("generation_record")


def _render_outcome(case: dict[str, Any]) -> dict[str, Any]:
    try:
        return {"output": template.render(case["template"], case["variables"], case["engine"])}
    except MissingVariableError as error:
        return {"error": "missing_variable", "variable": error.variable}
    except TemplateSyntaxError:
        return {"error": "parse_error"}
    except RenderError:
        return {"error": "render_error"}


@pytest.mark.parametrize("case", TEMPLATE["cases"], ids=lambda c: c["name"])
def test_template_render(case: dict[str, Any]) -> None:
    got = _render_outcome(case)
    if case.get("normative", True):
        assert got == case["expect"], case.get("note", "")
    else:
        # Reference behaviour this SDK need not reproduce; the case still must not crash.
        assert got


@pytest.mark.parametrize("case", TEMPLATE["lint_cases"], ids=lambda c: c["name"])
def test_template_lint(case: dict[str, Any]) -> None:
    reasons = [reason.as_dict() for reason in template.lint(case["template"])]
    if case["expect"]["lint"] == "ok":
        assert reasons == []
    else:
        assert reasons == case["expect"]["reasons"]


@pytest.mark.parametrize("case", TEMPLATE["variables_cases"], ids=lambda c: c["name"])
def test_template_variables(case: dict[str, Any]) -> None:
    assert template.variables_of(case["template"]) == case["expect"]["variables"]


def test_template_allowed_sets_match_the_contract() -> None:
    assert sorted(template.ALLOWED_TAGS) == sorted(TEMPLATE["allowed_tags"])
    assert sorted(template.ALLOWED_FILTERS) == sorted(TEMPLATE["allowed_filters"])


SNAPSHOTS = {name: SnapshotData.from_mapping(doc) for name, doc in RESOLVE["snapshots"].items()}


def _resolve_outcome(case: dict[str, Any]) -> dict[str, Any]:
    snapshot = SNAPSHOTS[case["snapshot_ref"]]
    try:
        resolution = resolver.resolve(snapshot, case["use_case"], case.get("prompt"))
    except UnknownUseCaseError:
        return {"error": "unknown_use_case"}
    except UnresolvedError:
        return {"error": "unresolved"}
    except UnknownPromptError as error:
        return {
            "error": "unknown_prompt",
            "prompt": error.prompt,
            "available_prompts": error.available_prompts,
        }

    got: dict[str, Any] = {
        "kind": resolution.kind,
        "deployment_id": resolution.deployment_id,
        "revision": resolution.deployment_revision,
        "model": resolution.model,
        "model_id": resolution.model_id,
        "provider": resolution.provider,
        "prompt": resolution.prompt,
        "prompts": list(resolution.available_prompts),
        "prompt_version": resolution.prompt_version,
        "effective_params": resolution.effective_params,
        "effective_provider_options": resolution.effective_provider_options,
        "warnings": list(resolution.warnings),
    }
    variables = case.get("variables")
    try:
        if resolution.messages is not None:
            got["messages"] = (
                template.render_messages(resolution.messages, variables, resolution.engine)
                if variables is not None
                else [dict(message) for message in resolution.messages]
            )
        if resolution.text_template is not None:
            got["text"] = (
                template.render(resolution.text_template, variables, resolution.engine)
                if variables is not None
                else resolution.text_template
            )
    except MissingVariableError as error:
        return {"error": "missing_variable", "variable": error.variable}
    return got


@pytest.mark.parametrize("case", RESOLVE["cases"], ids=lambda c: c["name"])
def test_resolve(case: dict[str, Any]) -> None:
    assert _resolve_outcome(case) == case["expect"], case.get("note", "")


@pytest.mark.parametrize("case", TRUNCATION["cases"], ids=lambda c: c["name"])
def test_truncation(case: dict[str, Any]) -> None:
    config = case.get("config") or {}
    got = payload.apply_policy(
        case["generation"],
        case["policy"],
        hash_end_user=bool(config.get("hash_end_user")),
    )
    assert got == case["expect"]["generation"], case.get("note", "")


@pytest.mark.parametrize("case", TRUNCATION["sampling"]["buckets"], ids=lambda c: repr(c["id"]))
def test_sampling_buckets(case: dict[str, Any]) -> None:
    assert payload.bucket(case["id"]) == case["bucket"]


@pytest.mark.parametrize("case", STOP_KIND["cases"], ids=lambda c: repr(c["finish_reason"]))
def test_stop_kind(case: dict[str, Any]) -> None:
    assert stop_kind.normalize(case["finish_reason"]) == case["stop_kind"], case["source"]
    assert stop_kind.truncated(case["finish_reason"]) is case["truncated"]


def test_stop_kind_values_match_the_contract() -> None:
    assert list(stop_kind.STOP_KINDS) == STOP_KIND["stop_kinds"]


def _golden(name: str) -> dict[str, Any]:
    for entry in GENERATION_RECORD["records"]:
        if entry["name"] == name:
            return entry["record"]
    raise AssertionError(f"no golden record named {name}")


def test_generation_record_shape_matches_the_golden_chat_success() -> None:
    """Rebuild the golden ``chat/success`` record and compare it field for field."""
    golden = _golden("chat/success")
    resolution = resolver.Resolution(
        use_case="greeting",
        kind="chat",
        prompt="default",
        deployment_id=golden["deployment_id"],
        deployment_revision=golden["deployment_revision"],
        prompt_version_id=golden["prompt_version_id"],
        prompt_version_number=2,
        engine="liquid",
        model_id=None,
        model=golden["model"],
        provider=golden["provider"],
        effective_params=golden["params"],
        resolution_source="remote",
    )
    record = build_record(
        resolution,
        CallMeta(
            id=golden["id"],
            variables=golden["input"]["variables"],
            input_messages=golden["input"]["messages"],
            end_user_ref=golden["end_user_ref"],
            trace_id=golden["trace_id"],
            sequence=golden["sequence"],
            context=golden["context"],
            metadata={"job_id": 8842, "attempt": 1},
        ),
        status="ok",
        started_at=golden["started_at"],
        latency_ms=golden["latency_ms"],
        outcome=Outcome(
            content=golden["output"]["content"],
            finish_reason=golden["finish_reason"],
            input_tokens=golden["usage"]["input_tokens"],
            output_tokens=golden["usage"]["output_tokens"],
            usage_raw=golden["usage"]["raw"],
            cost_usd=golden["usage"]["cost_usd"],
            cost_source=golden["usage"]["cost_source"],
            is_byok=False,
            model_used=golden["model_used"],
            upstream_provider=golden["upstream_provider"],
        ),
    )
    # The SDK name is this package's, not the reference implementation's.
    assert record.pop("sdk") == {"name": "prompton-python", "version": "0.1.0"}
    expected = {key: value for key, value in golden.items() if key != "sdk"}
    assert record == expected


def test_generation_record_error_without_output() -> None:
    golden = _golden("chat/error_without_output")
    resolution = resolver.Resolution(
        use_case="greeting",
        kind="chat",
        prompt="default",
        deployment_id=golden["deployment_id"],
        deployment_revision=golden["deployment_revision"],
        prompt_version_id=golden["prompt_version_id"],
        prompt_version_number=2,
        engine="liquid",
        model_id=None,
        model=golden["model"],
        provider=golden["provider"],
        effective_params=golden["params"],
        resolution_source="remote",
    )
    record = build_record(
        resolution,
        CallMeta(
            id=golden["id"],
            variables=golden["input"]["variables"],
            input_messages=golden["input"]["messages"],
            trace_id=golden["trace_id"],
            sequence=golden["sequence"],
        ),
        status="error",
        started_at=golden["started_at"],
        latency_ms=golden["latency_ms"],
        error=golden["error"],
    )
    record.pop("sdk")
    assert record == {key: value for key, value in golden.items() if key != "sdk"}
    assert "output" not in record


def test_generation_record_required_fields_are_present() -> None:
    for entry in GENERATION_RECORD["records"]:
        for name in GENERATION_RECORD["field_rules"]["required"]:
            assert name in entry["record"], f"{entry['name']} is missing {name}"


def test_batch_envelope_shape() -> None:
    envelope = GENERATION_RECORD["batch_envelope"]["request"]
    assert list(envelope) == ["generations"]
    assert len(envelope["generations"]) == len(GENERATION_RECORD["records"])
