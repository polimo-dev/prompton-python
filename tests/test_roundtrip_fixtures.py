from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from prompton.resolver import resolve
from prompton.snapshot_data import InvalidUseCaseDocumentError, UseCaseDocument
from prompton.template import render, render_messages

ROUNDTRIP = Path(__file__).with_name("fixtures") / "schema1"


def fixture(name: str) -> dict[str, Any]:
    return json.loads((ROUNDTRIP / name).read_text())


def render_value(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return render(value, variables)
    if isinstance(value, list):
        return [render_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, variables) for key, item in value.items()}
    return value


def test_real_schema1_index_is_rejected_as_configuration() -> None:
    with pytest.raises(InvalidUseCaseDocumentError, match="index responses are discovery only"):
        UseCaseDocument.from_mapping(fixture("index.json"))


def test_real_chat_detail_decodes_and_renders_like_the_server() -> None:
    detail = fixture("chat-detail.json")
    variables = fixture("chat-variables.json")
    rendered = fixture("chat-render.json")
    snapshot = UseCaseDocument.from_mapping(detail)
    assert {
        "name": "question",
        "type": "string",
        "required": True,
        "description": "Question to answer",
        "example": "refund",
    } in [variable.__dict__ for variable in snapshot.use_cases["support_reply"].variables]
    resolution = resolve(snapshot, rendered["key"])

    assert (
        render_messages(resolution.messages, variables, resolution.engine) == rendered["messages"]
    )
    assert resolution.deployment == {"id": None, "revision": rendered["deployment"]["revision"]}
    assert resolution.model == rendered["model"]
    assert resolution.prompt_version == {"id": None, "number": detail["prompt_version"]["number"]}
    assert resolution.request_path == detail["deployment"]["request_path"]


def test_real_decision_detail_decodes_and_renders_like_the_server() -> None:
    detail = fixture("decision-detail.json")
    variables = fixture("decision-variables.json")
    rendered = fixture("decision-render.json")
    snapshot = UseCaseDocument.from_mapping(detail)
    assert {
        "name": "ticket",
        "type": "string",
        "required": True,
        "description": "Ticket summary",
        "example": "refund",
    } in [variable.__dict__ for variable in snapshot.use_cases["decision_triage"].variables]
    resolution = resolve(snapshot, rendered["key"])

    assert resolution.kind == "decision"
    assert render_value(resolution.decision, variables) == rendered["decision"]
    assert resolution.deployment == {"id": None, "revision": rendered["deployment"]["revision"]}
    assert resolution.model == rendered["model"]
    assert resolution.request_path == detail["deployment"]["request_path"]
