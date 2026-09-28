import json
from pathlib import Path

from prompton import template
from prompton.resolver import resolve
from prompton.snapshot_data import UseCaseDocument


def test_schema7_tools_and_message_slots_preserve_native_messages() -> None:
    doc = UseCaseDocument.from_mapping(
        {
            "schema_version": 7,
            "prompts": {"chat": {"kind": "chat", "default_params": {}}},
            "deployments": {
                "chat": {
                    "prompt_key": "chat",
                    "model_id": "m",
                    "api": "chat_completions",
                    "request_path": "/api/v1/chat/completions",
                    "template_pins": {"default": "v"},
                }
            },
            "prompt_versions": {
                "v": {
                    "id": "v",
                    "kind": "chat",
                    "engine": "liquid",
                    "messages": [
                        {"role": "system", "content": "Hi {{ name }}"},
                        {"type": "slot", "name": "history"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": [{"type": "text", "text": "ok"}],
                        },
                    ],
                    "tools": {
                        "definitions": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "parameters": {"type": "object"},
                                    "output_schema": {"type": "object"},
                                },
                            }
                        ],
                        "tool_choice": "auto",
                        "parallel_tool_calls": True,
                    },
                }
            },
            "models": {
                "m": {
                    "id": "m",
                    "provider": "openrouter",
                    "model_id": "openai/gpt-4o-mini",
                }
            },
        }
    )

    resolution = resolve(doc, "chat")

    assert resolution.api == "chat_completions"
    assert resolution.tools["tool_choice"] == "auto"

    messages = template.render_messages(
        resolution.messages,
        {"name": "Ada", "history": [{"role": "user", "content": "past", "tool_call_id": "keep"}]},
    )

    assert messages[0] == {"role": "system", "content": "Hi Ada"}
    assert messages[1] == {"role": "user", "content": "past", "tool_call_id": "keep"}
    assert messages[2]["content"] is None
    assert messages[2]["tool_calls"][0]["id"] == "call_1"
    assert messages[3]["content"] == [{"type": "text", "text": "ok"}]


def test_preview_http_contract_preserves_native_history_and_tools() -> None:
    fixture = json.loads((Path(__file__).parent / "conformance" / "http_contract.json").read_text())
    doc = UseCaseDocument.from_mapping(fixture["snapshot"])
    resolution = resolve(doc, fixture["render"]["key"])

    messages = template.render_messages(
        resolution.messages,
        {
            "locale": "ko-KR",
            "topic": "park walks",
            "history": fixture["render"]["request"]["body"]["messages"][1:4],
        },
    )

    assert messages == fixture["render"]["request"]["body"]["messages"]
    assert resolution.tools["tool_choice"] == "auto"
    assert resolution.tools["parallel_tool_calls"] is False
    definition = resolution.tools["definitions"][0]
    assert definition["output_schema"]["type"] == "object"
    assert definition["type"] == "function"
