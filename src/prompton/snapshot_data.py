"""Decode schema v1 prompt detail and cached aggregate prompt documents.

A deployment revision is a **pin, not a router**: one model plus one pinned prompt version per
prompt name. Lightweight prompt indexes are discovery-only and are rejected here. Older, newer and
legacy schema shapes are refused: this SDK reads exactly ``schema_version: 1``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import PromptOnError
from .params import stringify_keys

__all__ = [
    "SCHEMA_VERSION",
    "Deployment",
    "InvalidPromptDocumentError",
    "Model",
    "Prompt",
    "PromptDocument",
    "PromptVersion",
    "UnsupportedSchemaVersionError",
]

SCHEMA_VERSION = 1

_KINDS = ("chat", "decision", "text", "embedding")
_ENGINES = ("liquid", "raw")
_PAYLOAD_MODES = ("full", "hash", "none")
_VARIABLE_TYPES = ("string", "number", "boolean", "list", "map")

DEFAULT_MAX_BYTES = 262_144


class InvalidPromptDocumentError(PromptOnError):
    """The document is not a prompt document the SDK can read."""


class UnsupportedSchemaVersionError(InvalidPromptDocumentError):
    """The document announces a schema version this SDK does not read."""

    def __init__(self, version: int) -> None:
        super().__init__(f"unsupported schema_version {version}; this SDK reads schema_version 1")
        self.version = version


@dataclass(frozen=True)
class VariableDefinition:
    """One variable definition declared by a prompt."""

    name: str | None
    type: str = "string"
    required: bool = False
    description: str | None = None
    example: Any = None


@dataclass(frozen=True)
class PayloadPolicy:
    """How much of a prompt and completion the monitoring log may carry."""

    mode: str = "full"
    sample_rate: float = 1.0
    max_bytes: int = DEFAULT_MAX_BYTES
    retention_days: int | None = None
    encrypt: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "sample_rate": self.sample_rate,
            "max_bytes": self.max_bytes,
            "retention_days": self.retention_days,
            "encrypt": self.encrypt,
        }


@dataclass(frozen=True)
class Deployment:
    """The live pin for one prompt in one environment."""

    id: str | None
    prompt_key: str
    revision: str | None = None
    model_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    template_pins: dict[str, str] = field(default_factory=dict)
    api: str | None = None
    request_path: str | None = None


@dataclass(frozen=True)
class Prompt:
    """One LLM call site, as the control plane describes it."""

    id: str | None
    key: str
    kind: str = "chat"
    variables: tuple[VariableDefinition, ...] = ()
    default_params: dict[str, Any] = field(default_factory=dict)
    payload_policy: PayloadPolicy | None = None


@dataclass(frozen=True)
class PromptVersion:
    """An immutable prompt version: chat messages or a single text template."""

    id: str | None
    lookup_key: str | None = None
    prompt_id: str | None = None
    number: int | None = None
    engine: str = "liquid"
    messages: tuple[dict[str, Any], ...] | None = None
    text_template: str | None = None
    kind: str | None = None
    decision: dict[str, Any] | None = None
    tools: dict[str, Any] | None = None


@dataclass(frozen=True)
class Model:
    """A catalog model. ``model_id`` is the provider-side string your app sends to the provider."""

    id: str | None
    lookup_key: str | None = None
    provider: str | None = None
    model_id: str | None = None
    display_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    pricing: Any = None
    context_length: int | None = None
    status: str | None = None


@dataclass(frozen=True)
class PromptDocument:
    """A decoded prompt document."""

    schema_version: int = SCHEMA_VERSION
    project: str | None = None
    environment: str | None = None
    prompts: dict[str, Prompt] = field(default_factory=dict)
    deployments: dict[str, Deployment] = field(default_factory=dict)
    prompt_versions: dict[str, PromptVersion] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def deployment_for(self, prompt_key: str) -> Deployment | None:
        return self.deployments.get(prompt_key)

    @classmethod
    def from_json(cls, raw: bytes | str) -> PromptDocument:
        """Decode a JSON document. Raises :class:`InvalidPromptDocumentError`."""
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise InvalidPromptDocumentError(
                f"prompt document is not valid JSON: {error}"
            ) from error
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(cls, document: Any) -> PromptDocument:
        """Decode an already-parsed document."""
        if isinstance(document, PromptDocument):
            return document
        if not isinstance(document, Mapping):
            raise InvalidPromptDocumentError("prompt document must be an object")

        warnings: list[str] = []
        version = _schema_version(document)

        normalized = _normalize_prompt_document(document)

        for collection in ("prompts", "deployments", "prompt_versions", "models"):
            if not isinstance(normalized.get(collection), Mapping):
                raise InvalidPromptDocumentError(
                    f"{collection} is required and must be a keyed object"
                )

        raw_prompts = normalized.get("prompts")

        prompts = {
            str(key): _decode_prompt(str(key), value, warnings)
            for key, value in raw_prompts.items()
            if isinstance(value, Mapping)
        }
        deployments = _decode_deployments(normalized.get("deployments"), warnings)
        prompt_versions = _decode_prompt_versions(normalized.get("prompt_versions"), warnings)
        models = _decode_models(normalized.get("models"), warnings)

        return cls(
            schema_version=version,
            project=_as_str(normalized.get("project")),
            environment=_as_str(normalized.get("environment")),
            prompts=prompts,
            deployments=deployments,
            prompt_versions=prompt_versions,
            models=models,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------


def _normalize_prompt_document(document: Mapping[str, Any]) -> dict[str, Any] | Mapping[str, Any]:
    if isinstance(document.get("prompts"), list):
        raise InvalidPromptDocumentError(
            "prompt index responses are discovery only and cannot be loaded as configuration"
        )
    has_single = any(k in document for k in ("prompt", "deployment", "prompt_version", "model"))
    if not has_single:
        return document

    for plural in ("prompts", "deployments", "prompt_versions", "models"):
        if plural in document:
            raise InvalidPromptDocumentError(
                f"schema_version 1 prompt detail must use singular resources; found {plural}"
            )
    prompt = document.get("prompt")
    deployment = document.get("deployment")
    version = document.get("prompt_version")
    model = document.get("model")
    if not isinstance(prompt, Mapping):
        raise InvalidPromptDocumentError("schema v1 prompt document prompt is required")
    if not isinstance(deployment, Mapping):
        raise InvalidPromptDocumentError("schema v1 prompt document deployment is required")
    if not isinstance(version, Mapping):
        raise InvalidPromptDocumentError("schema v1 prompt document prompt_version is required")
    if not isinstance(model, Mapping):
        raise InvalidPromptDocumentError("schema v1 prompt document model is required")

    key = _strict_str(prompt.get("key"))
    if key is None or key.strip() == "":
        raise InvalidPromptDocumentError("schema v1 prompt.key is required")
    version_prompt = _strict_str(version.get("prompt"))
    if version_prompt != key:
        raise InvalidPromptDocumentError("schema v1 prompt_version.prompt must match prompt.key")
    number = _strict_positive_int(version.get("number"))
    if number is None:
        raise InvalidPromptDocumentError(
            "schema v1 prompt_version.number must be a positive integer"
        )
    if deployment.get("prompt_version") != number:
        raise InvalidPromptDocumentError(
            "schema v1 deployment.prompt_version must match prompt_version.number"
        )
    provider = _strict_str(model.get("provider"))
    model_name = _strict_str(model.get("model"))
    if provider is None or model_name is None:
        raise InvalidPromptDocumentError("schema v1 model provider and model are required")
    if deployment.get("provider") != provider or deployment.get("model") != model_name:
        raise InvalidPromptDocumentError("schema v1 deployment provider/model must match model")

    prompt_body = dict(prompt)
    prompt_body.pop("key", None)
    return {
        "schema_version": 1,
        "project": document.get("project"),
        "environment": document.get("environment"),
        "prompts": {key: prompt_body},
        "deployments": {key: deployment},
        "prompt_versions": {f"{key}@{number}": version},
        "models": {f"{provider}:{model_name}": model},
    }


def _schema_version(document: Mapping[str, Any]) -> int:
    raw = document.get("schema_version")
    if raw is None:
        raise InvalidPromptDocumentError("schema_version is required")
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise InvalidPromptDocumentError(
            f"schema_version must be integer {SCHEMA_VERSION}, got {raw!r}"
        )
    if raw == SCHEMA_VERSION:
        return raw
    raise UnsupportedSchemaVersionError(raw)


def _as_str(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return None


def _strict_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_revision(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise InvalidPromptDocumentError(
        f"deployment {key} revision must be a string like v2026.09.30-1"
    )


def _as_int(value: Any, default: int | None = None) -> int | None:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _strict_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value > 0 else None


def _model_key(provider: Any, model: Any) -> str | None:
    provider_text = _strict_str(provider)
    model_text = _strict_str(model)
    if provider_text is None or model_text is None:
        return None
    return f"{provider_text}:{model_text}"


def _enum(value: Any, allowed: tuple[str, ...], default: str, tag: str, warnings: list[str]) -> str:
    if value is None:
        return default
    text = _as_str(value)
    if text is None:
        warnings.append(f"{tag}: {value!r}")
        return default
    if text not in allowed:
        warnings.append(f"{tag}: {text}")
    return text


def _decode_prompt(key: str, raw: Mapping[str, Any], warnings: list[str]) -> Prompt:
    return Prompt(
        id=None,
        key=key,
        kind=_enum(raw.get("kind"), _KINDS, "chat", "unknown_kind", warnings),
        variables=tuple(_decode_variables(raw, warnings)),
        default_params=stringify_keys(raw.get("default_params")),
        payload_policy=_decode_payload_policy(raw.get("payload_policy"), warnings),
    )


def _decode_variables(
    raw_prompt: Mapping[str, Any], warnings: list[str]
) -> list[VariableDefinition]:
    if "input_schema" in raw_prompt:
        raise InvalidPromptDocumentError("prompt variables must use variables, not input_schema")
    raw = raw_prompt.get("variables")
    if not isinstance(raw, list):
        return []
    variables = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            warnings.append(f"invalid_variable: {entry!r}")
            continue
        variables.append(
            VariableDefinition(
                name=_as_str(entry.get("name")),
                type=_enum(
                    entry.get("type"), _VARIABLE_TYPES, "string", "unknown_variable_type", warnings
                ),
                required=entry.get("required") is True,
                description=_as_str(entry.get("description")),
                example=entry.get("example"),
            )
        )
    return variables


def _decode_payload_policy(raw: Any, warnings: list[str]) -> PayloadPolicy | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        warnings.append(f"invalid_payload_policy: {raw!r}")
        return None
    sample_rate = raw.get("sample_rate")
    return PayloadPolicy(
        mode=_enum(raw.get("mode"), _PAYLOAD_MODES, "full", "unknown_payload_mode", warnings),
        sample_rate=float(sample_rate)
        if isinstance(sample_rate, (int, float)) and not isinstance(sample_rate, bool)
        else 1.0,
        max_bytes=_as_int(raw.get("max_bytes"), DEFAULT_MAX_BYTES) or DEFAULT_MAX_BYTES,
        retention_days=_as_int(raw.get("retention_days")),
        encrypt=raw.get("encrypt") is True,
    )


def _decode_deployments(raw: Any, warnings: list[str]) -> dict[str, Deployment]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        warnings.append(f"invalid_deployments: {raw!r}")
        return {}
    deployments: dict[str, Deployment] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            warnings.append(f"invalid_deployment: {key}")
            continue
        pins: dict[str, str] = {}
        raw_pins = value.get("prompt_pins")
        if isinstance(raw_pins, Mapping):
            for name, raw_number in raw_pins.items():
                name_str = _as_str(name)
                pin_number = _strict_positive_int(raw_number)
                if name_str and pin_number is not None:
                    pins[name_str] = f"{key}@{pin_number}"
                else:
                    warnings.append(f"invalid_prompt_pin: {key}/{name}")
        else:
            number = _strict_positive_int(value.get("prompt_version"))
            if number is not None:
                pins["default"] = f"{key}@{number}"
            elif value.get("prompt_version") is not None:
                warnings.append(f"invalid_prompt_pin: {key}/default")
        model_id = _model_key(value.get("provider"), value.get("model"))
        if model_id is None:
            warnings.append(f"invalid_model_ref: {key}")
        deployments[str(key)] = Deployment(
            id=None,
            prompt_key=(
                _as_str(value.get("prompt_key")) or _as_str(value.get("prompt_key")) or str(key)
            ),
            revision=_as_revision(value.get("revision"), str(key)),
            model_id=f"invalid:{key}" if model_id is None else model_id,
            params=stringify_keys(value.get("params")),
            provider_options=stringify_keys(value.get("provider_options")),
            template_pins=pins,
            api=_as_str(value.get("api")),
            request_path=_as_str(value.get("request_path")),
        )
    return deployments


def _decode_prompt_versions(raw: Any, warnings: list[str]) -> dict[str, PromptVersion]:
    versions: dict[str, PromptVersion] = {}
    for entry_id, entry in _iter_by_id(raw, warnings):
        messages = _decode_messages(entry.get("messages"), warnings)
        prompt_key = _strict_str(entry.get("prompt"))
        number = _strict_positive_int(entry.get("number"))
        readable_key = f"{prompt_key}@{number}" if prompt_key and number is not None else None
        if (prompt_key is not None or "prompt" in entry) and readable_key is None:
            warnings.append(f"invalid_prompt_version: {entry_id}")
            continue
        if readable_key is not None and readable_key != entry_id:
            warnings.append(f"invalid_prompt_version: {entry_id}")
            continue
        if readable_key is None:
            warnings.append(f"invalid_prompt_version: {entry_id}")
            continue
        if readable_key in versions:
            warnings.append(f"invalid_entry: duplicate:{readable_key}")
            continue
        versions[readable_key] = PromptVersion(
            id=None,
            lookup_key=readable_key,
            prompt_id=prompt_key,
            number=number,
            engine=_enum(entry.get("engine"), _ENGINES, "liquid", "unknown_engine", warnings),
            messages=messages,
            text_template=_as_str(entry.get("text_template")),
            kind=_as_str(entry.get("kind")),
            decision=(
                stringify_keys(entry.get("decision"))
                if isinstance(entry.get("decision"), Mapping)
                else None
            ),
            tools=_decode_tools(entry.get("tools"), warnings),
        )
    return versions


def _decode_messages(raw: Any, warnings: list[str]) -> tuple[dict[str, Any], ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        warnings.append(f"invalid_messages: {raw!r}")
        return None
    messages: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            warnings.append(f"invalid_message: {entry!r}")
            continue
        messages.append(dict(entry))
    return tuple(messages)


def _decode_tools(raw: Any, warnings: list[str]) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        warnings.append(f"invalid_tools: {raw!r}")
        return None
    definitions = raw.get("definitions")
    if not isinstance(definitions, list):
        warnings.append("invalid_tools: definitions")
        return None
    tools = {"definitions": [dict(item) for item in definitions if isinstance(item, Mapping)]}
    if "tool_choice" in raw:
        tools["tool_choice"] = raw.get("tool_choice")
    if isinstance(raw.get("parallel_tool_calls"), bool):
        tools["parallel_tool_calls"] = raw.get("parallel_tool_calls")
    return tools


def _decode_models(raw: Any, warnings: list[str]) -> dict[str, Model]:
    models: dict[str, Model] = {}
    for entry_id, entry in _iter_by_id(raw, warnings):
        capabilities = entry.get("capabilities")
        if not isinstance(capabilities, list):
            capabilities = [] if capabilities is None else [capabilities]
        provider = _strict_str(entry.get("provider"))
        model = _strict_str(entry.get("model"))
        readable_key = f"{provider}:{model}" if provider and model else None
        if readable_key is None or readable_key != entry_id:
            warnings.append(f"invalid_entry: {entry_id}")
            continue
        if readable_key in models:
            warnings.append(f"invalid_entry: duplicate:{readable_key}")
            continue
        models[readable_key] = Model(
            id=None,
            lookup_key=readable_key,
            provider=provider,
            model_id=model,
            display_name=_as_str(entry.get("display_name")),
            metadata=stringify_keys(entry.get("metadata")),
            provider_options=stringify_keys(entry.get("provider_options")),
            capabilities=tuple(c for c in (_as_str(x) for x in capabilities) if c is not None),
            pricing=entry.get("pricing"),
            context_length=_as_int(entry.get("context_length")),
            status=_as_str(entry.get("status")),
        )
    return models


def _iter_by_id(raw: Any, warnings: list[str]):
    """Accept both the ``{id: entry}`` map and the ``[entry]`` list shape."""
    if raw is None:
        return
    if isinstance(raw, Mapping):
        for key, entry in raw.items():
            if isinstance(entry, Mapping):
                yield str(key), entry
            else:
                warnings.append(f"invalid_entry: {key}")
        return
    warnings.append(f"invalid_collection: {raw!r}")
