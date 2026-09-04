"""Decoding ``GET /snapshot`` (schema v3) into the structures the resolver reads.

A deployment revision is a **pin, not a router**: one model plus one pinned prompt version per
prompt name. v1 and v2 documents - a stale disk cache, an old bundle - are refused, and the SDK
keeps polling for a v3 one. A schema version newer than 3 decodes only the known fields and leaves
a warning, because v1 only ever adds fields.
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
    "InvalidSnapshotError",
    "Model",
    "PromptVersion",
    "SnapshotData",
    "UnsupportedSchemaVersionError",
    "UseCase",
]

SCHEMA_VERSION = 3

_KINDS = ("chat", "text", "embedding")
_ENGINES = ("liquid", "raw")
_PAYLOAD_MODES = ("full", "hash", "none")
_VARIABLE_TYPES = ("string", "number", "boolean", "list", "map")

DEFAULT_MAX_BYTES = 262_144


class InvalidSnapshotError(PromptOnError):
    """The document is not a snapshot the SDK can read."""


class UnsupportedSchemaVersionError(InvalidSnapshotError):
    """The document announces a schema version older than the one this SDK reads."""

    def __init__(self, version: int) -> None:
        super().__init__(
            f"unsupported snapshot schema_version {version}; this SDK reads v{SCHEMA_VERSION}"
        )
        self.version = version


@dataclass(frozen=True)
class InputVariable:
    """One entry of a use case's ``input_schema``."""

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
    """The live pin for one use case in one environment."""

    id: str | None
    use_case_key: str
    revision: int | None = None
    model_id: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    provider_options: dict[str, Any] = field(default_factory=dict)
    prompt_pins: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class UseCase:
    """One LLM call site, as the control plane describes it."""

    id: str | None
    key: str
    kind: str = "chat"
    input_schema: tuple[InputVariable, ...] = ()
    default_params: dict[str, Any] = field(default_factory=dict)
    payload_policy: PayloadPolicy | None = None


@dataclass(frozen=True)
class PromptVersion:
    """An immutable prompt version: chat messages or a single text template."""

    id: str
    prompt_id: str | None = None
    number: int | None = None
    engine: str = "liquid"
    messages: tuple[dict[str, Any], ...] | None = None
    text_template: str | None = None


@dataclass(frozen=True)
class Model:
    """A catalog model. ``model_id`` is the provider-side string your app sends to the provider."""

    id: str
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
class SnapshotData:
    """A decoded snapshot document."""

    schema_version: int = SCHEMA_VERSION
    project: str | None = None
    environment: str | None = None
    use_cases: dict[str, UseCase] = field(default_factory=dict)
    deployments: dict[str, Deployment] = field(default_factory=dict)
    prompt_versions: dict[str, PromptVersion] = field(default_factory=dict)
    models: dict[str, Model] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def deployment_for(self, use_case_key: str) -> Deployment | None:
        return self.deployments.get(use_case_key)

    @classmethod
    def from_json(cls, raw: bytes | str) -> SnapshotData:
        """Decode a JSON document. Raises :class:`InvalidSnapshotError`."""
        try:
            document = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as error:
            raise InvalidSnapshotError(f"snapshot is not valid JSON: {error}") from error
        return cls.from_mapping(document)

    @classmethod
    def from_mapping(cls, document: Any) -> SnapshotData:
        """Decode an already-parsed document."""
        if isinstance(document, SnapshotData):
            return document
        if not isinstance(document, Mapping):
            raise InvalidSnapshotError("snapshot must be an object")

        warnings: list[str] = []
        version = _schema_version(document, warnings)

        raw_use_cases = document.get("use_cases")
        if not isinstance(raw_use_cases, Mapping):
            raise InvalidSnapshotError("use_cases is required and must be an object")

        use_cases = {
            str(key): _decode_use_case(str(key), value, warnings)
            for key, value in raw_use_cases.items()
            if isinstance(value, Mapping)
        }
        deployments = _decode_deployments(document.get("deployments"), warnings)
        prompt_versions = _decode_prompt_versions(document.get("prompt_versions"), warnings)
        models = _decode_models(document.get("models"), warnings)

        return cls(
            schema_version=version,
            project=_as_str(document.get("project")),
            environment=_as_str(document.get("environment")),
            use_cases=use_cases,
            deployments=deployments,
            prompt_versions=prompt_versions,
            models=models,
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------


def _schema_version(document: Mapping[str, Any], warnings: list[str]) -> int:
    raw = document.get("schema_version", document.get("version"))
    if raw is None:
        if isinstance(document.get("deployments"), Mapping):
            return SCHEMA_VERSION
        raise InvalidSnapshotError("schema_version is required")
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise InvalidSnapshotError(f"schema_version must be a positive integer, got {raw!r}")
    if raw == SCHEMA_VERSION:
        return raw
    if raw > SCHEMA_VERSION:
        warnings.append(f"unknown_schema_version: {raw}")
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


def _decode_use_case(key: str, raw: Mapping[str, Any], warnings: list[str]) -> UseCase:
    return UseCase(
        id=_as_str(raw.get("id")),
        key=key,
        kind=_enum(raw.get("kind"), _KINDS, "chat", "unknown_kind", warnings),
        input_schema=tuple(_decode_input_schema(raw.get("input_schema"), warnings)),
        default_params=stringify_keys(raw.get("default_params")),
        payload_policy=_decode_payload_policy(raw.get("payload_policy"), warnings),
    )


def _decode_input_schema(raw: Any, warnings: list[str]) -> list[InputVariable]:
    if not isinstance(raw, list):
        return []
    variables = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            warnings.append(f"invalid_variable: {entry!r}")
            continue
        variables.append(
            InputVariable(
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
            for name, version_id in raw_pins.items():
                name_str, id_str = _as_str(name), _as_str(version_id)
                if name_str and id_str:
                    pins[name_str] = id_str
                else:
                    warnings.append(f"invalid_prompt_pin: {key}/{name}")
        elif raw_pins is not None:
            warnings.append(f"invalid_prompt_pins: {key}")
        deployments[str(key)] = Deployment(
            id=_as_str(value.get("id")),
            use_case_key=_as_str(value.get("use_case_key")) or str(key),
            revision=_as_int(value.get("revision")),
            model_id=_as_str(value.get("model_id")),
            params=stringify_keys(value.get("params")),
            provider_options=stringify_keys(value.get("provider_options")),
            prompt_pins=pins,
        )
    return deployments


def _decode_prompt_versions(raw: Any, warnings: list[str]) -> dict[str, PromptVersion]:
    versions: dict[str, PromptVersion] = {}
    for entry_id, entry in _iter_by_id(raw, warnings):
        messages = _decode_messages(entry.get("messages"), warnings)
        versions[entry_id] = PromptVersion(
            id=entry_id,
            prompt_id=_as_str(entry.get("prompt_id")),
            number=_as_int(entry.get("number")),
            engine=_enum(entry.get("engine"), _ENGINES, "liquid", "unknown_engine", warnings),
            messages=messages,
            text_template=_as_str(entry.get("text_template")),
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
        message: dict[str, Any] = {
            "role": _as_str(entry.get("role")),
            "content": _as_str(entry.get("content")) or "",
        }
        name = _as_str(entry.get("name"))
        if name is not None:
            message["name"] = name
        messages.append(message)
    return tuple(messages)


def _decode_models(raw: Any, warnings: list[str]) -> dict[str, Model]:
    models: dict[str, Model] = {}
    for entry_id, entry in _iter_by_id(raw, warnings):
        capabilities = entry.get("capabilities")
        if not isinstance(capabilities, list):
            capabilities = [] if capabilities is None else [capabilities]
        models[entry_id] = Model(
            id=entry_id,
            provider=_as_str(entry.get("provider")),
            model_id=_as_str(entry.get("model_id")),
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
                yield _as_str(entry.get("id")) or str(key), entry
            else:
                warnings.append(f"invalid_entry: {key}")
        return
    if isinstance(raw, list):
        for entry in raw:
            entry_id = _as_str(entry.get("id")) if isinstance(entry, Mapping) else None
            if entry_id:
                yield entry_id, entry
            else:
                warnings.append(f"invalid_entry: {entry!r}")
        return
    warnings.append(f"invalid_collection: {raw!r}")
