"""Building a monitoring-log record, and the wrapper that times a provider call.

``build_record`` turns a :class:`~prompton.resolver.Resolution`, the call metadata and the outcome
into the exact JSON shape ``POST /api/v1/generations`` accepts. Top-level keys whose value is
``None`` are omitted; the nulls inside ``usage`` are sent as-is and accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._version import SDK_NAME, VERSION
from .errors import ProviderError
from .params import merge, stringify_keys
from .resolver import Resolution
from .stop_kind import normalize as normalize_stop_kind
from .uuidv7 import uuid7

__all__ = ["ERROR_KINDS", "Outcome", "build_record", "iso_timestamp"]

ERROR_KINDS = ("http_4xx", "http_5xx", "rate_limited", "timeout", "transport", "parse", "app")

REQUIRED_FIELDS = ("id", "use_case", "model", "status", "started_at")


@dataclass
class Outcome:
    """What the provider answered, in the shape the monitoring log wants.

    Return one of these from the function you pass to ``with_generation``. Every field is optional:
    fill in what your provider actually reports. ``result`` is yours - the SDK carries it back to
    you untouched, so a wrapper can hand you the parsed answer.
    """

    content: str | None = None
    tool_calls: list[Any] | None = None
    finish_reason: str | None = None
    stop_kind: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    usage_raw: Any = None
    cost_usd: float | None = None
    cost_source: str | None = None  # provider | catalog | unknown
    is_byok: bool | None = None
    model_used: str | None = None
    upstream_provider: str | None = None
    result: Any = None

    @classmethod
    def coerce(cls, value: Any) -> Outcome | None:
        """Accept an Outcome, a mapping in the same shape, a plain string, or nothing."""
        if value is None or isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(content=value)
        if isinstance(value, Mapping):
            usage = value.get("usage") if isinstance(value.get("usage"), Mapping) else {}
            return cls(
                content=value.get("content"),
                tool_calls=value.get("tool_calls"),
                finish_reason=value.get("finish_reason"),
                stop_kind=value.get("stop_kind"),
                input_tokens=usage.get("input_tokens", value.get("input_tokens")),
                output_tokens=usage.get("output_tokens", value.get("output_tokens")),
                usage_raw=usage.get("raw", value.get("usage_raw")),
                cost_usd=value.get("cost_usd"),
                cost_source=value.get("cost_source"),
                is_byok=value.get("is_byok"),
                model_used=value.get("model_used"),
                upstream_provider=value.get("upstream_provider"),
                result=value.get("result"),
            )
        return None


@dataclass
class CallMeta:
    """The call-site facts a monitoring log needs that the resolution cannot know."""

    id: str | None = None
    variables: Mapping[str, Any] | None = None
    input_messages: Any = None
    input_text: str | None = None
    end_user_ref: Any = None
    trace_id: str | None = None
    sequence: int | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    params: Mapping[str, Any] | None = None


def iso_timestamp(moment: datetime | None = None) -> str:
    """ISO 8601 UTC with microseconds and a ``Z``, the shape the ingest endpoint expects."""
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def build_record(
    resolution: Resolution | None,
    meta: CallMeta,
    *,
    status: str,
    started_at: str,
    latency_ms: int | None = None,
    outcome: Outcome | None = None,
    error: BaseException | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one monitoring-log record. Top-level ``None`` values are dropped."""
    usage_source = outcome or Outcome()
    metadata = stringify_keys(meta.metadata)
    if usage_source.is_byok is not None:
        metadata = {**metadata, "is_byok": usage_source.is_byok}

    record: dict[str, Any] = {
        "id": meta.id or uuid7(),
        "use_case": resolution.use_case if resolution else None,
        "deployment_id": resolution.deployment_id if resolution else None,
        "deployment_revision": resolution.deployment_revision if resolution else None,
        "prompt": resolution.prompt if resolution else None,
        "prompt_version_id": resolution.prompt_version_id if resolution else None,
        "model_id": resolution.model_id if resolution else None,
        "resolution_source": resolution.resolution_source if resolution else "manual",
        "context": stringify_keys(meta.context),
        "kind": resolution.kind if resolution else None,
        "model": resolution.model if resolution else None,
        "model_used": usage_source.model_used,
        "provider": resolution.provider if resolution else None,
        "upstream_provider": usage_source.upstream_provider,
        "params": merge(resolution.effective_params if resolution else None, meta.params),
        "input": _build_input(meta),
        "output": _build_output(outcome),
        "status": status,
        "finish_reason": usage_source.finish_reason,
        "stop_kind": _stop_kind(outcome),
        "error": _build_error(error),
        "usage": {
            "input_tokens": usage_source.input_tokens,
            "output_tokens": usage_source.output_tokens,
            "cost_usd": usage_source.cost_usd,
            "cost_source": usage_source.cost_source or "unknown",
            "raw": usage_source.usage_raw,
        },
        "latency_ms": latency_ms,
        "started_at": started_at,
        "trace_id": meta.trace_id,
        "sequence": meta.sequence,
        "end_user_ref": None if meta.end_user_ref is None else str(meta.end_user_ref),
        "metadata": metadata,
        "sdk": {"name": SDK_NAME, "version": VERSION},
    }
    return {key: value for key, value in record.items() if value is not None}


def _build_input(meta: CallMeta) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    if meta.variables is not None:
        payload["variables"] = stringify_keys(meta.variables)
    if meta.input_messages is not None:
        payload["messages"] = meta.input_messages
    if meta.input_text is not None:
        payload["text"] = meta.input_text
    return payload or None


def _build_output(outcome: Outcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    payload: dict[str, Any] = {}
    if outcome.content is not None:
        payload["content"] = outcome.content
    if outcome.tool_calls is not None:
        payload["tool_calls"] = outcome.tool_calls
    return payload or None


def _stop_kind(outcome: Outcome | None) -> str | None:
    if outcome is None:
        return None
    if outcome.stop_kind is not None:
        return normalize_stop_kind(outcome.stop_kind)
    if outcome.finish_reason is not None:
        return normalize_stop_kind(outcome.finish_reason)
    return None


def _build_error(error: BaseException | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, ProviderError):
        kind, status, message = error.kind, error.status, str(error)
    elif isinstance(error, Mapping):
        kind = error.get("kind")
        status = error.get("status")
        message = error.get("message")
    elif isinstance(error, BaseException):
        kind, status = "app", None
        message = f"{type(error).__name__}: {error}"
    else:  # pragma: no cover - defensive
        kind, status, message = "app", None, str(error)

    built: dict[str, Any] = {"kind": kind if kind in ERROR_KINDS else "app"}
    if isinstance(status, int) and not isinstance(status, bool):
        built["status"] = status
    if message is not None:
        built["message"] = message if isinstance(message, str) else repr(message)
    return built
