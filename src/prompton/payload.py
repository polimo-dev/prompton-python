"""The payload policy: what of a prompt and completion actually leaves the process.

The server re-checks all of this, but the SDK applies it first so the raw text never travels over
the network at all. The order matters, because the steps interact:

1. **keep decision** - always keep on ``status == "error"`` or ``stop_kind == "length"``, otherwise
   keep when ``bucket(id) < round(sample_rate * 10000)``. The bucket is a pure function of the
   record id, so a resend makes the same decision and the server reaches the same answer on its
   own.
2. **wrapping** - a string ``input`` becomes ``{"text": ...}`` and a string ``output``
   ``{"content": ...}``.
3. **mode** - ``none`` drops input and output, ``hash`` replaces them with a digest of the wrapped
   value, ``full`` truncates to the caps below.
4. ``error.message`` capped at 2048 bytes.
5. ``end_user_ref`` hashed when ``hash_end_user`` is set.
6. the app's ``redact`` hook, last.

Caps, all derived from the use case's ``max_bytes`` (default 262144):

=================================  =============================
one message ``content``            ``max(max_bytes / 8, 64)``
``input.messages`` (whole list)    ``max_bytes``
``input.text``                     ``max_bytes``
``input.variables``                ``max(max_bytes / 4, 64)``
``output.content``                 ``max(max_bytes / 4, 64)``
``output.tool_calls``              ``max(max_bytes / 4, 64)``
``error.message``                  2048, fixed
=================================  =============================
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from typing import Any

from ._json import canonical_json, json_size, list_json_size, sha256_hex

__all__ = ["DEFAULT_POLICY", "apply_policy", "bucket", "should_keep", "truncate_bytes"]

log = logging.getLogger("prompton")

ERROR_MESSAGE_MAX = 2048
SAMPLE_SCALE = 10_000
DEFAULT_MAX_BYTES = 262_144

DEFAULT_POLICY: dict[str, Any] = {
    "mode": "full",
    "sample_rate": 1.0,
    "max_bytes": DEFAULT_MAX_BYTES,
}


def bucket(record_id: Any) -> int:
    """``first 4 bytes of sha256(id)`` as an unsigned big-endian integer, mod 10000."""
    digest = hashlib.sha256(str(record_id or "").encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % SAMPLE_SCALE


def should_keep(record: Mapping[str, Any], sample_rate: float) -> bool:
    """Whether this record's raw text is kept.

    Errors and truncated answers are kept whatever the rate: an error you cannot see is worse than
    a storage bill, and a cut-off answer is the one you most need the text of.
    """
    if record.get("status") == "error":
        return True
    if record.get("stop_kind") == "length":
        return True
    if sample_rate >= 1.0:
        return True
    if sample_rate <= 0.0:
        return False
    return bucket(record.get("id")) < round(sample_rate * SAMPLE_SCALE)


def normalize_policy(policy: Any, defaults: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot policy layered over the SDK defaults, with the values clamped."""
    base = dict(defaults or DEFAULT_POLICY)
    given: dict[str, Any] = {}
    if policy is not None:
        given = (
            dict(policy) if isinstance(policy, Mapping) else dict(getattr(policy, "__dict__", {}))
        )
        if hasattr(policy, "as_dict"):
            given = policy.as_dict()

    def pick(key: str) -> Any:
        value = given.get(key)
        return base.get(key) if value is None else value

    mode = pick("mode")
    if mode not in ("full", "hash", "none"):
        mode = "full"

    rate = pick("sample_rate")
    rate = float(rate) if isinstance(rate, (int, float)) and not isinstance(rate, bool) else 1.0
    rate = min(max(rate, 0.0), 1.0)

    max_bytes = pick("max_bytes")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        max_bytes = DEFAULT_MAX_BYTES

    return {"mode": mode, "sample_rate": rate, "max_bytes": max_bytes}


def apply_policy(
    record: Mapping[str, Any],
    policy: Any = None,
    *,
    defaults: Mapping[str, Any] | None = None,
    hash_end_user: bool = False,
    redact: Callable[[dict[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Return ``record`` with the payload policy applied. Never raises."""
    resolved = normalize_policy(policy, defaults)
    result = dict(record)

    if resolved["mode"] == "none" or not should_keep(result, resolved["sample_rate"]):
        result.pop("input", None)
        result.pop("output", None)
    else:
        _wrap_strings(result)
        if resolved["mode"] == "hash":
            _hash_payload(result)
        else:
            _truncate_payload(result, resolved["max_bytes"])

    _cap_error_message(result)

    if hash_end_user and result.get("end_user_ref") is not None:
        result["end_user_ref"] = sha256_hex(str(result["end_user_ref"]))

    if redact is not None:
        try:
            redacted = redact(result)
        except Exception as error:  # noqa: BLE001 - a broken hook must not lose the record
            log.warning("prompton: redact hook raised %s; dropping the payload", error)
            result.pop("input", None)
            result.pop("output", None)
            return result
        if isinstance(redacted, Mapping):
            return dict(redacted)
        log.warning("prompton: redact hook returned %r; dropping the payload", type(redacted))
        result.pop("input", None)
        result.pop("output", None)
    return result


# ---------------------------------------------------------------------------
# wrapping and hashing


def _wrap_strings(record: dict[str, Any]) -> None:
    if isinstance(record.get("input"), str):
        record["input"] = {"text": record["input"]}
    if isinstance(record.get("output"), str):
        record["output"] = {"content": record["output"]}


def _hash_payload(record: dict[str, Any]) -> None:
    for key in ("input", "output"):
        if record.get(key) is None:
            continue
        encoded = canonical_json(record[key])
        record[key] = {"sha256": sha256_hex(encoded), "bytes": len(encoded), "hashed": True}


# ---------------------------------------------------------------------------
# truncation


def _truncate_payload(record: dict[str, Any], max_bytes: int) -> None:
    if isinstance(record.get("input"), Mapping):
        record["input"] = _truncate_input(dict(record["input"]), max_bytes)
    if isinstance(record.get("output"), Mapping):
        record["output"] = _truncate_output(dict(record["output"]), max_bytes)


def _truncate_input(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    per_message = max(max_bytes // 8, 64)
    variable_limit = max(max_bytes // 4, 64)
    cut = False

    if isinstance(payload.get("messages"), list):
        payload["messages"], changed = _truncate_messages(
            payload["messages"], per_message, max_bytes
        )
        cut = cut or changed
    if isinstance(payload.get("text"), str):
        payload["text"], changed = truncate_bytes(payload["text"], max_bytes)
        cut = cut or changed
    if payload.get("variables") is not None:
        encoded = canonical_json(payload["variables"])
        if len(encoded) > variable_limit:
            # variables are never partially cut: the whole map becomes a digest
            payload["variables"] = {
                "truncated": True,
                "sha256": sha256_hex(encoded),
                "bytes": len(encoded),
            }
            cut = True

    if cut:
        payload["truncated"] = True
    return payload


def _truncate_output(payload: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    limit = max(max_bytes // 4, 64)
    cut = False

    if isinstance(payload.get("content"), str):
        payload["content"], changed = truncate_bytes(payload["content"], limit)
        cut = cut or changed
    if isinstance(payload.get("tool_calls"), list):
        payload["tool_calls"], changed = _truncate_tool_calls(payload["tool_calls"], limit)
        cut = cut or changed

    if cut:
        payload["truncated"] = True
    return payload


def _truncate_messages(
    messages: list[Any], per_message: int, total_limit: int
) -> tuple[list[Any], bool]:
    cut = False
    shrunk: list[Any] = []
    for message in messages:
        message, changed = _truncate_message(message, per_message)
        cut = cut or changed
        shrunk.append(message)

    if list_json_size(shrunk) <= total_limit:
        return shrunk, cut

    stubbed = _stub_middle(shrunk, total_limit)
    if list_json_size(stubbed) <= total_limit:
        return stubbed, True
    return _drop_middle(shrunk, total_limit), True


def _truncate_message(message: Any, limit: int) -> tuple[Any, bool]:
    if not isinstance(message, Mapping):
        return message, False
    content = message.get("content")
    if content is None:
        return message, False
    if isinstance(content, str):
        shrunk, changed = truncate_bytes(content, limit)
        if not changed:
            return message, False
        result = dict(message)
        result["content"] = shrunk
        result["truncated"] = True
        return result, True
    encoded = canonical_json(content)
    if len(encoded) <= limit:
        return message, False
    shrunk_bytes, _ = truncate_bytes(encoded.decode("utf-8", "replace"), limit)
    result = dict(message)
    result["content"] = shrunk_bytes
    result["truncated"] = True
    return result, True


def _message_content_bytes(message: Any) -> int:
    if not isinstance(message, Mapping):
        return 0
    content = message.get("content")
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    return json_size(content)


def _stub_middle(messages: list[Any], limit: int) -> list[Any]:
    """Empty middle messages into byte-count stubs from the front until the list fits.

    The first message (the system prompt) and the last one (the newest turn) are always preserved,
    so a later middle message can survive intact.
    """
    count = len(messages)
    running = list_json_size(messages)
    result: list[Any] = []
    for index, message in enumerate(messages):
        if 0 < index < count - 1 and running > limit and isinstance(message, Mapping):
            stub = dict(message)
            stub["content"] = f"…[truncated {_message_content_bytes(message)} bytes]…"
            stub["truncated"] = True
            running = running - json_size(message) + json_size(stub)
            result.append(stub)
        else:
            result.append(message)
    return result


def _drop_middle(messages: list[Any], limit: int) -> list[Any]:
    """First message, one ``…[N messages truncated]…`` marker, and as much of the tail as fits."""
    if not messages:
        return []
    first, rest = messages[0], messages[1:]
    marker = {
        "role": "system",
        "content": _marker_text(len(rest)),
        "truncated": True,
    }
    base = list_json_size([first, marker])
    if base <= limit:
        kept = _tail_within(rest, limit - base)
        marker["content"] = _marker_text(len(rest) - len(kept))
        return [first, marker, *kept]

    smaller = _shrink(first)
    if smaller is not None:
        return _drop_middle([smaller, *rest], limit)
    marker["content"] = _marker_text(len(rest) + 1)
    return [marker] if list_json_size([marker]) <= limit else []


def _shrink(message: Any) -> Any | None:
    """Halve a message's content, then reduce it to its role. ``None`` when nothing is left."""
    if not isinstance(message, Mapping):
        return None
    size = _message_content_bytes(message)
    if size > 0:
        shrunk, changed = _truncate_message(message, size // 2)
        return shrunk if changed else None
    role = message.get("role")
    minimal: dict[str, Any] = {"truncated": True}
    if role is not None:
        minimal["role"] = role
    return None if minimal == dict(message) else minimal


def _tail_within(messages: list[Any], budget: int) -> list[Any]:
    kept: list[Any] = []
    for message in reversed(messages):
        size = json_size(message) + 1
        if size > budget:
            break
        kept.insert(0, message)
        budget -= size
    return kept


def _marker_text(dropped: int) -> str:
    return f"…[{dropped} messages truncated]…"


def _truncate_tool_calls(calls: list[Any], limit: int) -> tuple[list[Any], bool]:
    if json_size(calls) <= limit:
        return calls, False
    overhead = json_size([_with_arguments(call, "") for call in calls])
    budget = max(limit - overhead, 0) // max(len(calls), 1)
    return _shrink_tool_calls(calls, budget, limit), True


def _shrink_tool_calls(calls: list[Any], budget: int, limit: int) -> list[Any]:
    while budget >= 32:
        shrunk = []
        for call in calls:
            arguments = _arguments_of(call)
            if arguments is None:
                shrunk.append(call)
            else:
                shrunk.append(_with_arguments(call, truncate_bytes(arguments, budget)[0]))
        if json_size(shrunk) <= limit:
            return shrunk
        budget //= 2
    return [{"truncated": True, "bytes": json_size(calls)}]


def _arguments_of(call: Any) -> str | None:
    if isinstance(call, Mapping) and isinstance(call.get("function"), Mapping):
        arguments = call["function"].get("arguments")
        if isinstance(arguments, str):
            return arguments
    return None


def _with_arguments(call: Any, arguments: str) -> Any:
    if _arguments_of(call) is None:
        return call
    result = dict(call)
    function = dict(result["function"])
    function["arguments"] = arguments
    result["function"] = function
    return result


def truncate_bytes(text: str, limit: int) -> tuple[str, bool]:
    """Cut ``text`` to ``limit`` bytes, keeping head and tail, never splitting a character.

    The marker is ``\\n…[truncated N bytes]…\\n`` where ``N = original_bytes - limit``. The budget
    left after the marker is split 60% head / 40% tail, then each side is trimmed back to a UTF-8
    boundary, so the result is never longer than the cap and always valid UTF-8.
    """
    raw = text.encode("utf-8")
    if len(raw) <= limit:
        return text, False

    marker = f"\n…[truncated {len(raw) - limit} bytes]…\n".encode()
    if len(marker) > limit:
        return _trim_trailing_partial(raw[:limit]).decode("utf-8", "ignore"), True

    budget = limit - len(marker)
    head_length = budget * 6 // 10
    tail_length = budget - head_length
    head = _trim_trailing_partial(raw[:head_length])
    tail = _trim_leading_partial(raw[len(raw) - tail_length :]) if tail_length else b""
    return (head + marker + tail).decode("utf-8", "ignore"), True


def _trim_trailing_partial(raw: bytes, tries: int = 3) -> bytes:
    while raw:
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            if tries == 0:
                return b""
            raw = raw[:-1]
            tries -= 1
            continue
        return raw
    return raw


def _trim_leading_partial(raw: bytes) -> bytes:
    index = 0
    while index < len(raw) and 0x80 <= raw[index] < 0xC0:
        index += 1
    return raw[index:]


def _cap_error_message(record: dict[str, Any]) -> None:
    error = record.get("error")
    if not isinstance(error, Mapping):
        return
    message = error.get("message")
    if not isinstance(message, str):
        return
    if len(message.encode("utf-8")) <= ERROR_MESSAGE_MAX:
        return
    capped = dict(error)
    capped["message"] = truncate_bytes(message, ERROR_MESSAGE_MAX)[0]
    record["error"] = capped
