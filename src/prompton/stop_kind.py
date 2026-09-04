"""Provider ``finish_reason`` to PromptOn ``stop_kind``.

Two traps live here. Google's ``SAFETY`` and ``RECITATION`` map to ``other``, not
``content_filter`` - only the literal string ``content_filter`` lands there. And ``tool_calls`` is
*not* a truncation: only ``length`` sets :func:`truncated`, and the truncation rate, the evaluator
and the alerts all depend on that.
"""

from __future__ import annotations

__all__ = ["STOP_KINDS", "normalize", "truncated"]

STOP_KINDS = ("stop", "length", "tool_call", "content_filter", "other")

_STOP = frozenset({"stop", "end_turn", "stop_sequence"})
_LENGTH = frozenset({"length", "max_tokens"})
_TOOL_CALL = frozenset({"tool_call", "tool_calls", "tool_use"})
_CONTENT_FILTER = frozenset({"content_filter"})


def normalize(finish_reason: object) -> str:
    """Normalize a raw finish reason into one of :data:`STOP_KINDS`.

    Comparison lowercases and trims, so Google's ``STOP`` and ``MAX_TOKENS`` map correctly.
    Normalization is idempotent - feeding a ``stop_kind`` back in returns itself - which matters
    because the server re-normalizes whatever the client sent.
    """
    if finish_reason is None:
        return "other"
    if not isinstance(finish_reason, str):
        finish_reason = str(finish_reason)
    value = finish_reason.strip().lower()
    if value in _STOP:
        return "stop"
    if value in _LENGTH:
        return "length"
    if value in _TOOL_CALL:
        return "tool_call"
    if value in _CONTENT_FILTER:
        return "content_filter"
    return "other"


def truncated(finish_reason_or_stop_kind: object) -> bool:
    """Whether the output was cut off. True only for ``length``."""
    return normalize(finish_reason_or_stop_kind) == "length"
