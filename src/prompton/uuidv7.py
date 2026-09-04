"""RFC 9562 UUIDv7, implemented here because the standard library has no generator for it.

A monitoring-log ``id`` is the idempotency key the app issues *before* the provider call, and the
PromptOn column is a UUIDv7 type: a v4 id passes request validation and then fails on write, coming
back in ``rejected`` with a message that does not say why. Layout: 48-bit unix milliseconds,
version nibble 7, 12 random bits, variant ``10``, 62 random bits.
"""

from __future__ import annotations

import os
import time

__all__ = ["timestamp_ms", "uuid7"]


def uuid7(unix_ms: int | None = None) -> str:
    """A new UUIDv7 string, lowercase with dashes.

    Ids generated in the same millisecond have no defined order between them; across milliseconds
    string order is time order.
    """
    if unix_ms is None:
        unix_ms = time.time_ns() // 1_000_000
    if unix_ms < 0:
        raise ValueError("unix_ms must not be negative")

    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = (rand >> 62) & 0xFFF
    rand_b = rand & ((1 << 62) - 1)

    value = (unix_ms & 0xFFFFFFFFFFFF) << 80
    value |= 0x7 << 76
    value |= rand_a << 64
    value |= 0b10 << 62
    value |= rand_b

    hexed = f"{value:032x}"
    return f"{hexed[0:8]}-{hexed[8:12]}-{hexed[12:16]}-{hexed[16:20]}-{hexed[20:32]}"


def timestamp_ms(value: str) -> int | None:
    """The unix milliseconds encoded in a UUIDv7, or ``None`` when it is not one."""
    raw = value.replace("-", "")
    if len(raw) != 32:
        return None
    try:
        return int(raw[:12], 16)
    except ValueError:
        return None
