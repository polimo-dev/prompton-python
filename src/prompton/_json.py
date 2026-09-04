"""Canonical JSON, byte sizes and digests.

Every byte count in the truncation arithmetic is measured on the same encoding the reference
implementation uses: no whitespace, keys sorted, raw UTF-8 (no ``\\uXXXX`` escaping). Keeping that
in one place is what makes a Python SDK and an Elixir SDK agree on whether a payload fits.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode ``value`` the way the contract measures it: compact, sorted keys, UTF-8."""
    try:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError):
        return repr(value).encode("utf-8", "replace")


def json_size(value: Any) -> int:
    """Byte size of ``value`` under :func:`canonical_json`."""
    return len(canonical_json(value))


def list_json_size(values: list[Any]) -> int:
    """Byte size of a JSON array, computed element by element.

    ``1 + sum(size(el) + 1)`` is exactly ``2 + sum(sizes) + (n - 1)``, the real array size.
    """
    if not values:
        return 2
    total = 1
    for element in values:
        total += json_size(element) + 1
    return total


def sha256_hex(data: bytes | str) -> str:
    """Lowercase hex SHA-256 of ``data``."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
