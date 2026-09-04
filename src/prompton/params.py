"""Shallow merge for parameter maps.

``params = use_case.default_params <- deployment.params`` and
``provider_options = model.provider_options <- deployment.provider_options``. Both are
**shallow** merges where the right side wins: a nested map on the right replaces the left side
whole. An override value of ``None`` is kept as ``None``, not deleted - apps rely on sending
``"only": null`` to clear a provider restriction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["merge", "stringify_keys"]


def stringify_keys(value: Any) -> dict[str, Any]:
    """Top-level keys as strings. A non-mapping (including ``None``) becomes ``{}``."""
    if not isinstance(value, Mapping):
        return {}
    return {(key if isinstance(key, str) else str(key)): item for key, item in value.items()}


def merge(base: Any, override: Any) -> dict[str, Any]:
    """Shallow-merge ``override`` onto ``base``. ``override`` wins; ``None`` values are kept."""
    merged = stringify_keys(base)
    merged.update(stringify_keys(override))
    return merged
