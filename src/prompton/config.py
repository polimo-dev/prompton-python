"""Runtime configuration, and where each value comes from.

Precedence is always **explicit option > environment variable > default**. The base URL is a host
(``PTN_HOST``); the SDK appends ``/api/v1`` itself, so you configure the same value you would paste
into a browser.
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ._version import USER_AGENT
from .errors import ConfigurationError

__all__ = ["DEFAULT_HOST", "Config", "default_disk_cache_path", "project_from_api_key"]

DEFAULT_HOST = "https://app.prompton.ai"
DEFAULT_ENVIRONMENT = "production"
API_PREFIX = "/api/v1"

Mode = Literal["live", "test", "offline"]

_UNSET = object()


def _env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else None


def _env_bool(name: str) -> bool | None:
    raw = _env(name)
    if raw is None:
        return None
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str) -> float | None:
    raw = _env(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from error


def _env_int(name: str) -> int | None:
    value = _env_float(name)
    return None if value is None else int(value)


def project_from_api_key(api_key: str | None) -> str | None:
    """A runtime key is ``ptn_<project_slug>_<random>``, so the key names its own project."""
    if not api_key or not api_key.startswith("ptn_"):
        return None
    parts = api_key.split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[1:-1]) or None


def default_disk_cache_path(project: str, environment: str) -> Path:
    """``<os cache dir>/prompton/<project>-<environment>.json``.

    Several processes on one host may share this file: writes are atomic (tmp + rename), readers
    tolerate a concurrent rename, and a corrupt or partial file is ignored rather than raised.
    """
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        root = Path(base)
    elif sys.platform == "darwin":  # pragma: no cover - platform specific
        root = Path.home() / "Library" / "Caches"
    elif sys.platform.startswith("win"):  # pragma: no cover - platform specific
        root = Path(os.environ.get("LOCALAPPDATA") or tempfile.gettempdir())
    else:  # pragma: no cover - platform specific
        root = Path.home() / ".cache"
    if not root.is_dir():  # pragma: no cover - unwritable home
        root = Path(tempfile.gettempdir())
    name = f"{project}-{environment}"
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in name)
    return root / "prompton" / f"{safe}.json"


@dataclass(frozen=True)
class Config:
    """Everything the client needs, already resolved. Build it with :meth:`Config.build`."""

    api_key: str | None = None
    host: str = DEFAULT_HOST
    environment: str = DEFAULT_ENVIRONMENT
    project: str | None = None
    timeout: float = 5.0
    cache_ttl: float = 10.0
    poll: bool = True
    disk_cache_path: Path | None = None
    bundle_path: Path | None = None
    mode: Mode = "live"
    hash_end_user: bool = False
    user_agent: str = USER_AGENT
    max_backoff: float = 300.0
    flush_interval: float = 2.0
    flush_size: int = 100
    flush_bytes: int = 1_000_000
    max_queue: int = 10_000
    max_send_attempts: int = 8
    redact: Callable[[dict[str, Any]], Any] | None = None
    payload_defaults: dict[str, Any] = field(
        default_factory=lambda: {"mode": "full", "sample_rate": 1.0, "max_bytes": 262_144}
    )

    @property
    def base_url(self) -> str:
        """The API root, ``<host>/api/v1``."""
        return self.host.rstrip("/") + API_PREFIX

    @property
    def remote_enabled(self) -> bool:
        """Whether the SDK may talk to PromptOn at all."""
        return self.mode == "live" and bool(self.api_key)

    @classmethod
    def build(
        cls,
        *,
        api_key: Any = _UNSET,
        host: Any = _UNSET,
        environment: Any = _UNSET,
        project: Any = _UNSET,
        timeout: Any = _UNSET,
        cache_ttl: Any = _UNSET,
        poll: Any = _UNSET,
        disk_cache: Any = _UNSET,
        bundle: Any = _UNSET,
        mode: Any = _UNSET,
        hash_end_user: Any = _UNSET,
        max_backoff: Any = _UNSET,
        flush_interval: Any = _UNSET,
        flush_size: Any = _UNSET,
        flush_bytes: Any = _UNSET,
        max_queue: Any = _UNSET,
        max_send_attempts: Any = _UNSET,
        redact: Any = _UNSET,
        payload_defaults: Any = _UNSET,
    ) -> Config:
        """Resolve options, environment variables and defaults into a frozen config."""
        api_key_value = _pick(api_key, _env("PTN_API_KEY"), None)
        host_value = str(_pick(host, _env("PTN_HOST"), DEFAULT_HOST)).rstrip("/")
        if not host_value.startswith(("http://", "https://")):
            raise ConfigurationError(
                f"host must start with http:// or https://, got {host_value!r}"
            )
        environment_value = str(_pick(environment, _env("PTN_ENVIRONMENT"), DEFAULT_ENVIRONMENT))
        if not environment_value:
            raise ConfigurationError("environment must not be empty")

        mode_value = str(_pick(mode, _env("PTN_MODE"), "live"))
        if mode_value not in ("live", "test", "offline"):
            raise ConfigurationError(f"mode must be live, test or offline, got {mode_value!r}")

        project_value = _pick(
            project, _env("PTN_PROJECT"), project_from_api_key(api_key_value) or "default"
        )

        disk_value = _pick(disk_cache, _env("PTN_DISK_CACHE"), True)
        disk_path = _resolve_disk_cache(disk_value, str(project_value), environment_value)

        bundle_value = _pick(bundle, _env("PTN_BUNDLE"), None)
        bundle_path = Path(bundle_value) if bundle_value else None

        if mode_value == "test":
            # A test-mode client must start empty and behave the same on every machine, so it
            # reads neither the developer's OS cache nor a bundle: load_snapshot() is the only way
            # to put a document in it. (offline mode is the one that reads them for real.)
            disk_path = None
            bundle_path = None

        return cls(
            api_key=api_key_value,
            host=host_value,
            environment=environment_value,
            project=str(project_value),
            timeout=float(_pick(timeout, _env_float("PTN_TIMEOUT"), 5.0)),
            cache_ttl=float(_pick(cache_ttl, _env_float("PTN_CACHE_TTL"), 10.0)),
            poll=bool(_pick(poll, _env_bool("PTN_POLL"), True)),
            disk_cache_path=disk_path,
            bundle_path=bundle_path,
            mode=mode_value,  # type: ignore[arg-type]
            hash_end_user=bool(_pick(hash_end_user, _env_bool("PTN_HASH_END_USER"), False)),
            max_backoff=float(_pick(max_backoff, _env_float("PTN_MAX_BACKOFF"), 300.0)),
            flush_interval=float(_pick(flush_interval, _env_float("PTN_FLUSH_INTERVAL"), 2.0)),
            flush_size=int(_pick(flush_size, _env_int("PTN_FLUSH_SIZE"), 100)),
            flush_bytes=int(_pick(flush_bytes, _env_int("PTN_FLUSH_BYTES"), 1_000_000)),
            max_queue=int(_pick(max_queue, _env_int("PTN_MAX_QUEUE"), 10_000)),
            max_send_attempts=int(_pick(max_send_attempts, _env_int("PTN_MAX_SEND_ATTEMPTS"), 8)),
            redact=_pick(redact, None, None),
            payload_defaults=dict(
                _pick(
                    payload_defaults,
                    None,
                    {"mode": "full", "sample_rate": 1.0, "max_bytes": 262_144},
                )
            ),
        )


def _pick(explicit: Any, from_env: Any, default: Any) -> Any:
    if explicit is not _UNSET and explicit is not None:
        return explicit
    if from_env is not None:
        return from_env
    return default


def _resolve_disk_cache(value: Any, project: str, environment: str) -> Path | None:
    if value is False or value is None:
        return None
    if value is True:
        return default_disk_cache_path(project, environment)
    text = str(value)
    if text.strip().lower() in ("0", "off", "false", "no", "none", ""):
        return None
    if text.strip().lower() in ("1", "on", "true", "yes"):
        return default_disk_cache_path(project, environment)
    return Path(text)
