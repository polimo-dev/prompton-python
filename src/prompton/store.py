"""The snapshot store: three tiers, one file, no external services.

Memory is always the fast path. Disk is on by default, written atomically, and shared safely by
several processes on one host. A bundle - a snapshot JSON committed inside the app - is the
last resort. Load order on start is **memory, disk, bundle, remote**, and the tier that answered
is reported as ``resolution_source``.

The caching rules this file implements, in one place because they are the whole resilience story:

* a **10-second cache**: within the TTL every resolve is served from memory with no HTTP call;
* past the TTL the document is refreshed with ``If-None-Match`` - in a poll thread, or as a
  stale-while-revalidate refresh triggered by the next call. **A refresh never blocks or fails a
  generation**: while it is in flight, and if it fails, the previous document is used;
* on ``429`` the SDK reads ``Retry-After`` and does not contact the server again before it has
  elapsed; on ``5xx``, timeouts and transport errors it backs off ×2 from the TTL to five minutes.
  The caller never sees any of it;
* a document for another environment or project is never used. The file records both, and a
  mismatch is ignored rather than raised;
* resolution fails only when no tier has a document at all.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .config import Config
from .errors import PromptOnError, SnapshotUnavailableError, TransportError
from .http import (
    HttpResponse,
    Transport,
    build_headers,
    parse_api_error,
    retry_after_seconds,
    urlencode,
)
from .snapshot_data import SnapshotData

__all__ = ["SnapshotEntry", "SnapshotStore"]

log = logging.getLogger("prompton")

Source = Literal["remote", "disk", "bundle", "manual"]


@dataclass
class SnapshotEntry:
    """One cached snapshot document plus where it came from and how fresh it is."""

    data: SnapshotData
    raw: bytes
    source: Source
    etag: str | None = None
    last_modified: str | None = None
    fetched_at: float = 0.0
    checked_at: float = 0.0
    stale_since: float | None = None

    @property
    def environment(self) -> str | None:
        return self.data.environment

    @property
    def project(self) -> str | None:
        return self.data.project


class SnapshotStore:
    """Holds the current snapshot and decides when to go and get a new one."""

    def __init__(self, config: Config, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._lock = threading.RLock()
        self._entry: SnapshotEntry | None = None
        self._failures = 0
        self._not_before = 0.0
        self._refreshing = False
        self._poll_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._warned_no_key = False
        self.last_error: BaseException | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Load the local tiers, then start polling when polling is enabled."""
        self.load_local()
        if self._config.mode == "test":
            return
        if not self._config.remote_enabled:
            self._warn_no_remote()
            return
        if self._config.poll and self._poll_thread is None:
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="prompton-snapshot", daemon=True
            )
            self._poll_thread.start()

    def close(self) -> None:
        self._stop.set()
        thread = self._poll_thread
        self._poll_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _warn_no_remote(self) -> None:
        if self._warned_no_key:
            return
        self._warned_no_key = True
        if self._config.mode == "offline":
            log.info("prompton: offline mode - serving the disk cache and bundle only")
        else:
            log.warning(
                "prompton: no API key (PTN_API_KEY) - no remote calls will be made; "
                "serving the disk cache and bundle only"
            )

    # -- reading -----------------------------------------------------------

    def current(self) -> SnapshotEntry:
        """The document to resolve against, refreshing in the background when it is stale.

        Raises :class:`~prompton.errors.SnapshotUnavailableError` only when no tier has anything.
        """
        with self._lock:
            entry = self._entry
            due = entry is None or (time.monotonic() - entry.checked_at) >= self._config.cache_ttl

        if entry is not None:
            if due:
                self._revalidate_in_background()
            return entry

        # Nothing cached at all: this one call waits for the first fetch.
        self.load_local()
        with self._lock:
            if self._entry is not None:
                return self._entry
        if self._config.remote_enabled:
            self.refresh(raise_errors=False)
        with self._lock:
            if self._entry is not None:
                return self._entry
        raise SnapshotUnavailableError(
            "PromptOn is unreachable and nothing is cached: no snapshot in memory, on disk "
            f"({self._config.disk_cache_path or 'disabled'}) or in a bundle "
            f"({self._config.bundle_path or 'none'}) for environment "
            f"{self._config.environment!r}"
        )

    def peek(self) -> SnapshotEntry | None:
        """The current entry without triggering any refresh."""
        with self._lock:
            return self._entry

    def info(self) -> dict[str, Any]:
        """A small dict for health endpoints and debugging."""
        with self._lock:
            entry = self._entry
            failures = self._failures
            not_before = self._not_before
        if entry is None:
            return {
                "source": "none",
                "etag": None,
                "environment": self._config.environment,
                "project": self._config.project,
                "fetched_at": None,
                "age_seconds": None,
                "stale": True,
                "failures": failures,
                "retry_after_seconds": max(not_before - time.monotonic(), 0.0),
            }
        return {
            "source": entry.source,
            "etag": entry.etag,
            "last_modified": entry.last_modified,
            "environment": entry.environment,
            "project": entry.project,
            "fetched_at": entry.fetched_at,
            "age_seconds": max(time.time() - entry.fetched_at, 0.0),
            "stale": entry.stale_since is not None or entry.source != "remote",
            "failures": failures,
            "retry_after_seconds": max(not_before - time.monotonic(), 0.0),
        }

    # -- local tiers -------------------------------------------------------

    def load_local(self) -> SnapshotEntry | None:
        """Load the disk cache, then the bundle. Returns the entry installed, if any."""
        with self._lock:
            if self._entry is not None:
                return self._entry
        for path, source in self._local_candidates():
            entry = self._read_file(path, source)
            if entry is not None:
                with self._lock:
                    if self._entry is None:
                        self._entry = entry
                log.info("prompton: loaded snapshot from %s (%s)", source, path)
                return entry
        return None

    def _local_candidates(self) -> list[tuple[Path, Source]]:
        candidates: list[tuple[Path, Source]] = []
        if self._config.disk_cache_path is not None:
            candidates.append((self._config.disk_cache_path, "disk"))
        if self._config.bundle_path is not None:
            candidates.append((self._config.bundle_path, "bundle"))
        return candidates

    def _read_file(self, path: Path, source: Source) -> SnapshotEntry | None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            log.warning("prompton: could not read the %s snapshot %s: %s", source, path, error)
            return None
        try:
            data = SnapshotData.from_json(raw)
        except PromptOnError as error:
            # A corrupt, partial or outdated file is ignored, never raised.
            log.warning("prompton: ignoring the %s snapshot %s: %s", source, path, error)
            return None
        if not self._matches_scope(data, path, source):
            return None
        meta = _read_sidecar(path)
        return SnapshotEntry(
            data=data,
            raw=raw,
            source=source,
            etag=_as_text(meta.get("etag")),
            last_modified=_as_text(meta.get("last_modified")),
            fetched_at=_as_epoch(meta.get("fetched_at"), path),
            checked_at=0.0,
            stale_since=time.monotonic(),
        )

    def _matches_scope(self, data: SnapshotData, path: Path, source: Source) -> bool:
        """A snapshot for another environment or project is never used."""
        if data.environment and data.environment != self._config.environment:
            log.warning(
                "prompton: ignoring the %s snapshot %s: it is for environment %r, this client "
                "reads %r",
                source,
                path,
                data.environment,
                self._config.environment,
            )
            return False
        if (
            data.project
            and self._config.project
            and self._config.project != "default"
            and data.project != self._config.project
        ):
            log.warning(
                "prompton: ignoring the %s snapshot %s: it is for project %r, this client reads %r",
                source,
                path,
                data.project,
                self._config.project,
            )
            return False
        return True

    # -- remote ------------------------------------------------------------

    def refresh(self, *, raise_errors: bool = True) -> bool:
        """Fetch once, now, and wait for the answer. ``True`` when a new document was installed.

        This is the synchronous entry point for scripts and for a warm-up at boot. A ``304`` counts
        as success and returns ``False``.
        """
        if self._config.mode == "test":
            return False
        if not self._config.remote_enabled:
            self._warn_no_remote()
            if self._config.mode == "offline":
                # offline mode re-reads the files instead of the network
                with self._lock:
                    self._entry = None
                return self.load_local() is not None
            if raise_errors:
                raise SnapshotUnavailableError(
                    "no API key configured: set PTN_API_KEY or pass api_key= to use the network"
                )
            return False

        with self._lock:
            entry = self._entry
            blocked = time.monotonic() < self._not_before
        if blocked and not raise_errors:
            return False

        try:
            response = self._get_snapshot(entry.etag if entry else None)
        except TransportError as error:
            self._record_failure(error)
            if raise_errors:
                raise
            return False

        if response.status == 304:
            self._record_success(refreshed=False, response=response)
            return False
        if response.status == 200:
            return self._install(response)

        error = parse_api_error(response)
        if response.status == 429 or response.status >= 500:
            self._record_failure(error, retry_after=retry_after_seconds(response))
        else:
            # 401/403/404 are configuration mistakes: back off, but say so loudly.
            self._record_failure(error)
            log.error("prompton: snapshot fetch failed: %s", error)
        if raise_errors:
            raise error
        return False

    def _get_snapshot(self, etag: str | None) -> HttpResponse:
        query = urlencode({"environment": self._config.environment})
        url = f"{self._config.base_url}/snapshot?{query}"
        headers = build_headers(self._config.api_key, self._config.user_agent)
        if etag:
            headers["if-none-match"] = etag
        return self._transport.request("GET", url, headers=headers, timeout=self._config.timeout)

    def _install(self, response: HttpResponse) -> bool:
        try:
            data = SnapshotData.from_json(response.body)
        except PromptOnError as error:
            self._record_failure(error)
            log.error("prompton: the server returned a snapshot this SDK cannot read: %s", error)
            return False
        if not self._matches_scope(data, Path("<response>"), "remote"):
            self._record_failure(PromptOnError("snapshot scope mismatch"))
            return False

        now = time.time()
        entry = SnapshotEntry(
            data=data,
            raw=response.body,
            source="remote",
            etag=response.etag,
            last_modified=response.last_modified,
            fetched_at=now,
            checked_at=time.monotonic(),
        )
        with self._lock:
            self._entry = entry
            self._failures = 0
            self._not_before = 0.0
            self.last_error = None
        self._write_disk(entry)
        log.info(
            "prompton: snapshot updated (environment=%s etag=%s)", data.environment, entry.etag
        )
        return True

    def _record_success(self, *, refreshed: bool, response: HttpResponse) -> None:
        with self._lock:
            self._failures = 0
            self._not_before = 0.0
            self.last_error = None
            if self._entry is not None:
                self._entry.checked_at = time.monotonic()
                self._entry.stale_since = None
                self._entry.source = "remote"
                if response.etag:
                    self._entry.etag = response.etag

    def _record_failure(self, error: BaseException, retry_after: float | None = None) -> None:
        with self._lock:
            self._failures += 1
            self.last_error = error
            delay = retry_after
            if delay is None:
                base = max(self._config.cache_ttl, 1.0)
                delay = min(base * (2 ** (self._failures - 1)), self._config.max_backoff)
            self._not_before = time.monotonic() + delay
            if self._entry is not None:
                self._entry.checked_at = time.monotonic()
                if self._entry.stale_since is None:
                    self._entry.stale_since = time.monotonic()
        log.warning(
            "prompton: snapshot refresh failed (attempt %s, retrying in %.0fs), serving the "
            "cached document: %s",
            self._failures,
            delay,
            error,
        )

    # -- background --------------------------------------------------------

    def _revalidate_in_background(self) -> None:
        if not self._config.remote_enabled or self._config.poll:
            return
        with self._lock:
            if self._refreshing or time.monotonic() < self._not_before:
                return
            self._refreshing = True
        threading.Thread(
            target=self._background_refresh, name="prompton-refresh", daemon=True
        ).start()

    def _background_refresh(self) -> None:
        try:
            self.refresh(raise_errors=False)
        except Exception as error:  # noqa: BLE001 - a refresh may not escape into the app
            log.warning("prompton: background snapshot refresh failed: %s", error)
        finally:
            with self._lock:
                self._refreshing = False

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh(raise_errors=False)
            except Exception as error:  # noqa: BLE001 - the poll thread must never die
                log.warning("prompton: snapshot poll failed: %s", error)
            with self._lock:
                wait = max(self._not_before - time.monotonic(), self._config.cache_ttl)
            if self._stop.wait(wait):
                return

    # -- disk --------------------------------------------------------------

    def _write_disk(self, entry: SnapshotEntry) -> None:
        path = self._config.disk_cache_path
        if path is None:
            return
        meta = {
            "etag": entry.etag,
            "last_modified": entry.last_modified,
            "environment": entry.environment,
            "project": entry.project,
            "fetched_at": entry.fetched_at,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, entry.raw)
            _atomic_write(_sidecar_path(path), json.dumps(meta).encode("utf-8"))
        except OSError as error:
            log.warning("prompton: could not write the disk cache %s: %s", path, error)

    def export(self, path: str | os.PathLike[str]) -> Path:
        """Write the current document to ``path`` so it can be committed as a bundle."""
        entry = self.peek()
        if entry is None:
            raise SnapshotUnavailableError("no snapshot to export: fetch one first")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, entry.raw)
        _atomic_write(
            _sidecar_path(target),
            json.dumps(
                {
                    "etag": entry.etag,
                    "last_modified": entry.last_modified,
                    "environment": entry.environment,
                    "project": entry.project,
                    "fetched_at": entry.fetched_at,
                }
            ).encode("utf-8"),
        )
        return target

    def install(
        self, data: SnapshotData, *, raw: bytes | None = None, source_name: Source = "remote"
    ) -> None:
        """Put a document straight into memory. Used by test mode and by ``load_snapshot``."""
        with self._lock:
            self._entry = SnapshotEntry(
                data=data,
                raw=raw if raw is not None else b"",
                source=source_name,
                fetched_at=time.time(),
                checked_at=time.monotonic(),
            )
            self._failures = 0
            self._not_before = 0.0

    def clear(self) -> None:
        with self._lock:
            self._entry = None
            self._failures = 0
            self._not_before = 0.0


def _as_text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _as_epoch(value: Any, path: Path) -> float:
    """A sidecar written by another SDK may hold an ISO timestamp, or nothing we understand."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:  # pragma: no cover - the file was just read
        return time.time()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta.json")


def _read_sidecar(path: Path) -> dict[str, Any]:
    try:
        raw = _sidecar_path(path).read_bytes()
    except OSError:
        return {}
    try:
        meta = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _atomic_write(path: Path, content: bytes) -> None:
    """tmp + rename, so a reader never sees a half-written file."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{threading.get_ident()}")
    try:
        tmp.write_bytes(content)
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
