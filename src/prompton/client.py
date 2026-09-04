"""The client: resolve, render, log.

Three things happen here and nothing else. You ask which prompt, model and params to use; you
render that prompt with this call's variables; you send back what happened. Your provider key and
your HTTP client never leave your process - PromptOn is config-fetch, not a proxy, so it is never
in the request path and an outage costs you nothing but fresher configuration.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from ._version import SDK_NAME, VERSION
from .buffer import BufferStats, LogBuffer
from .config import Config
from .errors import NoTemplateError, ProviderError
from .generation import CallMeta, Outcome, build_record, iso_timestamp
from .http import Transport, UrllibTransport
from .payload import apply_policy
from .resolve_client import RemoteResolution, ResolveClient
from .resolver import Resolution
from .resolver import prompt_names as _prompt_names
from .resolver import resolve as _resolve
from .snapshot_data import SnapshotData
from .store import SnapshotStore
from .template import render as render_template
from .template import render_messages
from .uuidv7 import uuid7

__all__ = ["PromptOn"]

log = logging.getLogger("prompton")


class PromptOn:
    """A PromptOn client.

    One instance per process is normal; it is thread-safe and holds one background thread for
    snapshot polling and one for sending monitoring logs. Create it once, keep it, and call
    :meth:`close` (or use it as a context manager) at shutdown so the last logs are sent.

    >>> client = PromptOn(api_key="ptn_myproject_...")           # doctest: +SKIP
    >>> resolution = client.resolve("support_reply")             # doctest: +SKIP
    >>> messages = client.render(resolution, {"question": "..."})  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        config: Config | None = None,
        transport: Transport | None = None,
        autostart: bool = True,
        **options: Any,
    ) -> None:
        self.config = config or Config.build(**options)
        self._transport = transport or UrllibTransport()
        self._store = SnapshotStore(self.config, self._transport)
        self._resolve_client = ResolveClient(self.config, self._transport)
        self._buffer = LogBuffer(self.config, self._transport)
        self._captured: list[dict[str, Any]] = []
        self._captured_lock = threading.Lock()
        self._closed = False
        self._atexit_registered = False
        if autostart:
            self.start()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> PromptOn:
        """Load the local snapshot tiers and start the background threads."""
        self._store.start()
        self._buffer.start()
        if not self._atexit_registered:
            atexit.register(self._atexit_close)
            self._atexit_registered = True
        return self

    def close(self, timeout: float = 5.0) -> None:
        """Flush the monitoring logs, best effort, and stop the background threads."""
        if self._closed:
            return
        self._closed = True
        if self._atexit_registered:
            self._atexit_registered = False
            atexit.unregister(self._atexit_close)
        with contextlib.suppress(Exception):
            self._buffer.close(timeout=timeout)
        with contextlib.suppress(Exception):
            self._store.close()

    def _atexit_close(self) -> None:  # pragma: no cover - interpreter shutdown
        with contextlib.suppress(Exception):
            self.close(timeout=2.0)

    def __enter__(self) -> PromptOn:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- resolving ---------------------------------------------------------

    def resolve(self, use_case: str, prompt: str | None = None) -> Resolution:
        """Which prompt version, model and params to use for this call site.

        Served from memory within the cache TTL, with no HTTP call. Raises
        :class:`~prompton.errors.UnknownUseCaseError`, :class:`~prompton.errors.UnresolvedError`
        or :class:`~prompton.errors.UnknownPromptError` for a mistake in the app or the deployment,
        and :class:`~prompton.errors.SnapshotUnavailableError` only when PromptOn is unreachable
        *and* nothing is cached.
        """
        entry = self._store.current()
        return _resolve(
            entry.data,
            use_case,
            prompt,
            resolution_source=entry.source,
            etag=entry.etag,
        )

    def prompt_names(self, use_case: str) -> list[str]:
        """The prompt names the live deployment pins - exactly what ``resolve`` accepts."""
        return _prompt_names(self._store.current().data, use_case)

    def render(
        self, resolution: Resolution, variables: Mapping[str, Any] | None = None
    ) -> list[dict[str, Any]] | str:
        """Render the pinned prompt: a message list for ``chat``, a string for ``text``.

        Raises :class:`~prompton.errors.MissingVariableError` naming the variable that was not
        supplied, and :class:`~prompton.errors.NoTemplateError` for an embedding use case.
        """
        if resolution.kind == "chat" and resolution.messages is not None:
            return render_messages(resolution.messages, variables, resolution.engine)
        if resolution.kind == "text" and resolution.text_template is not None:
            return render_template(resolution.text_template, variables, resolution.engine)
        raise NoTemplateError(
            f"use case {resolution.use_case!r} of kind {resolution.kind!r} has no prompt template"
        )

    def resolve_remote(
        self,
        use_case: str,
        *,
        prompt: str | None = None,
        variables: Mapping[str, Any] | None = None,
        environment: str | None = None,
        render_locally: bool = True,
    ) -> RemoteResolution:
        """Resolve through ``POST /resolve`` instead of the local snapshot.

        The simple path and the smoke test. The raw answer is cached for the same TTL as the
        snapshot and rendered locally, so this stays cheap when you call it repeatedly. Pass
        ``render_locally=False`` to let the server render instead - a request every time, and the
        exact reference behaviour.
        """
        return self._resolve_client.resolve(
            use_case,
            prompt=prompt,
            variables=variables,
            environment=environment,
            render_locally=render_locally,
        )

    # -- snapshot ----------------------------------------------------------

    def refresh(self, *, force: bool = False) -> bool:
        """Fetch the snapshot once, now, and wait. ``True`` when a new document was installed.

        An active ``Retry-After`` pause is honoured here too, so calling this from a readiness
        probe cannot keep a rate-limited server busy; ``force=True`` overrides it.
        """
        return self._store.refresh(raise_errors=True, force=force)

    def snapshot(self) -> SnapshotData:
        """The decoded document currently in use."""
        return self._store.current().data

    def snapshot_info(self) -> dict[str, Any]:
        """Where the current document came from and how fresh it is."""
        return self._store.info()

    def export_snapshot(self, path: str | os.PathLike[str]) -> Path:
        """Write the current document to ``path``, to be committed as a bundle."""
        return self._store.export(path)

    def load_snapshot(
        self, source: Mapping[str, Any] | SnapshotData | str | os.PathLike[str]
    ) -> None:
        """Put a document straight into memory: a mapping, a decoded snapshot, or a JSON file."""
        if isinstance(source, SnapshotData):
            self._store.install(source, source_name="manual")
            return
        if isinstance(source, Mapping):
            self._store.install(SnapshotData.from_mapping(source), source_name="manual")
            return
        raw = Path(source).read_bytes()
        self._store.install(SnapshotData.from_json(raw), raw=raw, source_name="bundle")

    # -- monitoring logs ---------------------------------------------------

    def generation_id(self) -> str:
        """A UUIDv7 to use as a record id, issued before the provider call."""
        return uuid7()

    def log(
        self,
        record: Mapping[str, Any],
        *,
        resolution: Resolution | None = None,
        policy: Any = None,
    ) -> str:
        """Enqueue one monitoring log the app built itself, and return its id.

        Returns immediately. ``id`` (a UUIDv7), ``sdk`` and ``started_at`` are filled in when
        absent, and passing ``resolution`` fills in ``resolution_source`` and the deployment and
        prompt evidence. Raises ``ValueError`` when a required field is missing - that is a bug in
        the calling code, worth catching straight away; everything after this point (the network,
        the server, a full queue) is counted, never raised.
        """
        item = dict(record)
        item.setdefault("id", uuid7())
        item.setdefault("started_at", iso_timestamp())
        item.setdefault("sdk", {"name": SDK_NAME, "version": VERSION})

        if resolution is not None:
            item.setdefault("use_case", resolution.use_case)
            item.setdefault("kind", resolution.kind)
            item.setdefault("model", resolution.model)
            item.setdefault("resolution_source", resolution.resolution_source)
            for key, value in (
                ("deployment_id", resolution.deployment_id),
                ("deployment_revision", resolution.deployment_revision),
                ("prompt", resolution.prompt),
                ("prompt_version_id", resolution.prompt_version_id),
                ("model_id", resolution.model_id),
                ("provider", resolution.provider),
            ):
                if value is not None:
                    item.setdefault(key, value)

        missing = [
            name
            for name in ("use_case", "model", "status", "started_at")
            if item.get(name) in (None, "")
        ]
        if missing:
            raise ValueError(
                f"a monitoring log needs {', '.join(missing)}; pass resolution= to fill in "
                "use_case and model from a Resolution"
            )
        if item["status"] not in ("ok", "error"):
            raise ValueError(f"status must be 'ok' or 'error', got {item['status']!r}")

        self._enqueue(item, policy if policy is not None else self._policy_for(item, resolution))
        return str(item["id"])

    def _policy_for(self, item: Mapping[str, Any], resolution: Resolution | None) -> Any:
        if resolution is not None:
            return resolution.payload_policy
        entry = self._store.peek()
        if entry is None:
            return None
        use_case = entry.data.use_cases.get(str(item.get("use_case")))
        return use_case.payload_policy if use_case else None

    def _enqueue(self, item: dict[str, Any], policy: Any) -> None:
        try:
            prepared = apply_policy(
                item,
                policy,
                defaults=self.config.payload_defaults,
                hash_end_user=self.config.hash_end_user,
                redact=self.config.redact,
            )
        except Exception as error:  # noqa: BLE001 - monitoring must not break the app
            log.warning("prompton: could not prepare a monitoring log: %s", error)
            return
        if self.config.mode == "test":
            with self._captured_lock:
                self._captured.append(prepared)
            return
        self._buffer.enqueue(prepared)

    def flush(self, timeout: float = 5.0) -> BufferStats:
        """Send everything queued now and wait for the result. Returns the buffer counters."""
        return self._buffer.flush(timeout=timeout)

    @property
    def stats(self) -> BufferStats:
        """Monitoring-log counters: enqueued, sent, accepted, duplicates, rejected, dropped."""
        return self._buffer.stats

    # -- the wrapper -------------------------------------------------------

    def with_generation(
        self,
        resolution: Resolution,
        call: Callable[[], Any],
        *,
        id: str | None = None,
        variables: Mapping[str, Any] | None = None,
        input_messages: Any = None,
        input_text: str | None = None,
        end_user_ref: Any = None,
        trace_id: str | None = None,
        sequence: int | None = None,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run ``call``, time it, and log what happened. Returns whatever ``call`` returned.

        Return an :class:`~prompton.generation.Outcome` (or a mapping in the same shape, or the
        completion text) to record usage, cost and the stop reason. Raise
        :class:`~prompton.errors.ProviderError` for a typed failure - pass ``outcome=`` on it when
        the provider did answer and the usage is still worth keeping, as with a parse failure. Any
        exception is logged as ``status: error`` and then **re-raised unchanged**.
        """
        meta = CallMeta(
            id=id or uuid7(),
            variables=variables,
            input_messages=input_messages,
            input_text=input_text,
            end_user_ref=end_user_ref,
            trace_id=trace_id,
            sequence=sequence,
            context=context or {},
            metadata=metadata or {},
            params=params,
        )
        started_at = iso_timestamp()
        started = time.monotonic()
        try:
            result = call()
        except ProviderError as error:
            self._log_from_wrapper(
                resolution,
                meta,
                started_at,
                started,
                status="error",
                outcome=Outcome.coerce(error.outcome),
                error=error,
            )
            raise
        except BaseException as error:
            self._log_from_wrapper(
                resolution, meta, started_at, started, status="error", outcome=None, error=error
            )
            raise
        self._log_from_wrapper(
            resolution,
            meta,
            started_at,
            started,
            status="ok",
            outcome=Outcome.coerce(result),
            error=None,
        )
        return result

    @contextlib.contextmanager
    def track(self, resolution: Resolution, **meta: Any) -> Iterator[dict[str, Any]]:
        """Context-manager form of :meth:`with_generation`.

        >>> with client.track(resolution, variables=variables) as call:   # doctest: +SKIP
        ...     answer = my_provider(...)
        ...     call["outcome"] = Outcome(content=answer.text, finish_reason=answer.stop)
        """
        slot: dict[str, Any] = {"outcome": None}
        started_at = iso_timestamp()
        started = time.monotonic()
        call_meta = CallMeta(
            id=meta.get("id") or uuid7(),
            variables=meta.get("variables"),
            input_messages=meta.get("input_messages"),
            input_text=meta.get("input_text"),
            end_user_ref=meta.get("end_user_ref"),
            trace_id=meta.get("trace_id"),
            sequence=meta.get("sequence"),
            context=meta.get("context") or {},
            metadata=meta.get("metadata") or {},
            params=meta.get("params"),
        )
        try:
            yield slot
        except ProviderError as error:
            self._log_from_wrapper(
                resolution,
                call_meta,
                started_at,
                started,
                status="error",
                outcome=Outcome.coerce(error.outcome or slot.get("outcome")),
                error=error,
            )
            raise
        except BaseException as error:
            self._log_from_wrapper(
                resolution,
                call_meta,
                started_at,
                started,
                status="error",
                outcome=Outcome.coerce(slot.get("outcome")),
                error=error,
            )
            raise
        self._log_from_wrapper(
            resolution,
            call_meta,
            started_at,
            started,
            status="ok",
            outcome=Outcome.coerce(slot.get("outcome")),
            error=None,
        )

    def _log_from_wrapper(
        self,
        resolution: Resolution,
        meta: CallMeta,
        started_at: str,
        started: float,
        *,
        status: str,
        outcome: Outcome | None,
        error: BaseException | None,
    ) -> None:
        try:
            record = build_record(
                resolution,
                meta,
                status=status,
                started_at=started_at,
                latency_ms=int((time.monotonic() - started) * 1000),
                outcome=outcome,
                error=error,
            )
        except Exception as failure:  # noqa: BLE001 - monitoring must not break the app
            log.warning("prompton: could not build a monitoring log: %s", failure)
            return
        self._enqueue(record, resolution.payload_policy)

    # -- test mode ---------------------------------------------------------

    @property
    def captured(self) -> list[dict[str, Any]]:
        """In test mode, the monitoring logs that would have been sent."""
        with self._captured_lock:
            return list(self._captured)

    def clear_captured(self) -> None:
        with self._captured_lock:
            self._captured.clear()
