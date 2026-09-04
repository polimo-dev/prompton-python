"""The monitoring-log buffer: batching, retries, and the rules for giving up.

The contract in one paragraph. Records are batched on size, bytes or time, never one HTTP call per
generation. Each request carries at most 200 records and 5 MB, with the ``environment`` forced onto
the whole batch, so there is one batch per environment. Ids are app-generated UUIDv7 and are the
idempotency key: a resend is counted as a duplicate, never stored twice, which is why a retry
resends *the same batch with the same ids*. ``429`` and any ``5xx`` are retried (honouring
``Retry-After``, otherwise backing off ×2 from one second to five minutes, for a bounded number of
attempts); a ``413`` batch is split in half; every other ``4xx`` is dropped, counted and logged once
- retrying a rejected record only rejects it again. Partial acceptance is read from ``rejected``:
accepted records are never resent.

None of this is allowed to affect the app. ``log`` returns immediately, the queue is bounded and
drops the oldest with a counter, and a full queue or a dead server costs a counter, not an
exception.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from . import _fork
from ._json import canonical_json
from .config import Config
from .errors import TransportError
from .http import Transport, build_headers, parse_api_error, retry_after_seconds, urlencode

__all__ = ["MAX_BATCH_BYTES", "MAX_BATCH_RECORDS", "BufferStats", "LogBuffer"]

log = logging.getLogger("prompton")

MAX_BATCH_RECORDS = 200
MAX_BATCH_BYTES = 5_000_000
_ENVELOPE_OVERHEAD = 20  # {"generations":[]} plus a little room


@dataclass
class BufferStats:
    """Counters worth exporting to your own metrics."""

    enqueued: int = 0
    sent: int = 0
    accepted: int = 0
    duplicates: int = 0
    rejected: int = 0
    dropped_queue_full: int = 0
    dropped_too_large: int = 0
    dropped_invalid: int = 0
    dropped_client_error: int = 0
    dropped_give_up: int = 0
    dropped_no_remote: int = 0
    send_failures: int = 0
    batches_sent: int = 0  # batches the server accepted; a 429/5xx/413 is not one
    queued: int = 0  # everything still held: the queue, retried batches, the batch on the wire

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class _Item:
    record: dict[str, Any]
    size: int
    at: float


@dataclass(eq=False)  # identity, so a batch can be found in _in_flight by `is`
class _Batch:
    records: list[dict[str, Any]]
    sizes: list[int]
    attempts: int = 0

    @property
    def bytes(self) -> int:
        return sum(self.sizes)


class LogBuffer:
    """Queues monitoring logs and sends them in batches from a worker thread."""

    def __init__(
        self,
        config: Config,
        transport: Transport,
        *,
        sender: Callable[[list[dict[str, Any]]], Any] | None = None,
    ) -> None:
        self._config = config
        self._transport = transport
        self._sender = sender
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._queue: deque[_Item] = deque()
        self._queued_bytes = 0
        self._pending: deque[_Batch] = deque()  # split or retried batches, sent before the queue
        self._in_flight: list[_Batch] = []  # handed to _deliver: in neither queue nor pending
        self._not_before = 0.0
        self._force = False
        self._stop = False
        self._idle = threading.Event()
        self._idle.set()
        self._warned_client_error = False
        self._warned_no_remote = False
        self.stats = BufferStats()
        self._worker: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def _can_send(self) -> bool:
        return self._sender is not None or self._config.remote_enabled

    def start(self) -> None:
        """Start the worker thread. A no-op when nothing can be sent (test or offline mode)."""
        if self._worker is not None or self._config.mode == "test" or not self._can_send:
            return
        _fork.register(self)
        self._worker = threading.Thread(target=self._run, name="prompton-logs", daemon=True)
        self._worker.start()

    def close(self, timeout: float = 5.0) -> None:
        """Flush what is queued, best effort, then stop the worker.

        ``timeout`` is the whole budget: whatever the flush leaves is what the worker gets to
        finish the batch it is on.
        """
        deadline = time.monotonic() + timeout
        try:
            self.flush(timeout=timeout)
        finally:
            with self._lock:
                self._stop = True
                self._wake.notify_all()
            worker = self._worker
            self._worker = None
            if worker is not None and worker.is_alive():
                worker.join(timeout=max(deadline - time.monotonic(), 0.1))

    def _restart_after_fork(self) -> None:
        """Rebuild the inherited locks and restart the sender in a forked child.

        Records queued at fork time stay queued in both processes. Their ids are the idempotency
        key, so the copy that arrives second is counted as a duplicate rather than stored twice -
        which is the safe direction: losing the records would not be.
        """
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._idle = threading.Event()
        self._in_flight = []
        self._force = False
        if not self._queue and not self._pending:
            self._idle.set()
        worker, self._worker = self._worker, None
        if worker is None or self._stop:
            return
        self.start()

    # -- producing ---------------------------------------------------------

    def enqueue(self, record: dict[str, Any]) -> None:
        """Add one record. Returns immediately; failures are counted, never raised."""
        if not self._can_send:
            # offline, or no API key: no remote calls at all, so the record is dropped and counted
            self.stats.dropped_no_remote += 1
            if not self._warned_no_remote:
                self._warned_no_remote = True
                log.warning(
                    "prompton: no API key (or offline mode) - monitoring logs are dropped, not sent"
                )
            return
        try:
            size = len(canonical_json(record))
            json.dumps(record, allow_nan=False)
        except (TypeError, ValueError) as error:
            self.stats.dropped_invalid += 1
            log.warning("prompton: dropping a monitoring log that cannot be encoded: %s", error)
            return

        if size > MAX_BATCH_BYTES - _ENVELOPE_OVERHEAD:
            self.stats.dropped_too_large += 1
            log.warning(
                "prompton: dropping a %s-byte monitoring log, over the %s-byte request limit",
                size,
                MAX_BATCH_BYTES,
            )
            return

        with self._lock:
            while len(self._queue) >= self._config.max_queue:
                oldest = self._queue.popleft()
                self._queued_bytes -= oldest.size
                self.stats.dropped_queue_full += 1
                if self.stats.dropped_queue_full % 1000 == 1:
                    log.warning(
                        "prompton: the monitoring-log queue is full (%s records); "
                        "dropping the oldest",
                        self._config.max_queue,
                    )
            self._queue.append(_Item(record, size, time.monotonic()))
            self._queued_bytes += size
            self.stats.enqueued += 1
            self.stats.queued = self._backlog_locked()
            self._idle.clear()
            self._wake.notify_all()

    # -- consuming ---------------------------------------------------------

    def _backlog_locked(self) -> int:
        """Every record still held: queued, waiting out a retry, or on the wire right now."""
        return (
            len(self._queue)
            + sum(len(batch.records) for batch in self._pending)
            + sum(len(batch.records) for batch in self._in_flight)
        )

    def flush(self, timeout: float = 5.0) -> BufferStats:
        """Send everything queued now and wait for the result.

        This is the entry point for shutdown, tests and scripts. It waits for the batch already on
        the wire as well as for the queue, so the counters it returns describe what really
        happened. A pending ``Retry-After`` is honoured, so a flush during a rate-limit pause waits
        rather than hammering the server.
        """
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if not self._queue and not self._pending and not self._in_flight:
                    break
                self._force = True
                self._wake.notify_all()
                pause = self._not_before - time.monotonic()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    left = self._backlog_locked()
                log.warning(
                    "prompton: flush timed out with %s monitoring log(s) still queued", left
                )
                break
            if self._worker is not None:
                self._idle.wait(min(remaining, 0.05))
            elif pause > 0:
                # no worker to wait on, so this thread waits out the pause itself
                time.sleep(min(pause, remaining, 0.05))
            else:
                self._send_once()
        with self._lock:
            self._force = False
            self.stats.queued = self._backlog_locked()
        return self.stats

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stop:
                    return
                wait = self._delay_until_send_locked()
                if wait is None:
                    if not self._in_flight:
                        self._idle.set()
                    self._wake.wait(self._config.flush_interval)
                    continue
                if wait > 0:
                    self._wake.wait(wait)
                    continue
            self._send_once()

    def _delay_until_send_locked(self) -> float | None:
        """Seconds until the next send, ``0`` for now, ``None`` when there is nothing to do."""
        if not self._queue and not self._pending:
            return None
        now = time.monotonic()
        if now < self._not_before:
            return self._not_before - now
        if self._pending or self._force:
            return 0.0
        if (
            len(self._queue) >= self._config.flush_size
            or self._queued_bytes >= self._config.flush_bytes
        ):
            return 0.0
        return max(self._queue[0].at + self._config.flush_interval - now, 0.0)

    def _next_batch(self) -> _Batch | None:
        """Take the next batch, and hold it in ``_in_flight`` until the send is over.

        A batch that lives in neither the queue nor ``_pending`` is invisible to ``flush``, which
        is how a shutdown used to abandon the request it had just started.
        """
        with self._lock:
            if self._pending:
                batch = self._pending.popleft()
                self._in_flight.append(batch)
                return batch
            if not self._queue:
                return None
            records: list[dict[str, Any]] = []
            sizes: list[int] = []
            total = _ENVELOPE_OVERHEAD
            while self._queue and len(records) < MAX_BATCH_RECORDS:
                item = self._queue[0]
                if records and total + item.size + 1 > MAX_BATCH_BYTES:
                    break
                self._queue.popleft()
                self._queued_bytes -= item.size
                records.append(item.record)
                sizes.append(item.size)
                total += item.size + 1
            batch = _Batch(records, sizes)
            self._in_flight.append(batch)
            return batch

    def _release_locked(self, batch: _Batch) -> None:
        self._in_flight = [held for held in self._in_flight if held is not batch]

    def _send_once(self) -> None:
        batch = self._next_batch()
        if batch is None:
            self._mark_idle_if_empty()
            return
        try:
            if not batch.records:
                return
            batch.attempts += 1
            try:
                self._deliver(batch)
            except Exception as error:  # noqa: BLE001 - the worker thread must never die
                log.warning("prompton: unexpected error while sending monitoring logs: %s", error)
                self._retry(batch, None, error)
        finally:
            with self._lock:
                self._release_locked(batch)
            self._mark_idle_if_empty()

    def _mark_idle_if_empty(self) -> None:
        with self._lock:
            self.stats.queued = self._backlog_locked()
            if not self._queue and not self._pending and not self._in_flight:
                self._force = False
                self._idle.set()

    def _deliver(self, batch: _Batch) -> None:
        if self._sender is not None:
            self._sender(batch.records)
            self.stats.sent += len(batch.records)
            self.stats.accepted += len(batch.records)
            self.stats.batches_sent += 1
            self._clear_backoff()
            return

        query = urlencode({"environment": self._config.environment})
        url = f"{self._config.base_url}/generations?{query}"
        body = json.dumps({"generations": batch.records}, ensure_ascii=False).encode("utf-8")
        headers = build_headers(self._config.api_key, self._config.user_agent)
        headers["content-type"] = "application/json"
        try:
            response = self._transport.request(
                "POST", url, headers=headers, body=body, timeout=self._config.timeout
            )
        except TransportError as error:
            self._retry(batch, None, error)
            return

        if 200 <= response.status < 300:
            self._accepted(batch, response.json())
            return
        if response.status == 413:
            self._split(batch)
            return
        if response.status == 429 or response.status >= 500:
            self._retry(batch, retry_after_seconds(response), parse_api_error(response))
            return

        error = parse_api_error(response)
        self.stats.dropped_client_error += len(batch.records)
        if not self._warned_client_error:
            self._warned_client_error = True
            log.error(
                "prompton: the server rejected a batch of %s monitoring log(s) with %s (%s); "
                "dropping it and not retrying",
                len(batch.records),
                response.status,
                error.message or error.code,
            )
        self._clear_backoff()

    def _accepted(self, batch: _Batch, body: Any) -> None:
        self.stats.sent += len(batch.records)
        self.stats.batches_sent += 1
        if isinstance(body, dict):
            self.stats.accepted += int(body.get("accepted") or 0)
            self.stats.duplicates += int(body.get("duplicates") or 0)
            rejected = body.get("rejected")
            if isinstance(rejected, list) and rejected:
                self.stats.rejected += len(rejected)
                log.warning(
                    "prompton: %s monitoring log(s) were rejected and will not be resent: %s",
                    len(rejected),
                    rejected[:3],
                )
        else:
            self.stats.accepted += len(batch.records)
        self._clear_backoff()

    def _split(self, batch: _Batch) -> None:
        if len(batch.records) <= 1:
            self.stats.dropped_too_large += len(batch.records)
            log.error(
                "prompton: the server rejected a single %s-byte monitoring log with 413, "
                "dropping it",
                batch.bytes,
            )
            self._clear_backoff()
            return
        middle = len(batch.records) // 2
        first = _Batch(batch.records[:middle], batch.sizes[:middle])
        second = _Batch(batch.records[middle:], batch.sizes[middle:])
        log.warning(
            "prompton: a batch of %s monitoring log(s) was rejected with 413, splitting it in half",
            len(batch.records),
        )
        with self._lock:
            self._release_locked(batch)
            self._pending.appendleft(second)
            self._pending.appendleft(first)
            self._wake.notify_all()

    def _retry(self, batch: _Batch, retry_after: float | None, error: BaseException | None) -> None:
        self.stats.send_failures += 1
        if batch.attempts >= self._config.max_send_attempts:
            self.stats.dropped_give_up += len(batch.records)
            log.error(
                "prompton: giving up on %s monitoring log(s) after %s attempts: %s",
                len(batch.records),
                batch.attempts,
                error,
            )
            self._clear_backoff()
            return
        delay = retry_after
        if delay is None:
            delay = min(2.0 ** (batch.attempts - 1), self._config.max_backoff)
        with self._lock:
            self._release_locked(batch)
            self._pending.appendleft(batch)
            self._not_before = time.monotonic() + delay
            self._wake.notify_all()
        log.warning(
            "prompton: could not send %s monitoring log(s) (attempt %s), retrying in %.0fs: %s",
            len(batch.records),
            batch.attempts,
            delay,
            error,
        )

    def _clear_backoff(self) -> None:
        with self._lock:
            self._not_before = 0.0
