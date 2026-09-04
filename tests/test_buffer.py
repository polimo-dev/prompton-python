"""The monitoring-log buffer: every batching and retry rule the contract names."""

from __future__ import annotations

import threading
import time

import pytest

from prompton.buffer import MAX_BATCH_RECORDS, LogBuffer
from prompton.config import Config
from prompton.uuidv7 import uuid7

from .conftest import FakeTransport, json_response, transport_error

ACCEPTED = {"accepted": 1, "duplicates": 0, "rejected": []}


def record(**overrides):
    item = {
        "id": uuid7(),
        "use_case": "greeting",
        "model": "openai/gpt-4o-mini",
        "status": "ok",
        "started_at": "2026-09-04T00:00:00.000000Z",
    }
    item.update(overrides)
    return item


def build(transport, **options) -> LogBuffer:
    settings = {
        "api_key": "ptn_demo_key",
        "host": "http://localhost:4000",
        "disk_cache": False,
        "flush_interval": 0.02,
        "flush_size": 100,
    }
    settings.update(options)
    return LogBuffer(Config.build(**settings), transport)


def drain(buffer: LogBuffer, timeout: float = 2.0) -> None:
    buffer.flush(timeout=timeout)


@pytest.fixture
def started():
    """Build a buffer with its worker thread running - the code path that actually ships.

    The triggers are turned off so only ``flush`` releases a batch, which keeps the retry, split
    and partial-acceptance assertions deterministic.
    """
    made: list[LogBuffer] = []

    def make(transport, **options) -> LogBuffer:
        options.setdefault("flush_interval", 60.0)
        options.setdefault("flush_size", 1000)
        buffer = build(transport, **options)
        buffer.start()
        made.append(buffer)
        return buffer

    yield make
    for buffer in made:
        buffer.close(timeout=2.0)


def wait_for_request(transport: FakeTransport, count: int = 1, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while len(transport.generation_requests) < count and time.monotonic() < deadline:
        time.sleep(0.005)


class TestBatching:
    def test_records_go_out_in_one_request_not_one_per_generation(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport)
        for _ in range(50):
            buffer.enqueue(record())
        drain(buffer)
        assert len(transport.generation_requests) == 1
        assert len(transport.generation_requests[0]["body"]["generations"]) == 50

    def test_the_size_trigger_flushes_without_a_flush_call(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, flush_size=5, flush_interval=60.0)
        buffer.start()
        try:
            for _ in range(5):
                buffer.enqueue(record())
            deadline = time.monotonic() + 2
            while not transport.generation_requests and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(transport.generation_requests) == 1
        finally:
            buffer.close(timeout=1.0)

    def test_the_time_trigger_flushes_a_partial_batch(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, flush_size=1000, flush_interval=0.05)
        buffer.start()
        try:
            buffer.enqueue(record())
            deadline = time.monotonic() + 2
            while not transport.generation_requests and time.monotonic() < deadline:
                time.sleep(0.01)
            assert len(transport.generation_requests) == 1
        finally:
            buffer.close(timeout=1.0)

    def test_a_batch_never_carries_more_than_two_hundred_records(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport)
        for _ in range(450):
            buffer.enqueue(record())
        drain(buffer)
        sizes = [len(call["body"]["generations"]) for call in transport.generation_requests]
        assert sizes == [MAX_BATCH_RECORDS, MAX_BATCH_RECORDS, 50]

    def test_the_environment_is_forced_onto_the_whole_batch(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, environment="staging")
        buffer.enqueue(record())
        drain(buffer)
        assert "environment=staging" in transport.generation_requests[0]["url"]

    def test_a_single_record_over_the_request_limit_is_dropped_and_counted(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport)
        buffer.enqueue(record(input={"text": "x" * 5_000_001}))
        assert buffer.stats.dropped_too_large == 1
        drain(buffer)
        assert transport.generation_requests == []

    def test_an_unencodable_record_is_dropped_and_counted(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport)
        buffer.enqueue(record(metadata={"bad": object()}))
        assert buffer.stats.dropped_invalid == 1


class TestRetries:
    def test_a_429_resends_the_same_batch_with_the_same_ids(self, started):
        answers = [
            json_response(429, {"error": {"code": "rate_limited"}}, **{"retry-after": "0"}),
            json_response(202, ACCEPTED),
        ]
        transport = FakeTransport(
            lambda call: answers.pop(0) if answers else json_response(202, ACCEPTED)
        )
        buffer = started(transport)
        item = record()
        buffer.enqueue(item)
        drain(buffer)
        assert len(transport.generation_requests) == 2
        first, second = transport.generation_requests
        assert first["body"] == second["body"]
        assert second["body"]["generations"][0]["id"] == item["id"]

    def test_a_5xx_is_retried_and_a_transport_failure_too(self, started):
        answers = [
            json_response(503, {"error": {"code": "unavailable"}}, **{"retry-after": "0"}),
            transport_error(),
            json_response(202, ACCEPTED),
        ]
        transport = FakeTransport(lambda call: answers.pop(0))
        buffer = started(transport, max_backoff=0.01)
        buffer.enqueue(record())
        drain(buffer)
        assert buffer.stats.accepted == 1
        assert buffer.stats.send_failures == 2

    def test_a_413_batch_is_split_in_half(self, started):
        seen: list[int] = []

        def handler(call):
            size = len(call["body"]["generations"])
            seen.append(size)
            if size > 1:
                return json_response(413, {"error": {"code": "payload_too_large"}})
            return json_response(202, ACCEPTED)

        transport = FakeTransport(handler)
        buffer = started(transport)
        for _ in range(4):
            buffer.enqueue(record())
        drain(buffer)
        # 4 -> [2, 2]; each half is split again, depth first
        assert seen == [4, 2, 1, 1, 2, 1, 1]
        assert buffer.stats.accepted == 4

    def test_another_4xx_drops_the_batch_and_does_not_retry(self, started, caplog):
        transport = FakeTransport(
            lambda call: json_response(400, {"error": {"code": "invalid_request", "message": "no"}})
        )
        buffer = started(transport)
        for _ in range(3):
            buffer.enqueue(record())
        with caplog.at_level("ERROR"):
            drain(buffer)
        assert len(transport.generation_requests) == 1
        assert buffer.stats.dropped_client_error == 3
        assert sum("rejected a batch" in message for message in caplog.messages) == 1

    def test_after_the_attempt_limit_the_batch_is_dropped_and_counted(self, started):
        transport = FakeTransport(lambda call: transport_error())
        buffer = started(transport, max_send_attempts=3, max_backoff=0.001)
        buffer.enqueue(record())
        drain(buffer, timeout=3.0)
        assert buffer.stats.dropped_give_up == 1
        assert len(transport.generation_requests) == 3


class TestPartialAcceptance:
    def test_rejected_records_are_counted_and_never_resent(self, started):
        body = {
            "accepted": 1,
            "duplicates": 0,
            "rejected": [
                {"index": 0, "id": "x", "code": "invalid_request", "message": "id must be a UUID"}
            ],
        }
        transport = FakeTransport(lambda call: json_response(202, body))
        buffer = started(transport)
        buffer.enqueue(record())
        buffer.enqueue(record())
        drain(buffer)
        assert len(transport.generation_requests) == 1
        assert buffer.stats.rejected == 1
        assert buffer.stats.accepted == 1

    def test_duplicates_are_counted_on_a_resend(self, started):
        transport = FakeTransport(
            lambda call: json_response(202, {"accepted": 0, "duplicates": 2, "rejected": []})
        )
        buffer = started(transport)
        buffer.enqueue(record())
        drain(buffer)
        assert buffer.stats.duplicates == 2


class TestBoundedQueue:
    def test_the_oldest_records_are_dropped_and_counted(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, max_queue=10)
        ids = [uuid7() for _ in range(15)]
        for value in ids:
            buffer.enqueue(record(id=value))
        assert buffer.stats.dropped_queue_full == 5
        drain(buffer)
        sent = [item["id"] for item in transport.generation_requests[0]["body"]["generations"]]
        assert sent == ids[5:]


class TestShutdown:
    def test_close_flushes_what_is_queued(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, flush_interval=60.0)
        buffer.start()
        buffer.enqueue(record())
        buffer.close(timeout=2.0)
        assert len(transport.generation_requests) == 1

    def test_flush_waits_for_the_batch_already_on_the_wire(self, started):
        """The size trigger fires first in a busy app; flush must still wait for the answer."""
        answered = threading.Event()

        def handler(call):
            time.sleep(0.3)
            answered.set()
            return json_response(202, ACCEPTED)

        transport = FakeTransport(handler)
        buffer = started(transport, flush_size=1, flush_interval=0.01)
        buffer.enqueue(record())
        wait_for_request(transport)  # the worker has the batch; the queue is empty

        stats = buffer.flush(timeout=5.0)
        assert answered.is_set(), "flush returned while the batch was still on the wire"
        assert stats.accepted == 1
        assert stats.queued == 0

    def test_close_drains_a_batch_the_server_first_refused(self, started):
        answers = [json_response(500, {"error": {"code": "boom"}})]
        transport = FakeTransport(
            lambda call: answers.pop(0) if answers else json_response(202, ACCEPTED)
        )
        buffer = started(transport, flush_size=1, flush_interval=0.01, max_backoff=0.05)
        buffer.enqueue(record())
        wait_for_request(transport)

        buffer.close(timeout=5.0)
        assert len(transport.generation_requests) == 2, "the scheduled retry never ran"
        assert buffer.stats.accepted == 1


class TestFlushWithoutAWorker:
    def test_a_flush_without_a_worker_still_honours_retry_after(self):
        """``autostart=False`` and post-``close`` flushes send from the calling thread."""
        transport = FakeTransport(
            lambda call: json_response(
                429, {"error": {"code": "rate_limited"}}, **{"retry-after": "30"}
            )
        )
        buffer = build(transport)
        buffer.enqueue(record())

        started_at = time.monotonic()
        stats = buffer.flush(timeout=0.4)
        elapsed = time.monotonic() - started_at

        assert len(transport.generation_requests) == 1, "no request may go out inside the pause"
        assert 0.3 < elapsed < 5.0, "the flush waits out its budget instead of hammering"
        assert stats.send_failures == 1
        assert stats.dropped_give_up == 0
        assert stats.queued == 1, "the record is still buffered, not lost"


class TestCounters:
    def test_queued_counts_a_batch_waiting_out_a_retry(self):
        transport = FakeTransport(
            lambda call: json_response(
                503, {"error": {"code": "unavailable"}}, **{"retry-after": "30"}
            )
        )
        buffer = build(transport)
        for _ in range(3):
            buffer.enqueue(record())
        buffer.flush(timeout=0.2)
        assert buffer.stats.queued == 3

    def test_batches_sent_counts_only_what_the_server_accepted(self, started):
        answers = [json_response(503, {"error": {"code": "unavailable"}}, **{"retry-after": "0"})]
        transport = FakeTransport(
            lambda call: answers.pop(0) if answers else json_response(202, ACCEPTED)
        )
        buffer = started(transport, max_backoff=0.01)
        buffer.enqueue(record())
        drain(buffer)
        assert len(transport.generation_requests) == 2
        assert buffer.stats.batches_sent == 1


class TestNoRemote:
    def test_without_an_api_key_records_are_dropped_and_counted_not_sent(self, caplog):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, api_key=None)
        buffer.start()
        with caplog.at_level("WARNING"):
            for _ in range(3):
                buffer.enqueue(record())
        drain(buffer)
        assert transport.requests == []
        assert buffer.stats.dropped_no_remote == 3
        assert sum("monitoring logs are dropped" in message for message in caplog.messages) == 1

    def test_offline_mode_sends_nothing_either(self):
        transport = FakeTransport(lambda call: json_response(202, ACCEPTED))
        buffer = build(transport, mode="offline")
        buffer.enqueue(record())
        drain(buffer)
        assert transport.requests == []
        assert buffer.stats.dropped_no_remote == 1
