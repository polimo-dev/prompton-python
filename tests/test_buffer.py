"""The monitoring-log buffer: every batching and retry rule the contract names."""

from __future__ import annotations

import time

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
    def test_a_429_resends_the_same_batch_with_the_same_ids(self):
        answers = [
            json_response(429, {"error": {"code": "rate_limited"}}, **{"retry-after": "0"}),
            json_response(202, ACCEPTED),
        ]
        transport = FakeTransport(
            lambda call: answers.pop(0) if answers else json_response(202, ACCEPTED)
        )
        buffer = build(transport)
        item = record()
        buffer.enqueue(item)
        drain(buffer)
        assert len(transport.generation_requests) == 2
        first, second = transport.generation_requests
        assert first["body"] == second["body"]
        assert second["body"]["generations"][0]["id"] == item["id"]

    def test_a_5xx_is_retried_and_a_transport_failure_too(self):
        answers = [
            json_response(503, {"error": {"code": "unavailable"}}, **{"retry-after": "0"}),
            transport_error(),
            json_response(202, ACCEPTED),
        ]
        transport = FakeTransport(lambda call: answers.pop(0))
        buffer = build(transport, max_backoff=0.01)
        buffer.enqueue(record())
        drain(buffer)
        assert buffer.stats.accepted == 1
        assert buffer.stats.send_failures == 2

    def test_a_413_batch_is_split_in_half(self):
        seen: list[int] = []

        def handler(call):
            size = len(call["body"]["generations"])
            seen.append(size)
            if size > 1:
                return json_response(413, {"error": {"code": "payload_too_large"}})
            return json_response(202, ACCEPTED)

        transport = FakeTransport(handler)
        buffer = build(transport)
        for _ in range(4):
            buffer.enqueue(record())
        drain(buffer)
        # 4 -> [2, 2]; each half is split again, depth first
        assert seen == [4, 2, 1, 1, 2, 1, 1]
        assert buffer.stats.accepted == 4

    def test_another_4xx_drops_the_batch_and_does_not_retry(self, caplog):
        transport = FakeTransport(
            lambda call: json_response(400, {"error": {"code": "invalid_request", "message": "no"}})
        )
        buffer = build(transport)
        for _ in range(3):
            buffer.enqueue(record())
        with caplog.at_level("ERROR"):
            drain(buffer)
        assert len(transport.generation_requests) == 1
        assert buffer.stats.dropped_client_error == 3
        assert sum("rejected a batch" in message for message in caplog.messages) == 1

    def test_after_the_attempt_limit_the_batch_is_dropped_and_counted(self):
        transport = FakeTransport(lambda call: transport_error())
        buffer = build(transport, max_send_attempts=3, max_backoff=0.001)
        buffer.enqueue(record())
        drain(buffer, timeout=3.0)
        assert buffer.stats.dropped_give_up == 1
        assert len(transport.generation_requests) == 3


class TestPartialAcceptance:
    def test_rejected_records_are_counted_and_never_resent(self):
        body = {
            "accepted": 1,
            "duplicates": 0,
            "rejected": [
                {"index": 0, "id": "x", "code": "invalid_request", "message": "id must be a UUID"}
            ],
        }
        transport = FakeTransport(lambda call: json_response(202, body))
        buffer = build(transport)
        buffer.enqueue(record())
        buffer.enqueue(record())
        drain(buffer)
        assert len(transport.generation_requests) == 1
        assert buffer.stats.rejected == 1
        assert buffer.stats.accepted == 1

    def test_duplicates_are_counted_on_a_resend(self):
        transport = FakeTransport(
            lambda call: json_response(202, {"accepted": 0, "duplicates": 2, "rejected": []})
        )
        buffer = build(transport)
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
