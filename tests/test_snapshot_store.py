"""Every caching rule the contract names, tested one at a time."""

from __future__ import annotations

import json
import time

import pytest

from prompton.config import Config
from prompton.errors import APIError, SnapshotUnavailableError
from prompton.http import HttpResponse
from prompton.store import SnapshotStore
from prompton.testing import make_snapshot

from .conftest import FakeTransport, json_response, transport_error

ETAG = '"sha256-abc"'


def snapshot_ok(document: dict, etag: str = ETAG) -> HttpResponse:
    return HttpResponse(
        status=200,
        body=json.dumps(document).encode("utf-8"),
        headers={"etag": etag, "last-modified": "Fri, 04 Sep 2026 00:21:48 GMT"},
    )


@pytest.fixture
def document() -> dict:
    return make_snapshot(
        project="demo",
        environment="production",
        greeting={"messages": [{"role": "user", "content": "hi"}]},
    )


def build(tmp_path, transport, **options) -> SnapshotStore:
    settings = {
        "api_key": "ptn_demo_key",
        "host": "http://localhost:4000",
        "project": "demo",
        "disk_cache": str(tmp_path / "snapshot.json"),
        "poll": False,
        "cache_ttl": 10.0,
    }
    settings.update(options)
    return SnapshotStore(Config.build(**settings), transport)


class TestTenSecondCache:
    def test_within_the_ttl_every_resolve_is_served_from_memory(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()

        for _ in range(10):
            assert store.current().data.project == "demo"
        assert len(transport.snapshot_requests) == 1

    def test_past_the_ttl_the_next_call_revalidates_in_the_background(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport, cache_ttl=0.01)
        store.start()
        store.current()

        transport.push(HttpResponse(status=304, headers={"etag": ETAG}))
        time.sleep(0.02)
        store.current()  # triggers the background refresh, returns immediately
        deadline = time.monotonic() + 2
        while len(transport.snapshot_requests) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(transport.snapshot_requests) == 2
        assert transport.snapshot_requests[1]["headers"]["if-none-match"] == ETAG

    def test_a_304_costs_nothing_and_keeps_the_document(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        first = store.current().data

        transport.push(HttpResponse(status=304, headers={"etag": ETAG}))
        assert store.refresh() is False
        assert store.current().data is first


class TestRateLimiting:
    def test_a_429_pauses_for_retry_after_and_keeps_serving(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport, cache_ttl=0.01)
        store.start()
        store.current()

        transport.push(
            HttpResponse(
                status=429,
                body=b'{"error":{"code":"rate_limited","message":"slow down","details":{}}}',
                headers={"retry-after": "30"},
            )
        )
        store.refresh(raise_errors=False)
        assert store.info()["retry_after_seconds"] > 25

        # the caller never sees an error, and no further request is made
        before = len(transport.snapshot_requests)
        for _ in range(5):
            assert store.current().data.project == "demo"
        store.refresh(raise_errors=False)
        assert len(transport.snapshot_requests) == before

    def test_retry_after_falls_back_to_the_error_details(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        store.current()

        transport.push(
            json_response(
                429,
                {
                    "error": {
                        "code": "rate_limited",
                        "message": "no",
                        "details": {"retry_after": 42},
                    }
                },
            )
        )
        store.refresh(raise_errors=False)
        assert 40 < store.info()["retry_after_seconds"] <= 42

    def test_failures_back_off_by_doubling_up_to_the_ceiling(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport, cache_ttl=10.0, max_backoff=300.0)
        store.start()
        store.current()

        delays = []
        for _ in range(7):
            transport.push(transport_error())
            store._not_before = 0.0  # skip the wait; we are measuring the schedule
            store.refresh(raise_errors=False)
            delays.append(round(store.info()["retry_after_seconds"]))
        assert delays[:5] == [10, 20, 40, 80, 160]
        assert delays[5] == delays[6] == 300


class TestServerDown:
    def test_the_previous_document_is_served_when_a_refresh_fails(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        store.current()

        transport.push(transport_error())
        store.refresh(raise_errors=False)
        assert store.current().data.project == "demo"
        assert store.info()["stale"] is True

    def test_a_cold_start_reads_the_disk_cache_written_by_an_earlier_run(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        first = build(tmp_path, transport)
        first.start()
        first.current()
        assert (tmp_path / "snapshot.json").exists()
        assert (tmp_path / "snapshot.json.meta.json").exists()

        offline = FakeTransport()
        offline.push(transport_error())
        second = build(tmp_path, offline)
        second.start()
        entry = second.current()
        assert entry.source == "disk"
        assert entry.etag == ETAG

    def test_the_bundle_is_used_when_memory_and_disk_are_empty(self, tmp_path, document):
        bundle = tmp_path / "bundle.json"
        bundle.write_text(json.dumps(document))
        transport = FakeTransport()
        transport.push(transport_error())
        config = Config.build(
            api_key="ptn_demo_key",
            host="http://localhost:4000",
            project="demo",
            disk_cache=str(tmp_path / "missing.json"),
            bundle=str(bundle),
            poll=False,
        )
        store = SnapshotStore(config, transport)
        store.start()
        assert store.current().source == "bundle"

    def test_with_nothing_cached_resolution_fails_with_a_clear_message(self, tmp_path):
        transport = FakeTransport()
        transport.push(transport_error())
        store = build(tmp_path, transport)
        store.start()
        with pytest.raises(SnapshotUnavailableError) as error:
            store.current()
        assert "unreachable" in str(error.value)
        assert "nothing is cached" in str(error.value)

    def test_a_corrupt_or_partial_file_is_ignored_not_raised(self, tmp_path, document):
        (tmp_path / "snapshot.json").write_text('{"schema_version": 3, "use_ca')
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        assert store.current().source == "remote"


class TestForeignSidecars:
    def test_a_sidecar_written_by_another_sdk_is_tolerated(self, tmp_path, document):
        """Several processes on one host share the file; not all of them are this SDK."""
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        (tmp_path / "snapshot.json.meta.json").write_text(
            json.dumps(
                {
                    "etag": ETAG,
                    "last_modified": None,
                    "environment": "production",
                    "project": "demo",
                    "fetched_at": "2026-09-04T01:10:30.115160Z",
                }
            )
        )
        transport = FakeTransport()
        transport.push(transport_error())
        store = build(tmp_path, transport)
        store.start()
        entry = store.current()
        assert entry.source == "disk"
        assert entry.etag == ETAG
        assert entry.fetched_at > 0

    def test_a_corrupt_sidecar_costs_the_metadata_not_the_snapshot(self, tmp_path, document):
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        (tmp_path / "snapshot.json.meta.json").write_text("{oh no")
        transport = FakeTransport()
        transport.push(transport_error())
        store = build(tmp_path, transport)
        store.start()
        entry = store.current()
        assert entry.source == "disk"
        assert entry.etag is None


class TestScopeGuard:
    def test_a_snapshot_for_another_environment_is_never_used(self, tmp_path, document):
        document["environment"] = "staging"
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        transport = FakeTransport()
        transport.push(transport_error())
        store = build(tmp_path, transport, environment="production")
        store.start()
        with pytest.raises(SnapshotUnavailableError):
            store.current()

    def test_a_snapshot_for_another_project_is_never_used(self, tmp_path, document):
        document["project"] = "someone-else"
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        transport = FakeTransport()
        transport.push(transport_error())
        store = build(tmp_path, transport)
        store.start()
        with pytest.raises(SnapshotUnavailableError):
            store.current()


class TestDiskWrites:
    def test_writes_are_atomic_and_leave_no_temporary_file(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        store.current()
        names = sorted(p.name for p in tmp_path.iterdir())
        assert names == ["snapshot.json", "snapshot.json.meta.json"]

    def test_the_sidecar_carries_the_etag_and_the_scope(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        store.current()
        meta = json.loads((tmp_path / "snapshot.json.meta.json").read_text())
        assert meta["etag"] == ETAG
        assert meta["environment"] == "production"
        assert meta["project"] == "demo"

    def test_export_writes_a_bundle_that_loads_back(self, tmp_path, document):
        transport = FakeTransport()
        transport.push(snapshot_ok(document))
        store = build(tmp_path, transport)
        store.start()
        store.current()
        exported = store.export(tmp_path / "bundle" / "snapshot.json")
        assert json.loads(exported.read_text())["project"] == "demo"


class TestModes:
    def test_offline_mode_never_touches_the_network(self, tmp_path, document):
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        transport = FakeTransport()
        config = Config.build(
            api_key="ptn_demo_key",
            host="http://localhost:4000",
            project="demo",
            disk_cache=str(tmp_path / "snapshot.json"),
            mode="offline",
        )
        store = SnapshotStore(config, transport)
        store.start()
        assert store.current().source == "disk"
        store.refresh(raise_errors=False)
        assert transport.requests == []

    def test_without_an_api_key_no_remote_call_is_made(self, tmp_path, document, caplog):
        (tmp_path / "snapshot.json").write_text(json.dumps(document))
        transport = FakeTransport()
        config = Config.build(
            api_key=None,
            host="http://localhost:4000",
            project="demo",
            disk_cache=str(tmp_path / "snapshot.json"),
        )
        store = SnapshotStore(config, transport)
        with caplog.at_level("WARNING"):
            store.start()
        assert transport.requests == []
        assert store.current().source == "disk"
        assert sum("no API key" in message for message in caplog.messages) == 1


class TestErrors:
    def test_a_configuration_mistake_is_raised_by_an_explicit_refresh(self, tmp_path):
        transport = FakeTransport()
        transport.push(
            json_response(401, {"error": {"code": "unauthorized", "message": "invalid key"}})
        )
        store = build(tmp_path, transport)
        with pytest.raises(APIError) as error:
            store.refresh()
        assert error.value.status == 401
        assert error.value.code == "unauthorized"
