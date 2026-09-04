"""What happens to the background threads when a prefork server forks.

``gunicorn --preload``, uWSGI and any other prefork runtime build the app - and usually the client -
in a parent process and then ``fork()``. Threads do not survive that call, so without the at-fork
handler a worker would keep serving the snapshot it inherited for the rest of its life and would
never send a monitoring log on the time trigger.

The child writes one byte to a pipe and leaves with ``os._exit`` so it never runs pytest's
teardown.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from prompton.buffer import LogBuffer
from prompton.config import Config
from prompton.errors import TransportError
from prompton.http import HttpResponse
from prompton.store import SnapshotStore
from prompton.testing import make_use_case_document

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="fork() is POSIX only")

DOCUMENT = make_use_case_document(
    project="demo",
    environment="production",
    greeting={"messages": [{"role": "user", "content": "hi"}]},
)
BODY = json.dumps(DOCUMENT).encode("utf-8")


class CountingTransport:
    """Deliberately lock-free: a lock held by another thread at fork time deadlocks the child."""

    def __init__(self, *, fail: bool = False) -> None:
        self.count = 0
        self.fail = fail

    def request(self, method, url, *, headers, body=None, timeout=5.0) -> HttpResponse:
        self.count += 1
        if self.fail:
            raise TransportError("could not reach PromptOn")
        return HttpResponse(status=200, body=BODY, headers={"etag": '"sha256-fork"'})


def run_in_child(work) -> bool:
    """Run ``work()`` in a forked child and return what it reported."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports coverage
        code = 1
        try:
            os.close(read_fd)
            code = 0 if work() else 1
            os.write(write_fd, b"1" if code == 0 else b"0")
        except BaseException:  # noqa: BLE001 - the child reports, it never raises
            os.write(write_fd, b"0")
        finally:
            os._exit(code)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        answer = pipe.read(1)
    _, status = os.waitpid(pid, 0)
    return answer == b"1" and os.waitstatus_to_exitcode(status) == 0


def test_a_forked_child_keeps_refreshing_its_snapshot(tmp_path):
    transport = CountingTransport()
    config = Config.build(
        api_key="ptn_demo_key",
        host="http://localhost:4000",
        project="demo",
        disk_cache=str(tmp_path / "snapshot.json"),
        poll=True,
        cache_ttl=0.05,
    )
    store = SnapshotStore(config, transport)
    store.start()
    store.current()

    def work() -> bool:
        before = transport.count
        deadline = time.monotonic() + 3
        while transport.count <= before and time.monotonic() < deadline:
            store.current()
            time.sleep(0.02)
        return transport.count > before

    try:
        assert run_in_child(work), "the forked worker never refreshed its snapshot again"
    finally:
        store.close()


def test_a_forked_child_still_sends_its_monitoring_logs(tmp_path):
    sent: list[int] = []
    config = Config.build(
        api_key="ptn_demo_key",
        host="http://localhost:4000",
        disk_cache=False,
        flush_interval=0.05,
        flush_size=1000,
    )
    buffer = LogBuffer(
        config, CountingTransport(), sender=lambda records: sent.append(len(records))
    )
    buffer.start()

    def work() -> bool:
        buffer.enqueue(
            {
                "id": "0198f2a1-1111-7000-8000-0000000000ff",
                "use_case": "greeting",
                "model": "openai/gpt-4o-mini",
                "status": "ok",
                "started_at": "2026-09-04T09:00:00.000000Z",
            }
        )
        # no flush(): only a live worker thread can honour the time trigger
        deadline = time.monotonic() + 3
        while not sent and time.monotonic() < deadline:
            time.sleep(0.02)
        return bool(sent)

    try:
        assert run_in_child(work), "the forked worker never sent its monitoring logs"
    finally:
        buffer.close(timeout=1.0)
