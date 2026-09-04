"""End-to-end tests against a real local HTTP server, through the real urllib transport.

The fixture server is only used by the live integration test. Everything that needs a *failing*
server - a refused connection, a 429 with ``Retry-After``, a 304 - is done here with a stub, so the
suite is hermetic and nobody's shared fixture has to be broken on purpose.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from prompton import PromptOn
from prompton.errors import TransportError, UseCaseDocumentUnavailableError
from prompton.http import UrllibTransport
from prompton.testing import make_use_case_document

DOCUMENT = make_use_case_document(
    project="stub",
    environment="production",
    greeting={
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say hello to {{ name }}."}],
    },
)
BODY = json.dumps(DOCUMENT).encode("utf-8")
ETAG = '"sha256-stub"'


class StubServer:
    """A tiny PromptOn stand-in whose behaviour each test scripts."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.lock = threading.Lock()
        self.snapshot_status = 200
        self.retry_after: str | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # silence the default stderr logging
                pass

            def _record(self, body=None):
                with outer.lock:
                    outer.calls.append(
                        {
                            "method": self.command,
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "body": json.loads(body) if body else None,
                        }
                    )

            def do_GET(self):  # BaseHTTPRequestHandler's interface
                self._record()
                if outer.snapshot_status == 429:
                    payload = json.dumps(
                        {"error": {"code": "rate_limited", "message": "slow down", "details": {}}}
                    ).encode()
                    self.send_response(429)
                    if outer.retry_after:
                        self.send_header("retry-after", outer.retry_after)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if self.headers.get("if-none-match") == ETAG:
                    self.send_response(304)
                    self.send_header("etag", ETAG)
                    self.send_header("content-length", "0")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("etag", ETAG)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)

            def do_POST(self):  # BaseHTTPRequestHandler's interface
                length = int(self.headers.get("content-length") or 0)
                self._record(self.rfile.read(length))
                payload = json.dumps({"accepted": 1, "duplicates": 0, "rejected": []}).encode()
                self.send_response(202)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self) -> StubServer:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


@pytest.fixture
def stub():
    with StubServer() as server:
        yield server


def client_for(host: str, tmp_path, **options) -> PromptOn:
    settings = {
        "api_key": "ptn_stub_key",
        "host": host,
        "project": "stub",
        "disk_cache": str(tmp_path / "cache.json"),
        "poll": False,
        "timeout": 2.0,
    }
    settings.update(options)
    return PromptOn(**settings)


def test_a_real_round_trip_resolves_renders_and_logs(stub, tmp_path):
    client = client_for(stub.host, tmp_path)
    try:
        use_case = client.use_case("greeting")
        assert use_case.messages({"name": "Ada"})[0]["content"] == "Say hello to Ada."
        use_case.track(lambda: "hello", variables={"name": "Ada"})
        stats = client.flush(timeout=5)
        assert stats.accepted == 1
        posts = [call for call in stub.calls if call["method"] == "POST"]
        assert len(posts) == 1
        assert posts[0]["path"] == "/api/v1/logs?environment=production"
        assert posts[0]["headers"]["authorization"] == "Bearer ptn_stub_key"
        assert posts[0]["headers"]["user-agent"].startswith("prompton-python/")
    finally:
        client.close(timeout=1)


def test_a_repoll_sends_if_none_match_and_gets_a_304(stub, tmp_path):
    client = client_for(stub.host, tmp_path, cache_ttl=0.0)
    try:
        client.use_case("greeting")
        assert client.refresh() is False  # 304
        gets = [call for call in stub.calls if call["method"] == "GET"]
        assert gets[1]["headers"]["if-none-match"] == ETAG
    finally:
        client.close(timeout=1)


def test_a_429_pauses_and_the_caller_never_sees_an_error(stub, tmp_path):
    client = client_for(stub.host, tmp_path, cache_ttl=0.0)
    try:
        client.use_case("greeting")
        stub.snapshot_status = 429
        stub.retry_after = "45"
        client._store.refresh(raise_errors=False)
        assert client.use_cases_info()["retry_after_seconds"] > 40

        before = len([call for call in stub.calls if call["method"] == "GET"])
        for _ in range(3):
            assert client.use_case("greeting").model == "openai/gpt-4o-mini"
        after = len([call for call in stub.calls if call["method"] == "GET"])
        assert after == before, "no request may be made before Retry-After has elapsed"
    finally:
        client.close(timeout=1)


def test_when_the_server_is_down_the_disk_cache_keeps_the_app_running(stub, tmp_path):
    warm = client_for(stub.host, tmp_path)
    warm.use_case("greeting")
    warm.close(timeout=1)

    dead_host = stub.host
    stub.server.shutdown()
    stub.server.server_close()

    cold = client_for(dead_host, tmp_path)
    try:
        use_case = cold.use_case("greeting")
        assert use_case.source == "disk"
        assert use_case.model == "openai/gpt-4o-mini"
    finally:
        cold.close(timeout=1)


def test_with_nothing_cached_a_dead_server_fails_with_a_clear_message(stub, tmp_path):
    dead_host = stub.host
    stub.server.shutdown()
    stub.server.server_close()
    client = client_for(dead_host, tmp_path)
    try:
        with pytest.raises(UseCaseDocumentUnavailableError) as error:
            client.use_case("greeting")
        assert "unreachable" in str(error.value)
    finally:
        client.close(timeout=1)


def test_the_transport_turns_a_refused_connection_into_a_transport_error():
    transport = UrllibTransport()
    with pytest.raises(TransportError):
        transport.request("GET", "http://127.0.0.1:1/api/v1/use-cases", headers={}, timeout=0.5)
