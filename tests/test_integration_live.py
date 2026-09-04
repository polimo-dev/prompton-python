"""Live integration tests against a real PromptOn server.

Skipped unless ``PTN_API_KEY`` is set::

    PTN_HOST=http://localhost:4000 PTN_API_KEY=ptn_sdkfixture_... \\
        python -m pytest tests/test_integration_live.py -v

They expect the ``sdkfixture`` project: use cases ``greeting`` (chat, prompt names ``default`` and
``ko``), ``summarize`` (text) and ``embed`` (embedding), in the ``production`` and ``staging``
environments. The point of these tests is the one thing a stub cannot prove: that **local
resolution agrees with the server's own prompt endpoint**, byte for byte.
"""

from __future__ import annotations

import os

import pytest

from prompton import PromptOn
from prompton.errors import APIError, MissingVariableError, UnknownPromptError, UnknownUseCaseError
from prompton.uuidv7 import uuid7

pytestmark = pytest.mark.skipif(
    not os.environ.get("PTN_API_KEY"),
    reason="set PTN_API_KEY (and PTN_HOST) to run the live integration tests",
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    instance = PromptOn(
        host=os.environ.get("PTN_HOST", "http://localhost:4000"),
        api_key=os.environ["PTN_API_KEY"],
        disk_cache=str(tmp_path_factory.mktemp("prompton") / "snapshot.json"),
        poll=False,
        timeout=10.0,
    )
    yield instance
    instance.close(timeout=5.0)


class TestSnapshot:
    def test_the_snapshot_fetch_carries_an_etag_and_the_expected_use_cases(self, client):
        assert client.refresh() is True
        info = client.use_cases_info()
        assert info["source"] == "remote"
        assert info["etag"].startswith('"sha256-')
        assert info["environment"] == "production"
        snapshot = client.use_cases()
        assert {"greeting", "summarize", "embed"} <= set(snapshot.use_cases)

    def test_a_repoll_answers_304_and_changes_nothing(self, client):
        client.refresh()
        before = client.use_cases_info()["etag"]
        assert client.refresh() is False, "an unchanged snapshot must answer 304"
        assert client.use_cases_info()["etag"] == before

    def test_the_disk_cache_is_written_and_reused_by_a_cold_client(self, client, tmp_path):
        client.refresh()
        cache = client.config.disk_cache_path
        assert cache.exists() and cache.stat().st_size > 0

        cold = PromptOn(
            host="http://127.0.0.1:1",  # nothing listens here
            api_key=os.environ["PTN_API_KEY"],
            disk_cache=str(cache),
            poll=False,
            timeout=1.0,
        )
        try:
            assert cold.use_case("greeting").source == "disk"
        finally:
            cold.close(timeout=1.0)


class TestLocalResolutionMatchesTheServer:
    def _compare(self, client, use_case, prompt=None, variables=None, environment=None):
        local = client.use_case(use_case, prompt)
        remote = client.filled_prompt(
            use_case, prompt=prompt, variables=variables, environment=environment
        )
        assert local.deployment["id"] == remote.deployment["id"]
        assert local.deployment["revision"] == remote.deployment["revision"]
        assert local.model == remote.model
        assert local.model_id == remote.model_id
        assert local.provider == remote.provider
        assert local.params == remote.params
        assert local.provider_options == remote.provider_options
        assert local.prompt == remote.prompt
        assert list(local.prompt_names) == remote.prompt_names
        assert local.prompt_version == remote.prompt_version
        assert local.kind == remote.kind
        return local, remote

    def test_greeting_default(self, client):
        local, remote = self._compare(client, "greeting", variables={"name": "Ada"})
        assert local.messages({"name": "Ada"}) == remote.messages

    def test_greeting_ko(self, client):
        local, remote = self._compare(client, "greeting", prompt="ko", variables={"name": "아다"})
        assert local.messages({"name": "아다"}) == remote.messages

    def test_summarize_is_a_text_use_case(self, client):
        local, remote = self._compare(
            client, "summarize", variables={"items": ["alpha", "beta", "gamma"]}
        )
        assert local.kind == "text"
        assert local.text({"items": ["alpha", "beta", "gamma"]}) == remote.text

    def test_embed_resolves_the_model_only(self, client):
        local, remote = self._compare(client, "embed")
        assert local.kind == "embedding"
        assert local.prompt is None and remote.prompt is None
        assert remote.prompt_version is None
        assert remote.prompt_names == []

    def test_the_raw_template_comes_back_unrendered_without_variables(self, client):
        remote = client.filled_prompt("greeting")
        assert "{{ name }}" in remote.messages[-1]["content"]


class TestErrorsMatchTheServer:
    def test_an_unknown_use_case(self, client):
        with pytest.raises(UnknownUseCaseError):
            client.use_case("does_not_exist")
        with pytest.raises(APIError) as error:
            client.filled_prompt("does_not_exist")
        assert error.value.status == 404
        assert error.value.details["key"] == "does_not_exist"

    def test_an_unpinned_prompt_name(self, client):
        with pytest.raises(UnknownPromptError) as local_error:
            client.use_case("greeting", "fr")
        with pytest.raises(APIError) as remote_error:
            client.filled_prompt("greeting", prompt="fr")
        assert remote_error.value.status == 404
        assert remote_error.value.details["reason"] == "unknown_prompt"
        assert local_error.value.prompt_names == remote_error.value.details["prompt_names"]

    def test_a_missing_variable(self, client):
        with pytest.raises(MissingVariableError) as local_error:
            client.use_case("greeting").messages({})
        # filled_prompt renders locally, so it fails the same way...
        with pytest.raises(MissingVariableError):
            client.filled_prompt("greeting", variables={})
        # ...and the server itself answers 400 with the variable it wanted
        with pytest.raises(APIError) as remote_error:
            client.filled_prompt("greeting", variables={}, render_locally=False)
        assert remote_error.value.status == 400
        assert local_error.value.variable == remote_error.value.details["missing_variable"]

    def test_an_unknown_environment(self, client):
        with pytest.raises(APIError) as error:
            client.filled_prompt("greeting", environment="nope")
        assert error.value.status == 404
        assert error.value.details["environment"] == "nope"

    def test_an_invalid_key_is_401(self, client):
        broken = PromptOn(
            host=client.config.host,
            api_key="ptn_sdkfixture_wrong",
            disk_cache=False,
            poll=False,
            timeout=10.0,
        )
        try:
            with pytest.raises(APIError) as error:
                broken.refresh()
            assert error.value.status == 401
            assert error.value.code == "unauthorized"
        finally:
            broken.close(timeout=1.0)


class TestEnvironments:
    def test_staging_is_a_request_parameter_not_a_property_of_the_key(self, client, tmp_path):
        staging = PromptOn(
            host=client.config.host,
            api_key=os.environ["PTN_API_KEY"],
            environment="staging",
            disk_cache=str(tmp_path / "staging.json"),
            poll=False,
            timeout=10.0,
        )
        try:
            assert staging.refresh() is True
            assert staging.use_cases().environment == "staging"
            local = staging.use_case("greeting")
            remote = staging.filled_prompt("greeting", variables={"name": "Ada"})
            assert local.deployment["id"] == remote.deployment["id"]
            assert local.deployment["id"] != client.use_case("greeting").deployment["id"]
        finally:
            staging.close(timeout=1.0)


class TestGenerations:
    def test_a_batch_is_accepted_and_a_resend_is_absorbed_as_duplicates(self, client):
        use_case = client.use_case("greeting")
        messages = use_case.messages({"name": "Ada"})
        ids = [uuid7(), uuid7()]

        client.log(
            {
                "id": ids[0],
                "status": "ok",
                "finish_reason": "stop",
                "stop_kind": "stop",
                "latency_ms": 842,
                "input": {"variables": {"name": "Ada"}, "messages": messages},
                "output": {"content": "Hello, Ada!"},
                "usage": {
                    "input_tokens": 38,
                    "output_tokens": 6,
                    "cost_usd": 1.2e-05,
                    "cost_source": "provider",
                },
                "trace_id": "sdk-python:1",
                "sequence": 1,
                "end_user_ref": "user-42",
                "metadata": {"source": "prompton-python integration test"},
            },
            use_case=use_case,
        )
        client.log(
            {
                "id": ids[1],
                "status": "error",
                "latency_ms": 1503,
                "error": {
                    "kind": "rate_limited",
                    "status": 429,
                    "message": "rate limited by upstream provider",
                },
                "trace_id": "sdk-python:2",
                "sequence": 2,
            },
            use_case=use_case,
        )

        stats = client.flush(timeout=15)
        assert stats.accepted == 2, f"the server rejected records: {stats.as_dict()}"
        assert stats.rejected == 0

        # the id is the idempotency key: resending stores nothing new
        before = stats.duplicates
        for record_id, status in ((ids[0], "ok"), (ids[1], "error")):
            client.log(
                {
                    "id": record_id,
                    "status": status,
                    "error": None if status == "ok" else {"kind": "rate_limited"},
                },
                use_case=use_case,
            )
        stats = client.flush(timeout=15)
        assert stats.duplicates == before + 2
        assert stats.rejected == 0

    def test_a_malformed_record_is_rejected_by_index_and_the_rest_is_kept(self, client):
        use_case = client.use_case("greeting")
        before = client.stats.rejected
        client.log({"id": "not-a-uuid", "status": "ok"}, use_case=use_case)
        client.log({"id": uuid7(), "status": "ok"}, use_case=use_case)
        stats = client.flush(timeout=15)
        assert stats.rejected == before + 1
        assert stats.accepted >= 1

    def test_the_wrapper_sends_a_complete_record(self, client):
        use_case = client.use_case("greeting")
        before = client.stats.accepted
        rejected_before = client.stats.rejected
        use_case.track(
            lambda: "Hello, Ada!",
            variables={"name": "Ada"},
            input_messages=use_case.messages({"name": "Ada"}),
            trace_id="sdk-python:wrapper",
        )
        stats = client.flush(timeout=15)
        assert stats.accepted == before + 1
        assert stats.rejected == rejected_before
