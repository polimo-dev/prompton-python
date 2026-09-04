from __future__ import annotations

import json

import pytest

import prompton
from prompton import PromptOn, ProviderError, Result
from prompton.errors import MissingVariableError, NoTemplateError, UnknownUseCaseError
from prompton.testing import make_use_case_document

from .conftest import FakeTransport, json_response


@pytest.fixture
def client(snapshot_document):
    instance = PromptOn(mode="test", api_key=None, disk_cache=False, host="http://localhost:4000")
    instance.load_use_cases(snapshot_document)
    yield instance
    instance.close(timeout=0.1)


class TestResolveAndRender:
    def test_use_case_then_messages(self, client):
        use_case = client.use_case("greeting")
        assert use_case.key == "greeting"
        assert use_case.model == "openai/gpt-4o-mini"
        assert use_case.params == {"temperature": 0.2, "max_tokens": 256}
        assert use_case.messages({"name": "Ada"})[1]["content"] == "Say hello to Ada."

    def test_a_named_prompt_is_how_language_branching_works(self, client):
        use_case = client.use_case("greeting")
        assert (
            use_case.messages({"name": "아다"}, prompt="ko")[0]["content"] == "아다님에게 인사해줘."
        )

    def test_text_and_embedding_kinds(self, client):
        text = client.use_case("summarize").text({"items": ["a", "b"]})
        assert text == "Summarize:\n- a\n- b\n"
        with pytest.raises(NoTemplateError):
            client.use_case("embed").messages({})
        with pytest.raises(NoTemplateError):
            client.use_case("greeting").text({"name": "Ada"})

    def test_prompt_names_lists_exactly_what_resolve_accepts(self, client):
        assert client.prompt_names("greeting") == ["default", "ko"]
        for name in client.prompt_names("greeting"):
            assert client.use_case("greeting", name).prompt == name

    def test_an_unknown_use_case_is_an_error_not_a_fallback(self, client):
        with pytest.raises(UnknownUseCaseError):
            client.use_case("nope")


class TestWithGeneration:
    def test_a_successful_call_is_logged_with_the_resolution_evidence(self, client):
        use_case = client.use_case("greeting")
        result = use_case.track(
            lambda: Result(
                content="Hello, Ada!",
                finish_reason="stop",
                input_tokens=38,
                output_tokens=6,
                cost_usd=1.2e-05,
                cost_source="provider",
                model_used="openai/gpt-4o-mini",
                upstream_provider="OpenAI",
            ),
            variables={"name": "Ada"},
            input_messages=use_case.messages({"name": "Ada"}),
            trace_id="job:1",
            sequence=1,
            end_user_ref="user-42",
            context={"plan": "pro"},
            metadata={"job_id": 1},
        )
        assert result.content == "Hello, Ada!"
        [logged] = client.captured
        assert logged["status"] == "ok"
        assert logged["stop_kind"] == "stop"
        assert logged["deployment_id"] == use_case.deployment["id"]
        assert logged["prompt_version_id"] == use_case.prompt_version["id"]
        assert logged["source"] == "manual"
        assert logged["usage"]["cost_source"] == "provider"
        assert logged["input"]["variables"] == {"name": "Ada"}
        assert logged["output"]["content"] == "Hello, Ada!"
        assert logged["context"] == {"plan": "pro"}
        assert logged["sdk"] == {"name": "prompton-python", "version": prompton.VERSION}
        assert isinstance(logged["latency_ms"], int)

    def test_named_prompt_render_updates_following_track_evidence(self, client):
        use_case = client.use_case("greeting")

        assert (
            use_case.messages({"name": "아다"}, prompt="ko")[0]["content"] == "아다님에게 인사해줘."
        )
        use_case.track(lambda: Result(content="안녕", finish_reason="stop"))

        [logged] = client.captured
        assert logged["prompt"] == "ko"
        assert logged["prompt_version_id"] == use_case.prompt_version["id"]

    def test_failed_named_prompt_render_does_not_poison_following_track_evidence(self, client):
        use_case = client.use_case("greeting")

        with pytest.raises(MissingVariableError):
            use_case.messages({}, prompt="ko")
        use_case.track(lambda: Result(content="Hello", finish_reason="stop"))

        [logged] = client.captured
        assert use_case.prompt == "default"
        assert logged["prompt"] == "default"
        assert logged["prompt_version_id"] == use_case.prompt_version["id"]

    def test_a_provider_error_is_logged_with_its_kind_and_then_re_raised(self, client):
        use_case = client.use_case("greeting")

        def call():
            raise ProviderError("rate limited", kind="rate_limited", status=429)

        with pytest.raises(ProviderError):
            use_case.track(call)
        [logged] = client.captured
        assert logged["status"] == "error"
        assert logged["error"] == {
            "kind": "rate_limited",
            "status": 429,
            "message": "rate limited",
        }
        assert "output" not in logged

    def test_an_error_with_a_result_keeps_the_usage_as_a_quality_signal(self, client):
        use_case = client.use_case("greeting")
        result = Result(content='{"a":', finish_reason="length", input_tokens=38, output_tokens=512)

        def call():
            raise ProviderError("unexpected end of JSON input", kind="parse", result=result)

        with pytest.raises(ProviderError):
            use_case.track(call)
        [logged] = client.captured
        assert logged["status"] == "error"
        assert logged["stop_kind"] == "length"
        assert logged["output"]["content"] == '{"a":'
        assert logged["usage"]["output_tokens"] == 512

    def test_any_other_exception_is_recorded_as_app_and_propagates_unchanged(self, client):
        use_case = client.use_case("greeting")

        def call():
            raise ZeroDivisionError("boom")

        with pytest.raises(ZeroDivisionError, match="boom"):
            use_case.track(call)
        [logged] = client.captured
        assert logged["error"]["kind"] == "app"
        assert "ZeroDivisionError" in logged["error"]["message"]

    def test_a_plain_string_return_value_is_treated_as_the_completion(self, client):
        client.use_case("greeting").track(lambda: "hi there")
        assert client.captured[0]["output"] == {"content": "hi there"}

    def test_the_context_manager_form_records_the_same_thing(self, client):
        use_case = client.use_case("greeting")
        with use_case.track(variables={"name": "Ada"}) as log:
            log.result(Result(content="Hello", finish_reason="stop"))
        [logged] = client.captured
        assert logged["status"] == "ok"
        assert logged["output"]["content"] == "Hello"

    def test_the_context_manager_records_an_exception_and_re_raises(self, client):
        use_case = client.use_case("greeting")
        with pytest.raises(RuntimeError), use_case.track():
            raise RuntimeError("nope")
        assert client.captured[0]["status"] == "error"


class TestLog:
    def test_a_hand_built_record_is_completed_and_captured(self, client):
        record_id = client.log(
            {
                "use_case": "summarize",
                "model": "openai/gpt-4o-mini",
                "status": "ok",
                "kind": "text",
                "input": {"text": "Summarize: a, b"},
                "output": {"content": "a and b"},
            }
        )
        [logged] = client.captured
        assert logged["id"] == record_id
        assert logged["id"][14] == "7", "the id must be a UUIDv7"
        assert logged["started_at"].endswith("Z")
        assert logged["sdk"]["name"] == "prompton-python"

    def test_a_resolution_fills_in_the_evidence(self, client):
        use_case = client.use_case("greeting")
        client.log({"status": "ok"}, use_case=use_case)
        [logged] = client.captured
        assert logged["use_case"] == "greeting"
        assert logged["model"] == use_case.model
        assert logged["deployment_revision"] == use_case.deployment["revision"]
        assert logged["source"] == "manual"

    def test_a_missing_required_field_is_a_bug_in_the_caller(self, client):
        with pytest.raises(ValueError, match="use_case"):
            client.log({"model": "m", "status": "ok"})
        with pytest.raises(ValueError, match="status"):
            client.log({"use_case": "u", "model": "m", "status": "maybe"})

    def test_the_use_case_payload_policy_is_applied_before_the_record_is_queued(self):
        document = make_use_case_document(
            secret={
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "payload_policy": {"mode": "none", "sample_rate": 1.0, "max_bytes": 262144},
            }
        )
        client = PromptOn(mode="test", disk_cache=False)
        client.load_use_cases(document)
        client.use_case("secret").track(
            lambda: Result(content="a secret answer"),
            variables={"question": "a secret question"},
        )
        [logged] = client.captured
        assert "input" not in logged and "output" not in logged
        assert logged["use_case"] == "secret"
        client.close(timeout=0.1)

    def test_redaction_and_end_user_hashing_are_applied(self, snapshot_document):
        client = PromptOn(
            mode="test",
            disk_cache=False,
            hash_end_user=True,
            redact=lambda record: {**record, "input": {"text": "[redacted]"}},
        )
        client.load_use_cases(snapshot_document)
        client.use_case("greeting").track(
            lambda: Result(content="hi"),
            variables={"name": "Ada"},
            end_user_ref="user-42",
        )
        [logged] = client.captured
        assert logged["input"] == {"text": "[redacted]"}
        assert logged["end_user_ref"] == (
            "6d894aa3ee802549d7f340e7c1cf0d1c1cb14cd84f768d92ffaa6785337c4997"
        )
        client.close(timeout=0.1)


class TestSnapshotSurface:
    def test_use_cases_info_reports_the_tier(self, client):
        info = client.use_cases_info()
        assert info["source"] == "manual"
        assert info["environment"] == "production"

    def test_export_and_reload_a_bundle(self, tmp_path, snapshot_document):
        transport = FakeTransport()
        transport.push(json_response(200, snapshot_document, etag='"sha256-x"'))
        client = PromptOn(
            api_key="ptn_sdkfixture_k",
            host="http://localhost:4000",
            project="sdkfixture",
            disk_cache=str(tmp_path / "cache.json"),
            poll=False,
            transport=transport,
        )
        assert client.refresh() is True
        bundle = client.export_use_cases(tmp_path / "bundle.json")
        assert json.loads(bundle.read_text())["project"] == "sdkfixture"

        offline = PromptOn(
            api_key=None,
            mode="offline",
            host="http://localhost:4000",
            project="sdkfixture",
            disk_cache=False,
            bundle=str(bundle),
        )
        assert offline.use_case("greeting").source == "bundle"
        client.close(timeout=0.1)
        offline.close(timeout=0.1)


class TestModuleLevelClient:
    def test_configure_replaces_the_default_and_close_clears_it(self, snapshot_document):
        client = prompton.configure(mode="test", disk_cache=False)
        try:
            client.load_use_cases(snapshot_document)
            assert prompton.use_case("greeting").model == "openai/gpt-4o-mini"
            assert prompton.get_client() is client
            replacement = prompton.configure(mode="test", disk_cache=False)
            assert prompton.get_client() is replacement
        finally:
            prompton.close(timeout=0.1)
        assert prompton.get_client() is not client
        prompton.close(timeout=0.1)
