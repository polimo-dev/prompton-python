from __future__ import annotations

import pytest

from prompton.errors import UnknownPromptError, UnknownUseCaseError, UnresolvedError
from prompton.resolver import prompt_names, resolve
from prompton.snapshot_data import (
    SCHEMA_VERSION,
    InvalidSnapshotError,
    SnapshotData,
    UnsupportedSchemaVersionError,
)
from prompton.testing import make_snapshot


@pytest.fixture
def snapshot() -> SnapshotData:
    document = make_snapshot(
        greeting={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi {{ name }}"}],
            "prompts": {"ko": [{"role": "user", "content": "{{ name }}님"}]},
            "default_params": {"temperature": 0.5, "max_tokens": 512},
            "params": {"temperature": 0.2},
            "model_provider_options": {"only": ["OpenAI"], "sort": "price"},
            "provider_options": {"sort": None, "allow_fallbacks": False},
        },
        embed={"kind": "embedding", "model": "openai/text-embedding-3-small"},
    )
    document["use_cases"]["draft"] = {"id": "uc-draft", "kind": "chat", "default_params": {}}
    return SnapshotData.from_mapping(document)


def test_merges_params_shallowly_with_the_deployment_winning(snapshot):
    resolution = resolve(snapshot, "greeting")
    assert resolution.effective_params == {"temperature": 0.2, "max_tokens": 512}


def test_an_override_of_none_is_kept_not_deleted(snapshot):
    """Apps rely on sending `"only": null` to clear a provider restriction."""
    resolution = resolve(snapshot, "greeting")
    assert resolution.effective_provider_options == {
        "only": ["OpenAI"],
        "sort": None,
        "allow_fallbacks": False,
    }


def test_the_prompt_name_is_the_only_selection_axis(snapshot):
    assert resolve(snapshot, "greeting").prompt == "default"
    assert resolve(snapshot, "greeting", "ko").prompt == "ko"
    assert prompt_names(snapshot, "greeting") == ["default", "ko"]


def test_an_unpinned_prompt_name_never_falls_back_to_default(snapshot):
    with pytest.raises(UnknownPromptError) as error:
        resolve(snapshot, "greeting", "fr")
    assert error.value.prompt == "fr"
    assert error.value.available_prompts == ["default", "ko"]


def test_an_embedding_use_case_ignores_a_prompt_name(snapshot):
    resolution = resolve(snapshot, "embed", "ko")
    assert resolution.prompt is None
    assert resolution.prompt_version is None
    assert resolution.available_prompts == ()
    assert resolution.model == "openai/text-embedding-3-small"


def test_unknown_use_case_and_undeployed_use_case_are_different_errors(snapshot):
    with pytest.raises(UnknownUseCaseError):
        resolve(snapshot, "nope")
    with pytest.raises(UnresolvedError):
        resolve(snapshot, "draft")


def test_a_dangling_reference_resolves_with_warnings_rather_than_failing(snapshot):
    document = make_snapshot(greeting={"messages": [{"role": "user", "content": "hi"}]})
    document["deployments"]["greeting"]["model_id"] = "gone"
    document["deployments"]["greeting"]["prompt_pins"]["default"] = "also-gone"
    resolution = resolve(SnapshotData.from_mapping(document), "greeting")
    assert resolution.model is None
    assert resolution.prompt_version is None
    assert resolution.warnings == (
        "missing_prompt_version: also-gone",
        "missing_model: gone",
    )


def test_the_resolution_source_and_etag_travel_with_the_resolution(snapshot):
    resolution = resolve(snapshot, "greeting", resolution_source="disk", etag='"sha256-abc"')
    assert resolution.resolution_source == "disk"
    assert resolution.etag == '"sha256-abc"'


class TestSnapshotDecoding:
    def test_older_schema_versions_are_refused(self):
        with pytest.raises(UnsupportedSchemaVersionError):
            SnapshotData.from_mapping({"schema_version": 2, "use_cases": {}})

    def test_a_newer_schema_version_decodes_the_known_fields_with_a_warning(self):
        document = make_snapshot(greeting={"messages": []})
        document["schema_version"] = SCHEMA_VERSION + 1
        document["use_cases"]["greeting"]["brand_new_field"] = 1
        data = SnapshotData.from_mapping(document)
        assert "unknown_schema_version" in data.warnings[0]
        assert "greeting" in data.use_cases

    def test_an_environment_with_no_deployments_is_not_an_error(self):
        data = SnapshotData.from_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "environment": "staging",
                "use_cases": {"greeting": {"kind": "chat"}},
                "deployments": {},
                "prompt_versions": {},
                "models": {},
            }
        )
        with pytest.raises(UnresolvedError):
            resolve(data, "greeting")

    def test_invalid_json_and_a_non_object_are_rejected(self):
        with pytest.raises(InvalidSnapshotError):
            SnapshotData.from_json(b"{not json")
        with pytest.raises(InvalidSnapshotError):
            SnapshotData.from_mapping([1, 2, 3])
