from __future__ import annotations

import pytest

from prompton.errors import UnknownPromptError, UnknownUseCaseError, UnresolvedError
from prompton.resolver import prompt_names, resolve
from prompton.snapshot_data import (
    SCHEMA_VERSION,
    InvalidSnapshotError,
    UnsupportedSchemaVersionError,
    UseCaseDocument,
)
from prompton.testing import make_use_case_document


@pytest.fixture
def snapshot() -> UseCaseDocument:
    document = make_use_case_document(
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
    return UseCaseDocument.from_mapping(document)


def test_merges_params_shallowly_with_the_deployment_winning(snapshot):
    resolution = resolve(snapshot, "greeting")
    assert resolution.params == {"temperature": 0.2, "max_tokens": 512}


def test_an_override_of_none_is_kept_not_deleted(snapshot):
    """Apps rely on sending `"only": null` to clear a provider restriction."""
    resolution = resolve(snapshot, "greeting")
    assert resolution.provider_options == {
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
    assert error.value.prompt_names == ["default", "ko"]


def test_an_embedding_use_case_ignores_a_prompt_name(snapshot):
    resolution = resolve(snapshot, "embed", "ko")
    assert resolution.prompt is None
    assert resolution.prompt_version is None
    assert resolution.prompt_names == ()
    assert resolution.model == "openai/text-embedding-3-small"


def test_unknown_use_case_and_undeployed_use_case_are_different_errors(snapshot):
    with pytest.raises(UnknownUseCaseError):
        resolve(snapshot, "nope")
    with pytest.raises(UnresolvedError):
        resolve(snapshot, "draft")


def test_a_dangling_reference_resolves_with_warnings_rather_than_failing(snapshot):
    document = make_use_case_document(greeting={"messages": [{"role": "user", "content": "hi"}]})
    document["deployments"]["greeting"]["model_id"] = "gone"
    document["deployments"]["greeting"]["prompt_pins"]["default"] = "also-gone"
    resolution = resolve(UseCaseDocument.from_mapping(document), "greeting")
    assert resolution.model is None
    assert resolution.prompt_version is None
    assert resolution.warnings == (
        "missing_prompt_version: also-gone",
        "missing_model: gone",
    )


def test_the_source_and_etag_travel_with_the_resolution(snapshot):
    resolution = resolve(snapshot, "greeting", source="disk", etag='"sha256-abc"')
    assert resolution.source == "disk"
    assert resolution.etag == '"sha256-abc"'


class TestSnapshotDecoding:
    def test_older_schema_versions_are_refused(self):
        with pytest.raises(UnsupportedSchemaVersionError):
            UseCaseDocument.from_mapping({"schema_version": 2, "use_cases": {}})

    def test_newer_schema_versions_are_refused(self):
        document = make_use_case_document(greeting={"messages": []})
        document["schema_version"] = SCHEMA_VERSION + 1
        document["use_cases"]["greeting"]["brand_new_field"] = 1
        with pytest.raises(UnsupportedSchemaVersionError):
            UseCaseDocument.from_mapping(document)

    def test_legacy_version_field_is_not_a_schema_version(self):
        document = make_use_case_document(greeting={"messages": []})
        document["version"] = document.pop("schema_version")
        with pytest.raises(InvalidSnapshotError, match="schema_version is required"):
            UseCaseDocument.from_mapping(document)

    def test_missing_schema_version_is_refused_even_when_deployments_exist(self):
        document = make_use_case_document(greeting={"messages": []})
        document.pop("schema_version")
        with pytest.raises(InvalidSnapshotError, match="schema_version is required"):
            UseCaseDocument.from_mapping(document)

    def test_schema_version_must_be_exact_integer_four(self):
        document = make_use_case_document(greeting={"messages": []})
        document["schema_version"] = "4"
        with pytest.raises(InvalidSnapshotError, match="integer 4"):
            UseCaseDocument.from_mapping(document)

    def test_an_environment_with_no_deployments_is_not_an_error(self):
        data = UseCaseDocument.from_mapping(
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
            UseCaseDocument.from_json(b"{not json")
        with pytest.raises(InvalidSnapshotError):
            UseCaseDocument.from_mapping([1, 2, 3])
