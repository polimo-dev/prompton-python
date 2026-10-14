from __future__ import annotations

import pytest

from prompton.errors import UnknownPromptError, UnknownUseCaseError, UnresolvedError
from prompton.resolver import prompt_names, resolve
from prompton.snapshot_data import (
    SCHEMA_VERSION,
    InvalidUseCaseDocumentError,
    UnsupportedSchemaVersionError,
    UseCaseDocument,
)
from prompton.testing import make_use_case_document
from tests.conftest import readable_document, single_prompt_document


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
    document["prompts"]["draft"] = {"id": "uc-draft", "kind": "chat", "default_params": {}}
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


def test_a_dangling_reference_fails_closed(snapshot):
    document = make_use_case_document(greeting={"messages": [{"role": "user", "content": "hi"}]})
    document["deployments"]["greeting"]["model"] = "gone/model"
    document["deployments"]["greeting"]["prompt_pins"]["default"] = 99
    with pytest.raises((UnresolvedError, UnknownPromptError)):
        resolve(UseCaseDocument.from_mapping(document), "greeting")


def test_schema1_aggregate_requires_all_keyed_resource_maps():
    base = {
        "schema_version": 1,
        "project": "sdkfixture",
        "environment": "production",
        "prompts": {},
        "deployments": {},
        "prompt_versions": {},
        "models": {},
    }
    UseCaseDocument.from_mapping(base)
    for collection in ("deployments", "prompt_versions", "models"):
        missing = dict(base)
        missing.pop(collection)
        with pytest.raises(InvalidUseCaseDocumentError, match="keyed object"):
            UseCaseDocument.from_mapping(missing)

        array = dict(base)
        array[collection] = []
        with pytest.raises(InvalidUseCaseDocumentError, match="keyed object"):
            UseCaseDocument.from_mapping(array)


def test_decodes_variable_definitions_from_variables_and_rejects_old_input_schema_metadata(
    snapshot_document,
):
    document = UseCaseDocument.from_mapping(readable_document(snapshot_document))
    assert document.use_cases["greeting"].variables[0].name == "name"
    assert not hasattr(document.use_cases["greeting"], "input_schema")

    old = readable_document(snapshot_document)
    old["prompts"]["greeting"]["input_schema"] = old["prompts"]["greeting"].pop("variables")
    with pytest.raises(InvalidUseCaseDocumentError, match="not input_schema"):
        UseCaseDocument.from_mapping(old)


def test_schema1_prompt_id_is_not_read_from_legacy_uuid_field(snapshot_document):
    document = readable_document(snapshot_document)
    document["prompts"]["greeting"]["id"] = "0198f2a1-0000-7000-8000-00000000c001"
    snapshot = UseCaseDocument.from_mapping(document)
    assert snapshot.use_cases["greeting"].id is None


def test_schema1_readable_resources_have_no_public_resource_ids(snapshot_document):
    snapshot = UseCaseDocument.from_mapping(readable_document(snapshot_document))
    resolution = resolve(snapshot, "greeting")
    assert resolution.deployment == {"id": None, "revision": "v2026.09.30-1"}
    assert resolution.model_id is None
    assert resolution.model == "openai/gpt-4o-mini"
    assert resolution.provider == "openrouter"
    assert resolution.prompt_version == {"id": None, "number": 1}
    assert resolution.messages is not None
    assert resolution.messages[1]["content"] == "Say hello to {{ name }}."


def test_schema1_single_prompt_chat_and_decision_documents():
    greeting = UseCaseDocument.from_mapping(single_prompt_document("greeting"))
    resolution = resolve(greeting, "greeting")
    assert resolution.deployment == {"id": None, "revision": "v2026.09.30-1"}
    assert resolution.model_id is None
    assert resolution.model == "openai/gpt-4o-mini"
    assert resolution.provider == "openrouter"
    assert resolution.prompt_version == {"id": None, "number": 1}
    assert resolution.messages is not None
    assert resolution.messages[1]["content"] == "Say hello to {{ name }}."

    sentiment = UseCaseDocument.from_mapping(single_prompt_document("sentiment"))
    decision = resolve(sentiment, "sentiment")
    assert decision.kind == "decision"
    assert decision.model == "typesafe/jev-1.13"
    assert decision.request_path == "/api/v1/systemone"


def test_schema1_rejects_incomplete_mixed_or_mismatched_single_prompt_documents():
    missing = single_prompt_document("greeting")
    missing.pop("prompt_version")
    with pytest.raises(InvalidUseCaseDocumentError, match="prompt_version is required"):
        UseCaseDocument.from_mapping(missing)

    mixed = single_prompt_document("greeting", prompts={})
    with pytest.raises(InvalidUseCaseDocumentError, match="singular resources"):
        UseCaseDocument.from_mapping(mixed)

    mixed_null = single_prompt_document("greeting", prompts=None)
    with pytest.raises(InvalidUseCaseDocumentError, match="singular resources"):
        UseCaseDocument.from_mapping(mixed_null)

    bad_version_prompt = single_prompt_document("greeting")
    bad_version_prompt["prompt_version"]["prompt"] = "review"
    with pytest.raises(InvalidUseCaseDocumentError, match=r"prompt_version\.prompt"):
        UseCaseDocument.from_mapping(bad_version_prompt)

    bad_deployment_version = single_prompt_document("greeting")
    bad_deployment_version["deployment"]["prompt_version"] = 99
    with pytest.raises(InvalidUseCaseDocumentError, match=r"deployment\.prompt_version"):
        UseCaseDocument.from_mapping(bad_deployment_version)

    bad_model = single_prompt_document("greeting")
    bad_model["deployment"]["model"] = "different/model"
    with pytest.raises(InvalidUseCaseDocumentError, match="provider/model"):
        UseCaseDocument.from_mapping(bad_model)


def test_schema1_scopes_prompt_versions_by_prompt_and_models_by_provider(snapshot_document):
    document = readable_document(snapshot_document)
    document["prompts"]["review"] = {
        "kind": "chat",
        "variables": [],
        "default_params": {},
        "payload_policy": None,
    }
    document["deployments"]["review"] = {
        "revision": "v2026.09.30-7",
        "provider": "direct",
        "model": "openai/gpt-4o-mini",
        "prompt_version": 1,
        "params": {},
        "provider_options": {},
        "api": "chat_completions",
        "request_path": "/v1/chat/completions",
    }
    document["prompt_versions"]["review@1"] = {
        "prompt": "review",
        "number": 1,
        "kind": "chat",
        "engine": "liquid",
        "messages": [{"role": "user", "content": "Review {{ item }}."}],
        "text_template": None,
    }
    document["models"]["direct:openai/gpt-4o-mini"] = {
        "provider": "direct",
        "model": "openai/gpt-4o-mini",
        "display_name": "Direct GPT-4o mini",
        "metadata": {},
        "provider_options": {"base_url": "https://api.openai.example"},
        "capabilities": [],
        "status": "active",
    }
    snapshot = UseCaseDocument.from_mapping(document)
    assert resolve(snapshot, "greeting").provider_options == {"only": ["OpenAI"]}
    review = resolve(snapshot, "review")
    assert review.provider == "direct"
    assert review.provider_options == {"base_url": "https://api.openai.example"}
    assert review.messages == ({"role": "user", "content": "Review {{ item }}."},)


def test_schema1_invalid_pins_and_missing_references_fail_closed(snapshot_document):
    for prompt_version in ("2", "2abc", 2.9, 0):
        invalid = readable_document(snapshot_document)
        invalid["deployments"]["greeting"]["prompt_pins"]["default"] = prompt_version
        invalid_snapshot = UseCaseDocument.from_mapping(invalid)
        with pytest.raises(UnknownPromptError):
            resolve(invalid_snapshot, "greeting")

    missing_version = readable_document(snapshot_document)
    missing_version["deployments"]["greeting"]["prompt_pins"]["default"] = 99
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(missing_version), "greeting")

    missing_model = readable_document(snapshot_document)
    missing_model["deployments"]["greeting"]["model"] = "missing/model"
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(missing_model), "greeting")


def test_schema1_relabels_do_not_make_old_uuid_resources_resolvable(snapshot_document):
    document = readable_document(snapshot_document)
    document["deployments"]["greeting"] = {
        "revision": "v2026.09.30-1",
        "model_id": "0198f2a1-0000-7000-8000-00000000e001",
        "template_pins": {"default": "0198f2a1-0000-7000-8000-00000000a001"},
        "api": "chat_completions",
        "request_path": "/api/v1/chat/completions",
    }
    document["prompt_versions"]["0198f2a1-0000-7000-8000-00000000a001"] = {
        "id": "0198f2a1-0000-7000-8000-00000000a001",
        "prompt_template_id": "0198f2a1-0000-7000-8000-00000000b001",
        "number": 1,
        "kind": "chat",
        "engine": "liquid",
        "messages": [{"role": "user", "content": "Hi"}],
    }
    document["models"]["0198f2a1-0000-7000-8000-00000000e001"] = {
        "id": "0198f2a1-0000-7000-8000-00000000e001",
        "provider": "openrouter",
        "model_id": "openai/gpt-4o-mini",
    }
    with pytest.raises((UnresolvedError, UnknownPromptError)):
        resolve(UseCaseDocument.from_mapping(document), "greeting")


def test_schema1_composite_keys_must_match_embedded_fields(snapshot_document):
    bad_version = readable_document(snapshot_document)
    bad_version["prompt_versions"]["greeting@1"]["prompt"] = "review"
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(bad_version), "greeting")

    bad_number = readable_document(snapshot_document)
    bad_number["prompt_versions"]["greeting@1"]["number"] = 1.9
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(bad_number), "greeting")

    bad_model = readable_document(snapshot_document)
    bad_model["models"]["openrouter:openai/gpt-4o-mini"]["provider"] = "direct"
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(bad_model), "greeting")

    bad_deployment_model = readable_document(snapshot_document)
    bad_deployment_model["deployments"]["greeting"]["model"] = 123
    with pytest.raises(UnresolvedError):
        resolve(UseCaseDocument.from_mapping(bad_deployment_model), "greeting")

    duplicate = readable_document(snapshot_document)
    duplicate["models"]["direct:openai/gpt-4o-mini"] = {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "display_name": "alias",
        "metadata": {},
        "provider_options": {"only": ["Wrong"]},
        "capabilities": [],
        "status": "active",
    }
    assert resolve(UseCaseDocument.from_mapping(duplicate), "greeting").provider_options == {
        "only": ["OpenAI"]
    }


def test_the_source_and_etag_travel_with_the_resolution(snapshot):
    resolution = resolve(snapshot, "greeting", source="disk", etag='"sha256-abc"')
    assert resolution.source == "disk"
    assert resolution.etag == '"sha256-abc"'


class TestSnapshotDecoding:
    def test_older_schema_versions_are_refused(self):
        with pytest.raises(UnsupportedSchemaVersionError):
            UseCaseDocument.from_mapping({"schema_version": 2, "prompts": {}})

    def test_newer_schema_versions_are_refused(self):
        document = make_use_case_document(greeting={"messages": []})
        document["schema_version"] = SCHEMA_VERSION + 2
        document["prompts"]["greeting"]["brand_new_field"] = 1
        with pytest.raises(UnsupportedSchemaVersionError):
            UseCaseDocument.from_mapping(document)

    def test_legacy_version_field_is_not_a_schema_version(self):
        document = make_use_case_document(greeting={"messages": []})
        document["version"] = document.pop("schema_version")
        with pytest.raises(InvalidUseCaseDocumentError, match="schema_version is required"):
            UseCaseDocument.from_mapping(document)

    def test_missing_schema_version_is_refused_even_when_deployments_exist(self):
        document = make_use_case_document(greeting={"messages": []})
        document.pop("schema_version")
        with pytest.raises(InvalidUseCaseDocumentError, match="schema_version is required"):
            UseCaseDocument.from_mapping(document)

    def test_schema_version_must_be_supported_integer(self):
        document = make_use_case_document(greeting={"messages": []})
        document["schema_version"] = "4"
        with pytest.raises(InvalidUseCaseDocumentError, match="integer"):
            UseCaseDocument.from_mapping(document)

    def test_deployment_revision_must_be_a_string(self):
        document = make_use_case_document(greeting={"messages": []})
        document["deployments"]["greeting"]["revision"] = 1
        with pytest.raises(InvalidUseCaseDocumentError, match="revision must be a string"):
            UseCaseDocument.from_mapping(document)

    def test_an_environment_with_no_deployments_is_not_an_error(self):
        data = UseCaseDocument.from_mapping(
            {
                "schema_version": SCHEMA_VERSION,
                "environment": "staging",
                "prompts": {"greeting": {"kind": "chat"}},
                "deployments": {},
                "prompt_versions": {},
                "models": {},
            }
        )
        with pytest.raises(UnresolvedError):
            resolve(data, "greeting")

    def test_invalid_json_and_a_non_object_are_rejected(self):
        with pytest.raises(InvalidUseCaseDocumentError):
            UseCaseDocument.from_json(b"{not json")
        with pytest.raises(InvalidUseCaseDocumentError):
            UseCaseDocument.from_mapping([1, 2, 3])
