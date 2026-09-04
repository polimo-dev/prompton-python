from __future__ import annotations

import json

from prompton.payload import apply_policy, bucket, should_keep, truncate_bytes


def test_truncation_keeps_head_and_tail_and_names_the_lost_bytes():
    text = "a" * 400
    cut, changed = truncate_bytes(text, 64)
    assert changed
    assert len(cut.encode()) <= 64
    assert "…[truncated 336 bytes]…" in cut
    assert cut.startswith("a") and cut.endswith("a")


def test_truncation_never_splits_a_character():
    cut, _ = truncate_bytes("한" * 200, 100)
    assert cut.encode().decode()  # valid UTF-8
    assert len(cut.encode()) <= 100


def test_a_cap_too_small_for_the_marker_keeps_only_the_head():
    cut, changed = truncate_bytes("abcdefghij" * 10, 8)
    assert changed
    assert len(cut.encode()) <= 8


def test_errors_and_length_stops_are_kept_whatever_the_sample_rate():
    assert should_keep({"status": "error", "id": "x"}, 0.0)
    assert should_keep({"status": "ok", "stop_kind": "length", "id": "x"}, 0.0)
    assert not should_keep({"status": "ok", "id": "x"}, 0.0)


def test_sampling_is_deterministic_on_the_id():
    record = {"id": "0198f2a1-0000-7000-8000-00000000100f", "status": "ok"}
    rate = (bucket(record["id"]) + 1) / 10_000
    assert should_keep(record, rate)
    assert not should_keep(record, bucket(record["id"]) / 10_000)


def test_hash_mode_digests_the_wrapped_value_and_sends_no_text():
    result = apply_policy(
        {"id": "a", "status": "ok", "input": "secret prompt", "output": "secret answer"},
        {"mode": "hash", "sample_rate": 1.0, "max_bytes": 262144},
    )
    assert set(result["input"]) == {"sha256", "bytes", "hashed"}
    assert result["input"]["bytes"] == len(
        json.dumps({"text": "secret prompt"}, separators=(",", ":"))
    )
    assert "secret" not in json.dumps(result)


def test_none_mode_drops_the_payload_but_keeps_the_record():
    result = apply_policy(
        {"id": "a", "status": "ok", "input": {"text": "hi"}, "output": {"content": "yo"}},
        {"mode": "none"},
    )
    assert result == {"id": "a", "status": "ok"}


def test_variables_are_replaced_wholesale_rather_than_cut():
    result = apply_policy(
        {"id": "a", "status": "ok", "input": {"variables": {"blob": "x" * 500}}},
        {"mode": "full", "sample_rate": 1.0, "max_bytes": 256},
    )
    assert set(result["input"]["variables"]) == {"truncated", "sha256", "bytes"}
    assert result["input"]["truncated"] is True


def test_the_error_message_cap_is_independent_of_max_bytes():
    result = apply_policy(
        {"id": "a", "status": "error", "error": {"kind": "app", "message": "E" * 4000}},
        {"mode": "full", "sample_rate": 1.0, "max_bytes": 1_048_576},
    )
    assert len(result["error"]["message"].encode()) <= 2048


def test_the_redact_hook_runs_last_and_a_broken_one_costs_the_payload_not_the_record():
    def redact(record):
        record["input"] = {"text": "[redacted]"}
        return record

    result = apply_policy(
        {"id": "a", "status": "ok", "input": {"text": "secret"}},
        {"mode": "full", "sample_rate": 1.0, "max_bytes": 512},
        redact=redact,
    )
    assert result["input"] == {"text": "[redacted]"}

    def broken(record):
        raise RuntimeError("boom")

    result = apply_policy(
        {"id": "a", "status": "ok", "input": {"text": "secret"}},
        {"mode": "full"},
        redact=broken,
    )
    assert "input" not in result
    assert result["id"] == "a"

    result = apply_policy(
        {"id": "a", "status": "ok", "input": {"text": "secret"}},
        {"mode": "full"},
        redact=lambda record: "not a mapping",
    )
    assert "input" not in result


def test_hash_end_user_replaces_the_reference_with_a_digest():
    result = apply_policy(
        {"id": "a", "status": "ok", "end_user_ref": "user-42"}, None, hash_end_user=True
    )
    assert result["end_user_ref"] == (
        "6d894aa3ee802549d7f340e7c1cf0d1c1cb14cd84f768d92ffaa6785337c4997"
    )


def test_the_policy_defaults_apply_when_the_snapshot_has_none():
    result = apply_policy({"id": "a", "status": "ok", "input": {"text": "x" * 100}}, None)
    assert result["input"]["text"] == "x" * 100
