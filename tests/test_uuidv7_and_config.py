from __future__ import annotations

import time
from pathlib import Path

import pytest

from prompton.config import Config, default_disk_cache_path, project_from_api_key
from prompton.errors import ConfigurationError
from prompton.uuidv7 import timestamp_ms, uuid7


class TestUuidV7:
    def test_layout_is_version_7_variant_10(self):
        value = uuid7()
        assert len(value) == 36
        assert value[14] == "7", "the version nibble must be 7, not 4"
        assert value[19] in "89ab", "the variant bits must be 10"

    def test_ids_sort_by_time_across_milliseconds(self):
        first = uuid7(1_700_000_000_000)
        second = uuid7(1_700_000_000_001)
        assert first < second

    def test_the_timestamp_round_trips(self):
        now = int(time.time() * 1000)
        assert timestamp_ms(uuid7(now)) == now
        assert timestamp_ms("not-a-uuid") is None

    def test_ids_are_unique(self):
        assert len({uuid7() for _ in range(5000)}) == 5000


class TestConfig:
    def test_precedence_is_option_then_environment_then_default(self, monkeypatch):
        monkeypatch.delenv("PTN_HOST", raising=False)
        assert Config.build().host == "https://app.prompton.ai"

        monkeypatch.setenv("PTN_HOST", "http://from-env:4000")
        assert Config.build().host == "http://from-env:4000"
        assert Config.build(host="http://explicit:9000").host == "http://explicit:9000"

    def test_the_sdk_appends_the_api_prefix_itself(self):
        assert (
            Config.build(host="http://localhost:4000/").base_url == "http://localhost:4000/api/v1"
        )

    def test_a_host_without_a_scheme_is_refused(self):
        with pytest.raises(ConfigurationError):
            Config.build(host="localhost:4000")

    def test_the_project_comes_from_the_key_when_not_given(self, monkeypatch):
        monkeypatch.delenv("PTN_PROJECT", raising=False)
        assert project_from_api_key("ptn_sdkfixture_abcdefghijkl") == "sdkfixture"
        assert project_from_api_key("ptn_my_project_abc") == "my_project"
        assert project_from_api_key(None) is None
        assert Config.build(api_key="ptn_sdkfixture_abc").project == "sdkfixture"

    def test_without_an_api_key_the_sdk_makes_no_remote_calls(self, monkeypatch):
        monkeypatch.delenv("PTN_API_KEY", raising=False)
        assert Config.build().remote_enabled is False
        assert Config.build(api_key="ptn_x_y").remote_enabled is True
        assert Config.build(api_key="ptn_x_y", mode="offline").remote_enabled is False

    def test_the_disk_cache_is_on_by_default_and_named_by_project_and_environment(self):
        config = Config.build(api_key="ptn_demo_x", environment="staging")
        assert config.disk_cache_path is not None
        assert config.disk_cache_path.name == "demo-staging.json"
        assert Config.build(disk_cache=False).disk_cache_path is None
        assert Config.build(disk_cache="/tmp/x.json").disk_cache_path == Path("/tmp/x.json")

    def test_the_disk_cache_can_be_disabled_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("PTN_DISK_CACHE", "off")
        assert Config.build().disk_cache_path is None

    def test_an_invalid_mode_is_refused(self):
        with pytest.raises(ConfigurationError):
            Config.build(mode="whatever")

    def test_default_cache_path_is_stable(self):
        assert default_disk_cache_path("a/b", "prod").name == "a_b-prod.json"
