"""Tests for the secrets management abstraction layer.

Covers all provider types, chaining, the factory function, the module-level
singleton, and edge cases around missing/empty secrets.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from aiswarm.utils.secrets import (
    ChainSecretsProvider,
    EnvSecretsProvider,
    FileSecretsProvider,
    SecretsProvider,
    create_secrets_provider,
    get_secrets_provider,
    reset_secrets_provider,
    set_secrets_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_provider() -> Iterator[None]:
    """Reset the module-level secrets provider after each test."""
    yield
    reset_secrets_provider()


# ---------------------------------------------------------------------------
# EnvSecretsProvider
# ---------------------------------------------------------------------------


class TestEnvSecretsProvider:
    def test_reads_existing_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_SECRET_ABC", "secret-value-123")
        provider = EnvSecretsProvider()
        assert provider.get_secret("TEST_SECRET_ABC") == "secret-value-123"

    def test_returns_none_for_missing_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_SECRET_XYZ", raising=False)
        provider = EnvSecretsProvider()
        assert provider.get_secret("NONEXISTENT_SECRET_XYZ") is None

    def test_returns_empty_string_when_set_to_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EMPTY_SECRET", "")
        provider = EnvSecretsProvider()
        assert provider.get_secret("EMPTY_SECRET") == ""

    def test_is_subclass_of_secrets_provider(self) -> None:
        assert issubclass(EnvSecretsProvider, SecretsProvider)


# ---------------------------------------------------------------------------
# FileSecretsProvider — JSON file
# ---------------------------------------------------------------------------


class TestFileSecretsProviderJsonFile:
    def test_reads_from_json_file(self) -> None:
        data = {"AIS_RISK_HMAC_SECRET": "hmac-from-file", "AIS_API_KEY": "key-from-file"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = f.name

        try:
            provider = FileSecretsProvider(path)
            assert provider.get_secret("AIS_RISK_HMAC_SECRET") == "hmac-from-file"
            assert provider.get_secret("AIS_API_KEY") == "key-from-file"
        finally:
            os.unlink(path)

    def test_returns_none_for_missing_key(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"existing_key": "value"}, f)
            f.flush()
            path = f.name

        try:
            provider = FileSecretsProvider(path)
            assert provider.get_secret("nonexistent_key") is None
        finally:
            os.unlink(path)

    def test_raises_on_nonexistent_path(self) -> None:
        with pytest.raises(FileNotFoundError, match="Secrets path not found"):
            FileSecretsProvider("/tmp/absolutely_does_not_exist_12345.json")

    def test_raises_on_non_object_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["not", "an", "object"], f)
            f.flush()
            path = f.name

        try:
            with pytest.raises(TypeError, match="JSON object"):
                FileSecretsProvider(path)
        finally:
            os.unlink(path)

    def test_coerces_non_string_values_to_string(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"numeric": 42, "boolean": True}, f)
            f.flush()
            path = f.name

        try:
            provider = FileSecretsProvider(path)
            assert provider.get_secret("numeric") == "42"
            assert provider.get_secret("boolean") == "True"
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# FileSecretsProvider — directory of files
# ---------------------------------------------------------------------------


class TestFileSecretsProviderDirectory:
    def test_reads_from_directory(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "AIS_RISK_HMAC_SECRET").write_text("hmac-from-dir\n")
        (d / "AIS_API_KEY").write_text("  key-from-dir  \n")

        provider = FileSecretsProvider(str(d))
        assert provider.get_secret("AIS_RISK_HMAC_SECRET") == "hmac-from-dir"
        assert provider.get_secret("AIS_API_KEY") == "key-from-dir"

    def test_skips_hidden_files(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / ".hidden_secret").write_text("should-be-skipped")
        (d / "visible_secret").write_text("included")

        provider = FileSecretsProvider(str(d))
        assert provider.get_secret(".hidden_secret") is None
        assert provider.get_secret("visible_secret") == "included"

    def test_returns_none_for_missing_key_in_directory(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "one_key").write_text("one_value")

        provider = FileSecretsProvider(str(d))
        assert provider.get_secret("another_key") is None

    def test_empty_directory_works(self) -> None:
        d = Path(tempfile.mkdtemp())
        provider = FileSecretsProvider(str(d))
        assert provider.get_secret("anything") is None


# ---------------------------------------------------------------------------
# ChainSecretsProvider
# ---------------------------------------------------------------------------


class TestChainSecretsProvider:
    def test_first_provider_wins(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "secrets.json").write_text(json.dumps({"KEY": "from-file"}))
        file_provider = FileSecretsProvider(str(d / "secrets.json"))

        # EnvSecretsProvider also has the key
        os.environ["KEY"] = "from-env"
        try:
            chain = ChainSecretsProvider([file_provider, EnvSecretsProvider()])
            assert chain.get_secret("KEY") == "from-file"
        finally:
            os.environ.pop("KEY", None)

    def test_falls_through_to_second_provider(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "secrets.json").write_text(json.dumps({"FILE_ONLY": "value1"}))
        file_provider = FileSecretsProvider(str(d / "secrets.json"))

        os.environ["ENV_ONLY"] = "value2"
        try:
            chain = ChainSecretsProvider([file_provider, EnvSecretsProvider()])
            # File provider does not have ENV_ONLY, env provider does
            assert chain.get_secret("ENV_ONLY") == "value2"
            # File provider has FILE_ONLY
            assert chain.get_secret("FILE_ONLY") == "value1"
        finally:
            os.environ.pop("ENV_ONLY", None)

    def test_returns_none_when_no_provider_has_key(self) -> None:
        chain = ChainSecretsProvider([EnvSecretsProvider()])
        assert chain.get_secret("DEFINITELY_DOES_NOT_EXIST_EVER_99") is None

    def test_raises_on_empty_providers_list(self) -> None:
        with pytest.raises(ValueError, match="at least one provider"):
            ChainSecretsProvider([])


# ---------------------------------------------------------------------------
# get_secret_required
# ---------------------------------------------------------------------------


class TestGetSecretRequired:
    def test_returns_value_when_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REQUIRED_KEY", "required-value")
        provider = EnvSecretsProvider()
        assert provider.get_secret_required("REQUIRED_KEY") == "required-value"

    def test_raises_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_REQUIRED_KEY", raising=False)
        provider = EnvSecretsProvider()
        with pytest.raises(ValueError, match="Required secret 'MISSING_REQUIRED_KEY' not found"):
            provider.get_secret_required("MISSING_REQUIRED_KEY")

    def test_raises_on_chain_when_all_miss(self) -> None:
        chain = ChainSecretsProvider([EnvSecretsProvider()])
        with pytest.raises(ValueError, match="Required secret"):
            chain.get_secret_required("CHAIN_MISSING_KEY_XYZ_999")


# ---------------------------------------------------------------------------
# Module-level singleton (get/set/reset)
# ---------------------------------------------------------------------------


class TestModuleLevelProvider:
    def test_default_is_env_provider(self) -> None:
        provider = get_secrets_provider()
        assert isinstance(provider, EnvSecretsProvider)

    def test_set_overrides_default(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "secrets.json").write_text(json.dumps({"CUSTOM": "custom-value"}))
        custom = FileSecretsProvider(str(d / "secrets.json"))

        set_secrets_provider(custom)
        assert get_secrets_provider() is custom
        assert get_secrets_provider().get_secret("CUSTOM") == "custom-value"

    def test_reset_clears_provider(self) -> None:
        custom = EnvSecretsProvider()
        set_secrets_provider(custom)
        assert get_secrets_provider() is custom

        reset_secrets_provider()
        # After reset, a fresh EnvSecretsProvider should be created
        new_provider = get_secrets_provider()
        assert new_provider is not custom
        assert isinstance(new_provider, EnvSecretsProvider)


# ---------------------------------------------------------------------------
# create_secrets_provider factory
# ---------------------------------------------------------------------------


class TestCreateSecretsProvider:
    def test_default_returns_env_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AIS_SECRETS_FILE", raising=False)
        monkeypatch.delenv("AIS_SECRETS_DIR", raising=False)
        provider = create_secrets_provider()
        assert isinstance(provider, EnvSecretsProvider)

    def test_with_secrets_file_returns_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"FILE_SECRET": "file-val"}, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setenv("AIS_SECRETS_FILE", path)
            monkeypatch.delenv("AIS_SECRETS_DIR", raising=False)
            provider = create_secrets_provider()

            assert isinstance(provider, ChainSecretsProvider)
            assert provider.get_secret("FILE_SECRET") == "file-val"
        finally:
            os.unlink(path)

    def test_with_secrets_dir_returns_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "DIR_SECRET").write_text("dir-val\n")

        monkeypatch.delenv("AIS_SECRETS_FILE", raising=False)
        monkeypatch.setenv("AIS_SECRETS_DIR", str(d))
        provider = create_secrets_provider()

        assert isinstance(provider, ChainSecretsProvider)
        assert provider.get_secret("DIR_SECRET") == "dir-val"

    def test_env_fallback_always_included(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"ONLY_IN_FILE": "val"}, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setenv("AIS_SECRETS_FILE", path)
            monkeypatch.delenv("AIS_SECRETS_DIR", raising=False)
            monkeypatch.setenv("ONLY_IN_ENV", "env-val")

            provider = create_secrets_provider()
            # File key
            assert provider.get_secret("ONLY_IN_FILE") == "val"
            # Env fallback
            assert provider.get_secret("ONLY_IN_ENV") == "env-val"
        finally:
            os.unlink(path)

    def test_file_takes_precedence_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"SHARED_KEY": "from-file"}, f)
            f.flush()
            path = f.name

        try:
            monkeypatch.setenv("AIS_SECRETS_FILE", path)
            monkeypatch.delenv("AIS_SECRETS_DIR", raising=False)
            monkeypatch.setenv("SHARED_KEY", "from-env")

            provider = create_secrets_provider()
            assert provider.get_secret("SHARED_KEY") == "from-file"
        finally:
            os.unlink(path)

    def test_both_file_and_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # File provider
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"FROM_FILE": "file-val"}, f)
            f.flush()
            file_path = f.name

        # Dir provider
        d = Path(tempfile.mkdtemp())
        (d / "FROM_DIR").write_text("dir-val")

        try:
            monkeypatch.setenv("AIS_SECRETS_FILE", file_path)
            monkeypatch.setenv("AIS_SECRETS_DIR", str(d))

            provider = create_secrets_provider()
            assert isinstance(provider, ChainSecretsProvider)
            assert provider.get_secret("FROM_FILE") == "file-val"
            assert provider.get_secret("FROM_DIR") == "dir-val"
        finally:
            os.unlink(file_path)
