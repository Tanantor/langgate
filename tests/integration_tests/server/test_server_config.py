"""Tests for environment configuration."""

import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from langgate.server.core.config import (
    ApiSettings,
    FixedYamlConfigSettingsSource,
    _get_api_env_file_path,
)
from tests.utils.config_utils import config_path_resolver
from tests.utils.server_utils import prevent_server_config_loading
from tests.utils.utils import capture_logs


def test_app_config_loading_from_yaml(
    mock_config_path_in_env: dict[str, str], mock_env_vars_from_env_file: dict[str, str]
):
    """Test that app config is correctly loaded from YAML file and logs are emitted."""
    with capture_logs() as caplogs:
        # Re-initialize settings to ensure fresh loading
        test_settings = ApiSettings()

        # Test that settings were properly loaded from the YAML file
        assert test_settings.PROJECT_NAME == "LangGate Test"
        assert test_settings.CORS_ORIGINS == ["http://localhost"]
        assert test_settings.HTTPS is False
        assert test_settings.SECRET_KEY.get_secret_value() == "test-secret-key"

        # Verify the log was emitted
        assert any(log["event"] == "loaded_app_config" for log in caplogs)


def test_warning_when_no_app_config(tmp_path: Path):
    """Test warning when YAML has no app_config section."""
    # Create a config file without app_config
    config_path = tmp_path / "no_app_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump({"some_other_key": "value"}, f)

    class SimpleSettings(BaseSettings):
        """Simple settings class for testing FixedYamlConfigSettingsSource."""

    with (
        mock.patch.dict(os.environ, {"LANGGATE_CONFIG": str(config_path)}),
        capture_logs() as caplogs,
    ):
        yaml_source = FixedYamlConfigSettingsSource(SimpleSettings)
        # Call it to trigger config loading
        yaml_source()

        # Check warning was logged
        assert any(log["event"] == "no_app_config_in_config_file" for log in caplogs)


def test_fixed_yaml_config_env_path():
    """Test that LANGGATE_CONFIG environment variable is used by FixedYamlConfigSettingsSource."""
    with mock.patch.dict(
        os.environ, {"LANGGATE_CONFIG": "/custom/path/langgate_config.yaml"}
    ):

        class TestSettings(BaseSettings):
            pass

        yaml_source = FixedYamlConfigSettingsSource(TestSettings)
        assert str(yaml_source.config_path) == "/custom/path/langgate_config.yaml"


@pytest.mark.parametrize(
    "source,expected_path",
    [
        # Skip arg path as FixedYamlConfigSettingsSource doesn't accept constructor args
        ("env", "/env/path/config.yaml"),
        ("cwd", "langgate_config.yaml"),
        ("package_dir", "default_config.yaml"),
    ],
    ids=["env_var", "cwd_path", "package_dir_path"],
)
def test_fixed_yaml_config_settings_source_paths(source, expected_path):
    """Test path resolution in FixedYamlConfigSettingsSource with different sources."""

    class TestSettings(BaseSettings):
        """Simple settings class for testing FixedYamlConfigSettingsSource."""

    with (
        prevent_server_config_loading(),
        config_path_resolver(source, "config_yaml", expected_path),
    ):
        # Initialize the YAML settings source
        yaml_source = FixedYamlConfigSettingsSource(TestSettings)

        # Verify the config path was correctly resolved
        assert expected_path in str(yaml_source.config_path)


def test_api_settings_customise_sources():
    """Test that ApiSettings customizes sources by adding FixedYamlConfigSettingsSource."""
    # Directly test the behavior of settings_customise_sources

    # Create mock settings sources
    init_settings = MagicMock()
    env_settings = MagicMock()
    dotenv_settings = MagicMock()
    file_secret_settings = MagicMock()

    settings_args = [init_settings, env_settings, dotenv_settings, file_secret_settings]

    # Get the actual settings sources
    sources = ApiSettings.settings_customise_sources(
        ApiSettings,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    )

    # Verify that we have settings sources (should be 5)
    assert len(sources) > 0

    # Verify that we're returning all the provided sources
    for arg in settings_args:
        assert arg in sources, f"{arg.__name__} should be included in sources"

    # Verify that FixedYamlConfigSettingsSource is included
    assert any(
        isinstance(source, FixedYamlConfigSettingsSource) for source in sources
    ), "FixedYamlConfigSettingsSource should be included in sources"

    # Verify the order of sources
    for i, (arg, source) in enumerate(zip(settings_args, sources, strict=False)):
        assert arg == source, f"""Source at index {i} mismatch:
         expected {arg.__class__.__name__}, got {source.__class__.__name__}"""

    assert isinstance(sources[-1], FixedYamlConfigSettingsSource), (
        "Fifth source should be FixedYamlConfigSettingsSource"
    )


def test_api_settings_env_file_source_resolution():
    """Test that ApiSettings uses correct env file resolution based on source."""

    # Since ApiSettings.model_config is evaluated at import time, we need to test that
    # the _get_api_env_file_path function returns the correct path, and then
    # verify that this path is what would be used in the model_config

    def _set_settings_config(env_file: Path | str | None):
        return SettingsConfigDict(
            case_sensitive=True,
            env_file=env_file,
            env_file_encoding="utf-8",
            extra="ignore",
        )

    # Test case 1: Using environment variable LANGGATE_ENV_FILE
    expected_path = "/custom/env/path/.env"
    with config_path_resolver("env", "env_file", expected_path):
        env_file_path = _get_api_env_file_path()
        assert str(env_file_path) == expected_path

        # Verify this path would be used in settings config
        config_dict = _set_settings_config(env_file_path)
        assert str(config_dict.get("env_file")) == expected_path

    # Test case 2: Using CWD when no LANGGATE_ENV_FILE is set
    expected_path = "/fake/cwd/.env"
    with config_path_resolver("cwd", "env_file", expected_path):
        env_file_path = _get_api_env_file_path()
        assert str(env_file_path) == expected_path

        config_dict = _set_settings_config(env_file_path)
        assert str(config_dict.get("env_file")) == expected_path

    # Test case 3: No LANGGATE_ENV_FILE env var and no env file in CWD
    with mock.patch("pathlib.Path.exists", return_value=False):
        env_file_path = _get_api_env_file_path()
        assert env_file_path is None

        # Verify None would be used in settings config
        config_dict = _set_settings_config(env_file_path)
        assert config_dict.get("env_file") is None
