"""Tests for environment configuration."""

import os
from pathlib import Path
from unittest import mock

import pytest
import yaml
from pydantic import ValidationError

from langgate.registry.config import RegistryConfig
from langgate.registry.models import ModelRegistry
from tests.utils.config_utils import config_path_resolver
from tests.utils.registry_utils import (
    patch_model_registry,
    prevent_registry_config_loading,
)


@pytest.mark.parametrize(
    "source,expected_path",
    [
        ("arg", "/arg/path/models.json"),
        ("env", "/env/path/models.json"),
        ("cwd", "langgate_models.json"),
        ("package_dir", "default_models.json"),
    ],
    ids=["arg_path", "env_var", "cwd_path", "package_dir_path"],
)
def test_registry_config_models_json_paths(source, expected_path):
    """Test path resolution for models JSON file with different sources."""
    # Reset singleton for each case
    ModelRegistry._instance = None

    # Use our unified config path resolver
    with (
        prevent_registry_config_loading(),
        config_path_resolver(source, "models_json", expected_path),
    ):
        # For arg source, we need to explicitly pass the path
        if source == "arg":
            config = RegistryConfig(models_data_path=Path(expected_path))
        else:
            config = RegistryConfig()

        assert expected_path in str(config.models_data_path)


@pytest.mark.parametrize(
    "source,expected_path",
    [
        ("arg", "/arg/path/config.yaml"),
        ("env", "/env/path/config.yaml"),
        ("cwd", "langgate_config.yaml"),
        ("package_dir", "default_config.yaml"),
    ],
    ids=["arg_path", "env_var", "cwd_path", "package_dir_path"],
)
def test_registry_config_yaml_config_paths(source, expected_path):
    """Test path resolution for config YAML file with different sources."""
    # Reset singleton for each case
    ModelRegistry._instance = None

    with (
        prevent_registry_config_loading(),
        config_path_resolver(source, "config_yaml", expected_path),
    ):
        # For arg source, we need to explicitly pass the path
        if source == "arg":
            config = RegistryConfig(config_path=Path(expected_path))
        else:
            config = RegistryConfig()

        assert expected_path in str(config.config_path)


@pytest.mark.parametrize(
    "source,expected_path",
    [
        ("arg", "/arg/path/.env"),
        ("env", "/env/path/.env"),
        ("cwd", ".env"),
    ],
    ids=["arg_path", "env_var", "cwd_path"],
)
def test_registry_config_env_file_paths(source, expected_path):
    """Test path resolution for .env file with different sources."""
    # Reset singleton for each case
    ModelRegistry._instance = None

    with (
        prevent_registry_config_loading(),
        config_path_resolver(source, "env_file", expected_path),
    ):
        # For arg source, we need to explicitly pass the path
        if source == "arg":
            config = RegistryConfig(env_file_path=Path(expected_path))
        else:
            config = RegistryConfig()

        assert expected_path in str(config.env_file_path)


def test_registry_config_env_path_vars():
    """Test that environment variables set the correct paths."""
    env_vars = {
        "LANGGATE_MODELS": "/custom/path/langgate_models.json",
        "LANGGATE_CONFIG": "/custom/path/langgate_config.yaml",
        "LANGGATE_ENV_FILE": "/custom/path/.env",
    }
    with patch_model_registry(env_vars):
        registry = ModelRegistry()
        assert (
            str(registry.config.models_data_path) == "/custom/path/langgate_models.json"
        )
        assert str(registry.config.config_path) == "/custom/path/langgate_config.yaml"
        assert str(registry.config.env_file_path) == "/custom/path/.env"


def test_registry_config_without_env_file(mock_registry_files: dict[str, Path]):
    """Test that ModelRegistry works when .env file doesn't exist."""
    # Use the proper fixture but with non-existent .env file path
    non_existent_env = mock_registry_files["env_file"].parent / "nonexistent.env"

    with mock.patch.dict(
        os.environ,
        {
            "LANGGATE_CONFIG": str(mock_registry_files["config_yaml"]),
            "LANGGATE_MODELS": str(mock_registry_files["models_json"]),
            "LANGGATE_ENV_FILE": str(non_existent_env),
        },
    ):
        # Reset the singleton for environment variables to take effect
        ModelRegistry._instance = None

        # This should work without a .env file
        registry = ModelRegistry()

        # Verify models loaded correctly
        models = registry.list_models()
        assert len(models) > 0
        assert any(model.id for model in models)


# Models merge mode tests


def test_merge_mode_default_behavior(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    mock_config_file: Path,
):
    """Test that merge mode is the default and merges models correctly."""
    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        config = RegistryConfig(
            models_data_path=mock_user_models_file, config_path=mock_config_file
        )

        # Should use merge mode by default
        assert config.models_merge_mode == "merge"

        # Should contain both default and user models
        assert len(config.models_data) == 3  # 2 default + 1 new custom

        # User model should override default
        assert config.models_data["openai/gpt-4o"]["name"] == "GPT-4o Custom"

        # Default model should remain
        assert (
            config.models_data["anthropic/claude-3-sonnet"]["name"]
            == "Claude 3 Sonnet Default"
        )

        # Custom model should be added
        assert config.models_data["custom/my-model"]["name"] == "My Custom Model"


def test_explicit_merge_mode(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    tmp_path: Path,
    merge_config_data: dict,
):
    """Test explicit merge mode configuration."""
    # Create config file with explicit merge mode
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(merge_config_data, f)

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        config = RegistryConfig(
            models_data_path=mock_user_models_file, config_path=config_file
        )

        assert config.models_merge_mode == "merge"
        assert len(config.models_data) == 3
        assert config.models_data["openai/gpt-4o"]["name"] == "GPT-4o Custom"


def test_replace_mode(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    tmp_path: Path,
    replace_config_data: dict,
):
    """Test replace mode (legacy behavior)."""
    # Create config file with replace mode
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(replace_config_data, f)

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        config = RegistryConfig(
            models_data_path=mock_user_models_file, config_path=config_file
        )

        assert config.models_merge_mode == "replace"
        # Should only contain user models
        assert len(config.models_data) == 2
        assert "anthropic/claude-3-sonnet" not in config.models_data
        assert config.models_data["openai/gpt-4o"]["name"] == "GPT-4o Custom"
        assert config.models_data["custom/my-model"]["name"] == "My Custom Model"


def test_extend_mode_success(
    mock_default_models_file: Path,
    tmp_path: Path,
    extend_config_data: dict,
    user_models_no_conflicts: dict,
):
    """Test extend mode with no conflicts."""
    # Create user models file with no conflicts
    user_models_file = tmp_path / "user_models.json"
    with open(user_models_file, "w") as f:
        import json

        json.dump(user_models_no_conflicts, f)

    # Create config file with extend mode
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(extend_config_data, f)

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        config = RegistryConfig(
            models_data_path=user_models_file, config_path=config_file
        )

        assert config.models_merge_mode == "extend"
        # Should contain all models (2 default + 1 custom)
        assert len(config.models_data) == 3
        assert config.models_data["openai/gpt-4o"]["name"] == "GPT-4o Default"
        assert (
            config.models_data["anthropic/claude-3-sonnet"]["name"]
            == "Claude 3 Sonnet Default"
        )
        assert config.models_data["custom/my-model"]["name"] == "My Custom Model"


def test_extend_mode_conflict_error(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    tmp_path: Path,
    extend_config_data: dict,
):
    """Test extend mode raises error on conflicts."""
    # Create config file with extend mode
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(extend_config_data, f)

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        with pytest.raises(ValueError) as exc_info:
            RegistryConfig(
                models_data_path=mock_user_models_file, config_path=config_file
            )

        assert "Model ID conflicts found in extend mode" in str(exc_info.value)
        assert "openai/gpt-4o" in str(exc_info.value)


def test_no_user_models_file(
    mock_default_models_file: Path,
    mock_config_file: Path,
):
    """Test behavior when no user models file exists."""
    non_existent_user_file = Path("/non/existent/path.json")

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        config = RegistryConfig(
            models_data_path=non_existent_user_file, config_path=mock_config_file
        )

        # Should only contain default models
        assert len(config.models_data) == 2
        assert config.models_data["openai/gpt-4o"]["name"] == "GPT-4o Default"
        assert (
            config.models_data["anthropic/claude-3-sonnet"]["name"]
            == "Claude 3 Sonnet Default"
        )


def test_invalid_merge_mode(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    tmp_path: Path,
    invalid_config_data: dict,
):
    """Test error handling for invalid merge mode."""
    # Create config file with invalid merge mode
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(invalid_config_data, f)

    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        # Should raise ValidationError from Pydantic during config validation
        with pytest.raises(ValidationError) as exc_info:
            RegistryConfig(
                models_data_path=mock_user_models_file, config_path=config_file
            )

        # Check the validation error message
        assert "Input should be 'merge', 'replace' or 'extend'" in str(exc_info.value)


def test_merge_mode_logging(
    mock_default_models_file: Path,
    mock_user_models_file: Path,
    mock_config_file: Path,
    caplog,
):
    """Test that merge operations are properly logged."""
    with mock.patch("langgate.registry.config.importlib.resources") as mock_resources:
        # Mock the registry resources to point to our default models file
        mock_files = mock.MagicMock()
        mock_files.joinpath.return_value = mock_default_models_file
        mock_resources.files.return_value = mock_files

        with caplog.at_level("DEBUG"):
            RegistryConfig(
                models_data_path=mock_user_models_file, config_path=mock_config_file
            )

        # Check that debug logging occurred for model override
        debug_messages = [
            record.message for record in caplog.records if record.levelname == "DEBUG"
        ]
        override_logs = [
            msg for msg in debug_messages if "overriding_default_model" in msg
        ]
        assert len(override_logs) > 0

        # Check that info logging occurred with merge mode
        info_messages = [
            record.message for record in caplog.records if record.levelname == "INFO"
        ]
        model_data_logs = [msg for msg in info_messages if "loaded_model_data" in msg]
        assert len(model_data_logs) > 0
