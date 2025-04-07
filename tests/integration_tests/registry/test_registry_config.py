"""Tests for environment configuration."""

import os
from pathlib import Path
from unittest import mock

import pytest

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
