"""Test utilities for configuration path resolution testing."""

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal
from unittest import mock

from tests.mocks.registry_mocks import create_mock_config


@contextmanager
def patch_load_yaml_config(custom_config=None):
    """Patch the load_yaml_config function to return a mock config."""
    mock_config = custom_config if custom_config else create_mock_config()
    with (
        mock.patch(
            "langgate.transform.local.load_yaml_config",
            return_value=mock_config,
        ),
        mock.patch(
            "langgate.registry.config.load_yaml_config",
            return_value=mock_config,
        ),
    ):
        yield mock_config


@contextmanager
def patch_config_paths_arg_source(arg_path: Path):
    """Mock path resolution for constructor arguments source.

    This utility patches only what's needed to test constructor argument paths.

    Args:
        arg_path: The path provided as constructor argument.
    """
    with mock.patch("pathlib.Path.exists", return_value=True), patch_load_yaml_config():
        yield


@contextmanager
def patch_config_paths_env_source(env_var: str, env_value: str):
    """Mock path resolution for environment variable source.

    This utility patches only what's needed to test environment variable paths.

    Args:
        env_var: The environment variable name to set.
        env_value: The value to set the environment variable to.
    """
    env_vars = {env_var: env_value}
    with (
        mock.patch.dict(os.environ, env_vars, clear=True),
        mock.patch("pathlib.Path.exists", return_value=True),
        patch_load_yaml_config(),
    ):
        yield


@contextmanager
def patch_config_paths_cwd_source(pattern: str):
    """Mock path resolution for current working directory source.

    This utility patches only what's needed to test CWD paths.

    Args:
        pattern: The filename pattern to match in the mocked CWD.
    """
    # Mock CWD to a known path
    with mock.patch("pathlib.Path.cwd", return_value=Path("/fake/cwd")):
        # Create a path-specific exists check
        def patched_exists(path_obj):
            path_str = str(path_obj)
            return pattern in path_str and "/fake/cwd" in path_str

        with (
            mock.patch.object(Path, "exists", patched_exists),
            patch_load_yaml_config(),
        ):
            yield


@contextmanager
def patch_config_paths_default_source():
    """Mock path resolution for package directory fallback source.

    This utility patches only what's needed to test package directory fallback paths.
    """
    # Make all explicit paths not exist to force default package dir
    with (
        mock.patch("pathlib.Path.exists", return_value=False),
        patch_load_yaml_config(),
    ):
        yield


@contextmanager
def config_path_resolver(
    source_type: Literal["arg", "env", "cwd", "package_dir"],
    file_type: Literal["models_json", "config_yaml", "env_file"],
    path_value: str | None = None,
):
    """Combined context manager for all path resolution types.

    This reuses the existing context managers but provides a more consistent interface.

    Args:
        source_type: The type of source to patch (arg, env, cwd, package_dir)
        file_type: The type of file being patched (models_json, config_yaml, env_file)
        path_value: The path value to use (required for all except package_dir)
    """
    env_var_map = {
        "models_json": "LANGGATE_MODELS",
        "config_yaml": "LANGGATE_CONFIG",
        "env_file": "LANGGATE_ENV_FILE",
    }

    default_pattern_map = {
        "models_json": "langgate_models.json",
        "config_yaml": "langgate_config.yaml",
        "env_file": ".env",
    }

    with ExitStack() as stack:
        if source_type == "arg":
            # For constructor argument paths, we just need to make Path.exists return True
            if path_value is None:
                raise ValueError("path_value cannot be None for arg source_type")
            stack.enter_context(patch_config_paths_arg_source(Path(path_value)))

        elif source_type == "env":
            # For environment variable paths
            if path_value is None:
                raise ValueError("path_value cannot be None for env source_type")
            env_var = env_var_map[file_type]
            stack.enter_context(patch_config_paths_env_source(env_var, path_value))

        elif source_type == "cwd":
            # For current working directory paths
            pattern = default_pattern_map[file_type]
            stack.enter_context(patch_config_paths_cwd_source(pattern))

        elif source_type == "package_dir":
            # For package directory fallback paths
            stack.enter_context(patch_config_paths_default_source())

        yield


@contextmanager
def prevent_server_config_loading():
    """Prevent actual config file loading in ServerConfig.

    Use this to avoid actual file operations during server config tests.
    """
    with mock.patch(
        "langgate.server.core.config.FixedYamlConfigSettingsSource.__call__",
        return_value={},
    ):
        yield
