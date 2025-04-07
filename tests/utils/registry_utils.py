import os
from contextlib import ExitStack, contextmanager
from unittest import mock

from langgate.registry.models import ModelRegistry


# Specific config implementations
@contextmanager
def prevent_registry_config_loading():
    """Prevent actual config file loading in RegistryConfig.

    Use this to avoid actual file operations during registry config tests.
    """
    with mock.patch(
        "langgate.registry.config.RegistryConfig._load_config", return_value=None
    ):
        yield


@contextmanager
def patch_model_registry(
    env_vars: dict[str, str] | None = None, reset_singleton: bool = True
):
    """Context manager for mocking ModelRegistry dependencies.

    Args:
        env_vars: Environment variables to mock.
        reset_singleton: Whether to reset the ModelRegistry singleton.
    """
    if reset_singleton:
        ModelRegistry._instance = None

    with ExitStack() as stack:
        stack.enter_context(mock.patch("builtins.open", mock.mock_open()))
        stack.enter_context(mock.patch("pathlib.Path.exists", return_value=True))
        stack.enter_context(prevent_registry_config_loading())

        if env_vars is not None:
            stack.enter_context(mock.patch.dict(os.environ, env_vars, clear=True))

        yield
