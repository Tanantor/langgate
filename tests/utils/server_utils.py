"""Test utilities for configuration path resolution testing."""

from contextlib import contextmanager
from unittest import mock


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
