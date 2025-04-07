"""Transformer test fixtures."""

import os
from unittest import mock

import pytest_asyncio

from langgate.transform.local import LocalTransformerClient


@pytest_asyncio.fixture
async def local_transformer_client(mock_config_yaml, mock_env_file):
    """Create a LocalTransformerClient instance with mock files."""
    # Reset singleton
    LocalTransformerClient._instance = None

    # Mock environment variables
    with mock.patch.dict(
        os.environ,
        {
            "LANGGATE_CONFIG": str(mock_config_yaml),
            "LANGGATE_ENV_FILE": str(mock_env_file),
            "OPENAI_API_KEY": "sk-test-123",
            "ANTHROPIC_API_KEY": "sk-ant-test-123",
        },
    ):
        # Create a fresh client instance
        client = LocalTransformerClient()
        yield client
        # Reset singleton after test
        LocalTransformerClient._instance = None
