"""Integration tests for LocalTransformerClient."""

import os
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
from pydantic.types import SecretStr

from langgate.transform.local import LocalTransformerClient
from tests.utils.config_utils import config_path_resolver


@pytest.mark.parametrize(
    "source,expected_path",
    [
        ("arg", "/arg/path/langgate_config.yaml"),
        ("env", "/env/path/langgate_config.yaml"),
        ("cwd", "langgate_config.yaml"),
        ("package_dir", "default_config.yaml"),
    ],
    ids=["arg_path", "env_var", "cwd_path", "package_dir_path"],
)
def test_transformer_config_yaml_paths(
    source: Literal["arg", "env", "cwd", "package_dir"], expected_path: str
):
    """Test path resolution for config YAML file with different sources."""
    # Reset singleton for each case
    LocalTransformerClient._instance = None

    with config_path_resolver(source, "config_yaml", expected_path):
        # For arg source, we need to explicitly pass the path
        if source == "arg":
            client = LocalTransformerClient(config_path=Path(expected_path))
        else:
            client = LocalTransformerClient()

        assert expected_path in str(client.config_path)


@pytest.mark.parametrize(
    "source,expected_path",
    [
        ("arg", "/arg/path/.env"),
        ("env", "/env/path/.env"),
        ("cwd", ".env"),
    ],
    ids=["arg_path", "env_var", "cwd_path"],
)
def test_transformer_env_file_paths(source, expected_path):
    """Test path resolution for .env file with different sources."""
    # Reset singleton for each case
    LocalTransformerClient._instance = None

    with config_path_resolver(source, "env_file", expected_path):
        # For arg source, we need to explicitly pass the path
        if source == "arg":
            client = LocalTransformerClient(env_file_path=Path(expected_path))
        else:
            client = LocalTransformerClient()

        assert expected_path in str(client.env_file_path)


@pytest.mark.asyncio
async def test_transformer_config_loading(
    local_transformer_client: LocalTransformerClient,
):
    """Test that configuration is properly loaded."""
    # Check service configs are loaded
    assert "openai" in local_transformer_client._service_config
    assert "anthropic" in local_transformer_client._service_config

    # Check global defaults are loaded
    assert "temperature" in local_transformer_client._global_config["default_params"]

    # Check model mappings are processed
    assert "gpt-4o" in local_transformer_client._model_mappings
    assert (
        "anthropic/claude-sonnet-4-reasoning"
        in local_transformer_client._model_mappings
    )


@pytest.mark.asyncio
async def test_transformer_global_defaults(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying global default parameters."""
    # Test with empty params
    result = await local_transformer_client.get_params("gpt-4o", {})

    # Global default should be applied
    assert result["temperature"] == 0.7


@pytest.mark.asyncio
async def test_transformer_service_provider_defaults(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying service provider default parameters."""
    # OpenAI service provider defaults
    result = await local_transformer_client.get_params("gpt-4o", {})
    assert result["max_tokens"] == 1000

    # Anthropic service provider defaults
    result = await local_transformer_client.get_params("anthropic/claude-sonnet-4", {})
    assert result["max_tokens"] == 2000


@pytest.mark.asyncio
async def test_transformer_model_pattern_params(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying model pattern specific parameters."""
    # Anthropic reasoning pattern
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", {}
    )

    # Pattern should apply thinking parameter and remove temperature
    assert result["thinking"]["type"] == "enabled"
    assert "temperature" not in result

    # Pattern should apply default max_tokens from model_patterns
    assert result["max_tokens"] == 64000


@pytest.mark.asyncio
async def test_transformer_model_pattern_defaults(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying default parameters at the model pattern level."""
    # Use empty params to ensure defaults are applied
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", {}
    )

    # The pattern has default_params with max_tokens
    assert result["max_tokens"] == 64000

    # User params should still override pattern defaults
    user_params = {"max_tokens": 1000}
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", user_params
    )
    assert result["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_transformer_model_pattern_renames(
    local_transformer_client: LocalTransformerClient,
):
    """Test parameter renaming at the model pattern level."""
    # Anthropic reasoning pattern has reasoning -> thinking rename
    user_params = {"reasoning": {"depth": "deep"}, "temperature": 0.7}
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", user_params
    )

    # Check parameter was renamed via pattern's rename_params
    assert "reasoning" not in result
    assert "thinking" in result
    assert result["thinking"]["depth"] == "deep"
    # Temperature should be removed by pattern's remove_params
    assert "temperature" not in result


@pytest.mark.asyncio
async def test_transformer_model_specific_overrides(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying model-specific override parameters."""
    # Model with specific thinking override
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", {}
    )

    # Check model-specific override is applied
    assert result["thinking"]["budget_tokens"] == 1024


@pytest.mark.asyncio
async def test_transformer_user_params_precedence(
    local_transformer_client: LocalTransformerClient,
):
    """Test that user parameters have precedence over defaults."""
    # User params should override defaults
    result = await local_transformer_client.get_params(
        "gpt-4o", {"temperature": 0.5, "max_tokens": 500}
    )
    assert result["temperature"] == 0.5  # User specified
    assert result["max_tokens"] == 500  # User specified


@pytest.mark.asyncio
async def test_transformer_api_key_resolution(
    local_transformer_client: LocalTransformerClient,
):
    """Test that API keys are resolved from environment variables."""
    # OpenAI API key
    result = await local_transformer_client.get_params("gpt-4o", {})
    assert isinstance(result["api_key"], SecretStr)
    assert result["api_key"].get_secret_value() == "sk-test-123"

    # Anthropic API key
    result = await local_transformer_client.get_params("anthropic/claude-sonnet-4", {})
    assert isinstance(result["api_key"], SecretStr)
    assert result["api_key"].get_secret_value() == "sk-ant-test-123"


@pytest.mark.asyncio
async def test_transformer_model_specific_renames(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying model-specific parameter renaming."""
    # The claude-sonnet-4 model has stop -> stop_sequences rename
    user_params = {"stop": ["END"], "temperature": 0.7}
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4", user_params
    )

    # Check parameter was renamed
    assert "stop" not in result
    assert "stop_sequences" in result
    assert result["stop_sequences"] == ["END"]


@pytest.mark.asyncio
async def test_transformer_remove_params(
    local_transformer_client: LocalTransformerClient,
):
    """Test removing parameters based on configuration."""
    # The anthropic/claude-sonnet-4 model removes response_format and reasoning
    user_params = {
        "response_format": {"type": "json_object"},
        "reasoning": {"depth": "deep"},
        "presence_penalty": 0.2,
    }
    result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4", user_params
    )

    # These params should be removed
    assert "response_format" not in result
    assert "reasoning" not in result
    # presence_penalty should remain
    if "presence_penalty" in result:
        assert result["presence_penalty"] == 0.2


@pytest.mark.asyncio
async def test_transformer_invalid_model(
    local_transformer_client: LocalTransformerClient,
):
    """Test error handling for invalid model ID."""
    invalid_model_id = "invalid-model"
    with pytest.raises(
        ValueError, match=f"Model '{invalid_model_id}' not found in configuration"
    ):
        await local_transformer_client.get_params(invalid_model_id, {})


@pytest.mark.asyncio
async def test_transformer_without_env_file():
    """Test that transformer works when .env file doesn't exist."""
    # Reset singleton
    LocalTransformerClient._instance = None

    with (
        mock.patch("pathlib.Path.exists", return_value=False),
        mock.patch.dict(os.environ, {"OPENAI_API_KEY": "direct-env-var"}),
    ):
        # Create client with non-existent files
        client = LocalTransformerClient()

        # Should still work with environment variables directly
        model_id = "gpt-4o"
        with pytest.raises(
            ValueError, match=f"Model '{model_id}' not found in configuration"
        ):
            await client.get_params(model_id, {})
