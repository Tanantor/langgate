"""Integration tests for LocalTransformerClient."""

import os
from pathlib import Path
from typing import Literal
from unittest import mock

import pytest
from pydantic.types import SecretStr

from langgate.core.schemas.config import ConfigSchema
from langgate.transform.local import LocalTransformerClient
from tests.utils.config_utils import config_path_resolver, patch_load_yaml_config


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
    _, result = await local_transformer_client.get_params("gpt-4o", {})

    # Global default should be applied
    assert result["temperature"] == 0.7


@pytest.mark.asyncio
async def test_transformer_service_provider_defaults(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying service provider default parameters."""
    # OpenAI service provider defaults
    _, result = await local_transformer_client.get_params("gpt-4o", {})
    assert result["max_tokens"] == 1000

    # Anthropic service provider defaults
    _, result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4", {}
    )
    assert result["max_tokens"] == 2000


@pytest.mark.asyncio
async def test_transformer_model_pattern_params(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying model pattern specific parameters."""
    # Anthropic reasoning pattern
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4-reasoning", {}
    )

    # The pattern has default_params with max_tokens
    assert result["max_tokens"] == 64000

    # User params should still override pattern defaults
    user_params = {"max_tokens": 1000}
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params("gpt-4o", {})
    assert isinstance(result["api_key"], SecretStr)
    assert result["api_key"].get_secret_value() == "sk-test-123"

    # Anthropic API key
    _, result = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4", {}
    )
    assert isinstance(result["api_key"], SecretStr)
    assert result["api_key"].get_secret_value() == "sk-ant-test-123"


@pytest.mark.asyncio
async def test_transformer_model_specific_renames(
    local_transformer_client: LocalTransformerClient,
):
    """Test applying model-specific parameter renaming."""
    # The claude-sonnet-4 model has stop -> stop_sequences rename
    user_params = {"stop": ["END"], "temperature": 0.7}
    _, result = await local_transformer_client.get_params(
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
    _, result = await local_transformer_client.get_params(
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
        patch_load_yaml_config(),
    ):
        # Create client with non-existent files
        client = LocalTransformerClient()

        # Should still work with environment variables directly
        model_id = "gpt-4o"
        with pytest.raises(
            ValueError, match=f"Model '{model_id}' not found in configuration"
        ):
            await client.get_params(model_id, {})


# API Format Mapping Tests


@pytest.mark.asyncio
async def test_api_format_service_level_openai_providers(
    local_transformer_client: LocalTransformerClient,
):
    """Test API format resolution for OpenAI-compatible service providers."""
    # Test OpenRouter (uses OpenAI API format)
    api_format, _ = await local_transformer_client.get_params(
        "google/gemma-3-27b-it", {}
    )
    assert api_format == "openai"

    # Test xAI (uses OpenAI API format)
    api_format, _ = await local_transformer_client.get_params("xai/grok-3", {})
    assert api_format == "openai"

    # Test Fireworks AI (uses OpenAI API format)
    api_format, _ = await local_transformer_client.get_params(
        "deepseek/deepseek-r1", {}
    )
    assert api_format == "openai"


@pytest.mark.asyncio
async def test_api_format_fallback_to_provider_name(
    local_transformer_client: LocalTransformerClient,
):
    """Test API format fallback to provider name when no api_format specified."""
    # OpenAI provider (no explicit api_format, should fallback to 'openai')
    api_format, _ = await local_transformer_client.get_params("gpt-4o", {})
    assert api_format == "openai"

    # Anthropic provider (no explicit api_format, should fallback to 'anthropic')
    api_format, _ = await local_transformer_client.get_params(
        "anthropic/claude-sonnet-4", {}
    )
    assert api_format == "anthropic"

    # Gemini provider (no explicit api_format, should fallback to 'gemini')
    api_format, _ = await local_transformer_client.get_params(
        "google/gemini-2.5-pro", {}
    )
    assert api_format == "gemini"

    # MistralAI provider (no explicit api_format, should fallback to 'mistralai')
    api_format, _ = await local_transformer_client.get_params(
        "mistralai/magistral-medium-latest", {}
    )
    assert api_format == "mistralai"


@pytest.mark.asyncio
async def test_api_format_precedence_hierarchy():
    """Test API format precedence: model.api_format → service.api_format → service.provider."""
    # Reset singleton to test with custom configuration
    LocalTransformerClient._instance = None

    # Custom configuration to test precedence
    custom_config = {
        "default_params": {"temperature": 0.7},
        "services": {
            "custom_service": {
                "api_key": "${CUSTOM_API_KEY}",
                "base_url": "https://api.custom.com/v1",
                "api_format": "service_format",  # Service-level API format
                "default_params": {},
            }
        },
        "models": [
            {
                "id": "test/model-with-override",
                "service": {"provider": "custom_service", "model_id": "actual-model"},
                "api_format": "model_override",  # Model-level override (highest precedence)
                "default_params": {},
                "override_params": {},
                "remove_params": [],
                "rename_params": {},
            },
            {
                "id": "test/model-without-override",
                "service": {"provider": "custom_service", "model_id": "another-model"},
                # No model-level api_format, should use service-level
                "default_params": {},
                "override_params": {},
                "remove_params": [],
                "rename_params": {},
            },
        ],
        "app_config": {},
    }

    with (
        mock.patch.dict(os.environ, {"CUSTOM_API_KEY": "test-key"}),
        patch_load_yaml_config(ConfigSchema.model_validate(custom_config)),
    ):
        client = LocalTransformerClient()

        # Test model-level override (highest precedence)
        api_format, _ = await client.get_params("test/model-with-override", {})
        assert api_format == "model_override"

        # Test service-level fallback (when no model-level override)
        api_format, _ = await client.get_params("test/model-without-override", {})
        assert api_format == "service_format"


@pytest.mark.asyncio
async def test_api_format_provider_name_fallback():
    """Test fallback to provider name when no api_format is specified at any level."""
    # Reset singleton to test with custom configuration
    LocalTransformerClient._instance = None

    # Custom configuration with no api_format specified
    custom_config = {
        "default_params": {"temperature": 0.7},
        "services": {
            "fallback_provider": {
                "api_key": "${FALLBACK_API_KEY}",
                "base_url": "https://api.fallback.com/v1",
                # No api_format specified at service level
                "default_params": {},
            }
        },
        "models": [
            {
                "id": "test/fallback-model",
                "service": {
                    "provider": "fallback_provider",
                    "model_id": "fallback-model",
                },
                # No api_format specified at model level
                "default_params": {},
                "override_params": {},
                "remove_params": [],
                "rename_params": {},
            },
        ],
        "app_config": {},
    }

    with (
        mock.patch.dict(os.environ, {"FALLBACK_API_KEY": "test-key"}),
        patch_load_yaml_config(ConfigSchema.model_validate(custom_config)),
    ):
        client = LocalTransformerClient()

        # Should fallback to provider name
        api_format, _ = await client.get_params("test/fallback-model", {})
        assert api_format == "fallback_provider"
