"""Unit tests for the parameter transformer."""

import os
from unittest import mock

from langgate.transform import ParamTransformer


def test_param_transformer_defaults():
    """Test applying default parameters."""
    transformer = ParamTransformer()
    transformer.with_defaults({"temperature": 0.7, "max_tokens": 1000})

    # Empty user params should get defaults
    result = transformer.transform({})
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 1000

    # User params should override defaults
    result = transformer.transform({"temperature": 0.5})
    assert result["temperature"] == 0.5
    assert result["max_tokens"] == 1000


def test_param_transformer_overrides():
    """Test applying override parameters."""
    transformer = ParamTransformer()
    transformer.with_defaults({"temperature": 0.7, "max_tokens": 1000})
    transformer.with_overrides({"max_tokens": 2000, "top_p": 0.9})

    # Overrides should take precedence over defaults and user params
    result = transformer.transform({"temperature": 0.5, "max_tokens": 500})
    assert result["temperature"] == 0.5  # User param not in overrides
    assert result["max_tokens"] == 2000  # Override takes precedence
    assert result["top_p"] == 0.9  # Override added


def test_param_transformer_removing():
    """Test removing parameters."""
    transformer = ParamTransformer()
    transformer.with_defaults({"temperature": 0.7, "max_tokens": 1000, "top_p": 0.9})
    transformer.removing(["top_p"])

    # Removed params should not appear in result
    result = transformer.transform({"temperature": 0.5, "top_p": 0.8})
    assert result["temperature"] == 0.5
    assert result["max_tokens"] == 1000
    assert "top_p" not in result


def test_param_transformer_renaming():
    """Test renaming parameters."""
    transformer = ParamTransformer()
    transformer.with_defaults({"temperature": 0.7, "max_tokens": 1000})
    transformer.renaming({"temperature": "temp", "max_tokens": "maxOutputTokens"})

    # Params should be renamed in result
    result = transformer.transform({"temperature": 0.5, "max_tokens": 500})
    assert "temperature" not in result
    assert "max_tokens" not in result
    assert result["temp"] == 0.5
    assert result["maxOutputTokens"] == 500


def test_param_transformer_model_id():
    """Test setting model ID."""
    transformer = ParamTransformer()
    transformer.with_model_id("gpt-4o")

    # Model ID should be set in result
    result = transformer.transform({})
    assert result["model"] == "gpt-4o"

    # User-provided model should be overridden
    result = transformer.transform({"model": "gpt-3.5-turbo"})
    assert result["model"] == "gpt-4o"


def test_param_transformer_env_vars():
    """Test substituting environment variables."""
    with mock.patch.dict(
        os.environ, {"API_KEY": "test-key", "BASE_URL": "http://test.com"}
    ):
        transformer = ParamTransformer()
        transformer.with_defaults(
            {
                "api_key": "${API_KEY}",
                "base_url": "${BASE_URL}",
                "other": "${MISSING_VAR}",
            }
        )
        transformer.with_env_vars()

        result = transformer.transform({})
        assert result["api_key"] == "test-key"
        assert result["base_url"] == "http://test.com"
        assert (
            result["other"] == "${MISSING_VAR}"
        )  # Missing vars should remain as placeholders


def test_param_transformer_chaining():
    """Test chaining multiple transformations."""
    with mock.patch.dict(os.environ, {"API_KEY": "test-key"}):
        transformer = ParamTransformer()

        # Chain multiple transformations
        transformer.with_defaults(
            {
                "temperature": 0.7,
                "max_tokens": 1000,
                "api_key": "${API_KEY}",
            }
        )
        transformer.with_overrides({"max_tokens": 2000})
        transformer.removing(["frequency_penalty"])
        transformer.renaming({"temperature": "temp"})
        transformer.with_model_id("gpt-4o")
        transformer.with_env_vars()

        # Apply all transformations
        result = transformer.transform(
            {
                "temperature": 0.5,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.1,
            }
        )

        # Check result
        assert "temperature" not in result
        assert result["temp"] == 0.5
        assert result["max_tokens"] == 2000
        assert result["model"] == "gpt-4o"
        assert result["api_key"] == "test-key"
        assert "frequency_penalty" not in result
        assert result["presence_penalty"] == 0.1
