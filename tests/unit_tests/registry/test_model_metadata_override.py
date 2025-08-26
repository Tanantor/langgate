"""Test metadata override functionality for models in YAML config."""

import tempfile
from pathlib import Path

from langgate.registry.config import RegistryConfig
from langgate.registry.models import ModelRegistry


def test_yaml_config_overrides_json_metadata_for_llm():
    """Test that metadata in YAML config overrides JSON data for LLM models."""
    # Create a temporary JSON file with model data
    models_json = {
        "gemini/gemini-2.5-pro": {
            "name": "Gemini 2.5 Pro",
            "mode": "text",
            "model_provider": "google",
            "model_provider_name": "Google",
            "capabilities": {
                "supports_tools": True,
                "supports_vision": True,
                "supports_reasoning": True,
                "supports_response_schema": True,
                "supports_system_messages": True,
                "supports_tool_choice": True,
            },
            "context": {"max_input_tokens": 1048576, "max_output_tokens": 65536},
            "costs": {
                "input_cost_per_token": "0.00000125",
                "output_cost_per_token": "0.00001",
            },
            "description": "Gemini 2.5 Pro is Google's state-of-the-art thinking model.",
        }
    }

    # Create a temporary YAML config file with overrides
    yaml_config = """
default_params:
  temperature: 0.7

services:
  gemini:
    api_key: "${GEMINI_API_KEY}"
    base_url: "https://api.gemini.ai"

models:
  text:
    - id: google/gemini-no-reasoning
      service:
        provider: gemini
        model_id: gemini-2.5-pro
      name: Gemini no reasoning
      description: Custom Gemini without reasoning capability
      capabilities:
        supports_tools: true
        supports_vision: true
        supports_response_schema: true
        supports_system_messages: true
        supports_tool_choice: true
        supports_reasoning: false  # Override: disable reasoning
      context:
        max_input_tokens: 100000  # Override: reduce context
        max_output_tokens: 65000  # Override: reduce output
      costs:
        input_cost_per_token: "0.00000200"  # Override: different pricing
        output_cost_per_token: "0.00001500"
      override_params:
        temperature: 0.2
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json

        json.dump(models_json, f)
        models_path = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_config)
        config_path = Path(f.name)

    try:
        # Create registry with test files
        config = RegistryConfig(
            models_data_path=models_path,
            config_path=config_path,
        )
        registry = ModelRegistry(config)

        # Get the model info
        model = registry.get_llm_info("google/gemini-no-reasoning")

        # Assert that YAML overrides are applied
        assert model.name == "Gemini no reasoning"
        assert model.description == "Custom Gemini without reasoning capability"

        # Check capabilities override
        assert model.capabilities.supports_reasoning is False  # Overridden
        assert model.capabilities.supports_tools is True
        assert model.capabilities.supports_vision is True

        # Check context override
        assert model.context_window.max_input_tokens == 100000  # Overridden
        assert model.context_window.max_output_tokens == 65000  # Overridden

        # Check costs override
        assert str(model.costs.input_cost_per_token) == "0.00000200"  # Overridden
        assert str(model.costs.output_cost_per_token) == "0.00001500"  # Overridden

    finally:
        models_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)


def test_yaml_config_partial_override_preserves_json_defaults():
    """Test that partial overrides in YAML preserve non-overridden JSON data."""
    # Create a temporary JSON file with model data
    models_json = {
        "openai/gpt-5": {
            "name": "GPT-5",
            "mode": "text",
            "model_provider": "openai",
            "model_provider_name": "OpenAI",
            "capabilities": {
                "supports_tools": True,
                "supports_vision": True,
                "supports_reasoning": True,
            },
            "context": {"max_input_tokens": 200000, "max_output_tokens": 4096},
            "costs": {
                "input_cost_per_token": "0.00001",
                "output_cost_per_token": "0.00002",
            },
            "description": "OpenAI's GPT-5 model.",
        }
    }

    # Create a YAML config with partial override (only capabilities)
    yaml_config = """
services:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"

models:
  text:
    - id: openai/gpt-5-no-vision
      service:
        provider: openai
        model_id: gpt-5
      capabilities:
        supports_vision: false  # Only override vision capability
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json

        json.dump(models_json, f)
        models_path = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_config)
        config_path = Path(f.name)

    try:
        config = RegistryConfig(
            models_data_path=models_path,
            config_path=config_path,
        )
        registry = ModelRegistry(config)

        model = registry.get_llm_info("openai/gpt-5-no-vision")

        # Original values should be preserved
        assert model.name == "GPT-5"  # From JSON
        assert model.description == "OpenAI's GPT-5 model."  # From JSON
        assert model.context_window.max_input_tokens == 200000  # From JSON
        assert model.context_window.max_output_tokens == 4096  # From JSON
        assert str(model.costs.input_cost_per_token) == "0.00001"  # From JSON

        # Overridden values should be applied
        assert model.capabilities.supports_vision is False  # Overridden
        assert model.capabilities.supports_tools is True  # From JSON
        assert model.capabilities.supports_reasoning is True  # From JSON

    finally:
        models_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)


def test_yaml_config_overrides_for_image_models():
    """Test that metadata overrides work for image generation models."""
    models_json = {
        "openai/dall-e-3": {
            "name": "DALL-E 3",
            "mode": "image",
            "model_provider": "openai",
            "model_provider_name": "OpenAI",
            "costs": {
                "image_generation": {
                    "quality_tiers": {
                        "standard": {
                            "1024x1024": "0.040",
                            "1024x1792": "0.080",
                        },
                        "hd": {
                            "1024x1024": "0.080",
                            "1024x1792": "0.120",
                        },
                    }
                }
            },
            "description": "DALL-E 3 image generation model.",
        }
    }

    yaml_config = """
services:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"

models:
  image:
    - id: openai/dall-e-3-budget
      service:
        provider: openai
        model_id: dall-e-3
      name: DALL-E 3 Budget
      description: Budget version with reduced pricing
      costs:
        image_generation:
          flat_rate: "0.020"  # Override with flat rate pricing
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json

        json.dump(models_json, f)
        models_path = Path(f.name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_config)
        config_path = Path(f.name)

    try:
        config = RegistryConfig(
            models_data_path=models_path,
            config_path=config_path,
        )
        registry = ModelRegistry(config)

        model = registry.get_image_model_info("openai/dall-e-3-budget")

        assert model.name == "DALL-E 3 Budget"
        assert model.description == "Budget version with reduced pricing"
        assert str(model.costs.image_generation.flat_rate) == "0.020"
        assert (
            model.costs.image_generation.quality_tiers is None
        )  # Replaced with flat rate

    finally:
        models_path.unlink(missing_ok=True)
        config_path.unlink(missing_ok=True)
