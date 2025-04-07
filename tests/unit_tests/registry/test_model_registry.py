"""Unit tests for the model registry."""

from pathlib import Path
from unittest import mock

from langgate.registry.models import ModelRegistry
from tests.factories.models import LLMInfoFactory
from tests.utils.registry_utils import patch_model_registry


def test_factory_generated_models():
    """Test that factory-generated models are valid."""
    # Create a model with the factory
    model_dict = LLMInfoFactory.create()

    # Validate model fields
    assert isinstance(model_dict, dict)
    assert "id" in model_dict
    assert "name" in model_dict
    assert "provider" in model_dict
    assert "costs" in model_dict
    assert "capabilities" in model_dict
    assert "context_window" in model_dict
    assert "updated_dt" in model_dict


def test_model_registry_works_without_env_file():
    """Test that the model registry works without an environment file."""
    # Create temporary config and models files
    with mock.patch.object(Path, "exists") as mock_exists:
        # Mock the env file to NOT exist
        def side_effect(path):
            # Convert both to strings to avoid Path equality issues
            return ".env" not in str(path)

        mock_exists.side_effect = side_effect

        # Use a mock registry to avoid actual file operations
        with patch_model_registry():
            # Should not raise any exceptions
            registry = ModelRegistry()

            # Verify we have models available
            models = registry.list_models()
            assert isinstance(models, list)

            # We may have 0 models in this test case since we're using a mock registry
            # The important part is that no exception was raised due to missing .env
