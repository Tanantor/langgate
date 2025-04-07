"""Tests for environment configuration."""

from langgate.registry.models import ModelRegistry
from tests.utils.registry_utils import (
    patch_model_registry,
)


def test_model_registry_singleton_instance():
    """Test that ModelRegistry is a singleton."""
    with patch_model_registry():
        registry1 = ModelRegistry()
        # Create second instance
        registry2 = ModelRegistry()
        # Test singleton behavior
        assert registry1 is registry2
        # Both instances should share the same state
        assert id(registry1._models_cache) == id(registry2._models_cache)
