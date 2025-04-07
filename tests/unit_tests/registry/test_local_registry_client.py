"""Unit tests for LocalRegistryClient."""

from langgate.registry.local import BaseLocalRegistryClient, LocalRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo, CustomLocalRegistryClient
from tests.utils.registry_utils import patch_model_registry


def test_local_registry_client_is_singleton():
    """Test that LocalRegistryClient implements the singleton pattern."""
    with patch_model_registry(reset_singleton=True):
        # Reset singleton for test
        LocalRegistryClient._instance = None

        # Create instances
        client1 = LocalRegistryClient()
        client2 = LocalRegistryClient()

        # Verify they are the same instance
        assert client1 is client2

        # Verify they share the same registry
        assert id(client1.registry) == id(client2.registry)


def test_custom_client_type_parameter_extraction():
    """Test that type parameters are correctly extracted from subclasses."""
    with patch_model_registry():
        # CustomLocalRegistryClient subclasses BaseLocalRegistryClient with CustomLLMInfo
        client = CustomLocalRegistryClient()

        # Verify the generic type parameter extraction worked
        assert client.model_info_cls is CustomLLMInfo


def test_explicit_model_class_parameter():
    """Test passing explicit model_info_cls to BaseLocalRegistryClient."""
    with patch_model_registry():
        # When using directly with type parameter, model_info_cls must be provided explicitly
        client = BaseLocalRegistryClient[CustomLLMInfo](model_info_cls=CustomLLMInfo)

        # Verify the model_info_cls is correctly set
        assert client.model_info_cls is CustomLLMInfo
