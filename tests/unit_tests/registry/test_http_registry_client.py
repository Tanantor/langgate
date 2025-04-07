"""Unit tests for HTTPRegistryClient."""

from langgate.client.http import BaseHTTPRegistryClient, HTTPRegistryClient
from tests.mocks.client_mocks import CustomHTTPRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo


def test_http_registry_client_is_singleton():
    """Test that HTTPRegistryClient implements the singleton pattern."""
    HTTPRegistryClient._instance = None

    # Different URLs shouldn't matter
    client1 = HTTPRegistryClient(base_url="http://test-server")
    client2 = HTTPRegistryClient(base_url="http://different-server")

    assert client1 is client2

    HTTPRegistryClient._instance = None


def test_custom_client_type_parameter_extraction():
    """Test that type parameters are correctly extracted from subclasses."""
    client = CustomHTTPRegistryClient(base_url="http://test-server")

    # Verify the generic type parameter extraction worked
    assert client.model_info_cls is CustomLLMInfo


def test_explicit_model_class_parameter():
    """Test passing explicit model_info_cls to BaseHTTPRegistryClient."""
    # When using directly with type parameter, model_info_cls must be provided explicitly
    client = BaseHTTPRegistryClient[CustomLLMInfo](
        base_url="http://test-server",
        model_info_cls=CustomLLMInfo,
    )

    assert client.model_info_cls is CustomLLMInfo
