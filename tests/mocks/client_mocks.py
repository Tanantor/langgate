"""Mock objects for client testing."""

from langgate.client.http import BaseHTTPRegistryClient
from langgate.core.models import ImageModelInfo
from tests.mocks.registry_mocks import CustomLLMInfo


class CustomHTTPRegistryClient(BaseHTTPRegistryClient[CustomLLMInfo, ImageModelInfo]):
    """Custom HTTP Registry Client implementation for testing.

    This client uses the CustomLLMInfo schema and default ImageModelInfo.
    """
