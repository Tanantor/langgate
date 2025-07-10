"""Mock objects for client testing."""

from langgate.client.http import BaseHTTPRegistryClient
from tests.mocks.registry_mocks import CustomImageModelInfo, CustomLLMInfo


class CustomHTTPRegistryClient(
    BaseHTTPRegistryClient[CustomLLMInfo, CustomImageModelInfo]
):
    """Custom HTTP Registry Client implementation for testing.

    This client uses custom schemas for both LLM and image models.
    """
