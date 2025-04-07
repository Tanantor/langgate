"""Mock objects for client testing."""

from langgate.client.http import BaseHTTPRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo


class CustomHTTPRegistryClient(BaseHTTPRegistryClient[CustomLLMInfo]):
    """Custom HTTP Registry Client implementation for testing.

    This client uses the CustomLLMInfo schema instead of the default LLMInfo.
    """
