"""Mock objects for registry testing."""

from langgate.core.models import LLMInfo
from langgate.registry.local import BaseLocalRegistryClient


class CustomLLMInfo(LLMInfo):
    """Custom LLMInfo class for testing subclass handling."""

    custom_field: str = "custom_value"


class CustomLocalRegistryClient(BaseLocalRegistryClient[CustomLLMInfo]):
    """Custom LocalRegistryClient implementation for testing.

    This is a non-singleton client that uses the CustomLLMInfo schema.
    """
