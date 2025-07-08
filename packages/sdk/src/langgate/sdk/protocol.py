"""Protocol definitions for LangGate clients."""

from langgate.client import RegistryClientProtocol
from langgate.core.models import ImageModelInfo, LLMInfo
from langgate.transform import TransformerClientProtocol


class LangGateLocalProtocol(
    RegistryClientProtocol[LLMInfo, ImageModelInfo], TransformerClientProtocol
):
    """Protocol for LangGate local clients supporting multiple modalities."""
