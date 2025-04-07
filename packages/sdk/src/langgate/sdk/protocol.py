"""Protocol definitions for LangGate clients."""

from langgate.client import RegistryClientProtocol
from langgate.transform import TransformerClientProtocol


class LangGateLocalProtocol(RegistryClientProtocol, TransformerClientProtocol):
    """Protocol for LangGate local clients."""
