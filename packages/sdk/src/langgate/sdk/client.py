"""Combined client for LangGate."""

from collections.abc import Sequence
from typing import Any

from langgate.client import RegistryClientProtocol
from langgate.core.logging import get_logger
from langgate.core.models import LLMInfo
from langgate.registry import LocalRegistryClient
from langgate.sdk.protocol import LangGateLocalProtocol
from langgate.transform import LocalTransformerClient, TransformerClientProtocol

logger = get_logger(__name__)


class LangGateLocal(LangGateLocalProtocol):
    """
    Combined client for LangGate providing access to both registry and transform functionality.

    This client is a convenience wrapper that gives access to both model information
    and parameter transformation in a single interface.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("creating_langgate_local_client_singleton")
        return cls._instance

    def __init__(
        self,
        registry: RegistryClientProtocol | None = None,
        transformer: TransformerClientProtocol | None = None,
    ):
        """Initialize the client with registry and transformer instances."""
        if not hasattr(self, "_initialized"):
            self.registry = registry or LocalRegistryClient()
            self.transformer = transformer or LocalTransformerClient()
            self._initialized = True
            logger.debug("initialized_langgate_local_client_singleton")

    async def get_model_info(self, model_id: str) -> LLMInfo:
        """Get model information by ID.

        Args:
            model_id: The ID of the model to get information for

        Returns:
            Information about the requested model

        Raises:
            ValueError: If the model is not found
        """
        return await self.registry.get_model_info(model_id)

    async def list_models(self) -> Sequence[LLMInfo]:
        """List all available models.

        Returns:
            A list of all available models
        """
        return await self.registry.list_models()

    async def get_params(
        self, model_id: str, input_params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Transform parameters for the specified model.

        Args:
            model_id: The ID of the model to transform parameters for
            input_params: The parameters to transform

        Returns:
            A tuple containing (api_format, transformed_parameters)

        Raises:
            ValueError: If the model is not found
        """
        return await self.transformer.get_params(model_id, input_params)
