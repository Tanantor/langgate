"""Combined client for LangGate."""

from collections.abc import Sequence
from typing import Any

from langgate.client import RegistryClientProtocol
from langgate.core.logging import get_logger
from langgate.core.models import ImageModelInfo, LLMInfo
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
        registry: RegistryClientProtocol[LLMInfo, ImageModelInfo] | None = None,
        transformer: TransformerClientProtocol | None = None,
    ):
        """Initialize the client with registry and transformer instances."""
        if not hasattr(self, "_initialized"):
            self.registry = registry or LocalRegistryClient()
            self.transformer = transformer or LocalTransformerClient()
            self._initialized = True
            logger.debug("initialized_langgate_local_client_singleton")

    # LLM methods
    async def get_llm_info(self, model_id: str) -> LLMInfo:
        """Get LLM information by ID.

        Args:
            model_id: The ID of the LLM to get information for

        Returns:
            Information about the requested LLM

        Raises:
            ValueError: If the LLM is not found
        """
        return await self.registry.get_llm_info(model_id)

    async def list_llms(self) -> Sequence[LLMInfo]:
        """List all available LLMs.

        Returns:
            A list of all available LLMs
        """
        return await self.registry.list_llms()

    # Image model methods
    async def get_image_model_info(self, model_id: str) -> ImageModelInfo:
        """Get image model information by ID.

        Args:
            model_id: The ID of the image model to get information for

        Returns:
            Information about the requested image model

        Raises:
            ValueError: If the image model is not found
        """
        return await self.registry.get_image_model_info(model_id)

    async def list_image_models(self) -> Sequence[ImageModelInfo]:
        """List all available image models.

        Returns:
            A list of all available image models
        """
        return await self.registry.list_image_models()

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
