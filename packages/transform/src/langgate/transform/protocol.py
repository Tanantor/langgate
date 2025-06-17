"""Protocol definitions for parameter transformers."""

from typing import Any, Protocol

from langgate.core.logging import get_logger

logger = get_logger(__name__)


class TransformerClientProtocol(Protocol):
    """Protocol for transformer clients."""

    async def get_params(
        self, model_id: str, input_params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Get transformed parameters for the specified model.

        Args:
            model_id: The ID of the model to get transformed parameters for
            input_params: The parameters to transform

        Returns:
            A tuple containing (api_format, transformed_parameters)
        """
        ...
