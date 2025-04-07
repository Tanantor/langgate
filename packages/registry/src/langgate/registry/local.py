"""LocalRegistryClient for direct registry access."""

from collections.abc import Sequence
from datetime import timedelta
from typing import Generic, cast, get_args

from langgate.client.protocol import BaseRegistryClient, LLMInfoT
from langgate.core.logging import get_logger
from langgate.core.models import LLMInfo
from langgate.registry.models import ModelRegistry

logger = get_logger(__name__)


class BaseLocalRegistryClient(BaseRegistryClient[LLMInfoT], Generic[LLMInfoT]):
    """
    Base local registry client that calls ModelRegistry directly.

    This client is used when you want to embed the registry in your application
    rather than connecting to a remote registry service.

    Type Parameters:
        LLMInfoT: The LLMInfo-derived model class to use for responses
    """

    __orig_bases__: tuple
    model_info_cls: type[LLMInfoT]

    def __init__(self, model_info_cls: type[LLMInfoT] | None = None):
        """Initialize the client with a ModelRegistry instance."""
        # Cache refreshing is no-op for local registry clients.
        # Since this client is local, we don't need to refresh the cache.
        # TODO: Move caching to the base HTTP client class instead.
        cache_ttl = timedelta(days=365)
        super().__init__(cache_ttl=cache_ttl)
        self.registry = ModelRegistry()

        # Set model_info_cls if provided, otherwise it is inferred from the class
        if model_info_cls is not None:
            self.model_info_cls = model_info_cls

        logger.debug("initialized_base_local_registry_client")

    def __init_subclass__(cls, **kwargs):
        """Set up model class when this class is subclassed."""
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "model_info_cls"):
            cls.model_info_cls = cls._get_model_info_class()

    @classmethod
    def _get_model_info_class(cls) -> type[LLMInfoT]:
        """Extract the model class from generic type parameters."""
        return get_args(cls.__orig_bases__[0])[0]

    async def _fetch_model_info(self, model_id: str) -> LLMInfoT:
        """Get information about a model directly from registry.

        Args:
            model_id: The ID of the model to get information for

        Returns:
            Information about the requested model with the type expected by this client
        """
        # Get the model info from the registry (always returns LLMInfo)
        info = self.registry.get_model_info(model_id)

        # If model_info_cls is LLMInfo (not a subclass), we can return it as-is
        if self.model_info_cls is LLMInfo:
            return cast(LLMInfoT, info)

        # Otherwise, validate against the subclass schema
        return self.model_info_cls.model_validate(info.model_dump())

    async def _fetch_all_models(self) -> Sequence[LLMInfoT]:
        """List all available models directly from registry.

        Returns:
            A sequence of model information objects of the type expected by this client.
        """
        models = self.registry.list_models()

        # If model_info_cls is LLMInfo (not a subclass), we can return the list as-is
        if self.model_info_cls is LLMInfo:
            return cast(Sequence[LLMInfoT], models)

        # Otherwise, we need to validate each model against the subclass schema
        return [
            self.model_info_cls.model_validate(model.model_dump()) for model in models
        ]


class LocalRegistryClient(BaseLocalRegistryClient[LLMInfo]):
    """
    Local registry client that calls ModelRegistry directly using the default LLMInfo schema.

    This is implemented as a singleton for convenient access across an application.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the client with a ModelRegistry instance."""
        if not hasattr(self, "_initialized"):
            super().__init__()
            self._initialized = True
            logger.debug("initialized_local_registry_client_singleton")
