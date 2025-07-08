"""LocalRegistryClient for direct registry access."""

from collections.abc import Sequence
from datetime import timedelta
from typing import Generic, cast, get_args

from langgate.client.protocol import BaseRegistryClient, ImageInfoT, LLMInfoT
from langgate.core.logging import get_logger
from langgate.core.models import ImageModelInfo, LLMInfo
from langgate.registry.models import ModelRegistry

logger = get_logger(__name__)


class BaseLocalRegistryClient(
    BaseRegistryClient[LLMInfoT, ImageInfoT], Generic[LLMInfoT, ImageInfoT]
):
    """
    Base local registry client that calls ModelRegistry directly.

    This client is used when you want to embed the registry in your application
    rather than connecting to a remote registry service.

    Type Parameters:
        LLMInfoT: The LLMInfo-derived model class to use for responses
        ImageInfoT: The ImageModelInfo-derived model class to use for responses
    """

    __orig_bases__: tuple
    llm_info_cls: type[LLMInfoT]
    image_info_cls: type[ImageInfoT]

    def __init__(
        self,
        llm_info_cls: type[LLMInfoT] | None = None,
        image_info_cls: type[ImageInfoT] | None = None,
    ):
        """Initialize the client with a ModelRegistry instance."""
        # Cache refreshing is no-op for local registry clients.
        # Since this client is local, we don't need to refresh the cache.
        # TODO: Move caching to the base HTTP client class instead.
        cache_ttl = timedelta(days=365)
        super().__init__(cache_ttl=cache_ttl)
        self.registry = ModelRegistry()

        # Set model info classes if provided, otherwise they are inferred from the class
        if llm_info_cls is not None:
            self.llm_info_cls = llm_info_cls
        if image_info_cls is not None:
            self.image_info_cls = image_info_cls

        logger.debug("initialized_base_local_registry_client")

    def __init_subclass__(cls, **kwargs):
        """Set up model classes when this class is subclassed."""
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "llm_info_cls") or not hasattr(cls, "image_info_cls"):
            llm_cls, image_cls = cls._get_model_info_classes()
            if not hasattr(cls, "llm_info_cls"):
                cls.llm_info_cls = llm_cls
            if not hasattr(cls, "image_info_cls"):
                cls.image_info_cls = image_cls

    @classmethod
    def _get_model_info_classes(cls) -> tuple[type[LLMInfoT], type[ImageInfoT]]:
        """Extract the model classes from generic type parameters."""
        args = get_args(cls.__orig_bases__[0])
        return args[0], args[1]

    async def _fetch_llm_info(self, model_id: str) -> LLMInfoT:
        """Get LLM information directly from registry.

        Args:
            model_id: The ID of the LLM to get information for

        Returns:
            Information about the requested LLM with the type expected by this client
        """
        # Get the LLM info from the registry
        info = self.registry.get_llm_info(model_id)

        # If llm_info_cls is LLMInfo (not a subclass), we can return it as-is
        if self.llm_info_cls is LLMInfo:
            return cast(LLMInfoT, info)

        # Otherwise, validate against the subclass schema
        return self.llm_info_cls.model_validate(info.model_dump())

    async def _fetch_image_model_info(self, model_id: str) -> ImageInfoT:
        """Get image model information directly from registry.

        Args:
            model_id: The ID of the image model to get information for

        Returns:
            Information about the requested image model with the type expected by this client
        """
        # Get the image model info from the registry
        info = self.registry.get_image_model_info(model_id)

        # If image_info_cls is ImageModelInfo (not a subclass), we can return it as-is
        if self.image_info_cls is ImageModelInfo:
            return cast(ImageInfoT, info)

        # Otherwise, validate against the subclass schema
        return self.image_info_cls.model_validate(info.model_dump())

    async def _fetch_all_llms(self) -> Sequence[LLMInfoT]:
        """List all available LLMs directly from registry.

        Returns:
            A sequence of LLM information objects of the type expected by this client.
        """
        models = self.registry.list_llms()

        # If llm_info_cls is LLMInfo (not a subclass), we can return the list as-is
        if self.llm_info_cls is LLMInfo:
            return cast(Sequence[LLMInfoT], models)

        # Otherwise, we need to validate each model against the subclass schema
        return [
            self.llm_info_cls.model_validate(model.model_dump()) for model in models
        ]

    async def _fetch_all_image_models(self) -> Sequence[ImageInfoT]:
        """List all available image models directly from registry.

        Returns:
            A sequence of image model information objects of the type expected by this client.
        """
        models = self.registry.list_image_models()

        # If image_info_cls is ImageModelInfo (not a subclass), we can return the list as-is
        if self.image_info_cls is ImageModelInfo:
            return cast(Sequence[ImageInfoT], models)

        # Otherwise, we need to validate each model against the subclass schema
        return [
            self.image_info_cls.model_validate(model.model_dump()) for model in models
        ]


class LocalRegistryClient(BaseLocalRegistryClient[LLMInfo, ImageModelInfo]):
    """
    Local registry client that calls ModelRegistry directly using the default schemas.

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
