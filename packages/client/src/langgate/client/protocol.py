"""Protocol definitions for LangGate clients."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Generic, Protocol, TypeVar

from langgate.core.logging import get_logger
from langgate.core.models import ImageModelInfo, LLMInfo

logger = get_logger(__name__)


LLMInfoT = TypeVar("LLMInfoT", bound=LLMInfo, covariant=True)
ImageInfoT = TypeVar("ImageInfoT", bound=ImageModelInfo, covariant=True)


class LLMRegistryClientProtocol(Protocol[LLMInfoT]):
    """Protocol for LLM registry clients."""

    async def get_llm_info(self, model_id: str) -> LLMInfoT:
        """Get LLM information by ID."""
        ...

    async def list_llms(self) -> Sequence[LLMInfoT]:
        """List all available LLMs."""
        ...


class ImageRegistryClientProtocol(Protocol[ImageInfoT]):
    """Protocol for image model registry clients."""

    async def get_image_model_info(self, model_id: str) -> ImageInfoT:
        """Get image model information by ID."""
        ...

    async def list_image_models(self) -> Sequence[ImageInfoT]:
        """List all available image models."""
        ...


class RegistryClientProtocol(
    LLMRegistryClientProtocol[LLMInfoT], ImageRegistryClientProtocol[ImageInfoT]
):
    """Protocol for clients supporting multiple modalities."""


class BaseRegistryClient(
    ABC,
    Generic[LLMInfoT, ImageInfoT],
    RegistryClientProtocol[LLMInfoT, ImageInfoT],
):
    """Base class for registry clients with common operations."""

    def __init__(self, cache_ttl: timedelta | None = None) -> None:
        """Initialize the client with cache settings."""
        self._llm_cache: dict[str, LLMInfoT] = {}
        self._image_cache: dict[str, ImageInfoT] = {}
        self._last_cache_refresh: datetime | None = None
        # TODO: allow this to be set via config or env var
        self._cache_ttl = cache_ttl or timedelta(minutes=60)

    # LLM methods
    async def get_llm_info(self, model_id: str) -> LLMInfoT:
        """Get LLM information by ID."""
        if self._should_refresh_cache():
            await self._refresh_cache()

        model = self._llm_cache.get(model_id)
        if model is None:
            # If not found after potential refresh, try fetching individually
            await logger.awarning(
                "cache_miss_fetching_individual_model",
                model_id=model_id,
                model_type="llm",
            )
            try:
                model = await self._fetch_llm_info(model_id)
                self._llm_cache[model_id] = model
            except Exception as exc:
                await logger.aexception(
                    "failed_to_fetch_individual_model",
                    model_id=model_id,
                    model_type="llm",
                )
                raise ValueError(
                    f"LLM model '{model_id}' not found in registry."
                ) from exc
        return model

    async def list_llms(self) -> Sequence[LLMInfoT]:
        """List all available LLMs."""
        if self._should_refresh_cache():
            await self._refresh_cache()
        return list(self._llm_cache.values())

    # Image model methods
    async def get_image_model_info(self, model_id: str) -> ImageInfoT:
        """Get image model information by ID."""
        if self._should_refresh_cache():
            await self._refresh_cache()

        model = self._image_cache.get(model_id)
        if model is None:
            # If not found after potential refresh, try fetching individually
            await logger.awarning(
                "cache_miss_fetching_individual_model",
                model_id=model_id,
                model_type="image",
            )
            try:
                model = await self._fetch_image_model_info(model_id)
                self._image_cache[model_id] = model
            except Exception as exc:
                await logger.aexception(
                    "failed_to_fetch_individual_model",
                    model_id=model_id,
                    model_type="image",
                )
                raise ValueError(
                    f"Image model '{model_id}' not found in registry."
                ) from exc
        return model

    async def list_image_models(self) -> Sequence[ImageInfoT]:
        """List all available image models."""
        if self._should_refresh_cache():
            await self._refresh_cache()
        return list(self._image_cache.values())

    @abstractmethod
    async def _fetch_llm_info(self, model_id: str) -> LLMInfoT:
        """Fetch LLM info from the source."""
        ...

    @abstractmethod
    async def _fetch_image_model_info(self, model_id: str) -> ImageInfoT:
        """Fetch image model info from the source."""
        ...

    @abstractmethod
    async def _fetch_all_llms(self) -> Sequence[LLMInfoT]:
        """Fetch all LLMs from the source."""
        ...

    @abstractmethod
    async def _fetch_all_image_models(self) -> Sequence[ImageInfoT]:
        """Fetch all image models from the source."""
        ...

    async def _refresh_cache(self) -> None:
        """Refresh both model caches."""
        await logger.adebug("refreshing_model_caches")
        try:
            # Refresh both caches in parallel
            llms = await self._fetch_all_llms()
            image_models = await self._fetch_all_image_models()

            self._llm_cache = {model.id: model for model in llms}
            self._image_cache = {model.id: model for model in image_models}

            self._last_cache_refresh = datetime.now()
            await logger.adebug(
                "refreshed_model_caches",
                llm_count=len(self._llm_cache),
                image_count=len(self._image_cache),
            )
        except Exception as exc:
            await logger.aexception("failed_to_refresh_model_caches")
            # Decide: Keep stale cache or clear it? Keeping stale might be better than empty.
            self._last_cache_refresh = None  # Force retry next time
            raise exc

    def _should_refresh_cache(self) -> bool:
        """Check if cache should be refreshed."""
        return (
            self._last_cache_refresh is None
            or (datetime.now() - self._last_cache_refresh) > self._cache_ttl
        )
