"""HTTP client for LangGate API."""

from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Generic, get_args

import httpx
from pydantic import SecretStr

from langgate.client.protocol import BaseRegistryClient, ImageInfoT, LLMInfoT
from langgate.core.logging import get_logger
from langgate.core.models import ImageModelInfo, LLMInfo

logger = get_logger(__name__)


def create_registry_http_client(
    base_url: str,
    api_key: SecretStr | None = None,
    timeout: float | httpx.Timeout | None = 10.0,
    **kwargs,
) -> httpx.AsyncClient:
    """
    Creates and configures an httpx.AsyncClient for the Registry API.
    """
    headers = kwargs.pop("headers", {})
    if api_key:
        headers["X-API-Key"] = api_key.get_secret_value()

    processed_base_url = base_url.rstrip("/")

    return httpx.AsyncClient(
        base_url=processed_base_url,
        headers=headers,
        timeout=timeout,
        **kwargs,
    )


class BaseHTTPRegistryClient(
    BaseRegistryClient[LLMInfoT, ImageInfoT], Generic[LLMInfoT, ImageInfoT]
):
    """
    Base HTTP client for the Model Registry API.
    Supports LLMInfo-derived and ImageModelInfo-derived schemas for response parsing and httpx client injection.

    Handles infrequent HTTP requests via temporary clients by default or uses an
    injected client as the request engine. Configuration (base_url, api_key)
    stored in this instance is always used for requests.

    Type Parameters:
        LLMInfoT: The LLMInfo-derived model class for response parsing
        ImageInfoT: The ImageModelInfo-derived model class for response parsing
    """

    __orig_bases__: tuple
    llm_info_cls: type[LLMInfoT]
    image_info_cls: type[ImageInfoT]

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None = None,
        cache_ttl: timedelta | None = None,
        llm_info_cls: type[LLMInfoT] | None = None,
        image_info_cls: type[ImageInfoT] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize the client.
        Args:
            base_url: The base URL of the registry service
            api_key: Registry server API key for authentication
            cache_ttl: Cache time-to-live
            llm_info_cls: Override for LLM info class
            image_info_cls: Override for image model info class
            http_client: Optional HTTP client for making requests
        """
        super().__init__(cache_ttl=cache_ttl)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._http_client = http_client

        # Set model info classes if provided, otherwise they are inferred from the class
        if llm_info_cls is not None:
            self.llm_info_cls = llm_info_cls
        if image_info_cls is not None:
            self.image_info_cls = image_info_cls

        logger.debug(
            "initialized_base_http_registry_client",
            base_url=self.base_url,
            api_key=self.api_key,
            llm_info_cls=self.llm_info_cls,
            image_info_cls=self.image_info_cls,
        )

    def __init_subclass__(cls, **kwargs):
        """Set up model classes when this class is subclassed."""
        super().__init_subclass__(**kwargs)

        # Extract the model classes from generic parameters
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

    @asynccontextmanager
    async def _get_client_for_request(self):
        """Provides the httpx client to use (injected or temporary)."""
        if self._http_client:
            yield self._http_client
        else:
            async with httpx.AsyncClient() as temp_client:
                yield temp_client

    async def _request(self, method: str, url_path: str, **kwargs) -> httpx.Response:
        """Makes an HTTP request using the appropriate client engine."""
        url = f"{self.base_url}{url_path}"
        headers = kwargs.pop("headers", {})
        if self.api_key:
            headers["X-API-Key"] = self.api_key.get_secret_value()

        async with self._get_client_for_request() as client:
            response = await client.request(method, url, headers=headers, **kwargs)
        return response

    async def _fetch_llm_info(self, model_id: str) -> LLMInfoT:
        """Fetch LLM info from the source via HTTP."""
        response = await self._request("GET", f"/models/llms/{model_id}")
        response.raise_for_status()
        return self.llm_info_cls.model_validate(response.json())

    async def _fetch_image_model_info(self, model_id: str) -> ImageInfoT:
        """Fetch image model info from the source via HTTP."""
        response = await self._request("GET", f"/models/images/{model_id}")
        response.raise_for_status()
        return self.image_info_cls.model_validate(response.json())

    async def _fetch_all_llms(self) -> list[LLMInfoT]:
        """Fetch all LLMs from the source via HTTP."""
        response = await self._request("GET", "/models/llms")
        response.raise_for_status()
        return [self.llm_info_cls.model_validate(model) for model in response.json()]

    async def _fetch_all_image_models(self) -> list[ImageInfoT]:
        """Fetch all image models from the source via HTTP."""
        response = await self._request("GET", "/models/images")
        response.raise_for_status()
        return [self.image_info_cls.model_validate(model) for model in response.json()]


class HTTPRegistryClient(BaseHTTPRegistryClient[LLMInfo, ImageModelInfo]):
    """HTTP client singleton for the Model Registry API using the default schemas."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            logger.debug("creating_http_registry_client_singleton")
        return cls._instance

    def __init__(
        self,
        base_url: str,
        api_key: SecretStr | None = None,
        cache_ttl: timedelta | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        if not hasattr(self, "_initialized"):
            super().__init__(base_url, api_key, cache_ttl, http_client=http_client)
            self._initialized = True
            logger.debug("initialized_http_registry_client_singleton")
