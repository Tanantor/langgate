"""Client API fixtures."""

from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from langgate.client.http import HTTPRegistryClient
from langgate.server.core.config import settings
from langgate.server.main import app
from tests.mocks.client_mocks import CustomHTTPRegistryClient


@pytest_asyncio.fixture
async def registry_api_client() -> AsyncGenerator[AsyncClient]:
    """Return an async client for testing the registry API."""
    # Create ASGITransport for testing
    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url=f"http://{settings.TEST_SERVER_HOST}{settings.API_V1_STR}",
    ) as client:
        yield client


@pytest_asyncio.fixture
async def http_registry_client(
    registry_api_client,
) -> AsyncGenerator[HTTPRegistryClient]:
    """Return an HTTPRegistryClient configured to use the test server."""
    # Save original singleton state
    original_instance = HTTPRegistryClient._instance
    HTTPRegistryClient._instance = None

    client = HTTPRegistryClient(
        base_url=f"http://{settings.TEST_SERVER_HOST}{settings.API_V1_STR}",
        api_key=None,
        http_client=registry_api_client,
    )

    yield client

    # Restore original singleton state
    HTTPRegistryClient._instance = original_instance


@pytest.fixture
def custom_http_registry_client(
    registry_api_client,
) -> Generator[CustomHTTPRegistryClient]:
    """Return a CustomHTTPRegistryClient instance for testing custom schema handling."""
    client = CustomHTTPRegistryClient(
        base_url=f"http://{settings.TEST_SERVER_HOST}{settings.API_V1_STR}",
        api_key=None,
        http_client=registry_api_client,
    )

    yield client
