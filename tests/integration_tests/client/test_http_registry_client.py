"""Integration tests for HTTPRegistryClient."""

from datetime import timedelta

import pytest

from langgate.client.http import HTTPRegistryClient
from langgate.core.models import LLMInfo
from tests.mocks.client_mocks import CustomHTTPRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo


@pytest.mark.asyncio
async def test_http_registry_client_get_llm(
    http_registry_client: HTTPRegistryClient,
):
    """Test getting a llm from the registry via the HTTP client."""
    llms = await http_registry_client.list_llms()
    assert len(llms) > 0

    first_llm = llms[0]

    llm = await http_registry_client.get_llm_info(first_llm.id)

    assert llm.id == first_llm.id
    assert llm.name == first_llm.name
    assert isinstance(llm, LLMInfo)
    assert llm.provider is not None


@pytest.mark.asyncio
async def test_http_registry_client_list_llms(
    http_registry_client: HTTPRegistryClient,
):
    """Test listing all llms from the registry."""
    llms = await http_registry_client.list_llms()

    assert len(llms) > 0

    for llm in llms:
        assert isinstance(llm, LLMInfo)
        assert llm.id is not None
        assert llm.name is not None
        assert llm.provider is not None
        assert llm.costs is not None


@pytest.mark.asyncio
async def test_http_registry_client_caching(
    http_registry_client: HTTPRegistryClient,
):
    """Test that llms are properly cached."""
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    llms = await http_registry_client.list_llms()
    assert len(llms) > 0

    # Verify cache state
    assert http_registry_client._last_cache_refresh is not None
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert len(http_registry_client._llm_cache) > 0

    # Get a specific llm ID to test
    model_id = llms[0].id

    # This should use the cache
    llm = await http_registry_client.get_llm_info(model_id)

    assert llm.id == model_id

    # Verify it's the same object reference (from cache)
    assert llm is http_registry_client._llm_cache[model_id]
    assert last_cache_refresh == http_registry_client._last_cache_refresh

    # Fetch the same llm again
    llm2 = await http_registry_client.get_llm_info(model_id)
    assert llm2.id == model_id
    assert llm2 is llm

    # Simulate cache expiration
    expired_last_refresh = (
        http_registry_client._last_cache_refresh
        - http_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    http_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the llm again, which should refresh the cache
    llm3 = await http_registry_client.get_llm_info(model_id)
    assert llm3.id == model_id
    # New llms from API should have different object references
    assert llm3 is not llm
    assert llm3 is http_registry_client._llm_cache[model_id]

    # Verify cache state
    assert http_registry_client._last_cache_refresh > expired_last_refresh
    assert http_registry_client._last_cache_refresh > last_cache_refresh


@pytest.mark.asyncio
async def test_http_registry_client_not_found(
    http_registry_client: HTTPRegistryClient,
):
    """Test requesting a non-existent llm returns the expected error."""
    with pytest.raises(ValueError, match="not found"):
        await http_registry_client.get_llm_info("non-existent-llm-id")


@pytest.mark.asyncio
async def test_custom_http_registry_client(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test using a custom HTTP client with a custom schema."""
    llms = await custom_http_registry_client.list_llms()
    assert len(llms) > 0

    # Verify custom llm type
    for llm in llms:
        assert isinstance(llm, CustomLLMInfo)
        assert llm.custom_field == "custom_value"

    first_llm = llms[0]
    llm = await custom_http_registry_client.get_llm_info(first_llm.id)

    assert isinstance(llm, CustomLLMInfo)
    assert llm.id == first_llm.id
    assert llm.name == first_llm.name
    assert llm.custom_field == "custom_value"


@pytest.mark.asyncio
async def test_custom_http_registry_client_caching(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test that llms are properly cached in the custom client."""
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    llms = await custom_http_registry_client.list_llms()
    assert len(llms) > 0

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh is not None
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert len(custom_http_registry_client._llm_cache) > 0

    # Get a specific llm ID to test
    model_id = llms[0].id

    # This should use the cache
    llm = await custom_http_registry_client.get_llm_info(model_id)

    assert llm.id == model_id

    # Verify it's the same object reference (from cache)
    assert llm is custom_http_registry_client._llm_cache[model_id]
    assert last_cache_refresh == custom_http_registry_client._last_cache_refresh

    # Fetch the same llm again
    llm2 = await custom_http_registry_client.get_llm_info(model_id)
    assert llm2.id == model_id
    assert llm2 is llm

    # Simulate cache expiration
    expired_last_refresh = (
        custom_http_registry_client._last_cache_refresh
        - custom_http_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    custom_http_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the llm again, which should refresh the cache
    llm3 = await custom_http_registry_client.get_llm_info(model_id)
    assert llm3.id == model_id
    # Custom llms should be revalidated, so new object
    assert llm3 is not llm
    assert llm3 is custom_http_registry_client._llm_cache[model_id]

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh > expired_last_refresh
    assert custom_http_registry_client._last_cache_refresh > last_cache_refresh
