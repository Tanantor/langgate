"""Integration tests for HTTPRegistryClient."""

from datetime import timedelta

import pytest

from langgate.client.http import HTTPRegistryClient
from langgate.core.models import LLMInfo
from tests.mocks.client_mocks import CustomHTTPRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo


@pytest.mark.asyncio
async def test_http_registry_client_get_model(
    http_registry_client: HTTPRegistryClient,
):
    """Test getting a model from the registry via the HTTP client."""
    models = await http_registry_client.list_models()
    assert len(models) > 0

    first_model = models[0]

    model = await http_registry_client.get_model_info(first_model.id)

    assert model.id == first_model.id
    assert model.name == first_model.name
    assert isinstance(model, LLMInfo)
    assert model.provider is not None


@pytest.mark.asyncio
async def test_http_registry_client_list_models(
    http_registry_client: HTTPRegistryClient,
):
    """Test listing all models from the registry."""
    models = await http_registry_client.list_models()

    assert len(models) > 0

    for model in models:
        assert isinstance(model, LLMInfo)
        assert model.id is not None
        assert model.name is not None
        assert model.provider is not None
        assert model.costs is not None


@pytest.mark.asyncio
async def test_http_registry_client_caching(
    http_registry_client: HTTPRegistryClient,
):
    """Test that models are properly cached."""
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    models = await http_registry_client.list_models()
    assert len(models) > 0

    # Verify cache state
    assert http_registry_client._last_cache_refresh is not None
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert len(http_registry_client._model_cache) > 0

    # Get a specific model ID to test
    model_id = models[0].id

    # This should use the cache
    model = await http_registry_client.get_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is http_registry_client._model_cache[model_id]
    assert last_cache_refresh == http_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await http_registry_client.get_model_info(model_id)
    assert model2.id == model_id
    assert model2 is model

    # Simulate cache expiration
    expired_last_refresh = (
        http_registry_client._last_cache_refresh
        - http_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    http_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the model again, which should refresh the cache
    model3 = await http_registry_client.get_model_info(model_id)
    assert model3.id == model_id
    # New models from API should have different object references
    assert model3 is not model
    assert model3 is http_registry_client._model_cache[model_id]

    # Verify cache state
    assert http_registry_client._last_cache_refresh > expired_last_refresh
    assert http_registry_client._last_cache_refresh > last_cache_refresh


@pytest.mark.asyncio
async def test_http_registry_client_not_found(
    http_registry_client: HTTPRegistryClient,
):
    """Test requesting a non-existent model returns the expected error."""
    with pytest.raises(ValueError, match="not found"):
        await http_registry_client.get_model_info("non-existent-model-id")


@pytest.mark.asyncio
async def test_custom_http_registry_client(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test using a custom HTTP client with a custom schema."""
    models = await custom_http_registry_client.list_models()
    assert len(models) > 0

    # Verify custom model type
    for model in models:
        assert isinstance(model, CustomLLMInfo)
        assert model.custom_field == "custom_value"

    first_model = models[0]
    model = await custom_http_registry_client.get_model_info(first_model.id)

    assert isinstance(model, CustomLLMInfo)
    assert model.id == first_model.id
    assert model.name == first_model.name
    assert model.custom_field == "custom_value"


@pytest.mark.asyncio
async def test_custom_http_registry_client_caching(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test that models are properly cached in the custom client."""
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    models = await custom_http_registry_client.list_models()
    assert len(models) > 0

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh is not None
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert len(custom_http_registry_client._model_cache) > 0

    # Get a specific model ID to test
    model_id = models[0].id

    # This should use the cache
    model = await custom_http_registry_client.get_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is custom_http_registry_client._model_cache[model_id]
    assert last_cache_refresh == custom_http_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await custom_http_registry_client.get_model_info(model_id)
    assert model2.id == model_id
    assert model2 is model

    # Simulate cache expiration
    expired_last_refresh = (
        custom_http_registry_client._last_cache_refresh
        - custom_http_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    custom_http_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the model again, which should refresh the cache
    model3 = await custom_http_registry_client.get_model_info(model_id)
    assert model3.id == model_id
    # Custom models should be revalidated, so new object
    assert model3 is not model
    assert model3 is custom_http_registry_client._model_cache[model_id]

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh > expired_last_refresh
    assert custom_http_registry_client._last_cache_refresh > last_cache_refresh
