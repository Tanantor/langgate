"""Integration tests for LocalRegistryClient."""

from datetime import timedelta

import pytest

from langgate.core.models import LLMInfo
from langgate.registry.local import LocalRegistryClient
from tests.mocks.registry_mocks import CustomLLMInfo, CustomLocalRegistryClient


@pytest.mark.asyncio
async def test_local_registry_client_get_model(
    local_registry_client: LocalRegistryClient,
):
    """Test getting a model from the registry via the client."""
    # First list all models
    models = await local_registry_client.list_models()
    assert len(models) > 0

    # Pick the first model
    first_model = models[0]

    # Then get that specific model by ID
    model = await local_registry_client.get_model_info(first_model.id)

    assert model.id == first_model.id
    assert model.name == first_model.name
    assert isinstance(model, LLMInfo)
    assert model.provider is not None


@pytest.mark.asyncio
async def test_local_registry_client_list_models(
    local_registry_client: LocalRegistryClient,
):
    """Test listing all models from the registry."""
    models = await local_registry_client.list_models()

    assert len(models) > 0

    for model in models:
        assert isinstance(model, LLMInfo)
        assert model.id is not None
        assert model.name is not None
        assert model.provider is not None
        assert model.costs is not None


@pytest.mark.asyncio
async def test_local_registry_client_caching(
    local_registry_client: LocalRegistryClient,
):
    """Test that models are properly cached.

    Note that caching at the client level is intended for
    HTTP clients. For simplicity, this logic lives in the base class
    `BaseRegistryClient` but refreshing the cache is essentially
    a no-op for the local registry client.
    """
    last_cache_refresh = local_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    models = await local_registry_client.list_models()
    assert len(models) > 0

    # Verify cache state
    assert local_registry_client._last_cache_refresh is not None
    last_cache_refresh = local_registry_client._last_cache_refresh
    assert len(local_registry_client._model_cache) > 0

    # Get a specific model ID to test
    model_id = models[0].id

    # This should use the cache
    model = await local_registry_client.get_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is local_registry_client._model_cache[model_id]
    assert last_cache_refresh == local_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await local_registry_client.get_model_info(model_id)
    assert model2.id == model_id
    assert model2 is model

    # Simulate cache expiration
    expired_last_refresh = (
        local_registry_client._last_cache_refresh
        - local_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    local_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the model again, which should refresh the cache
    model3 = await local_registry_client.get_model_info(model_id)
    assert model3.id == model_id
    # for the default local registry client, we expect refreshing to still return the same objects
    # which are perpetually cached in the ModelRegistry.
    assert model3 is model
    assert model3 is local_registry_client._model_cache[model_id]

    # Verify cache state
    assert local_registry_client._last_cache_refresh > expired_last_refresh
    assert local_registry_client._last_cache_refresh > last_cache_refresh


@pytest.mark.asyncio
async def test_custom_local_registry_client(
    custom_local_registry_client: CustomLocalRegistryClient,
):
    """Test using a custom client with a custom schema."""
    # List models with the custom client
    models = await custom_local_registry_client.list_models()
    assert len(models) > 0

    # Verify custom model type
    for model in models:
        assert isinstance(model, CustomLLMInfo)
        assert model.custom_field == "custom_value"

    # Get a specific model
    first_model = models[0]
    model = await custom_local_registry_client.get_model_info(first_model.id)

    # Verify it's the correct custom type
    assert isinstance(model, CustomLLMInfo)
    assert model.id == first_model.id
    assert model.name == first_model.name
    assert model.custom_field == "custom_value"


@pytest.mark.asyncio
async def test_custom_local_registry_client_caching(
    custom_local_registry_client: CustomLocalRegistryClient,
):
    """Test that models are properly cached."""
    last_cache_refresh = custom_local_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    models = await custom_local_registry_client.list_models()
    assert len(models) > 0

    # Verify cache state
    assert custom_local_registry_client._last_cache_refresh is not None
    last_cache_refresh = custom_local_registry_client._last_cache_refresh
    assert len(custom_local_registry_client._model_cache) > 0

    # Get a specific model ID to test
    model_id = models[0].id

    # This should use the cache
    model = await custom_local_registry_client.get_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is custom_local_registry_client._model_cache[model_id]
    assert last_cache_refresh == custom_local_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await custom_local_registry_client.get_model_info(model_id)
    assert model2.id == model_id
    assert model2 is model

    # Simulate cache expiration
    expired_last_refresh = (
        custom_local_registry_client._last_cache_refresh
        - custom_local_registry_client._cache_ttl
        - timedelta(seconds=1)
    )
    custom_local_registry_client._last_cache_refresh = expired_last_refresh

    # Fetch the model again, which should refresh the cache
    model3 = await custom_local_registry_client.get_model_info(model_id)
    assert model3.id == model_id
    # for a custom local registry client, we expect refreshing to return new objects
    # as they will be revalidated against the custom schema.
    assert model3 is not model
    assert model3 is custom_local_registry_client._model_cache[model_id]

    # Verify cache state
    assert custom_local_registry_client._last_cache_refresh > expired_last_refresh
    assert custom_local_registry_client._last_cache_refresh > last_cache_refresh
