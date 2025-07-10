"""Integration tests for HTTPRegistryClient."""

from datetime import timedelta

import pytest

from langgate.client.http import HTTPRegistryClient
from langgate.core.models import ImageModelInfo, LLMInfo
from tests.mocks.client_mocks import CustomHTTPRegistryClient
from tests.mocks.registry_mocks import CustomImageModelInfo, CustomLLMInfo


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


# Image model tests


@pytest.mark.asyncio
async def test_http_registry_client_get_image_model(
    http_registry_client: HTTPRegistryClient,
):
    """Test getting an image model from the registry via the HTTP client."""
    image_models = await http_registry_client.list_image_models()
    assert len(image_models) > 0

    first_model = image_models[0]

    model = await http_registry_client.get_image_model_info(first_model.id)

    assert model.id == first_model.id
    assert model.name == first_model.name
    assert isinstance(model, ImageModelInfo)
    assert model.provider is not None
    assert model.costs is not None
    assert model.costs.image_generation is not None


@pytest.mark.asyncio
async def test_http_registry_client_list_image_models(
    http_registry_client: HTTPRegistryClient,
):
    """Test listing all image models from the registry."""
    image_models = await http_registry_client.list_image_models()

    assert len(image_models) > 0

    for model in image_models:
        assert isinstance(model, ImageModelInfo)
        assert model.id is not None
        assert model.name is not None
        assert model.provider is not None
        assert model.costs is not None
        assert model.costs.image_generation is not None


@pytest.mark.asyncio
async def test_http_registry_client_image_model_caching(
    http_registry_client: HTTPRegistryClient,
):
    """Test that image models are properly cached."""
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    image_models = await http_registry_client.list_image_models()
    assert len(image_models) > 0

    # Verify cache state
    assert http_registry_client._last_cache_refresh is not None
    last_cache_refresh = http_registry_client._last_cache_refresh
    assert len(http_registry_client._image_cache) > 0

    # Get a specific model ID to test
    model_id = image_models[0].id

    # This should use the cache
    model = await http_registry_client.get_image_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is http_registry_client._image_cache[model_id]
    assert last_cache_refresh == http_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await http_registry_client.get_image_model_info(model_id)
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
    model3 = await http_registry_client.get_image_model_info(model_id)
    assert model3.id == model_id
    # New models from API should have different object references
    assert model3 is not model
    assert model3 is http_registry_client._image_cache[model_id]

    # Verify cache state
    assert http_registry_client._last_cache_refresh > expired_last_refresh
    assert http_registry_client._last_cache_refresh > last_cache_refresh


@pytest.mark.asyncio
async def test_http_registry_client_image_model_not_found(
    http_registry_client: HTTPRegistryClient,
):
    """Test requesting a non-existent image model returns the expected error."""
    with pytest.raises(ValueError, match="not found"):
        await http_registry_client.get_image_model_info("non-existent-image-model-id")


@pytest.mark.asyncio
async def test_custom_http_registry_client_image_models(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test using a custom HTTP client with a custom image model schema."""
    image_models = await custom_http_registry_client.list_image_models()
    assert len(image_models) > 0

    # Verify custom model type
    for model in image_models:
        assert isinstance(model, CustomImageModelInfo)
        assert model.custom_field == "custom_value"

    first_model = image_models[0]
    model = await custom_http_registry_client.get_image_model_info(first_model.id)

    assert isinstance(model, CustomImageModelInfo)
    assert model.id == first_model.id
    assert model.name == first_model.name
    assert model.custom_field == "custom_value"


@pytest.mark.asyncio
async def test_custom_http_registry_client_image_model_caching(
    custom_http_registry_client: CustomHTTPRegistryClient,
):
    """Test that image models are properly cached in the custom client."""
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert last_cache_refresh is None

    # Initial request populates cache
    image_models = await custom_http_registry_client.list_image_models()
    assert len(image_models) > 0

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh is not None
    last_cache_refresh = custom_http_registry_client._last_cache_refresh
    assert len(custom_http_registry_client._image_cache) > 0

    # Get a specific model ID to test
    model_id = image_models[0].id

    # This should use the cache
    model = await custom_http_registry_client.get_image_model_info(model_id)

    assert model.id == model_id

    # Verify it's the same object reference (from cache)
    assert model is custom_http_registry_client._image_cache[model_id]
    assert last_cache_refresh == custom_http_registry_client._last_cache_refresh

    # Fetch the same model again
    model2 = await custom_http_registry_client.get_image_model_info(model_id)
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
    model3 = await custom_http_registry_client.get_image_model_info(model_id)
    assert model3.id == model_id
    # Custom models should be revalidated, so new object
    assert model3 is not model
    assert model3 is custom_http_registry_client._image_cache[model_id]

    # Verify cache state
    assert custom_http_registry_client._last_cache_refresh > expired_last_refresh
    assert custom_http_registry_client._last_cache_refresh > last_cache_refresh
