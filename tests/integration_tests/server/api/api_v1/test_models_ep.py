"""Integration tests for the models endpoint."""

from pathlib import Path
from unittest import mock

import pytest
from httpx import AsyncClient

MODELS_URL = "/models"


@pytest.mark.asyncio
async def test_get_models(registry_api_client: AsyncClient) -> None:
    """Test retrieving all models."""
    response = await registry_api_client.get(MODELS_URL)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

    for model in data:
        assert isinstance(model, dict)
        assert "id" in model
        assert "name" in model
        assert "provider" in model
        assert "costs" in model
        assert "capabilities" in model
        assert "context_window" in model

        provider = model["provider"]
        assert "id" in provider
        assert "name" in provider


@pytest.mark.asyncio
async def test_get_model_by_id(registry_api_client: AsyncClient) -> None:
    """Test retrieving a specific model by ID."""
    # First get all models
    response = await registry_api_client.get(MODELS_URL)
    assert response.status_code == 200

    models = response.json()
    assert len(models) > 0
    test_model = models[0]

    # Test API response for specific model
    response = await registry_api_client.get(f"{MODELS_URL}/{test_model['id']}")
    assert response.status_code == 200

    model = response.json()
    assert model["id"] == test_model["id"]
    assert model["name"] == test_model["name"]
    assert model["provider"]["id"] == test_model["provider"]["id"]

    # Test with non-existent model
    response = await registry_api_client.get(f"{MODELS_URL}/non-existent-model")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_model_schema(registry_api_client: AsyncClient) -> None:
    """Test that the model schema validation works correctly."""
    # Get models to verify schema
    response = await registry_api_client.get(MODELS_URL)
    assert response.status_code == 200

    models = response.json()
    assert len(models) > 0

    # Attempt to validate model structure using Pydantic
    for model_data in models:
        # We can't directly validate here, but we should check key fields
        assert isinstance(model_data["id"], str)
        assert isinstance(model_data["name"], str)

        # Check costs structure
        costs = model_data["costs"]
        assert any(k.startswith("input_") for k in costs)
        assert any(k.startswith("output_") for k in costs)

        # Check capabilities structure
        capabilities = model_data["capabilities"]
        if capabilities:
            assert all(k.startswith("supports_") for k in capabilities)

        # Check context window
        context = model_data["context_window"]
        assert "max_input_tokens" in context
        assert "max_output_tokens" in context


@pytest.mark.asyncio
async def test_models_api_works_without_env_file(
    registry_api_client: AsyncClient,
) -> None:
    """Test that the models API endpoints work without env files."""
    # Make Path.exists return False for .env files
    with mock.patch.object(Path, "exists") as mock_exists:

        def side_effect(path):
            # Return False for .env files, True for everything else
            return ".env" not in str(path)

        mock_exists.side_effect = side_effect

        # Test the models route still returns successfully
        response = await registry_api_client.get(MODELS_URL)
        assert response.status_code == 200

        # Test that we can get one of the models by ID
        models = response.json()
        assert models, "No models found in registry"
        test_model = models[0]
        model_id = test_model["id"]

        response = await registry_api_client.get(f"{MODELS_URL}/{model_id}")
        assert response.status_code == 200
        assert response.json()["id"] == model_id
