"""Tests for environment configuration."""

import pytest

from langgate.core.models import ImageModelInfo, LLMInfo
from langgate.registry.models import ModelRegistry
from tests.utils.registry_utils import (
    patch_model_registry,
)


def test_model_registry_singleton_instance():
    """Test that ModelRegistry is a singleton."""
    with patch_model_registry():
        registry1 = ModelRegistry()
        # Create second instance
        registry2 = ModelRegistry()
        # Test singleton behavior
        assert registry1 is registry2
        # Both instances should share the same state
        assert id(registry1._model_caches) == id(registry2._model_caches)


def test_model_registry_list_llms(model_registry: ModelRegistry):
    """Test listing LLMs from the registry."""
    llms = model_registry.list_llms()

    assert isinstance(llms, list)
    assert len(llms) > 0

    for llm in llms:
        assert isinstance(llm, LLMInfo)
        assert llm.id is not None
        assert llm.name is not None
        assert llm.provider is not None
        assert llm.costs is not None


def test_model_registry_get_llm_info(model_registry: ModelRegistry):
    """Test getting specific LLM info from the registry."""
    llms = model_registry.list_llms()
    assert len(llms) > 0

    # Test getting first LLM
    first_llm = llms[0]
    llm_info = model_registry.get_llm_info(first_llm.id)

    assert isinstance(llm_info, LLMInfo)
    assert llm_info.id == first_llm.id
    assert llm_info.name == first_llm.name

    # Test non-existent LLM
    with pytest.raises(ValueError, match="not found"):
        model_registry.get_llm_info("non-existent-llm")


# Image model tests


def test_model_registry_list_image_models(model_registry: ModelRegistry):
    """Test listing image models from the registry."""
    image_models = model_registry.list_image_models()

    assert isinstance(image_models, list)
    assert len(image_models) > 0

    for model in image_models:
        assert isinstance(model, ImageModelInfo)
        assert model.id is not None
        assert model.name is not None
        assert model.provider is not None
        assert model.costs is not None
        assert model.costs.image_generation is not None


def test_model_registry_get_image_model_info(model_registry: ModelRegistry):
    """Test getting specific image model info from the registry."""
    image_models = model_registry.list_image_models()
    assert len(image_models) > 0

    # Test getting first image model
    first_model = image_models[0]
    model_info = model_registry.get_image_model_info(first_model.id)

    assert isinstance(model_info, ImageModelInfo)
    assert model_info.id == first_model.id
    assert model_info.name == first_model.name
    assert model_info.costs.image_generation is not None

    # Test non-existent image model
    with pytest.raises(ValueError, match="not found"):
        model_registry.get_image_model_info("non-existent-image-model")


def test_model_registry_image_model_costs_validation(model_registry: ModelRegistry):
    """Test that image model costs follow the validation rules."""
    image_models = model_registry.list_image_models()
    assert len(image_models) > 0

    for model in image_models:
        costs = model.costs.image_generation

        # Verify exactly one pricing model is set (as per business rule)
        pricing_models = [
            costs.flat_rate,
            costs.quality_tiers,
            costs.cost_per_megapixel,
            costs.cost_per_second,
        ]
        non_null_models = [p for p in pricing_models if p is not None]
        assert len(non_null_models) == 1, (
            f"Model {model.id} should have exactly 1 pricing model, got {len(non_null_models)}"
        )


def test_model_registry_generic_methods(model_registry: ModelRegistry):
    """Test generic get_model_info and list_models methods with modality."""
    # Test text modality (LLMs)
    text_models = model_registry.list_models("text")
    assert isinstance(text_models, list)
    assert len(text_models) > 0
    assert all(isinstance(model, LLMInfo) for model in text_models)

    first_text_model = text_models[0]
    text_model_info = model_registry.get_model_info(first_text_model.id, "text")
    assert isinstance(text_model_info, LLMInfo)
    assert text_model_info.id == first_text_model.id

    # Test image modality
    image_models = model_registry.list_models("image")
    assert isinstance(image_models, list)
    assert len(image_models) > 0
    assert all(isinstance(model, ImageModelInfo) for model in image_models)

    first_image_model = image_models[0]
    image_model_info = model_registry.get_model_info(first_image_model.id, "image")
    assert isinstance(image_model_info, ImageModelInfo)
    assert image_model_info.id == first_image_model.id

    # Test invalid modality
    with pytest.raises(ValueError):
        model_registry.list_models("invalid")  # type: ignore[call-overload]

    with pytest.raises(ValueError):
        model_registry.get_model_info("any-id", "invalid")  # type: ignore[call-overload]


def test_model_registry_modality_separation(model_registry: ModelRegistry):
    """Test that LLMs and image models are properly separated by modality."""
    llms = model_registry.list_llms()
    image_models = model_registry.list_image_models()

    # Get all model IDs
    llm_ids = {llm.id for llm in llms}
    image_model_ids = {model.id for model in image_models}

    # Verify no overlap between LLM and image model IDs
    assert len(llm_ids.intersection(image_model_ids)) == 0, (
        "LLM and image model IDs should not overlap"
    )

    # Verify we can access models through their respective methods
    for llm_id in llm_ids:
        llm = model_registry.get_llm_info(llm_id)
        assert isinstance(llm, LLMInfo)

        # Should not be accessible via image model method
        with pytest.raises(ValueError, match="not found"):
            model_registry.get_image_model_info(llm_id)

    for image_model_id in image_model_ids:
        image_model = model_registry.get_image_model_info(image_model_id)
        assert isinstance(image_model, ImageModelInfo)

        # Should not be accessible via LLM method
        with pytest.raises(ValueError, match="not found"):
            model_registry.get_llm_info(image_model_id)
