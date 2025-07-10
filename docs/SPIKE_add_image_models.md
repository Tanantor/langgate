# SPIKE: Adding Image Model Support to Langgate

**Date:** July 2025
**Status:** Proposed

## 1. Executive Summary

This SPIKE outlines the implementation plan for adding image generation model support to Langgate. The goal is to extend Langgate's current LLM-focused architecture to support image models while maintaining its clean separation of concerns and configuration-driven approach.

**Key Principles:**
- Breaking changes are acceptable
- Maintain clean architecture with proper type safety
- Support multiple modalities with extensibility for future additions

## 2. Current Langgate Architecture

Langgate currently consists of:

1. **Core Models** (`langgate.core.models`):
   - `LLMInfo`: Model metadata, costs, capabilities
   - `ModelProvider`: Creator/vendor information
   - `ModelCost`: Cost information structure

2. **Registry** (`langgate.registry`):
   - Singleton pattern with cached model information
   - Configuration-driven from YAML and JSON files
   - Model mappings: exposed ID → service provider/model ID

3. **Parameter Transformation** (`langgate.transform`):
   - Fluent interface for parameter manipulation
   - Supports defaults, overrides, renames, removes

4. **Configuration**:
   - YAML-based model mappings in `langgate_config.yaml`
   - JSON-based model metadata in `langgate_models.json`

## 3. Proposed Changes

### 3.1 Core Models (`langgate/core/models.py`)

Add modality enum and image model class:

```python
from enum import Enum

class Modality(str, Enum):
    """Supported model modalities."""
    TEXT = "text"
    IMAGE = "image"
    # Future: AUDIO = "audio", VIDEO = "video"

class TokenCosts(BaseModel):
    """Token-based cost information for models that charge for input/output tokens."""
    input_cost_per_token: Decimal
    output_cost_per_token: Decimal
    input_cached_cost_per_token: Decimal | None = None

class ImageGenerationCost(BaseModel):
    """Cost information for image generation with support for multiple pricing models."""
    # For simple flat-rate pricing (most providers)
    flat_rate: Decimal | None = None

    # For dimension-based pricing (OpenAI)
    quality_tiers: dict[str, dict[str, Decimal]] | None = None

    # For usage-based pricing
    cost_per_megapixel: Decimal | None = None
    cost_per_second: Decimal | None = None

    @model_validator(mode="after")
    def validate_exactly_one_pricing_model(self) -> Self:
        """Ensure exactly one pricing model is specified."""
        pricing_models = [
            self.flat_rate, self.quality_tiers,
            self.cost_per_megapixel, self.cost_per_second
        ]
        if sum(p is not None for p in pricing_models) != 1:
            raise ValueError("Exactly one pricing model must be set.")
        return self

class ImageModelCost(BaseModel):
    """Cost information for image generation models."""
    token_costs: TokenCosts | None = None
    image_generation: ImageGenerationCost

class BaseModelInfo(BaseModel):
    """Base class for all model types with common fields."""
    id: str = Field(...)
    name: str = Field(...)
    provider_id: ModelProviderId = Field(...)
    provider: ModelProvider
    description: Optional[str] = None
    updated_dt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_provider_id(self) -> Self:
        """Ensure provider_id matches provider.id."""
        self.provider_id = self.provider.id
        return self

class ImageModelInfo(BaseModelInfo):
    """Information about an image generation model."""
    costs: ImageModelCost = Field(default_factory=ImageModelCost)
```

### 3.2 Registry Updates (`langgate/registry/models.py`)

Update registry to support multiple modalities:

```python
class ModelRegistry:
    """Registry for managing model information across modalities."""

    def __init__(self, config: RegistryConfig | None = None):
        # Separate caches by modality
        self._model_caches: dict[Modality, dict[str, Any]] = {
            Modality.TEXT: {},
            Modality.IMAGE: {},
        }

        self._build_model_caches()

    def _build_model_caches(self) -> None:
        """Build cached model information for all modalities."""
        for model_id, mapping in self.config.model_mappings.items():
            # Get model data from JSON to determine modality
            service_provider = mapping["service_provider"]
            service_model_id = mapping["service_model_id"]
            full_service_model_id = f"{service_provider}/{service_model_id}"
            model_data = self.config.models_data.get(full_service_model_id, {})

            # Map "mode" field from JSON to Modality enum
            mode = model_data.get("mode")
            if not mode:
                raise ValueError(f"Model {model_id} does not specify a 'mode' in configuration")
            modality = Modality.IMAGE if mode == "image" else Modality.TEXT

            if modality == Modality.TEXT:
                model_info = self._build_llm_info(model_id, mapping)
                self._model_caches[Modality.TEXT][model_id] = model_info
            elif modality == Modality.IMAGE:
                model_info = self._build_image_model_info(model_id, mapping)
                self._model_caches[Modality.IMAGE][model_id] = model_info

    def _build_image_model_info(self, model_id: str, mapping: dict) -> ImageModelInfo:
        """Build ImageModelInfo from configuration."""
        # Follow same pattern as _build_llm_info
        service_provider = mapping["service_provider"]
        service_model_id = mapping["service_model_id"]

        full_service_model_id = f"{service_provider}/{service_model_id}"
        model_data = self.config.models_data.get(full_service_model_id, {})

        costs = ImageModelCost.model_validate(model_data.get("costs", {}))

        return ImageModelInfo(
            id=model_id,
            name=name,s
            description=description,
            provider_id=provider_id,
            provider=provider,
            costs=costs,
        )

    # Generic accessor methods with overloads for type safety
    @overload
    def get_model_info(self, model_id: str, modality: Literal[Modality.TEXT]) -> LLMInfo: ...

    @overload
    def get_model_info(self, model_id: str, modality: Literal[Modality.IMAGE]) -> ImageModelInfo: ...

    def get_model_info(self, model_id: str, modality: Modality) -> BaseModelInfo:
        """Get model information by ID and modality with proper typing."""
        if model_id not in self._model_caches[modality]:
            raise ValueError(f"{modality.value.capitalize()} model {model_id} not found")
        return self._model_caches[modality][model_id]

    @overload
    def list_models(self, modality: Literal[Modality.TEXT]) -> list[LLMInfo]: ...

    @overload
    def list_models(self, modality: Literal[Modality.IMAGE]) -> list[ImageModelInfo]: ...

    def list_models(self, modality: Modality) -> list[BaseModelInfo]:
        """List all available models for a given modality with proper typing."""
        return list(self._model_caches[modality].values())
```

### 3.3 Configuration Schema (`langgate/core/schemas/config.py`)

Add modality field to model configuration:

```python
class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    id: str
    service: ModelServiceConfig
    modality: Modality
    name: Optional[str] = None
    description: Optional[str] = None
    api_format: Optional[str] = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    override_params: dict[str, Any] = Field(default_factory=dict)
    remove_params: list[str] = Field(default_factory=list)
    rename_params: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")
```

## 4. Configuration Examples

### 4.1 YAML Configuration (`langgate_config.yaml`)

```yaml
services:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"

  replicate:
    api_key: "${REPLICATE_API_KEY}"
    base_url: "https://api.replicate.com/v1"

models:
  # LLM models
  - id: openai/gpt-4
    modality: text
    service:
      provider: openai
      model_id: gpt-4

  # Image models
  # NOTE: modality is NOT specified here - it's inferred from the "mode" field
  # in langgate_models.json during registry initialization
  - id: openai/dall-e-3
    service:
      provider: openai
      model_id: dall-e-3
    default_params:
      size: "1024x1024"
      quality: "standard"
      n: 1

  - id: black-forest-labs/flux-dev
    service:
      provider: replicate
      model_id: black-forest-labs/flux-dev
    default_params:
      disable_safety_checker: true

  - id: stability-ai/sd-3.5-large
    service:
      provider: replicate
      model_id: stability-ai/stable-diffusion-3.5-large
    default_params:
      width: 1024
      height: 1024
      num_inference_steps: 30
      guidance_scale: 6.5
```

### 4.2 Model Data JSON (`langgate_models.json`)

```json
{
  "openai/gpt-image-1": {
    "name": "GPT Image 1",
    "mode": "image",
    "service_provider": "openai",
    "model_provider": "openai",
    "model_provider_name": "OpenAI",
    "description": "Advanced text-to-image model with token-based input costs",
    "costs": {
      "token_costs": {
        "input_cost_per_token": 0.00001,
        "output_cost_per_token": 0.00004,
        "input_cached_cost_per_token": 0.0000025
      },
      "image_generation": {
        "quality_tiers": {
          "low": {
            "1024x1024": 0.011,
            "1024x1536": 0.016,
            "1536x1024": 0.016
          },
          "medium": {
            "1024x1024": 0.042,
            "1024x1536": 0.063,
            "1536x1024": 0.063
          },
          "high": {
            "1024x1024": 0.167,
            "1024x1536": 0.25,
            "1536x1024": 0.25
          }
        }
      }
    }
  },
  "openai/dall-e-3": {
    "name": "DALL-E 3",
    "mode": "image",
    "service_provider": "openai",
    "model_provider": "openai",
    "model_provider_name": "OpenAI",
    "description": "Advanced text-to-image model with improved coherence",
    "costs": {
      "image_generation": {
        "quality_tiers": {
          "standard": {
            "1024x1024": 0.04,
            "1024x1792": 0.08,
            "1792x1024": 0.08
          },
          "hd": {
            "1024x1024": 0.08,
            "1024x1792": 0.12,
            "1792x1024": 0.12
          }
        }
      }
    }
  },
  "replicate/black-forest-labs/flux-dev": {
    "name": "FLUX.1 [dev]",
    "mode": "image",
    "service_provider": "replicate",
    "model_provider": "black-forest-labs",
    "model_provider_name": "Black Forest Labs",
    "description": "FLUX.1 dev is an open-weight, 12 billion parameter rectified flow transformer, distilled from FLUX.1 [pro], FLUX.1 [dev] obtains similar quality and prompt adherence capabilities, while being more efficient and faster.",
    "costs": {
      "image_generation": {
        "flat_rate": 0.025
      }
    }
  },
  "replicate/stability-ai/stable-diffusion-3.5-large": {
    "name": "Stable Diffusion 3.5 Large",
    "mode": "image",
    "service_provider": "replicate",
    "model_provider": "stability-ai",
    "model_provider_name": "Stability AI",
    "description": "High-quality text-to-image generation",
    "costs": {
      "image_generation": {
        "flat_rate": 0.065
      }
    }
  }
}
```

## 5. Image Model Cost Structure Solution

### 5.1 Problem Analysis

Image generation providers have fundamentally different pricing models:

1. **OpenAI's gpt-image-1**: Charges for BOTH input/output tokens AND image generation costs (complex dual pricing)
2. **OpenAI's DALL-E 2/3**: Only charges per image based on quality and dimensions
3. **Other providers**: May charge per image, per megapixel, per compute second, etc.

### 5.2 Solution Architecture

The solution uses a **nested cost structure** that separates concerns:

- **TokenCosts**: Handles input/output token pricing (for hybrid models like gpt-image-1)
- **ImageGenerationCost**: Handles image generation pricing with multiple models:
  - `flat_rate`: Simple per-image pricing
  - `quality_tiers`: Quality/dimension-based pricing (OpenAI DALL-E)
  - `cost_per_megapixel`: Usage-based pricing
  - `cost_per_second`: Compute-time pricing

### 5.3 Key Benefits

1. **Type Safety**: Pydantic validation ensures data integrity
2. **Flexibility**: Supports all current and anticipated pricing models
3. **Clarity**: JSON structure mirrors provider pricing documentation
4. **Extensibility**: Easy to add new pricing models
5. **Maintainability**: Clear separation of concerns

### 5.4 Validation Strategy

The schema enforces exactly one pricing model per provider using Pydantic validators:
- Prevents invalid combinations (e.g., both flat_rate and quality_tiers)
- Ensures data consistency across configurations
- Provides clear error messages for misconfigurations

## 6. Implementation Steps

### 6.1 Phase 1: Core Model Support

1. **Add Modality Enum** (`core/models.py`)
2. **Create ImageModelInfo Class** (`core/models.py`)
3. **Update Config Schema** (`core/schemas/config.py`)

### 6.2 Phase 2: Registry Updates

1. **Refactor Registry Storage** (`registry/models.py`):
   - Change to `_model_caches[Modality]`
   - Update `_build_model_cache` to handle multiple modalities

2. **Add Typed Accessors**:
   - `get_llm_info()` → `LLMInfo`
   - `get_image_model_info()` → `ImageModelInfo`
   - `list_llms()` → `list[LLMInfo]`
   - `list_image_models()` → `list[ImageModelInfo]`

### 6.3 Phase 3: Integration Points

1. **LocalTransformerClient Updates** (`transform/local.py`):
   - No changes needed for image models

2. **Client Protocol Updates** (`client/protocol.py`):
   ```python
   class RegistryClientProtocol(Protocol):
       async def get_llm_info(self, model_id: str) -> LLMInfo: ...
       async def get_image_model_info(self, model_id: str) -> ImageModelInfo: ...
       async def list_llm_models(self) -> list[LLMInfo]: ...
       async def list_image_models(self) -> list[ImageModelInfo]: ...
   ```

## 7. Usage Example

After implementation, business applications can use langgate for image models:

```python
from langgate.registry import ModelRegistry
from langgate.transform import LocalTransformerClient

# Initialize components
registry = ModelRegistry()
transformer = LocalTransformerClient()

# Get image model info
model_info = registry.get_image_model_info("openai/dall-e-3")
assert isinstance(model_info, ImageModelInfo)

# Transform parameters
params = {
    "prompt": "A beautiful sunset",
    "size": "1024x1024",
    "quality": "hd"
}
api_format, transformed = await transformer.get_params("opanai/dall-e-3", params)

# Business application handles schema validation
# This happens OUTSIDE langgate
```

## 8. Testing Strategy

### 8.1 Unit Tests

1. **Model Validation Tests**:
   ```python
   def test_image_info_validation():
       """Test ImageModelInfo model validation."""
       info = ImageModelInfo(
           id="test-model",
           name="Test Model",
           provider_id="test-provider",
           provider=ModelProvider(id="test-provider", name="Test")
       )
       assert info.id == "test-model"
   ```

2. **Registry Tests**:
   ```python
   def test_registry_modality_separation():
       """Test models are stored separately by modality."""
       registry = ModelRegistry(test_config)

       # Should not mix modalities
       with pytest.raises(ValueError):
           registry.get_llm_info("openai/dall-e-3")  # Image model

       # Correct accessor works
       image_info = registry.get_image_model_info("openai/dall-e-3")
       assert isinstance(image_info, ImageModelInfo)
   ```

### 8.2 Integration Tests

1. **Configuration Loading**:
   ```python
   def test_mixed_modality_config():
       """Test loading config with both LLM and image models."""
       config = load_test_config("mixed_models.yaml")
       registry = ModelRegistry(config)

       llm_models = registry.list_llm_models()
       image_models = registry.list_image_models()

       assert len(llm_models) > 0
       assert len(image_models) > 0
   ```

## 9. Summary

This implementation plan:
1. Maintains Langgate's clean architecture
2. Provides strong typing with modality-specific models
3. Requires no schema management in Langgate
4. Supports breaking changes as requested
5. Enables future extensibility for other modalities

The approach balances immediate needs (image support) with long-term maintainability and extensibility. By using separate typed accessors and maintaining clear boundaries between modalities, we ensure type safety while keeping the system flexible for future enhancements.
