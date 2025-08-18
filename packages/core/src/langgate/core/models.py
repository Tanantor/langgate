"""Core domain data models for LangGate."""

from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, NewType, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from langgate.core.fields import NormalizedDecimal

# Type aliases for flexibility while maintaining naming compatibility
ServiceProviderId = NewType("ServiceProviderId", str)
# Model provider might differ from the inference service provider
# The service provider is not intended to be exposed to external consumers of the registry
# The service provider is used by the proxy for routing requests to the correct service
ModelProviderId = NewType("ModelProviderId", str)

# Common model providers for convenience
MODEL_PROVIDER_OPENAI = ModelProviderId("openai")
MODEL_PROVIDER_ANTHROPIC = ModelProviderId("anthropic")
MODEL_PROVIDER_META = ModelProviderId("meta")
MODEL_PROVIDER_GOOGLE = ModelProviderId("google")
MODEL_PROVIDER_MINIMAX = ModelProviderId("minimax")
MODEL_PROVIDER_DEEPSEEK = ModelProviderId("deepseek")
MODEL_PROVIDER_MISTRAL = ModelProviderId("mistralai")
MODEL_PROVIDER_ALIBABA = ModelProviderId("alibaba")
MODEL_PROVIDER_XAI = ModelProviderId("xai")
MODEL_PROVIDER_COHERE = ModelProviderId("cohere")
MODEL_PROVIDER_ELEUTHERIA = ModelProviderId("eleutheria")


class Modality(str, Enum):
    """Supported model modalities."""

    TEXT = "text"
    IMAGE = "image"


class ServiceProvider(BaseModel):
    """Information about a service provider (API service)."""

    id: ServiceProviderId
    base_url: str
    api_key: SecretStr
    default_params: dict[str, Any] = Field(default_factory=dict)


class ModelProviderBase(BaseModel):
    id: ModelProviderId | None = None
    name: str | None = None
    description: str | None = None

    model_config = ConfigDict(extra="allow")


class ModelProvider(ModelProviderBase):
    """Information about a model provider (creator)."""

    id: ModelProviderId = Field(default=...)
    name: str = Field(default=...)


class ContextWindow(BaseModel):
    """Context window information for a model."""

    max_input_tokens: int = 0
    max_output_tokens: int = 0

    model_config = ConfigDict(extra="allow")


class ModelCapabilities(BaseModel):
    """Capabilities of a language model."""

    supports_tools: bool | None = None
    supports_parallel_tool_calls: bool | None = None
    supports_vision: bool | None = None
    supports_audio_input: bool | None = None
    supports_audio_output: bool | None = None
    supports_prompt_caching: bool | None = None
    supports_reasoning: bool | None = None
    supports_response_schema: bool | None = None
    supports_system_messages: bool | None = None

    model_config = ConfigDict(extra="allow")


TokenCost = Annotated[NormalizedDecimal, "TokenCost"]
Percentage = Annotated[NormalizedDecimal, "Percentage"]
TokenUsage = Annotated[NormalizedDecimal, "TokenUsage"]


class BaseModelInfo(BaseModel):
    """Base class for all model types with common fields."""

    id: str | None = None
    name: str | None = None
    provider_id: ModelProviderId | None = None
    description: str | None = None

    model_config = ConfigDict(extra="allow")


class TokenCosts(BaseModel):
    """Token-based cost information for models that charge for input/output tokens."""

    input_cost_per_token: TokenCost = Field(default_factory=Decimal)
    output_cost_per_token: TokenCost = Field(default_factory=Decimal)
    input_cost_per_token_batches: TokenCost | None = None
    output_cost_per_token_batches: TokenCost | None = None
    cache_read_input_token_cost: TokenCost | None = None
    input_cached_cost_per_token: TokenCost | None = None

    model_config = ConfigDict(extra="allow")


class LLMInfoBase(BaseModelInfo):
    costs: TokenCosts | None = None
    capabilities: ModelCapabilities | None = None
    context_window: ContextWindow | None = None


class LLMInfo(LLMInfoBase):
    """Information about a language model."""

    id: str = Field(default=...)  # "gpt-5"
    name: str = Field(default=...)
    provider_id: ModelProviderId = Field(default=...)

    provider: ModelProvider  # Who created it (shown to users)
    description: str | None = None
    costs: TokenCosts = Field(default_factory=TokenCosts)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    context_window: ContextWindow = Field(default_factory=ContextWindow)
    updated_dt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_provider_id(self):
        self.provider_id = self.provider.id
        return self


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
            self.flat_rate,
            self.quality_tiers,
            self.cost_per_megapixel,
            self.cost_per_second,
        ]
        if sum(p is not None for p in pricing_models) != 1:
            raise ValueError("Exactly one pricing model must be set.")
        return self


class ImageModelCost(BaseModel):
    """Cost information for image generation models."""

    token_costs: TokenCosts | None = None
    image_generation: ImageGenerationCost


class ImageModelInfoBase(BaseModelInfo):
    costs: ImageModelCost | None = None


class ImageModelInfo(ImageModelInfoBase):
    """Information about an image generation model."""

    id: str = Field(default=...)
    name: str = Field(default=...)
    provider_id: ModelProviderId = Field(default=...)

    provider: ModelProvider
    description: str | None = None
    costs: ImageModelCost = Field(default=...)
    updated_dt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_provider_id(self) -> Self:
        """Ensure provider_id matches provider.id."""
        self.provider_id = self.provider.id
        return self
