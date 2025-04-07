"""Core domain data models for LangGate."""

from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, NewType

from pydantic import BaseModel, ConfigDict, Field, SecretStr

# Type aliases for flexibility while maintaining naming compatibility
ServiceProviderId = NewType("ServiceProviderId", str)
# Model provider might differ from the inference service provider
# The service provider is not intended to be exposed to external consumers of the registry
# The service provider is used by the proxy for routing requests to the correct service
ModelProviderId = NewType("ModelProviderId", str)

# Common inference service providers for convenience
SERVICE_PROVIDER_OPENAI = ServiceProviderId("openai")
SERVICE_PROVIDER_ANTHROPIC = ServiceProviderId("anthropic")
SERVICE_PROVIDER_GEMINI = ServiceProviderId("gemini")
SERVICE_PROVIDER_VERTEX = ServiceProviderId("vertex")
SERVICE_PROVIDER_DEEPSEEK = ServiceProviderId("deepseek")
SERVICE_PROVIDER_ALIBABA = ServiceProviderId("alibaba")
SERVICE_PROVIDER_XAI = ServiceProviderId("xai")
SERVICE_PROVIDER_ELEUTHERIA_VLLM = ServiceProviderId("eleutheria/vllm")
SERVICE_PROVIDER_GROQ = ServiceProviderId("groq")
SERVICE_PROVIDER_OPENROUTER = ServiceProviderId("openrouter")
SERVICE_PROVIDER_FIREWORKS = ServiceProviderId("fireworks_ai")
SERVICE_PROVIDER_HUGGINGFACE = ServiceProviderId("huggingface")
SERVICE_PROVIDER_COHERE = ServiceProviderId("cohere")
SERVICE_PROVIDER_BEDROCK = ServiceProviderId("bedrock")
SERVICE_PROVIDER_AZURE_OPENAI = ServiceProviderId("azure_openai")

# Common model providers for convenience
MODEL_PROVIDER_OPENAI = ModelProviderId("openai")
MODEL_PROVIDER_ANTHROPIC = ModelProviderId("anthropic")
MODEL_PROVIDER_META = ModelProviderId("meta")
MODEL_PROVIDER_GOOGLE = ModelProviderId("google")
MODEL_PROVIDER_DEEPSEEK = ModelProviderId("deepseek")
MODEL_PROVIDER_ALIBABA = ModelProviderId("alibaba")
MODEL_PROVIDER_XAI = ModelProviderId("xai")
MODEL_PROVIDER_COHERE = ModelProviderId("cohere")
MODEL_PROVIDER_ELEUTHERIA = ModelProviderId("eleutheria")


class ServiceProvider(BaseModel):
    """Information about a service provider (API service)."""

    id: ServiceProviderId
    base_url: str
    api_key: SecretStr
    default_params: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(BaseModel):
    """Information about a model provider (creator)."""

    id: ModelProviderId
    name: str
    description: str | None = None


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
    supports_response_schema: bool | None = None
    supports_system_messages: bool | None = None

    model_config = ConfigDict(extra="allow")


TokenCost = Annotated[Decimal, "TokenCost"]
Percentage = Annotated[Decimal, "Percentage"]
TokenUsage = Annotated[Decimal, "TokenUsage"]


class ModelCost(BaseModel):
    """Cost information for a language model."""

    input_cost_per_token: TokenCost = Field(default_factory=lambda: Decimal())
    output_cost_per_token: TokenCost = Field(default_factory=lambda: Decimal())
    input_cost_per_token_batches: TokenCost | None = None
    output_cost_per_token_batches: TokenCost | None = None
    cache_read_input_token_cost: TokenCost | None = None

    model_config = ConfigDict(extra="allow")


class LLMInfo(BaseModel):
    """Information about a language model."""

    id: str  # "gpt-4o"
    name: str
    provider: ModelProvider  # Who created it (shown to users)
    description: str | None = None
    costs: ModelCost = Field(default_factory=ModelCost)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    context_window: ContextWindow = Field(default_factory=ContextWindow)
    updated_dt: datetime = Field(default_factory=lambda: datetime.now(UTC))

    model_config = ConfigDict(extra="allow")
