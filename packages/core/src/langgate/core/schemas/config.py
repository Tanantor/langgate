"""Schema definitions for YAML configuration validation."""

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)

from langgate.core.fields import UrlOrEnvVar


class ServiceModelPatternConfig(BaseModel):
    """Configuration for model pattern matching within a service."""

    default_params: dict[str, Any] = Field(default_factory=dict)
    override_params: dict[str, Any] = Field(default_factory=dict)
    remove_params: list[str] = Field(default_factory=list)
    rename_params: dict[str, str] = Field(default_factory=dict)


class ServiceConfig(BaseModel):
    """Configuration for a service provider."""

    api_key: str | SecretStr
    base_url: UrlOrEnvVar | None = None
    api_format: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    override_params: dict[str, Any] = Field(default_factory=dict)
    remove_params: list[str] = Field(default_factory=list)
    rename_params: dict[str, str] = Field(default_factory=dict)
    model_patterns: dict[str, ServiceModelPatternConfig] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")

    @field_validator("api_key")
    def validate_api_key(cls, v):
        """Convert string API keys to SecretStr."""
        if isinstance(v, str):
            # Keep environment variable references as strings
            if v.startswith("${") and v.endswith("}"):
                return v
            return SecretStr(v)
        return v


class ModelServiceConfig(BaseModel):
    """Service configuration for a specific model."""

    provider: str
    model_id: str


class ModelConfig(BaseModel):
    """Configuration for a specific model."""

    id: str
    service: ModelServiceConfig
    name: str | None = None
    description: str | None = None
    api_format: str | None = None
    default_params: dict[str, Any] = Field(default_factory=dict)
    override_params: dict[str, Any] = Field(default_factory=dict)
    remove_params: list[str] = Field(default_factory=list)
    rename_params: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="allow")


class ConfigSchema(BaseModel):
    """Root schema for the configuration YAML."""

    default_params: dict[str, Any] = Field(default_factory=dict)
    services: dict[str, ServiceConfig] = Field(default_factory=dict)
    models: list[ModelConfig] = Field(default_factory=list)
    app_config: dict[str, Any] = Field(default_factory=dict)
    models_merge_mode: Literal["merge", "replace", "extend"] = "merge"

    @model_validator(mode="after")
    def validate_model_service_providers(self) -> "ConfigSchema":
        """Validate that all model service providers exist in the services configuration."""
        available_providers = set(self.services.keys())

        for model in self.models:
            if model.service.provider not in available_providers:
                raise ValueError(
                    f"Model '{model.id}' references service provider '{model.service.provider}' "
                    f"which is not defined in services. Available providers: {sorted(available_providers)}"
                )

        return self
