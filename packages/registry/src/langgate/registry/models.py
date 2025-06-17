"""Model information and registry implementation."""

import os
from typing import Any

from langgate.core.logging import get_logger
from langgate.core.models import (
    ContextWindow,
    LLMInfo,
    ModelCapabilities,
    ModelCost,
    ModelProvider,
    ModelProviderId,
)
from langgate.registry.config import RegistryConfig

logger = get_logger(__name__)


class ModelRegistry:
    """Registry for managing model information."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            logger.debug("creating_model_registry_singleton")
            cls._instance = super().__new__(cls)
        else:
            logger.debug(
                "reusing_registry_singleton",
                initialized=hasattr(cls._instance, "_initialized"),
            )
        return cls._instance

    def __init__(self, config: RegistryConfig | None = None):
        """Initialize the registry.

        Args:
            config: Optional configuration object, will create a new one if not provided
        """
        if not hasattr(self, "_initialized"):
            self.config = config or RegistryConfig()

            # Cached model info objects
            self._models_cache: dict[str, LLMInfo] = {}

            # Build the model cache
            try:
                self._build_model_cache()
            except Exception:
                logger.exception(
                    "failed_to_build_model_cache",
                    models_data_path=str(self.config.models_data_path),
                    config_path=str(self.config.config_path),
                    env_file_path=str(self.config.env_file_path),
                    env_file_exists=self.config.env_file_path.exists(),
                )
                if not self._models_cache:  # Only raise if we have no data
                    raise
            logger.debug(
                "initialized_model_registry_singleton",
                models_data_path=str(self.config.models_data_path),
                config_path=str(self.config.config_path),
                env_file_path=str(self.config.env_file_path),
                env_file_exists=self.config.env_file_path.exists(),
            )
            self._initialized = True

    def _build_model_cache(self) -> None:
        """Build the cached model information from configuration."""
        self._models_cache = {}

        for model_id, mapping in self.config.model_mappings.items():
            service_provider: str = mapping["service_provider"]
            service_model_id: str = mapping["service_model_id"]

            # Check if we have data for this service model ID
            full_service_model_id = f"{service_provider}/{service_model_id}"

            # Try to find model data
            model_data = {}
            if full_service_model_id in self.config.models_data:
                model_data = self.config.models_data[full_service_model_id].copy()
            else:
                logger.warning(
                    "no_model_data_found",
                    msg="No model data found for service provider model ID",
                    help="""Check that the model data file contains the correct service provider model ID.
                        To add new models for a service provider, add the model data to your langgate_models.json file.""",
                    full_service_model_id=full_service_model_id,
                    service_provider=service_provider,
                    service_model_id=service_model_id,
                    exposed_model_id=model_id,
                    available_models=list(self._models_cache.keys()),
                )

            # Extract context window if available
            context_window = ContextWindow.model_validate(model_data.get("context", {}))

            # Extract capabilities if available
            capabilities = ModelCapabilities.model_validate(
                model_data.get("capabilities", {})
            )

            # Extract costs if available
            costs = ModelCost.model_validate(model_data.get("costs", {}))

            # Determine the model provider (creator of the model)
            # Model provider might differ from the inference service provider
            # The service provider is not intended to be exposed to external consumers of the registry
            # The service provider is used by the proxy for routing requests to the correct service
            if "model_provider" in model_data:
                model_provider_id: str = model_data["model_provider"]
            else:
                raise ValueError(
                    f"Model {model_id} does not have a valid provider ID, Set `model_provider` in model data."
                )

            # Get the provider display name, either from data or fallback to ID
            provider_display_name = model_data.get(
                "model_provider_name", model_provider_id.title()
            )

            # Name can come from multiple sources in decreasing priority
            # Use the model name from the config if available, otherwise use the model data name,
            # If no name is provided, default to the model portion of the fully specified model ID
            # (if it contains any slashes), or else the entire model ID.
            model_name_setting = mapping.get("name") or model_data.get("name")
            if not model_name_setting:
                logger.warning(
                    "model_name_not_set",
                    msg="Model name not found in config or model data files",
                    model_id=model_id,
                )
            name = (
                model_name_setting or model_specifier
                if (model_specifier := model_id.split("/")[-1])
                else model_id
            )

            # Description can come from config mapping or model data
            description = mapping.get("description") or model_data.get("description")

            # Create complete model info
            self._models_cache[model_id] = LLMInfo(
                id=model_id,
                name=name,
                description=description,
                provider_id=ModelProviderId(model_provider_id),
                provider=ModelProvider(
                    id=ModelProviderId(model_provider_id),
                    name=provider_display_name,
                    description=None,
                ),
                costs=costs,
                capabilities=capabilities,
                context_window=context_window,
            )

        if not self._models_cache:
            logger.warning(
                "empty_model_registry",
                help="No models were loaded into the registry cache. Check configuration files.",
                models_data_path=str(self.config.models_data_path),
                config_path=str(self.config.config_path),
            )

    def get_model_info(self, model_id: str) -> LLMInfo:
        """Get model information by model ID."""
        if model_id not in self._models_cache:
            raise ValueError(f"Model {model_id} not found")

        return self._models_cache[model_id]

    def list_models(self) -> list[LLMInfo]:
        """List all available models."""
        return list(self._models_cache.values())

    def get_provider_info(self, model_id: str) -> dict[str, Any]:
        """Get provider information for a model to use in the proxy."""
        if not (mapping := self.config.model_mappings.get(model_id)):
            raise ValueError(f"Model {model_id} not found")

        service_provider = mapping["service_provider"]

        if not (service_config := self.config.service_config.get(service_provider)):
            raise ValueError(f"Service provider {service_provider} not found")

        provider_info = {"provider": service_provider}

        # Add base URL
        if "base_url" in service_config:
            provider_info["base_url"] = service_config["base_url"]

        # Add API key (resolve from environment)
        if "api_key" in service_config:
            api_key: str = service_config["api_key"]
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                if env_var in os.environ:
                    provider_info["api_key"] = os.environ[env_var]
                else:
                    logger.warning("env_variable_not_found", variable=env_var)
            else:
                provider_info["api_key"] = api_key

        return provider_info
