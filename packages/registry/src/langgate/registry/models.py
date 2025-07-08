"""Model information and registry implementation."""

import os
from typing import Any, Literal, overload

from langgate.core.logging import get_logger
from langgate.core.models import (
    ContextWindow,
    ImageGenerationCost,
    ImageModelCost,
    ImageModelInfo,
    LLMInfo,
    Modality,
    ModelCapabilities,
    ModelProvider,
    ModelProviderId,
    TokenCosts,
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

            # Separate caches by modality
            self._model_caches: dict[Modality, dict[str, Any]] = {
                Modality.TEXT: {},
                Modality.IMAGE: {},
            }

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
                if not any(
                    self._model_caches[modality] for modality in self._model_caches
                ):  # Only raise if we have no data
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
        """Build cached model information for all modalities."""
        # Clear all caches
        for modality in self._model_caches:
            self._model_caches[modality] = {}

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
                    available_models=list(self._model_caches[Modality.TEXT].keys()),
                )

            # Map "mode" field from JSON to Modality enum
            mode = model_data.get("mode")
            if not mode:
                raise ValueError(
                    f"Model {model_id} does not have a valid 'mode' field in model data."
                )
            modality = Modality.IMAGE if mode == "image" else Modality.TEXT

            if modality == Modality.TEXT:
                model_info = self._build_llm_info(model_id, mapping, model_data)
                self._model_caches[Modality.TEXT][model_id] = model_info
            elif modality == Modality.IMAGE:
                model_info = self._build_image_model_info(model_id, mapping, model_data)
                self._model_caches[Modality.IMAGE][model_id] = model_info

        # Check if registry is empty after building all caches
        if not any(self._model_caches[modality] for modality in self._model_caches):
            logger.warning(
                "empty_model_registry",
                help="No models were loaded into the registry cache. Check configuration files.",
                models_data_path=str(self.config.models_data_path),
                config_path=str(self.config.config_path),
            )

    def _get_model_metadata(
        self, model_id: str, mapping: dict[str, Any], model_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Get generic model metadata from configuration and model data."""
        # Determine the model provider (creator of the model)
        # Model provider might differ from the inference service provider
        # The service provider is not intended to be exposed to external consumers of the registry
        # The service provider is used by the proxy for routing requests to the correct service
        # Priority order:
        #   1. Explicitly set in mapping (takes precedence)
        #   2. Value coming from the service-level model data
        model_provider_id: str = mapping.get("model_provider") or model_data.get(
            "model_provider", ""
        )
        if not model_provider_id:
            raise ValueError(
                f"Model {model_id} does not have a valid provider ID, Set `model_provider` in model data."
            )

        # Get the provider display name, either from data or fallback to ID
        provider_display_name = mapping.get("model_provider_name") or model_data.get(
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

        return {
            "name": name,
            "description": description,
            "model_provider_id": model_provider_id,
            "provider_display_name": provider_display_name,
        }

    def _build_llm_info(
        self, model_id: str, mapping: dict[str, Any], model_data: dict[str, Any]
    ) -> LLMInfo:
        """Build LLMInfo from configuration data."""

        generic_metadata = self._get_model_metadata(model_id, mapping, model_data)

        # Extract context window if available
        context_window = ContextWindow.model_validate(model_data.get("context", {}))

        # Extract capabilities if available
        capabilities = ModelCapabilities.model_validate(
            model_data.get("capabilities", {})
        )

        # Extract costs if available
        costs = TokenCosts.model_validate(model_data.get("costs", {}))

        # Create complete model info
        return LLMInfo(
            id=model_id,
            name=generic_metadata["name"],
            description=generic_metadata["description"],
            provider_id=ModelProviderId(generic_metadata["model_provider_id"]),
            provider=ModelProvider(
                id=ModelProviderId(generic_metadata["model_provider_id"]),
                name=generic_metadata["provider_display_name"],
                description=None,
            ),
            costs=costs,
            capabilities=capabilities,
            context_window=context_window,
        )

    def _build_image_model_info(
        self, model_id: str, mapping: dict[str, Any], model_data: dict[str, Any]
    ) -> ImageModelInfo:
        """Build ImageModelInfo from configuration data."""

        generic_metadata = self._get_model_metadata(model_id, mapping, model_data)

        # Extract and validate image model costs
        costs_data = model_data.get("costs", {})

        # Handle token costs if present (for hybrid models like gpt-image-1)
        token_costs = None
        if "token_costs" in costs_data:
            token_costs = TokenCosts.model_validate(costs_data["token_costs"])

        # Handle image generation costs (required)
        image_generation_data = costs_data.get("image_generation", {})
        image_generation = ImageGenerationCost.model_validate(image_generation_data)

        costs = ImageModelCost(
            token_costs=token_costs, image_generation=image_generation
        )

        # Create complete model info
        return ImageModelInfo(
            id=model_id,
            name=generic_metadata["name"],
            description=generic_metadata["description"],
            provider_id=ModelProviderId(generic_metadata["model_provider_id"]),
            provider=ModelProvider(
                id=ModelProviderId(generic_metadata["model_provider_id"]),
                name=generic_metadata["provider_display_name"],
                description=None,
            ),
            costs=costs,
        )

    # Generic accessor methods with overloads for type safety
    @overload
    def get_model_info(self, model_id: str, modality: Literal["text"]) -> LLMInfo: ...

    @overload
    def get_model_info(
        self, model_id: str, modality: Literal["image"]
    ) -> ImageModelInfo: ...

    def get_model_info(self, model_id: str, modality: str) -> LLMInfo | ImageModelInfo:
        """Get model information by ID and modality with proper typing."""
        modality_enum = Modality(modality)
        if model_id not in self._model_caches[modality_enum]:
            raise ValueError(f"{modality.capitalize()} model {model_id} not found")
        return self._model_caches[modality_enum][model_id]

    @overload
    def list_models(self, modality: Literal["text"]) -> list[LLMInfo]: ...

    @overload
    def list_models(self, modality: Literal["image"]) -> list[ImageModelInfo]: ...

    def list_models(self, modality: str) -> list[LLMInfo] | list[ImageModelInfo]:
        """List all available models for a given modality with proper typing."""
        modality_enum = Modality(modality)
        return list(self._model_caches[modality_enum].values())

    # Backward compatibility methods for existing LLM functionality
    def get_llm_info(self, model_id: str) -> LLMInfo:
        """Get LLM information by model ID (backward compatibility)."""
        return self.get_model_info(model_id, "text")

    def get_image_model_info(self, model_id: str) -> ImageModelInfo:
        """Get image model information by model ID."""
        return self.get_model_info(model_id, "image")

    def list_llms(self) -> list[LLMInfo]:
        """List all available LLMs."""
        return self.list_models("text")

    def list_image_models(self) -> list[ImageModelInfo]:
        """List all available image models."""
        return self.list_models("image")

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
