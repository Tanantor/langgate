"""Local transformer client implementation."""

import importlib.resources
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import SecretStr

from langgate.core.logging import get_logger
from langgate.core.schemas.config import ConfigSchema, ModelConfig
from langgate.core.utils.config_utils import load_yaml_config, resolve_path
from langgate.transform.protocol import TransformerClientProtocol
from langgate.transform.transformer import ParamTransformer

logger = get_logger(__name__)


class LocalTransformerClient(TransformerClientProtocol):
    """
    Local transformer client for parameter transformations.

    This client handles parameter transformations based on local configuration.
    """

    _instance = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self, config_path: Path | None = None, env_file_path: Path | None = None
    ):
        """Initialize the client.

        Args:
            config_path: Path to the configuration file
            env_file_path: Path to the environment file
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        # Set up default paths
        cwd = Path.cwd()

        core_resources = importlib.resources.files("langgate.core")
        default_config_path = Path(
            str(core_resources.joinpath("data", "default_config.yaml"))
        )
        cwd_config_path = cwd / "langgate_config.yaml"
        cwd_env_path = cwd / ".env"

        self.config_path = resolve_path(
            "LANGGATE_CONFIG",
            config_path,
            cwd_config_path if cwd_config_path.exists() else default_config_path,
            "config_path",
            logger,
        )
        self.env_file_path = resolve_path(
            "LANGGATE_ENV_FILE",
            env_file_path,
            cwd_env_path,
            "env_file_path",
            logger,
        )

        # Cache for configs
        self._global_config: dict[str, Any] = {}
        self._service_config: dict[str, dict[str, Any]] = {}
        self._model_mappings: dict[str, dict[str, Any]] = {}

        # Load configuration
        self._load_config()
        self._initialized = True
        logger.debug("initialized_local_transformer_client")

    def _load_config(self) -> None:
        """Load configuration from file."""
        config = load_yaml_config(self.config_path, ConfigSchema, logger)

        # Extract validated data
        self._global_config = {"default_params": config.default_params}
        self._service_config = {
            k: v.model_dump(exclude_none=True) for k, v in config.services.items()
        }
        self._process_model_mappings(config.models)

        # Load environment variables from .env file if it exists
        if self.env_file_path.exists():
            load_dotenv(self.env_file_path)
            logger.debug("loaded_transformer_env_file", path=str(self.env_file_path))

    def _process_model_mappings(self, models_config: list[ModelConfig]) -> None:
        """Process model mappings from validated configuration.

        Args:
            models_config: List of validated model configurations
        """
        self._model_mappings = {}

        for model_config in models_config:
            model_data = model_config.model_dump(exclude_none=True)
            model_id = model_data["id"]
            service = model_data["service"]

            # Store mapping info
            self._model_mappings[model_id] = {
                "service_provider": service["provider"],
                "service_model_id": service["model_id"],
                "api_format": model_data.get("api_format"),
                "default_params": model_data["default_params"],
                "override_params": model_data["override_params"],
                "remove_params": model_data["remove_params"],
                "rename_params": model_data["rename_params"],
            }

    async def get_params(
        self, model_id: str, input_params: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Get transformed parameters for the specified model.

        Parameter precedence and transformation order:

        Defaults (applied only if key doesn't exist yet):
        1. Model-specific defaults (highest precedence for defaults)
        2. Pattern defaults (matching patterns applied in config order)
        3. Service provider defaults
        4. Global defaults (lowest precedence for defaults)

        Overrides/Removals/Renames (applied in order, later steps overwrite/modify earlier ones):
        1. Input parameters (initial state)
        2. Service-level API keys and base URLs
        3. Service-level overrides, removals, renames
        4. Pattern-level overrides, removals, renames (matching patterns applied in config order)
        5. Model-specific overrides, removals, renames (highest precedence)
        6. Model ID (always overwritten with service_model_id)
        7. Environment variable substitution (applied last to all string values)

        Args:
            model_id: The ID of the model to get transformed parameters for
            input_params: The parameters to transform

        Returns:
            A tuple containing (api_format, transformed_parameters)

        Raises:
            ValueError: If the model is not found in the configuration.
        """
        if model_id not in self._model_mappings:
            logger.error("model_not_found", model_id=model_id)
            raise ValueError(f"Model '{model_id}' not found in configuration")

        mapping = self._model_mappings[model_id]
        service_provider = mapping["service_provider"]
        service_model_id = mapping["service_model_id"]
        service_config = self._service_config.get(service_provider, {})

        transformer = ParamTransformer()

        # Collect pattern configurations
        matching_pattern_defaults = []
        matching_pattern_overrides = []
        matching_pattern_removals = []
        matching_pattern_renames = []

        model_patterns = service_config.get("model_patterns", {})
        for pattern, pattern_config in model_patterns.items():
            if pattern in model_id:
                if "default_params" in pattern_config:
                    matching_pattern_defaults.append(pattern_config["default_params"])
                if "override_params" in pattern_config:
                    matching_pattern_overrides.append(pattern_config["override_params"])
                if "remove_params" in pattern_config:
                    matching_pattern_removals.extend(pattern_config["remove_params"])
                if "rename_params" in pattern_config:
                    matching_pattern_renames.append(pattern_config["rename_params"])

        # Apply defaults (highest to lowest precedence)
        transformer.with_defaults(mapping.get("default_params", {}))
        for defaults in reversed(matching_pattern_defaults):
            transformer.with_defaults(defaults)
        transformer.with_defaults(service_config.get("default_params", {}))
        transformer.with_defaults(self._global_config.get("default_params", {}))

        # Apply overrides, removals, renames (service -> pattern -> model precedence)

        # Service level transformations
        api_key_val = service_config.get("api_key")
        if (
            api_key_val
            and isinstance(api_key_val, str)
            and api_key_val.startswith("${")
            and api_key_val.endswith("}")
        ):
            env_var = api_key_val[2:-1]
            env_val = os.environ.get(env_var)
            if env_val is not None:
                transformer.with_overrides({"api_key": SecretStr(env_val)})
            else:
                logger.warning(
                    "api_key_env_var_not_found",
                    variable=env_var,
                    service=service_provider,
                )
        elif api_key_val:
            # Handle non-env var API keys if needed (though config should use env vars)
            await logger.awarning(
                "api_key_set_directly_in_config",
                service=service_provider,
                msg="API key set directly in config, not recommended",
            )
            transformer.with_overrides({"api_key": api_key_val})

        base_url_val = service_config.get("base_url")
        if base_url_val and isinstance(base_url_val, str):
            transformer.with_overrides({"base_url": base_url_val})

        # Service overrides, removals, renames
        service_override_params = service_config.get("override_params", {})
        if isinstance(service_override_params, dict):
            transformer.with_overrides(service_override_params)
        service_remove_params = service_config.get("remove_params", [])
        if isinstance(service_remove_params, list):
            transformer.removing(service_remove_params)
        service_rename_params = service_config.get("rename_params", {})
        if isinstance(service_rename_params, dict):
            transformer.renaming(service_rename_params)

        # Pattern level transformations (apply collected configs)
        for overrides in matching_pattern_overrides:
            transformer.with_overrides(overrides)
        if matching_pattern_removals:
            transformer.removing(matching_pattern_removals)
        for renames in matching_pattern_renames:
            transformer.renaming(renames)

        # Model level transformations (highest precedence for overrides/removals/renames)
        transformer.with_overrides(mapping.get("override_params", {}))
        transformer.removing(mapping.get("remove_params", []))
        transformer.renaming(mapping.get("rename_params", {}))

        # Final transformations
        # Set the final model ID to be sent to the service provider
        transformer.with_model_id(service_model_id)

        # Substitute environment variables in string values
        transformer.with_env_vars()

        # Resolve API format based on fallback hierarchy: model.api_format → service.api_format → service.provider
        api_format = (
            mapping.get("api_format")
            or service_config.get("api_format")
            or service_provider
        )

        # Execute Transformation
        # Applies all chained transformations in order
        transformed_params = transformer.transform(input_params)

        return api_format, transformed_params
