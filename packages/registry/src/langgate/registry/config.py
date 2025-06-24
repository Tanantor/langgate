"""Configuration handling for the registry."""

import importlib.resources
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from langgate.core.logging import get_logger
from langgate.core.schemas.config import ConfigSchema
from langgate.core.utils.config_utils import load_yaml_config, resolve_path

logger = get_logger(__name__)


class RegistryConfig:
    """Configuration handler for the registry."""

    def __init__(
        self,
        models_data_path: Path | None = None,
        config_path: Path | None = None,
        env_file_path: Path | None = None,
    ):
        """
        Args:
            models_data_path: Path to the models data JSON file
            config_path: Path to the main configuration YAML file
            env_file_path: Path to a `.env` file for environment variables
        """
        # Set up default paths
        cwd = Path.cwd()
        # Get package resource paths
        registry_resources = importlib.resources.files("langgate.registry")
        core_resources = importlib.resources.files("langgate.core")
        self.default_models_path = Path(
            str(registry_resources.joinpath("data", "default_models.json"))
        )
        default_config_path = Path(
            str(core_resources.joinpath("data", "default_config.yaml"))
        )

        # Define default paths with priorities
        # Models data: args > env > cwd > package_dir
        cwd_models_path = cwd / "langgate_models.json"

        # Config: args > env > cwd > package_dir
        cwd_config_path = cwd / "langgate_config.yaml"

        # Env file: args > env > cwd
        cwd_env_path = cwd / ".env"

        # Resolve paths using priority order
        self.models_data_path = resolve_path(
            "LANGGATE_MODELS",
            models_data_path,
            cwd_models_path if cwd_models_path.exists() else self.default_models_path,
            "models_data_path",
        )

        self.config_path = resolve_path(
            "LANGGATE_CONFIG",
            config_path,
            cwd_config_path if cwd_config_path.exists() else default_config_path,
            "config_path",
        )

        self.env_file_path = resolve_path(
            "LANGGATE_ENV_FILE", env_file_path, cwd_env_path, "env_file_path"
        )

        # Load environment variables from .env file if it exists
        if self.env_file_path.exists():
            load_dotenv(self.env_file_path)
            logger.debug("loaded_env_file", path=str(self.env_file_path))

        # Initialize data structures
        self.models_data: dict[str, dict[str, Any]] = {}
        self.global_config: dict[str, Any] = {}
        self.service_config: dict[str, dict[str, Any]] = {}
        self.model_mappings: dict[str, dict[str, Any]] = {}
        self.models_merge_mode: str = "merge"  # Default value

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from files."""
        try:
            # Load main configuration first to get merge mode
            self._load_main_config()

            # Load model data with merge mode available
            self._load_model_data()

        except Exception:
            logger.exception(
                "failed_to_load_config",
                models_data_path=str(self.models_data_path),
                config_path=str(self.config_path),
            )
            raise

    def _has_user_models(self) -> bool:
        """Check if user has specified custom models."""
        return (
            self.models_data_path != self.default_models_path
            and self.models_data_path.exists()
        )

    def _load_json_file(self, path: Path) -> dict[str, dict[str, Any]]:
        """Load and return JSON data from file."""
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                "model_data_file_not_found",
                models_data_path=str(path),
            )
            return {}

    def _merge_models(
        self,
        default_models: dict[str, dict[str, Any]],
        user_models: dict[str, dict[str, Any]],
        merge_mode: str,
    ) -> dict[str, dict[str, Any]]:
        """Merge user models with default models based on merge mode."""
        if merge_mode == "merge":
            # User models override defaults, new models are added
            merged = default_models.copy()
            for model_id, model_config in user_models.items():
                if model_id in merged:
                    logger.debug(
                        "overriding_default_model",
                        model_id=model_id,
                        original_name=merged[model_id].get("name"),
                        new_name=model_config.get("name"),
                    )
                merged[model_id] = model_config
            return merged

        if merge_mode == "extend":
            # User models are added, conflicts with defaults cause errors
            conflicts = set(default_models.keys()) & set(user_models.keys())
            if conflicts:
                raise ValueError(
                    f"Model ID conflicts found in extend mode: {', '.join(conflicts)}. "
                    f"Use 'merge' mode to allow overrides or rename conflicting models."
                )
            return {**default_models, **user_models}

        raise ValueError(f"Unknown merge mode: {merge_mode}")

    def _load_model_data(self) -> None:
        """Load model data from JSON file(s) based on merge mode."""
        try:
            # Always load default models first
            default_models = self._load_json_file(self.default_models_path)

            has_user_models = self._has_user_models()

            if self.models_merge_mode == "replace" or not has_user_models:
                # Use only user models or default if no user models
                if has_user_models:
                    self.models_data = self._load_json_file(self.models_data_path)
                else:
                    self.models_data = default_models
            else:
                # merge user models with defaults
                user_models = (
                    self._load_json_file(self.models_data_path)
                    if has_user_models
                    else {}
                )
                self.models_data = self._merge_models(
                    default_models, user_models, self.models_merge_mode
                )

            logger.info(
                "loaded_model_data",
                models_data_path=str(self.models_data_path),
                merge_mode=self.models_merge_mode,
                model_count=len(self.models_data),
            )
        except Exception:
            logger.exception("failed_to_load_model_data")
            raise

    def _load_main_config(self) -> None:
        """Load main configuration from YAML file."""
        config = load_yaml_config(self.config_path, ConfigSchema, logger)

        # Extract merge mode
        self.models_merge_mode = config.models_merge_mode

        # Extract validated data
        self.global_config = {
            "default_params": config.default_params,
            "models_merge_mode": config.models_merge_mode,
        }

        # Extract service provider config
        self.service_config = {
            k: v.model_dump(exclude_none=True) for k, v in config.services.items()
        }

        # Process model mappings
        self._process_model_mappings(config.models)

    def _process_model_mappings(self, models_config) -> None:
        """Process model mappings from validated configuration.

        Args:
            models_config: List of validated model configurations
        """
        self.model_mappings = {}

        for model_config in models_config:
            model_data = model_config.model_dump(exclude_none=True)
            model_id = model_data["id"]
            service = model_data["service"]

            # Store mapping info with proper type handling
            self.model_mappings[model_id] = {
                "service_provider": service["provider"],
                "service_model_id": service["model_id"],
                "override_params": model_data.get("override_params", {}),
                "remove_params": model_data.get("remove_params", []),
                "rename_params": model_data.get("rename_params", {}),
                "name": model_data.get("name"),
                "description": model_data.get("description"),
            }
