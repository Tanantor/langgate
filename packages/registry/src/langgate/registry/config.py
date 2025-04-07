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
        default_models_path = Path(
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
            cwd_models_path if cwd_models_path.exists() else default_models_path,
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

        # Load configuration
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from files."""
        try:
            # Load model data
            self._load_model_data()

            # Load main configuration
            self._load_main_config()

        except Exception:
            logger.exception(
                "failed_to_load_config",
                models_data_path=str(self.models_data_path),
                config_path=str(self.config_path),
            )
            raise

    def _load_model_data(self) -> None:
        """Load model data from JSON file."""
        try:
            with open(self.models_data_path) as f:
                self.models_data = json.load(f)
            logger.info(
                "loaded_model_data",
                models_data_path=str(self.models_data_path),
                model_count=len(self.models_data),
            )
        except FileNotFoundError:
            logger.warning(
                "model_data_file_not_found",
                models_data_path=str(self.models_data_path),
            )
            self.models_data = {}

    def _load_main_config(self) -> None:
        """Load main configuration from YAML file."""
        config = load_yaml_config(self.config_path, ConfigSchema, logger)

        if config:
            # Extract validated data
            self.global_config = {
                "default_params": config.default_params,
            }

            # Extract service provider config
            self.service_config = {
                k: v.model_dump(exclude_none=True) for k, v in config.services.items()
            }

            # Process model mappings
            self._process_model_mappings(config.models)
        else:
            self._set_empty_config()

    def _set_empty_config(self) -> None:
        """Set empty/default config state, typically used in error scenarios."""
        self.global_config = {"default_params": {}}
        self.service_config = {}
        self.model_mappings = {}

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
