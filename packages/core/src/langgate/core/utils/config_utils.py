"""Configuration utilities for LangGate."""

import os
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from langgate.core.logging import StructLogger

ConfigSchemaT = TypeVar("ConfigSchemaT", bound=BaseModel)


def resolve_path(
    env_var: str,
    arg_path: Path | None = None,
    default_path: Path | None = None,
    path_desc: str = "path",
    logger: StructLogger | None = None,
) -> Path:
    """Resolve a file path based on priority: args > env > default.

    Args:
        env_var: Environment variable name to check
        arg_path: Path provided in constructor args
        default_path: Default path to use if others not provided
        path_desc: Description for logging
        logger: Optional logger instance for recording path resolution

    Returns:
        Resolved Path object
    """
    # Priority: args > env > default
    resolved_path = arg_path or Path(os.getenv(env_var, str(default_path)))

    # Log the resolved path and its existence
    if logger:
        exists = resolved_path.exists()
        logger.debug(
            f"resolved_{path_desc}",
            path=str(resolved_path),
            exists=exists,
            source="args" if arg_path else ("env" if os.getenv(env_var) else "default"),
        )

    return resolved_path


def load_yaml_config(
    config_path: Path,
    schema_class: type[ConfigSchemaT],
    logger: StructLogger | None = None,
) -> ConfigSchemaT:
    """Load and validate a YAML configuration file using a Pydantic schema.

    Args:
        config_path: Path to the YAML configuration file
        schema_class: The Pydantic schema class to validate against
        logger: Optional logger instance for recording validation results

    Returns:
        The validated schema instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is empty
        yaml.YAMLError: If YAML parsing fails
        ValidationError: If schema validation fails
    """
    try:
        if not config_path.exists():
            if logger:
                logger.error(
                    "config_file_not_found",
                    config_path=str(config_path),
                )
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            if logger:
                logger.error("config_file_is_empty", config_path=str(config_path))
            raise ValueError(f"Config file is empty: {config_path}")

        # Validate configuration using Pydantic schema
        try:
            config = schema_class.model_validate(raw_config)
            if logger:
                logger.info(
                    "loaded_config",
                    config_path=str(config_path),
                )
            return config
        except ValidationError as exc:
            if logger:
                logger.exception(
                    "invalid_config_format",
                    config_path=str(config_path),
                    errors=str(exc),
                )
            raise

    except yaml.YAMLError:
        if logger:
            logger.exception(
                "failed_to_parse_yaml_config", config_path=str(config_path)
            )
        raise
    except Exception:
        if logger:
            logger.exception("failed_to_load_config", config_path=str(config_path))
        raise
