"""Parameter transformation for LLM API calls."""

import os
from collections.abc import Callable
from typing import Any

from langgate.core.logging import get_logger

logger = get_logger(__name__)


class ParamTransformer:
    """
    Handle LLM API call parameter transformations with a fluent interface.

    This transformer allows for a series of transformations to be applied to
    a set of parameters in a specific order, including:
    - Adding default values
    - Applying overrides
    - Removing parameters
    - Renaming parameters
    - Setting the model ID
    - Substituting environment variables
    """

    def __init__(self, initial_params: dict[str, Any] | None = None):
        self.params = initial_params or {}
        self.transforms: list[Callable[[dict[str, Any]], dict[str, Any]]] = []

    def with_defaults(self, defaults: dict[str, Any]) -> "ParamTransformer":
        """Add default parameters that will be used if not present in input.

        Args:
            defaults: Dictionary of default parameters

        Returns:
            Self for method chaining
        """

        def apply_defaults(params: dict[str, Any]) -> dict[str, Any]:
            result = params.copy()
            for key, value in defaults.items():
                if key not in result:
                    result[key] = value
            return result

        self.transforms.append(apply_defaults)
        return self

    def with_overrides(self, overrides: dict[str, Any]) -> "ParamTransformer":
        """Add overrides that will replace any existing parameters."""

        def apply_overrides(params: dict[str, Any]) -> dict[str, Any]:
            result = params.copy()
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    # Deep merge for nested dicts
                    result[key] = {**result[key], **value}
                else:
                    result[key] = value
            return result

        self.transforms.append(apply_overrides)
        return self

    def removing(self, keys: list[str]) -> "ParamTransformer":
        """Remove specified parameters."""

        def apply_removals(params: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in params.items() if k not in keys}

        self.transforms.append(apply_removals)
        return self

    def renaming(self, key_map: dict[str, str]) -> "ParamTransformer":
        """Rename parameters according to the key_map."""

        def apply_renames(params: dict[str, Any]) -> dict[str, Any]:
            result = params.copy()
            for old_key, new_key in key_map.items():
                if old_key in result:
                    result[new_key] = result.pop(old_key)
            return result

        self.transforms.append(apply_renames)
        return self

    def with_model_id(self, model_id: str) -> "ParamTransformer":
        """Set the model parameter."""
        return self.with_overrides({"model": model_id})

    def with_env_vars(self) -> "ParamTransformer":
        """Substitute environment variables in string values."""

        def apply_env_vars(params: dict[str, Any]) -> dict[str, Any]:
            def substitute(value):
                if (
                    isinstance(value, str)
                    and value.startswith("${")
                    and value.endswith("}")
                ):
                    env_var = value[2:-1]
                    if env_var in os.environ:
                        return os.environ[env_var]
                    logger.warning("env_variable_not_found", variable=env_var)
                    return value
                if isinstance(value, dict):
                    return {k: substitute(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [substitute(item) for item in value]
                return value

            return {k: substitute(v) for k, v in params.items()}

        self.transforms.append(apply_env_vars)
        return self

    def transform(self, input_params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply all transformations to input parameters."""
        result = (input_params or {}).copy()

        for transform in self.transforms:
            result = transform(result)

        return result
