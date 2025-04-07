import importlib.resources
import os
import secrets
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from langgate.core.fields import HttpUrlStr
from langgate.core.logging import get_logger
from langgate.core.utils.config_utils import resolve_path

logger = get_logger(__name__)


class FixedYamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Settings source that loads from app_config namespace in langgate_config.yaml in project root."""

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)
        cwd = Path.cwd()
        # Config path: env > cwd > package_dir
        core_resources = importlib.resources.files("langgate.core")
        default_config_path = Path(
            str(core_resources.joinpath("data", "default_config.yaml"))
        )
        cwd_config_path = cwd / "langgate_config.yaml"

        self.config_path = resolve_path(
            "LANGGATE_CONFIG",
            None,
            cwd_config_path if cwd_config_path.exists() else default_config_path,
            "server_config_path",
            logger,
        )

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        """Get field value from app_config namespace in langgate_config.yaml file.

        This is required by the base class but not used in our implementation.
        We use __call__ instead to load all settings at once.
        """
        raise NotImplementedError

    def prepare_field_value(
        self, field_name: str, field: FieldInfo, value: Any, value_is_complex: bool
    ) -> Any:
        return value

    def __call__(self) -> dict[str, Any]:
        """Load settings from app_config namespace in langgate_config.yaml file."""
        if not self.config_path.exists():
            logger.warning(event="config_file_not_found", path=str(self.config_path))
            return {}

        try:
            with open(self.config_path) as f:
                config_data = yaml.safe_load(f)
                # Only return the app_config namespace
                app_config = config_data.get("app_config", {})
                if not app_config:
                    logger.warning(
                        event="no_app_config_in_config_file",
                        path=str(self.config_path),
                        help="Add app_config namespace to yaml config file to override default settings",
                    )
                logger.info("loaded_app_config", settings=app_config)
                return app_config
        except Exception:
            logger.exception(event="app_config_load_error", path=str(self.config_path))
            raise


def _get_api_env_file_path() -> Path | None:
    cwd = Path.cwd()
    cwd_env_path = cwd / ".env"
    env_file_path = resolve_path(
        "LANGGATE_ENV_FILE", None, cwd_env_path, "server_env_file", logger
    )
    return env_file_path if env_file_path.exists() else None


class ApiSettings(BaseSettings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    PROJECT_NAME: str = "LangGate"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: Literal["debug", "info", "warning", "error", "critical"] = "info"
    # Ensure this is set and fixed across instances in production
    SECRET_KEY: SecretStr = SecretStr(secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    HTTPS: bool
    CORS_ORIGINS: list[HttpUrlStr | Literal["*"]]
    TEST_SERVER_HOST: str = "test"
    JSON_LOGS: bool = False
    IS_TEST: bool = False

    @field_validator("LOG_LEVEL", mode="before")
    @classmethod
    def validate_log_level(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str):
            return v.lower()
        raise ValueError(f"Invalid log level {v}")

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: str | list[str]) -> list[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        if isinstance(v, list | str):
            return v
        raise ValueError(v)

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=_get_api_env_file_path(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        return (
            init_settings,  # Highest priority: constructor arguments
            env_settings,  # Second priority: environment variables
            dotenv_settings,  # Third priority: .env file
            file_secret_settings,  # Fourth priority: secrets from files
            # Lowest priority: YAML config file
            FixedYamlConfigSettingsSource(settings_cls),
        )

    def get_namespace(self, prefix: str, remove_prefix: bool = False) -> dict[str, Any]:
        namespace = {}
        for key, value in dict(self).items():
            if key.startswith(prefix):
                if remove_prefix:
                    namespace[key[len(prefix) :]] = value
                else:
                    namespace[key] = value
        return namespace


class TestSettings(ApiSettings):
    def __init__(self, *args, **kwargs):
        logger.info("using_test_settings")
        BaseSettings.__init__(self, *args, **kwargs)


settings = ApiSettings() if not os.getenv("IS_TEST") else TestSettings()
