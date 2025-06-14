"""Unit tests for default_config.yaml validation."""

from pathlib import Path

import pytest
import yaml

from langgate.core.schemas.config import ConfigSchema
from langgate.core.utils.config_utils import load_yaml_config


class TestDefaultConfig:
    """Test suite for default_config.yaml file validation."""

    @pytest.fixture
    def default_config_path(self) -> Path:
        """Get path to the default config file."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "packages"
            / "core"
            / "src"
            / "langgate"
            / "core"
            / "data"
            / "default_config.yaml"
        )

    def test_default_config_exists(self, default_config_path: Path) -> None:
        """Test that the default config file exists."""
        assert default_config_path.exists(), (
            f"Default config file not found at {default_config_path}"
        )

    def test_default_config_is_valid_yaml(self, default_config_path: Path) -> None:
        """Test that the default config file is valid YAML."""
        with open(default_config_path) as f:
            content = yaml.safe_load(f)

        assert content is not None, "Config file should not be empty"
        assert isinstance(content, dict), "Config file should contain a dictionary"

    def test_default_config_loads_with_schema(self, default_config_path: Path) -> None:
        """Test that the default config loads successfully with the ConfigSchema."""
        config = load_yaml_config(default_config_path, ConfigSchema)

        assert config is not None, "Config should load successfully"
        assert isinstance(config, ConfigSchema), (
            "Config should be a ConfigSchema instance"
        )

    def test_default_config_has_required_sections(
        self, default_config_path: Path
    ) -> None:
        """Test that the default config contains expected top-level sections."""
        config = load_yaml_config(default_config_path, ConfigSchema)

        assert config is not None, "Config should load successfully"

        # Check required sections exist
        assert hasattr(config, "default_params"), "Config should have default_params"
        assert hasattr(config, "services"), "Config should have services"
        assert hasattr(config, "models"), "Config should have models"
        assert hasattr(config, "app_config"), "Config should have app_config"

    def test_default_config_has_valid_services(self, default_config_path: Path) -> None:
        """Test that the default config contains valid service configurations."""
        config = load_yaml_config(default_config_path, ConfigSchema)

        assert config is not None, "Config should load successfully"
        assert len(config.services) > 0, "Config should have at least one service"

        # Check that common services are present
        expected_services = ["openai", "anthropic", "gemini"]
        for service in expected_services:
            assert service in config.services, (
                f"Service '{service}' should be present in config"
            )

    def test_default_config_has_valid_models(self, default_config_path: Path) -> None:
        """Test that the default config contains valid model configurations."""
        config = load_yaml_config(default_config_path, ConfigSchema)

        assert config is not None, "Config should load successfully"
        assert len(config.models) > 0, "Config should have at least one model"

        # Check that all models have required fields
        for model in config.models:
            assert model.id, "Each model should have an id"
            assert model.service, "Each model should have a service configuration"
            assert model.service.provider, "Each model service should have a provider"
            assert model.service.model_id, "Each model service should have a model_id"

    def test_default_config_app_config_section(self, default_config_path: Path) -> None:
        """Test that the app_config section is valid."""
        config = load_yaml_config(default_config_path, ConfigSchema)

        assert config is not None, "Config should load successfully"
        assert config.app_config is not None, "Config should have app_config section"
        assert isinstance(config.app_config, dict), "app_config should be a dictionary"
