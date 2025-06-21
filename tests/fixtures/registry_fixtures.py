"""Registry API fixtures."""

import json
import os
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest
import yaml

from langgate.core.models import LLMInfo
from langgate.registry.local import LocalRegistryClient
from langgate.registry.models import ModelRegistry
from tests.factories.models import LLMInfoFactory
from tests.mocks.registry_mocks import CustomLocalRegistryClient


@pytest.fixture
def mock_models_json(tmp_path: Path) -> Generator[Path]:
    """Create a mock langgate_models.json file for testing."""
    models_data = {
        "openai/gpt-4o": {
            "name": "GPT-4o",
            "mode": "chat",
            "service_provider": "openai",
            "model_provider": "openai",
            "model_provider_name": "OpenAI",
            "capabilities": {
                "supports_tools": True,
                "supports_parallel_tool_calls": True,
                "supports_vision": True,
                "supports_prompt_caching": True,
                "supports_response_schema": True,
                "supports_system_messages": True,
                "supports_tool_choice": True,
            },
            "context": {"max_input_tokens": 128000, "max_output_tokens": 16384},
            "costs": {
                "input_cost_per_token": "0.0000025",
                "output_cost_per_token": "0.00001",
                "input_cost_per_token_batches": "0.00000125",
                "output_cost_per_token_batches": "0.000005",
                "cache_read_input_token_cost": "0.00000125",
                "input_cost_per_image": "0.003613",
            },
            "description": "OpenAI's GP model",
            "_last_updated": "2025-03-21T21:40:54.742453+00:00",
            "_data_source": "openrouter",
            "_last_updated_from_id": "openai/gpt-4o",
        },
        "anthropic/claude-sonnet-4-0": {
            "name": "Claude-4 Sonnet",
            "mode": "chat",
            "service_provider": "anthropic",
            "model_provider": "anthropic",
            "model_provider_name": "Anthropic",
            "capabilities": {
                "supports_tools": True,
                "supports_vision": True,
                "supports_prompt_caching": True,
                "supports_response_schema": True,
                "supports_assistant_prefill": True,
                "supports_tool_choice": True,
            },
            "context": {"max_input_tokens": 200000, "max_output_tokens": 128000},
            "costs": {
                "input_cost_per_token": "0.000003",
                "output_cost_per_token": "0.000015",
                "cache_read_input_token_cost": "3E-7",
                "cache_creation_input_token_cost": "0.00000375",
                "input_cost_per_image": "0.0048",
            },
            "description": "Anthropic's Claude 4 Sonnet model",
            "openrouter_model_id": "anthropic/claude-4-sonnet",
            "_last_updated": "2025-03-21T21:40:54.743326+00:00",
            "_data_source": "openrouter",
            "_last_updated_from_id": "anthropic/claude-4-sonnet",
        },
    }
    models_json_path = tmp_path / "langgate_models.json"
    with open(models_json_path, "w") as f:
        json.dump(models_data, f)

    yield models_json_path


@pytest.fixture
def mock_config_yaml(tmp_path: Path) -> Generator[Path]:
    """Create a mock langgate_config.yaml file for testing."""
    config_data = {
        "app_config": {
            "PROJECT_NAME": "LangGate Test",
            "CORS_ORIGINS": ["http://localhost"],
            "HTTPS": False,
        },
        "default_params": {
            "temperature": 0.7,
        },
        "services": {
            "openai": {
                "api_key": "${OPENAI_API_KEY}",
                "base_url": "https://api.openai.com/v1",
                "default_params": {
                    "max_tokens": 1000,
                },
            },
            "anthropic": {
                "api_key": "${ANTHROPIC_API_KEY}",
                "base_url": "https://api.anthropic.com",
                "default_params": {
                    "max_tokens": 2000,
                },
                "model_patterns": {
                    "reasoning": {
                        "default_params": {"max_tokens": 64000},
                        "override_params": {
                            "thinking": {
                                "type": "enabled",
                            },
                        },
                        "remove_params": ["temperature"],
                        "rename_params": {"reasoning": "thinking"},
                    }
                },
            },
            "openrouter": {
                "api_key": "${OPENROUTER_API_KEY}",
                "base_url": "https://api.openrouter.ai/v1",
                "api_format": "openai",
                "default_params": {
                    "tiktoken_model_name": "gpt-4o",
                },
            },
            "xai": {
                "api_key": "${XAI_API_KEY}",
                "base_url": "https://api.x.ai/v1",
                "api_format": "openai",
            },
            "fireworks_ai": {
                "api_key": "${FIREWORKS_API_KEY}",
                "base_url": "https://api.fireworks.ai/inference/v1",
                "api_format": "openai",
                "default_params": {
                    "tiktoken_model_name": "gpt-4o",
                },
            },
            "gemini": {
                "api_key": "${GEMINI_API_KEY}",
            },
            "mistralai": {
                "api_key": "${MISTRAL_API_KEY}",
                "base_url": "https://api.mistral.ai/v1",
            },
        },
        "models": [
            {
                "id": "gpt-4o",
                "name": "GPT-4o",
                "service": {
                    "provider": "openai",
                    "model_id": "gpt-4o",
                },
            },
            {
                "id": "anthropic/claude-sonnet-4",
                "name": "Claude-3.7 Sonnet",
                "service": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4",
                },
                "remove_params": ["response_format", "reasoning"],
                "rename_params": {"stop": "stop_sequences"},
            },
            {
                "id": "anthropic/claude-sonnet-4-reasoning",
                "name": "Claude-3.7 Sonnet R",
                "description": "Claude 3.7 Sonnet with reasoning",
                "service": {
                    "provider": "anthropic",
                    "model_id": "claude-sonnet-4",
                },
                "remove_params": ["response_format"],
                "rename_params": {"stop": "stop_sequences"},
                "override_params": {
                    "thinking": {
                        "budget_tokens": 1024,
                    }
                },
            },
            {
                "id": "google/gemma-3-27b-it",
                "service": {
                    "provider": "openrouter",
                    "model_id": "google/gemma-3-27b-it:free",
                },
            },
            {
                "id": "xai/grok-3",
                "service": {
                    "provider": "xai",
                    "model_id": "grok-3-latest",
                },
            },
            {
                "id": "deepseek/deepseek-r1",
                "service": {
                    "provider": "fireworks_ai",
                    "model_id": "accounts/fireworks/models/deepseek-r1",
                },
            },
            {
                "id": "google/gemini-2.5-pro",
                "service": {
                    "provider": "gemini",
                    "model_id": "gemini-2.5-pro-preview",
                },
            },
            {
                "id": "mistralai/magistral-medium-latest",
                "service": {
                    "provider": "mistralai",
                    "model_id": "magistral-medium-latest",
                },
            },
        ],
    }
    config_yaml_path = tmp_path / "langgate_config.yaml"
    with open(config_yaml_path, "w") as f:
        yaml.dump(config_data, f)

    yield config_yaml_path


@pytest.fixture
def mock_env_file(tmp_path: Path) -> Generator[Path]:
    """Create a mock .env file for testing."""
    env_path = tmp_path / ".env"
    with open(env_path, "w") as f:
        f.write("OPENAI_API_KEY=sk-test-123\n")
        f.write("ANTHROPIC_API_KEY=sk-ant-test-123\n")
        f.write("OPENROUTER_API_KEY=sk-or-test-123\n")
        f.write("XAI_API_KEY=xai-test-123\n")
        f.write("FIREWORKS_API_KEY=fw-test-123\n")
        f.write("GEMINI_API_KEY=gm-test-123\n")
        f.write("MISTRAL_API_KEY=ms-test-123\n")
        f.write("SECRET_KEY=test-secret-key\n")

    yield env_path


@pytest.fixture
def mock_registry_files(
    mock_models_json: Path, mock_config_yaml: Path, mock_env_file: Path
) -> Generator[dict[str, Path]]:
    """Combine all mock registry files for testing."""
    result_dict = {
        "models_json": mock_models_json,
        "config_yaml": mock_config_yaml,
        "env_file": mock_env_file,
    }
    yield result_dict


@pytest.fixture
def mock_models_path_in_env(mock_models_json: Path) -> Generator[dict[str, str]]:
    """Mock only the langgate_models.json environment variable."""
    with mock.patch.dict(os.environ, {"LANGGATE_MODELS": str(mock_models_json)}):
        yield {"LANGGATE_MODELS": str(mock_models_json)}


@pytest.fixture
def mock_config_path_in_env(mock_config_yaml: Path) -> Generator[dict[str, str]]:
    """Mock only the langgate_config.yaml environment variable."""
    with mock.patch.dict(os.environ, {"LANGGATE_CONFIG": str(mock_config_yaml)}):
        yield {"LANGGATE_CONFIG": str(mock_config_yaml)}


@pytest.fixture
def mock_env_file_path_in_env(mock_env_file: Path) -> Generator[dict[str, str]]:
    """Mock only the .env file environment variable."""
    with mock.patch.dict(os.environ, {"LANGGATE_ENV_FILE": str(mock_env_file)}):
        yield {"LANGGATE_ENV_FILE": str(mock_env_file)}


@pytest.fixture
def mock_env_vars_from_env_file(mock_env_file: Path) -> Generator[dict[str, str]]:
    """Mock environment variables loaded from the .env file."""
    env_vars = {}
    with open(mock_env_file) as f:
        for _line in f:
            line = _line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                env_vars[key] = value

    with mock.patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_all_env_vars(
    mock_models_json: Path,
    mock_config_yaml: Path,
    mock_env_file: Path,
    mock_env_vars_from_env_file: dict[str, str],
) -> Generator[dict[str, str]]:
    """Mock all registry-related environment variables."""
    env_vars = mock_env_vars_from_env_file.copy()
    env_vars.update(
        {
            "LANGGATE_MODELS": str(mock_models_json),
            "LANGGATE_CONFIG": str(mock_config_yaml),
            "LANGGATE_ENV_FILE": str(mock_env_file),
        }
    )

    with mock.patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def model_registry(mock_all_env_vars: dict[str, str]) -> Generator[ModelRegistry]:
    """Create a ModelRegistry instance with mock files."""
    ModelRegistry._instance = None

    # Create a fresh registry instance
    registry = ModelRegistry()
    yield registry
    ModelRegistry._instance = None


@pytest.fixture
def mock_llm_info() -> LLMInfo:
    """Create a mock LLMInfo instance."""
    return LLMInfoFactory.create()


@pytest.fixture
def local_registry_client() -> Generator[LocalRegistryClient]:
    """Return a LocalRegistryClient instance."""
    original_instance = LocalRegistryClient._instance
    LocalRegistryClient._instance = None

    client = LocalRegistryClient()

    yield client

    LocalRegistryClient._instance = original_instance


@pytest.fixture
def custom_local_registry_client() -> Generator[CustomLocalRegistryClient]:
    """Return a CustomLocalRegistryClient instance."""
    client = CustomLocalRegistryClient()

    yield client


@pytest.fixture
def default_models_data() -> dict:
    """Sample default models data for merge testing."""
    return {
        "openai/gpt-4o": {
            "name": "GPT-4o Default",
            "service_provider": "openai",
            "model_provider": "openai",
            "description": "Default GPT-4o model",
        },
        "anthropic/claude-3-sonnet": {
            "name": "Claude 3 Sonnet Default",
            "service_provider": "anthropic",
            "model_provider": "anthropic",
            "description": "Default Claude 3 Sonnet model",
        },
    }


@pytest.fixture
def user_models_data() -> dict:
    """Sample user models data for merge testing."""
    return {
        "openai/gpt-4o": {
            "name": "GPT-4o Custom",
            "service_provider": "openai",
            "model_provider": "openai",
            "description": "Custom GPT-4o configuration",
        },
        "custom/my-model": {
            "name": "My Custom Model",
            "service_provider": "custom",
            "model_provider": "custom",
            "description": "My custom model",
        },
    }


@pytest.fixture
def base_config_data() -> dict:
    """Base configuration data for merge testing."""
    return {
        "default_params": {},
        "services": {
            "openai": {
                "api_key": "test-key",
                "base_url": "https://api.openai.com/v1",
            },
            "anthropic": {
                "api_key": "test-key",
                "base_url": "https://api.anthropic.com",
            },
            "custom": {"api_key": "test-key", "base_url": "https://api.custom.com"},
        },
        "models": [],
        "app_config": {},
    }


@pytest.fixture
def merge_config_data(base_config_data: dict) -> dict:
    """Configuration data with merge mode set."""
    config_data = base_config_data.copy()
    config_data["models_merge_mode"] = "merge"
    return config_data


@pytest.fixture
def replace_config_data(base_config_data: dict) -> dict:
    """Configuration data with replace mode set."""
    config_data = base_config_data.copy()
    config_data["models_merge_mode"] = "replace"
    return config_data


@pytest.fixture
def extend_config_data(base_config_data: dict) -> dict:
    """Configuration data with extend mode set."""
    config_data = base_config_data.copy()
    config_data["models_merge_mode"] = "extend"
    return config_data


@pytest.fixture
def invalid_config_data(base_config_data: dict) -> dict:
    """Configuration data with invalid merge mode."""
    config_data = base_config_data.copy()
    config_data["models_merge_mode"] = "invalid"
    return config_data


@pytest.fixture
def user_models_no_conflicts() -> dict:
    """User models data with no conflicts for extend mode testing."""
    return {
        "custom/my-model": {
            "name": "My Custom Model",
            "service_provider": "custom",
            "model_provider": "custom",
            "description": "My custom model",
        }
    }


@pytest.fixture
def mock_default_models_file(tmp_path: Path, default_models_data: dict) -> Path:
    """Create a temporary default models JSON file."""
    default_models_file = tmp_path / "default_models.json"
    with open(default_models_file, "w") as f:
        json.dump(default_models_data, f)
    return default_models_file


@pytest.fixture
def mock_user_models_file(tmp_path: Path, user_models_data: dict) -> Path:
    """Create a temporary user models JSON file."""
    user_models_file = tmp_path / "user_models.json"
    with open(user_models_file, "w") as f:
        json.dump(user_models_data, f)
    return user_models_file


@pytest.fixture
def mock_config_file(tmp_path: Path, base_config_data: dict) -> Path:
    """Create a temporary config YAML file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(base_config_data, f)
    return config_file
