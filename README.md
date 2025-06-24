# LangGate AI Gateway
<p align="left">
  <a href="https://pypi.org/project/langgate" target="_blank"><img src="https://img.shields.io/pypi/pyversions/langgate.svg" alt="Python versions"></a> <a href="https://pypi.org/project/langgate" target="_blank"><img src="https://img.shields.io/pypi/v/langgate" alt="PyPI"></a> <a href="https://github.com/Tanantor/langgate/actions?query=workflow%3A%22CI+Checks%22" target="_blank"><img src="https://github.com/Tanantor/langgate/actions/workflows/ci.yaml/badge.svg?event=push&branch=main" alt="CI Checks"></a>
  <a href="https://github.com/Tanantor/langgate/tree/main/tests" target="_blank"><img src="https://img.shields.io/badge/dynamic/xml?url=https://tanantor.github.io/langgate/coverage/test-count.xml&query=//testcount&label=tests&color=blue&style=flat" alt="Tests"></a> <a href="https://github.com/Tanantor/langgate/actions?query=workflow%3ACI" target="_blank"><img src="https://tanantor.github.io/langgate/coverage/coverage-badge.svg" alt="Coverage"></a>
</p>

LangGate is a lightweight, high-performance gateway for AI model inference.

LangGate adapts to your architecture: integrate it as a Python SDK, run it as a standalone registry, or deploy it as a complete proxy server.

LangGate works with any AI provider, without forcing standardization to a specific API format. Apply custom parameter mappings or none at all - you decide.

LangGate by default avoids unnecessary transformation.

## Core Features

- **Provider-Agnostic**: Works with any AI inference provider (OpenAI, Anthropic, Google, etc.)
- **Flexible Parameter Transformations**: Apply custom parameter mappings or none at all - you decide
- **High-Performance Proxying**: Uses Envoy for efficient request handling with direct response streaming
- **Simple Configuration**: Clean YAML configuration inspired by familiar formats
- **Minimal Architecture**: Direct integration with Envoy, without complex control plane overhead
- **SDK First Approach**: Use the registry as a standalone module without the proxy service

## Architecture

LangGate uses a simplified architecture with three main components:

1. **Envoy Proxy**: Front-facing proxy that receives API requests and handles response streaming
2. **External Processor**: gRPC service implementing Envoy's External Processing filter for request transformation and routing
3. **Registry Service**: Manages model mappings, parameter transformations, and provider configurations

The system works as follows:

1. **Request Flow**: Client sends request → Envoy → External Processor transforms request → Envoy routes to appropriate AI provider
2. **Response Flow**: AI provider response → Envoy streams directly to client

This architecture provides several advantages:
- No control plane overhead or complex deployment requirements
- Direct response streaming from providers through Envoy for optimal performance
- Flexible deployment options, from local development to production environments

## Getting Started

### Using the Registry SDK
The LangGate SDK is designed to be used as a standalone module, allowing you to integrate it into your existing applications without the need for the proxy service.
This is particularly useful for local development or when you want to use LangGate's features without deploying the full stack.
You probably won't need the proxy unless scaling your application to a microservice architecture or if you have multiple apps in a Kubernetes cluster that each depend on a registry.
You can switch from the SDK's local registry client to the remote registry client + proxy setup with minimal code changes.
#### Installation
We recommend using [uv](https://docs.astral.sh/uv/) to manage Python projects. In a uv project, add `langgate[sdk]` to dependencies by running:
```bash
uv add langgate[sdk]
```
Alternatively, using pip:

```bash
pip install langgate[sdk]
```

For more information on package components and installation options for specific use cases, see the  [packages documentation](packages/README.md).
#### Example Usage

The package includes a `LangGateLocal` client that can be used directly in your application without needing to run the proxy service. This client provides access to the model registry and parameter transformation features.

To get metadata for a model:

```py
from pprint import pprint as pp

from langgate.sdk import LangGateLocal

client = LangGateLocal()

# LangGate allows us to register "virtual models" - models with specific parameters.
# `langgate_config.yaml` defines this `claude-sonnet-4-reasoning` model
# which is a wrapper around the `claude-sonnet-4-latest` model,
# with specific parameters and metadata.
# In `langgate_config.yaml`, Anthropic is set as the inference service provider,
# but you could configure any backend API that offers the model, e.g. AWS Bedrock.
model_id = "anthropic/claude-sonnet-4-reasoning"

# get metadata for a model
model_info = await client.get_model_info(model_id)

# returns a Pydantic model instance (langgate.core.models.LLMInfo)
pp(model_info.model_dump(exclude_none=True))
```
```py
2025-04-01 19:33:35 [debug    ] creating_model_registry_singleton
2025-04-01 19:33:35 [debug    ] loaded_env_file                path=/your-working-directory/.env
2025-04-01 19:33:35 [info     ] loaded_model_data              model_count=35 models_data_path=/your-working-directory/.venv/lib/python3.13/site-packages/langgate/registry/data/default_models.json
2025-04-01 19:33:35 [info     ] loaded_config                  config_path=/your-working-directory/langgate_config.yaml
2025-04-01 19:33:35 [debug    ] initialized_local_registry_client
2025-04-01 19:33:35 [debug    ] resolved_config_path           exists=True path=/your-working-directory/langgate_config.yaml source=env
2025-04-01 19:33:35 [debug    ] resolved_env_file_path         exists=True path=/your-working-directory/.env source=default
2025-04-01 19:33:35 [info     ] loaded_config                  config_path=/your-working-directory/langgate_config.yaml
2025-04-01 19:33:35 [debug    ] initialized_local_transformer_client
2025-04-01 19:33:35 [debug    ] initialized_langgate_client
2025-04-01 19:33:35 [debug    ] refreshed_model_cache          model_count=34
{'capabilities': {'supports_assistant_prefill': True,
                  'supports_pdf_input': True,
                  'supports_prompt_caching': True,
                  'supports_response_schema': True,
                  'supports_tool_choice': True,
                  'supports_tools': True,
                  'supports_vision': True},
 'context_window': {'max_input_tokens': 200000, 'max_output_tokens': 128000},
 'costs': {'cache_creation_input_token_cost': Decimal('0.00000375'),
           'cache_read_input_token_cost': Decimal('3E-7'),
           'input_cost_per_image': Decimal('0.0048'),
           'input_cost_per_token': Decimal('0.000003'),
           'output_cost_per_token': Decimal('0.000015')},
 'description': 'Claude-3.7 Sonnet with standard reasoning capabilities '
                'optimized for complex problem-solving.',
 'id': 'anthropic/claude-sonnet-4-reasoning',
 'name': 'Claude-3.7 Sonnet R',
 'provider': {'id': 'anthropic', 'name': 'Anthropic'},
 'updated_dt': datetime.datetime(2025, 4, 1, 17, 7, 6, 737543, tzinfo=datetime.timezone.utc)}
```

To get paramaters for a model, as defined in your configuration mapping (or a default mapping we ship with the package):
```py
model_params = await client.get_params(model_id, {"temperature": 0.7, "stream": True})
pp(model_params)
```
```py
{'base_url': 'https://api.anthropic.com',
 'max_tokens': 64000,
 'model': 'claude-sonnet-4-0',
 'stream': True,
 'thinking': {'budget_tokens': 1024, 'type': 'enabled'}}
```
The `temperature` parameter is removed from the request, because `temperature` is not supported by Claude 3.7 Sonnet if reasoning is enabled. The `thinking` parameter is added to the request parameters, along with the `budget_tokens` we specify in `langgate_config.yaml`.


#### Example integration with Langchain:
The following is an example of how you might define a factory class to create a Langchain `BaseChatModel` instance configured via the `LangGateLocal` client:
```py
import os

# Ensure you have the required environment variables set
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

# The below environment variables are optional.

# The yaml config resolution priority is: args > env > cwd > package default.
# If you don't want to use either the package default (langgate/core/data/default_config.yaml)
# or a config in your cwd, set:
# os.environ["LANGGATE_CONFIG"] = "some_other_path_not_in_your_cwd/langgate_config.yaml"

# The models data resolution priority is: args > env > cwd > package default
# By default, any user-defined `langgate_models.json` files are merged with default models data. See `models_merge_mode` configuration.
# If you don't want to use either the package default (langgate/registry/data/default_models.json)
# or a models data file in your cwd, set:
# os.environ["LANGGATE_MODELS"] = "some_other_path_not_in_your_cwd/langgate_models.json"

# The .env file resolution priority is: args > env > cwd > None
# If you don't want to use either the package default or a .env file in your cwd, set:
# os.environ["LANGGATE_ENV_FILE"] = "some_other_path_not_in_your_cwd/.env"
```
```py
from typing import Any
from pprint import pprint as pp

from langchain.chat_models.base import BaseChatModel
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from langgate.sdk import LangGateLocal, LangGateLocalProtocol
from langgate.core.models import (
    # `ModelProviderId` is a string alias for better type safety
    ModelProviderId,
    # ids for common providers are included for convenience
    MODEL_PROVIDER_OPENAI,
    MODEL_PROVIDER_ANTHROPIC,
)

# Map providers to model classes
MODEL_CLASS_MAP: dict[ModelProviderId, type[BaseChatModel]] = {
    MODEL_PROVIDER_OPENAI: ChatOpenAI,
    MODEL_PROVIDER_ANTHROPIC: ChatAnthropic,
}


class ModelFactory:
    """
    Factory for creating a Langchain `BaseChatModel` instance
    with paramaters from LangGate.
    """

    def __init__(self, langgate_client: LangGateLocalProtocol | None = None):
        self.langgate_client = langgate_client or LangGateLocal()

    async def create_model(
        self, model_id: str, input_params: dict[str, Any] | None = None
    ) -> tuple[BaseChatModel, dict[str, Any]]:
        """Create a model instance for the given model ID."""
        params = {"temperature": 0.7, "streaming": True}
        if input_params:
            params.update(input_params)

        # Get model info from the registry cache
        model_info = await self.langgate_client.get_model_info(model_id)

        # Transform parameters using the transformer client
        # If switching to using the proxy, you would remove this line
        # and let the proxy handle the parameter transformation instead.
        api_format, model_params = await self.langgate_client.get_params(
            model_id, params
        )
        # api_format defaults to the provider id unless specified in the config.
        # e.g. Specify "openai" for OpenAI-compatible APIs, etc.
        print("API format:", api_format)
        pp(model_params)

        # Get the appropriate model class based on provider
        client_cls_key = ModelProviderId(api_format)
        model_class = MODEL_CLASS_MAP.get(client_cls_key)
        if not model_class:
            raise ValueError(f"No model class for provider {model_info.provider.id}")

        # Create model instance with parameters
        model = model_class(**model_params)

        # Create model info dict
        model_metadata = model_info.model_dump(exclude_none=True)

        return model, model_metadata
```
```py
model_factory = ModelFactory()
model_id = "openai/gpt-4o"
model = await model_factory.create_model(model_id, {"temperature": 0.7})
model
```
```text
API format: openai
{'api_key': SecretStr('**********'),
 'base_url': 'https://api.openai.com/v1',
 'model': 'gpt-4o',
 'streaming': True,
 'temperature': 0.7}
ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x121f66210>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x121f72210>, root_client=<openai.OpenAI object at 0x121f56210>, root_async_client=<openai.AsyncOpenAI object at 0x121f66350>, model_name='gpt-4o', temperature=0.7, model_kwargs={}, openai_api_key=SecretStr('**********'), openai_api_base='https://api.openai.com/v1', streaming=True)
```
If you want to use the LangGate Envoy proxy instead of `LangGateLocal`,  you can switch to the `HTTPRegistryClient` with minimal code changes.

For more usage patterns and detailed instructions, see  [examples](examples/README.md).

### Envoy Proxy Service (Coming Soon)

The LangGate proxy feature is currently in development. When completed, it will provide:

1. Centralized model registry accessible via API
2. Parameter transformation at the proxy level
3. API key management and request routing
4. High-performance response streaming via Envoy

## Configuration

LangGate uses a simple YAML configuration format:

```yaml
# Global default parameters (applied to all models unless overridden)
default_params:
  temperature: 0.7

# Service provider configurations
services:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
    model_patterns:
      # match any o-series model
      openai/o:
        remove_params:
          - temperature

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
    model_patterns:
      # match any model with reasoning in the id
      reasoning:
        override_params:
          thinking:
            type: enabled
        remove_params:
          - temperature

# Model-specific configurations and parameter overrides
models:
  - id: openai/gpt-4o
    service:
      provider: openai
      model_id: gpt-4o

  - id: openai/o1
    service:
      provider: openai
      model_id: o1

  # "virtual model" that wraps the o1 model with high-effort reasoning
  - id: openai/o1-high
    service:
      provider: openai
      model_id: o1
    name: o1-high
    description: o1-high applies high-effort reasoning for the o1 model
    override_params:
      reasoning_effort: high

  - id: anthropic/claude-sonnet-4
    service:
      provider: anthropic
      model_id: claude-sonnet-4-0

  # "virtual model" that wraps the claude-sonnet-4-0 model with reasoning
  - id: anthropic/claude-sonnet-4-reasoning
    service:
      provider: anthropic
      model_id: claude-sonnet-4-0
    name: Claude-3.7 Sonnet R
    description: "Claude-3.7 Sonnet with reasoning capabilities."
    override_params:
      thinking:
        budget_tokens: 1024

# Models merge mode for loading data from JSON files: "merge" (default), "replace", or "extend"
# - merge: User models override defaults, new models are added
# - replace: Only use user models (ignore defualt models file)
# - extend: Add user models to defaults, error on conflicts
models_merge_mode: merge

```

### Parameter Transformation Precedence

When transforming parameters for model requests, LangGate follows a specific precedence order:

#### Defaults (applied only if key doesn't exist yet):
1. Model-specific defaults (highest precedence for defaults)
2. Pattern defaults (matching patterns applied in config order)
3. Service provider defaults
4. Global defaults (lowest precedence for defaults)

#### Overrides/Removals/Renames (applied in order, later steps overwrite/modify earlier ones):
1. Input parameters (initial state)
2. Service-level API keys and base URLs
3. Service-level overrides, removals, renames
4. Pattern-level overrides, removals, renames (matching patterns applied in config order)
5. Model-specific overrides, removals, renames (highest precedence)
6. Model ID (always overwritten with service_model_id)
7. Environment variable substitution (applied last to all string values)

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| LANGGATE_CONFIG | Path to the main configuration file | ./langgate_config.yaml |
| LANGGATE_MODELS | Path to the models data JSON file | ./langgate_models.json |
| LANGGATE_ENV_FILE | Path to the .env file | ./.env |
| LOG_LEVEL | Logging level | info |

### Models Merge Behavior

You can add additional models to LangGate's model registry by creating a `langgate_models.json` file in your working directory, or by setting the `LANGGATE_MODELS` environment variable to point to a custom JSON file.

LangGate supports three modes for including extra models beyond those we ship with the package (`default_models.json`):
- **merge** (default): Your defined models are merged with default models, with your models taking precedence
- **replace**: Only your models are used
- **extend**: your models are added to defaults, conflicts cause errors

Configure this with `models_merge_mode` in your YAML configuration.

Note:
- If `langgate_models.json` is unset in your working directory, and no `LANGGATE_MODELS` environment variable is set, then the registry package default `langgate/registry/data/default_models.json` will be used. This file contains data on most major providers and models.
- If `langgate_config.yaml` is unset in your working directory, and no `LANGGATE_CONFIG` environment variable is set, then the core package default `langgate/core/data/default_config.yaml` will be used. This file contains a default configuration with common LLM providers.

## LangGate vs Alternatives

### LangGate vs Envoy AI Gateway

While both use Envoy for proxying, LangGate takes a more direct approach:

- **Simplified Architecture**: LangGate uses Envoy's ext_proc filter directly without a separate control plane
- **No Kubernetes Dependency**: Runs anywhere Docker runs, without requiring Kubernetes CRDs or custom resources
- **Configuration Simplicity**: Uses a straightforward YAML configuration instead of Kubernetes resources
- **Lightweight Deployment**: Deploy with Docker Compose or any container platform without complex orchestration

### LangGate vs Python-based Gateways

Unlike other Python-based gateways:

- **High-Performance Streaming**: Uses Envoy's native streaming capabilities instead of Python for response handling
- **Focused Functionality**: Handles request transformation in Python while letting Envoy manage the high-throughput parts
- **No Middleman for Responses**: Responses stream directly from providers to clients via Envoy

## Running with Docker

```bash
# Start the full LangGate stack
make compose-up

# Development mode with hot reloading
make compose-dev

# Local development (Python on host, Envoy in Docker)
make run-local

# Stop the stack
make compose-down

# Stop stack and remove volumes
make compose-breakdown
```

## Testing and Development

```bash
# Run all tests
make test

# Run lint checks
make lint
```

## Additional Documentation

- [Contributing Guide](CONTRIBUTING.md) - Development setup and guidelines
- [SDK Examples](examples/README.md) - Sample code for using the LangGate SDK
- [Deployment Guide](deployment/README.md) - Instructions for deploying to Kubernetes and other platforms

## Roadmap
- **Image Model Support**: Add explicit support for image generation models and include leading image models in the default model registry.
- **Pydantic Schema Validation**: Implement validation of parameters against Pydantic schemas representing the full API of the provider's model
- **TTS and ASR Model Support**: Include leading Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) models in the default model registry, with endpoints for fetching models filtered by modality (for modality-specific return typing) and schemas for these modalities.
- **Video Generation Model Support**: Add video generation models, similarly to the afformentioned modalities, with an explicit endpoint and schemas.
- **OpenAI API Standardization Option**: Introduce an option to standardize to the OpenAI API spec. This will involve mapping provider-specific Pydantic schemas to corresponding OpenAI API input schemas, offering a unified interface for diverse models.

## License

[MIT License](LICENSE)
