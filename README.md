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

The package includes a `LangGateLocal` client that can be used directly in your application without needing to run the proxy service. This client provides access to both language and image model registries, plus parameter transformation features.

**List Available Models:**

```py
from pprint import pprint as pp
from langgate.sdk import LangGateLocal

client = LangGateLocal()

# List available LLMs
llms = await client.list_llms()
print(f"Available LLMs: {len(llms)}")
for model in llms[:3]:
    print(f"- {model.id}: {model.name}")

# List available image models
image_models = await client.list_image_models()
print(f"Available Image Models: {len(image_models)}")
for model in image_models[:3]:
    print(f"- {model.id}: {model.name}")
```
```text
Available LLMs: 5
- openai/gpt-5-chat: ChatGPT-5
- openai/gpt-5: GPT-5
- openai/gpt-5-high: GPT-5 high
- anthropic/claude-sonnet-4: Claude-4 Sonnet
- anthropic/claude-sonnet-4-reasoning: Claude-4 Sonnet R

==================================================

Available Image Models: 4
- openai/gpt-image-1: GPT Image 1
- openai/dall-e-3: DALL-E 3
- black-forest-labs/flux-dev: FLUX.1 [dev]
- stability-ai/sd-3.5-large: SD 3.5 Large
```

**Get Model Information and Transform Parameters:**

```py
# LangGate allows us to register "virtual models" - models with specific parameters.
# `langgate_config.yaml` defines this `claude-sonnet-4-reasoning` model
# which is a wrapper around the `claude-sonnet-4-0` model,
# with specific parameters and metadata.
model_id = "anthropic/claude-sonnet-4-reasoning"

# Get model info
model_info = await client.get_llm_info(model_id)
print(f"Model: {model_info.name}")
print(f"Provider: {model_info.provider.name}")

# Transform parameters
input_params = {"temperature": 0.7, "stream": True}
api_format, transformed = await client.get_params(model_id, input_params)
print(f"API format: {api_format}")
pp(transformed)
```
```
Model: Claude-4 Sonnet R
Provider: Anthropic
Description: Claude-4 Sonnet with reasoning capabilities.

Transformed parameters:
('anthropic',
 {'api_key': SecretStr('**********'),
  'base_url': 'https://api.anthropic.com',
  'model': 'claude-sonnet-4-0',
  'stream': True,
  'thinking': {'budget_tokens': 1024, 'type': 'enabled'}})
```

The `temperature` parameter is removed because temperature is not supported by Claude models with reasoning enabled. The `thinking` parameter is added with the `budget_tokens` we specify in `langgate_config.yaml`. See the below [Configuration](#configuration) section for more details on how LangGate handles parameter transformations.

**Working with Image Models:**
Transforming parameters for image models is the exact same process as for LLMs.
```py
# Transform parameters for an image model
image_model_id = "openai/gpt-image-1"
image_params = {
    "prompt": "A beautiful sunset over the ocean",
    "size": "1024x1024",
    "quality": "medium",
}

api_format, transformed = await client.get_params(image_model_id, image_params)
print(f"API format: {api_format}")
pp(transformed)
```
```text
API format: openai
{'api_key': SecretStr('**********'),
 'base_url': 'https://api.openai.com/v1',
 'model': 'gpt-image-1',
 'prompt': 'A beautiful sunset over the ocean',
 'quality': 'medium',
 'size': '1024x1024'}
```

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
model_id = "openai/gpt-5"
model = await model_factory.create_model(model_id, {"temperature": 0.7})
model
```
```text
API format: openai

{'api_key': SecretStr('**********'),
 'base_url': 'https://api.openai.com/v1',
 'include_reasoning': True,
 'model': 'gpt-5',
 'streaming': True}

ChatOpenAI(client=<openai.resources.chat.completions.completions.Completions object at 0x10cbacec0>, async_client=<openai.resources.chat.completions.completions.AsyncCompletions object at 0x10cbad940>, root_client=<openai.OpenAI object at 0x10c40e270>, root_async_client=<openai.AsyncOpenAI object at 0x10cbad6a0>, model_name='gpt-5', model_kwargs={'include_reasoning': True}, openai_api_key=SecretStr('**********'), openai_api_base='https://api.openai.com/v1', streaming=True)
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
# langgate_config.yaml
# Main configuration file for LangGate

# Global default parameters by modality (applied to all models unless overridden)
default_params:
  text:
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

  replicate:
    api_key: "${REPLICATE_API_KEY}"

# Model-specific configurations organized by modality
models:
  text:
    - id: openai/gpt-5-chat
      service:
        provider: openai
        model_id: gpt-5-chat-latest

    - id: openai/gpt-5
      service:
        provider: openai
        model_id: gpt-5
      override_params:
        include_reasoning: true

    # "virtual model" that wraps the gpt-5 model with high-effort reasoning
    - id: openai/gpt-5-high
      service:
        provider: openai
        model_id: gpt-5
      name: GPT-5 high
      description: gpt-5-high applies high-effort reasoning for the gpt-5 model
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
      name: Claude-4 Sonnet R
      description: "Claude-4 Sonnet with reasoning capabilities."
      override_params:
        thinking:
          budget_tokens: 1024

  image:
    - id: openai/gpt-image-1
      service:
        provider: openai
        model_id: gpt-image-1

    - id: openai/dall-e-3
      service:
        provider: openai
        model_id: dall-e-3

    - id: black-forest-labs/flux-dev
      service:
        provider: replicate
        model_id: black-forest-labs/flux-dev
      default_params:
        disable_safety_checker: true

    - id: stability-ai/sd-3.5-large
      service:
        provider: replicate
        model_id: stability-ai/stable-diffusion-3.5-large

# Models merge mode for loading data from JSON files: "merge" (default), "replace", or "extend"
# - merge: User models override defaults, new models are added
# - replace: Only use user models (ignore default models file)
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
- **Pydantic Schema Validation**: Implement validation of parameters against Pydantic schemas representing the full API of the provider's model
- **TTS and ASR Model Support**: Include leading Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) models in the default model registry, with endpoints for fetching models filtered by modality (for modality-specific return typing) and schemas for these modalities.
- **Video Generation Model Support**: Add video generation models, similarly to the afformentioned modalities, with an explicit endpoint and schemas.
- **OpenAI API Standardization Option**: Introduce an option to standardize to the OpenAI API spec. This will involve mapping provider-specific Pydantic schemas to corresponding OpenAI API input schemas, offering a unified interface for diverse models.

## License

[MIT License](LICENSE)
