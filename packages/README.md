# LangGate SDK

LangGate is a lightweight, high-performance gateway for AI model inference. This SDK provides modular components for working with both language models (LLMs) and image generation models in various configurations, from embedding components directly in your application to connecting to a full LangGate proxy service.

## Installation

LangGate is designed to be modular. Install only the components you need:

### For applications using the registry only:

Using uv:
```bash
uv add langgate[registry]
```

Using pip:
```bash
pip install langgate[registry]
```

This provides access to model information and metadata.

### For applications using parameter transformation only:

Using uv:
```bash
uv add langgate[transform]
```

Using pip:
```bash
pip install langgate[transform]
```

This allows transforming model parameters based on your configuration.

### For applications needing both registry and transform functions:

Using uv:
```bash
uv add langgate[sdk]
```

Using pip:
```bash
pip install langgate[sdk]
```

Provides a convenient interface combining registry and transformation capabilities.

### For applications connecting to a remote LangGate service:

Using uv:
```bash
uv add langgate[client]
```

Using pip:
```bash
pip install langgate[client]
```

This installs just the HTTP client for connecting to a remote LangGate registry/proxy.

### For deploying a complete LangGate service:

Using uv:
```bash
uv add langgate[all]
```

Using pip:
```bash
pip install langgate[all]
```

This installs all components needed to run a full LangGate service.

## Usage

### Using the Registry

```python
from langgate.registry import LocalRegistryClient

# Initialize the client
client = LocalRegistryClient()

# List available LLMs
llms = await client.list_llms()

# List available image models
image_models = await client.list_image_models()

# Get LLM information
llm_info = await client.get_llm_info("openai/gpt-4o")

# Get image model information
image_info = await client.get_image_model_info("openai/dall-e-3")
```

### Using the Transformer

```python
from pprint import pprint as pp

from langgate.transform import LocalTransformerClient

# Initialize the transformer
transformer = LocalTransformerClient()

# Transform parameters for a specific model
transformed_params = await transformer.get_params(
    "openai/gpt-4o",
    {"temperature": 0.7, "stream": True}
)
```

### Using the Convenience SDK

```python
from langgate.sdk import LangGateLocal

# Initialize the combined client
client = LangGateLocal()

# Access both registry and transformer functions for LLMs
llms = await client.list_llms()
llm_info = await client.get_llm_info("openai/gpt-4o")
transformed_params = await client.get_params(
    "openai/gpt-4o",
    {"temperature": 0.7, "stream": True}
)

# Transform parameters for an image model
image_models = await client.list_image_models()
image_info = await client.get_image_model_info("openai/dall-e-3")
image_params = await client.get_params(
    "openai/dall-e-3",
    {"prompt": "A sunset over the ocean", "size": "1024x1024"}
)
```

### Connecting to a Remote LangGate Service

```python
from langgate.client import HTTPRegistryClient

# Initialize the client with the registry endpoint
client = HTTPRegistryClient("https://langgate.example.com/api/v1")

# Use the same interface as the local client
llms = await client.list_llms()
llm_info = await client.get_llm_info("openai/gpt-4o")

# Access image models
image_models = await client.list_image_models()
image_info = await client.get_image_model_info("openai/dall-e-3")
```

## Configuration

LangGate components use configuration from two main sources:

- `langgate_models.json`: Defines model metadata, capabilities, and costs for both LLMs and image models
- `langgate_config.yaml`: Defines service configurations, parameter mappings, and transformations with modality-aware structure for text and image models

These configurations are loaded from:
1. Paths specified in environment variables (`LANGGATE_MODELS`, `LANGGATE_CONFIG`)
2. Default paths in the current working directory
3. Default built-in configurations if no files are found

## Component Details

LangGate is composed of the following packages (PEP 420 implicitly namespaced by `langgate`):

- **core**: Shared data models and utilities
- **client**: HTTP client for remote LangGate services
- **registry**: Registry implementation for model information
- **transform**: Parameter transformation logic
- **processor**: Envoy external processor implementation
- **sdk**: Convenience package combining registry and transform
