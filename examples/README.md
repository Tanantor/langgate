# LangGate SDK Examples

This directory contains Jupyter notebooks demonstrating how to use the various components of the LangGate SDK.

## Jupyter Notebook Examples

We recommend using these notebook examples for an interactive exploration of the SDK:

1. [**registry_example.ipynb**](registry_example.ipynb):  Using the `LocalRegistryClient` to get model information
2. [**transformer_example.ipynb**](transformer_example.ipynb):  Using the `LocalTransformerClient` for parameter transformation
3. [**combined_example.ipynb**](combined_example.ipynb):  Using the combined `LangGateLocal` for both functionality
4. [**langchain_examples.ipynb**](langchain_examples.ipynb):  Integrating LangGate with Langchain
5. [**http_client_example.ipynb**](http_client_example.ipynb):  Using the `HTTPRegistryClient` to connect to a remote LangGate service
6. [**custom_clients.ipynb**](custom_clients.ipynb):  Creating custom clients with LangGate

## Running Examples

Make sure you have LangGate installed:

Using uv:
```bash
# Install the full SDK
uv add langgate[all]

# Or install just what you need
uv add langgate[registry]  # For registry examples
uv add langgate[transform]  # For transformer examples
uv add langgate[sdk]  # For combined client examples
uv add langgate[client]  # For HTTP client examples
```

Using pip:
```bash
# Install the full SDK
pip install langgate[all]

# Or install just what you need
pip install langgate[registry]  # For registry examples
pip install langgate[transform]  # For transformer examples
pip install langgate[sdk]  # For combined client examples
pip install langgate[client]  # For HTTP client examples
```

## Configuration

The examples expect a valid `langgate_config.yaml` file in the current directory. A minimal example config would look like:

```yaml
# Global default parameters
default_params:
  temperature: 0.7

# Service provider configurations
services:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"

# Model-specific configurations
models:
  - id: anthropic/claude-sonnet-4
    service:
      provider: anthropic
      model_id: claude-sonnet-4-0
```

See the [main documentation](../README.md) for full configuration details and parameter options.
