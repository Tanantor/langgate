# LangGate Helm Charts

This directory contains Helm charts for deploying LangGate in Kubernetes environments.

## Chart Structure

- `langgate`: Main umbrella chart that includes all components
- `langgate-envoy`: Envoy proxy component chart
- `langgate-processor`: External processor component chart
- `langgate-server`: Server/registry component chart
- `library/langgate-helpers`: Helper library for common template functions

## Installation

### Installing the complete stack

```bash
# Add the LangGate Helm repository
helm repo add langgate https://tanantor.github.io/langgate/charts
helm repo update

# Install with default configuration
helm install langgate langgate/langgate

# Install with custom values file
helm install langgate langgate/langgate -f values.yaml
```

### Installing individual components

You can install individual components separately if needed:

```bash
# Install just the server component (for registry functionality)
helm install langgate-server langgate/langgate-server

# Install just the proxy components
helm install langgate-envoy langgate/langgate-envoy
helm install langgate-processor langgate/langgate-processor
```

## Configuration

### Basic Configuration Options

#### Using an existing ConfigMap

```bash
# Create ConfigMap from your configuration files
kubectl create configmap langgate-config --from-file=langgate_config.yaml --from-file=langgate_models.json

# Reference it in your values.yaml
config:
  existingConfigMap: "langgate-config"
```

#### Using an existing Secret for API keys

```bash
# Create Secret with your API keys
kubectl create secret generic langgate-secrets --from-literal=OPENAI_API_KEY=sk-xxxx

# Reference it in your values.yaml
secrets:
  existingSecret: "langgate-secrets"
```

#### Using local configuration files during helm install

You can use the `--set-file` option to include configuration files directly from your filesystem:

```bash
# Install with config files
helm install langgate langgate/langgate \
  --set-file config.data.langgate_config=./my-config.yaml \
  --set-file config.data.langgate_models=./my-models.json
```

Alternatively, you can include the configuration content directly in your values.yaml file:

```yaml
# values.yaml
config:
  data:
    langgate_config: |
      default_params:
        temperature: 0.7
      models:
        - id: anthropic/claude-sonnet-4
          service:
            provider: anthropic
            model_id: claude-sonnet-4-0

    langgate_models: |
      {
        "anthropic/claude-sonnet-4": {
          "name": "Claude-4 Sonnet",
          "service_provider": "anthropic"
        }
      }
```

Then run:
```bash
helm install langgate langgate/langgate -f values.yaml
```

### Example values.yaml

```yaml
global:
  imageRegistry: ghcr.io/tanantor
  namespace: langgate
  env:
    APP_ENV: "k8s:prod"
    LOG_LEVEL: "info"

# Configuration
config:
  # Use one of these approaches:

  # 1. Reference an existing ConfigMap
  existingConfigMap: "my-langgate-config"

  # 2. Create ConfigMap (empty to use package defaults)
  data:
    langgate_config: ""
    langgate_models: ""

# Secrets
secrets:
  # Reference an existing Secret containing API keys
  existingSecret: "my-langgate-secrets"
```

## Publishing Docker Images

LangGate publishes official Docker images to GitHub Container Registry with image tags matching the chart version.

```yaml
global:
  imageRegistry: ghcr.io/tanantor
```

## Building Custom Images

If you need to build custom images, you can use the Dockerfiles in the `services` directory:

```bash
# Build images
docker build -t ghcr.io/your-org/langgate-envoy:0.1.0 -f services/envoy/Dockerfile .
docker build -t ghcr.io/your-org/langgate-processor:0.1.0 -f services/processor/Dockerfile .
docker build -t ghcr.io/your-org/langgate-server:0.1.0 -f services/server/Dockerfile .

# Push to your registry
docker push ghcr.io/your-org/langgate-envoy:0.1.0
```

Then reference your registry in Helm values:

```yaml
global:
  imageRegistry: ghcr.io/your-org
```
