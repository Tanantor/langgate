# LangGate Deployment

This directory contains deployment configurations for LangGate:

- `k8s/charts`: Helm charts for Kubernetes deployments
- `terraform`: Terraform configurations (future)

## Deployment Options

LangGate can be deployed in several ways:

1. **Python SDK Only**: The simplest option, using just the Python package with the local registry (see the [main documentation](../README.md))
2. **Docker Compose**: For local development or single-machine deployments
3. **Kubernetes**: For production deployments with Helm charts

## Kubernetes Deployment

For deploying LangGate to Kubernetes, see [charts documentation](k8s/charts/README.md) for detailed instructions on using the Helm charts.

## Docker Images

LangGate publishes the following Docker images:

- `langgate-server`: Registry and API service
- `langgate-processor`: Envoy external processor service
- `langgate-envoy`: Envoy proxy with LangGate configuration

These images are available on GitHub Container Registry:
```
ghcr.io/tanantor/langgate-server:latest
ghcr.io/tanantor/langgate-processor:latest
ghcr.io/tanantor/langgate-envoy:latest
```

Images are tagged with semantic versions (e.g., `0.1.0`) for stable releases.

## Building Images Locally

For development or testing, you can build the images locally:

```bash
# Build the server image
docker build -t langgate-server:latest -f services/server/Dockerfile .

# Build the processor image
docker build -t langgate-processor:latest -f services/processor/Dockerfile .

# Build the envoy image
docker build -t langgate-envoy:latest -f services/envoy/Dockerfile .
```

## Docker Compose Deployment

For single-machine deployments or local development, you can use Docker Compose:

```bash
# Start the stack
docker-compose up -d

# Or use the Makefile target
make compose-up
```

The Docker Compose configuration is in the root directory of the project.

## Terraform Deployment (Future)

Terraform modules for various deployments are planned for future releases.
