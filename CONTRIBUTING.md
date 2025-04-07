# Contributing to LangGate

This document provides instructions for setting up and contributing to the LangGate project.

## Workspace Structure

LangGate is organized as a uv workspace with multiple packages using a PEP-420 namespace package structure:

```
/
├── pyproject.toml            # Main project configuration with workspace definition
├── packages/
│   ├── core/                 # Core shared models and utilities
│   │   └── src/
│   │       └── langgate/
│   │           └── core/
│   ├── client/               # HTTP client for remote registry
│   │   └── src/
│   │       └── langgate/
│   │           └── client/
│   ├── registry/             # Registry for model information
│   │   └── src/
│   │       └── langgate/
│   │           └── registry/
│   ├── transform/            # Parameter transformation logic
│   │   └── src/
│   │       └── langgate/
│   │           └── transform/
│   ├── processor/            # Envoy external processor implementation
│   │   └── src/
│   │       └── langgate/
│   │           └── processor/
│   ├── server/               # API server implementation
│   │   └── src/
│   │       └── langgate/
│   │           └── server/
│   └── sdk/                  # Convenience package combining registry and transform
│       └── src/
│           └── langgate/
│               └── sdk/
├── examples/                 # Example usage of the SDK components
├── services/                 # Service-specific configurations
└── docs/                     # Documentation
```

Each package can be developed and published independently, while also working together through the shared `langgate` namespace. This PEP-420 namespace package structure allows for clean imports like `from langgate.core import X` or `from langgate.registry import Y`.

## Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) - We use uv for dependency management
- Docker (for running the full stack or tests)

### Initial Setup

1. **Fork and clone the repo**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/langgate.git
   cd langgate
   ```

2. **Set up the development environment**:
   This will create a virtual environment with all dependencies installed:
   ```bash
   make sync
   ```


1. **Verify your setup**:
   ```bash
   make test
   ```

## Development Workflow

### Running the Local Development Environment

For SDK development, you can work with just the Python code:

```bash
# Run tests
make test

# Run linting and type checking
make lint
make mypy
```

For full stack development (including Envoy):

```bash
# Start the full stack in development mode
make compose-dev

# Or run the Python services locally with only Envoy in Docker
make run-local
```

## Code Style

- Follow PEP 8 guidelines
- Strictly adhere to SOLID principles
- Use type hints for all functions and methods
- Include docstrings for all public functions, classes, and methods
- Protocol-based interfaces (for better testability and modularity)
- Avoid complexity unless there's a good reason for it

### Pre-commit hooks (Optional)

We have pre-commit hooks configured for code quality checks. These are optional for local development but will run in CI for all pull requests. To set up pre-commit locally:

```bash
# Install pre-commit hooks locally
make pre-commit-install

# To manually run all pre-commit hooks on all files
make pre-commit-run

# If needed, uninstall pre-commit hooks
make pre-commit-uninstall
```

Even if you don't install the hooks locally, running `make lint` will execute these checks.

## Pull Request Process

1. **Create a new branch** from the `main` branch for your changes
2. **Make your changes** following the code style guidelines
3. **Add tests** for any new functionality
4. **Ensure all tests pass** by running `make test`
5. **Run linting and type checking** with `make lint` and `make mypy`
6. **Submit a pull request** to the `main` branch
   - Include a clear description of the changes
   - Reference any related issues
   - Update documentation as needed

## Versioning

LangGate follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version (X.0.0) - Incompatible API changes
- **MINOR** version (0.X.0) - New backwards-compatible functionality
- **PATCH** version (0.0.X) - Backwards-compatible bug fixes

## Release Process

### Preparation

Create a PR that:

1. Updates version in `pyproject.toml`
2. Updates versions in Helm charts (`deployment/k8s/charts/*/Chart.yaml`)
3. Updates `CHANGELOG.md` with a summary of changes

### Creating a Release

After the version bump PR is merged, trigger the release:

1. **Create a Draft Release**:
   - Go to Actions → Create GitHub Release workflow
   - Enter version number (without 'v' prefix)
   - Set "Create as draft release" to `true`
   - Run workflow

2. **Publish the Release**:
   - Go to GitHub Releases page
   - Find your draft release
   - Click "Edit"
   - Click "Publish release"

Publishing a release automatically:
- Publishes Python packages to PyPI
- Builds and publishes Docker images
- Publishes Helm charts

### Published Artifacts

- **PyPI Packages**: `pip install langgate`
- **Docker Images**:
  - `ghcr.io/tanantor/langgate-server:VERSION`
  - `ghcr.io/tanantor/langgate-processor:VERSION`
  - `ghcr.io/tanantor/langgate-envoy:VERSION`
- **Helm Charts**: `helm repo add langgate https://tanantor.github.io/langgate/charts`

Thank you for contributing to LangGate!
