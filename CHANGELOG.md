# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.7] - 2025-07-01

### Added
- Aded MiniMax 01 and M1 models to package default models
- Add OpenAI o3 Pro to package default models

### Fixed
- Remove `supports_tools` from MiniMax M1 on OpenRouter - OpenRouter erroneously marks the MiniMax service API as not supporting tools


## [0.1.6] - 2025-06-24

### Added
- Models merge mode support for user-defined models JSON file

## [0.1.5] - 2025-06-18

### Added
- Reasoning support to model capabilities with updated model metadata
- Service provider API format inclusion when returning transformed parameters

### Changed
- Updated default LLMs and improved configuration validation checks
- Updated default registry config with additional model configurations for reasoning variants

### Removed
- Deprecated LLMs form the default config
- Mdels that are no longer supported from the default registry JSON file

## [0.1.3] - 2025-04-09

## Fixed
- Update version bump script to include dependency constraints between the monorepo's packages

## [0.1.2] - 2025-04-09

### Added
- Version bump script and corresponding Makefile targets

### Fixed
- Load environment variables from .env file in `LocalTransformerClient` initialisation - necessary when the registry and transformer clients are running in separate processes

## [0.1.1] - 2025-04-09

### Changed
- Synchronized published package versions with the current repository state.
- Validated and tested automated release workflows (PyPI, Docker, Helm).

## [0.1.0] - 2025-04-08

### Added
- Initial public release of LangGate
- Core functionality for LLM proxy and transformation
- Python client SDK for interacting with LangGate
- Processor service for Envoy integration (incomplete)
- Envoy configuration for routing and transformation
- Server API for registry management
- Helm charts for Kubernetes deployment
- Docker images for all components

### Changed
- Migrated from private repository to open source
