# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
