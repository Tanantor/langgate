# LangGate Registry

Model registry implementation for LangGate AI Gateway.

This package provides the core registry implementation used by LangGate, including:

1. The registry model system - defining model metadata, capabilities, and costs
2. Configuration loading and management - loading from YAML and environment variables
3. `LocalRegistryClient` - for embedding the registry directly in applications

This package is used both:
- As a standalone library for applications embedding the registry directly
- As a core component of the LangGate server when deployed as a microservice
