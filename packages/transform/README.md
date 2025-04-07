# LangGate Transform

Parameter transformation utilities for LangGate AI Gateway.

This package provides the core parameter transformation logic used by both:
- Local registry clients to transform parameters for direct model calls
- The Envoy external processor to transform parameters in the proxy

It implements a declarative approach to parameter transformation that can be:
1. Used directly in Python applications
2. Potentially reimplemented in other languages (like Go) for external processors
3. Extended with custom transformation rules

## Parameter Transformation Precedence

The transformation module follows a specific precedence order when applying parameter transformations:

### Defaults (applied only if key doesn't exist yet):
1. Model-specific defaults (highest precedence for defaults)
2. Pattern defaults (matching patterns applied in config order)
3. Service provider defaults
4. Global defaults (lowest precedence for defaults)

### Overrides/Removals/Renames (applied in order, later steps overwrite/modify earlier ones):
1. Input parameters (initial state)
2. Service-level API keys and base URLs
3. Service-level overrides, removals, renames
4. Pattern-level overrides, removals, renames (matching patterns applied in config order)
5. Model-specific overrides, removals, renames (highest precedence)
6. Model ID (always overwritten with service_model_id)
7. Environment variable substitution (applied last to all string values)
