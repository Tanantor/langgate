## SPIKE: Image Model Support in LangGate

**Date:** 7 June 2025
**Status:** Proposed

**1. Background & Goals**

LangGate currently provides robust support for Large Language Models (LLMs). This SPIKE outlines extending LangGate to support other model types, starting with Image Generation Models.

The primary goals are:

*   Allow users to define various model configurations (text, image, audio, etc.) in `langgate_config.yaml`.
*   Provide a generic parameter transformation pipeline.
*   Enable validation of parameters against Pydantic schemas representing the full API of the provider's model.
*   Store and serve rich metadata (costs, capabilities) for all configured models.
*   Maintain LangGate's principles of simplicity, explicitness, and provider-agnosticism.

**Clarification on `modality` Tag:**
*   The `modality` field in a `langgate_config.yaml` model entry (e.g., "text", "image", "audio") serves as a **tag**.
*   Its primary purpose is for **client-side filtering** (e.g., an application querying LangGate for all configurations tagged as "image").
*   It does **not** alter LangGate's internal parameter transformation pipeline, nor does it influence default Pydantic schema selection.
*   A model's comprehensive, inherent abilities (e.g., understanding vision input, supporting multiple output types) are detailed in the `capabilities` object within its metadata JSON file.

**2. Proposed Changes & Implementation Plan**

**2.1. Core (`packages/core/src/langgate/core`)**

*   **`models.py`:**
    *   Refactor `LLMInfo` into `BaseModelInfo`.
        *   `BaseModelInfo` fields: `id: str` (the LangGate unique ID from YAML), `name: str`, `description: str | None`, `provider: ModelProvider` (display info), `modality: str | None` (the tag from YAML), `capabilities: dict` (raw capabilities data from JSON), `costs: dict` (raw cost data from JSON), `service_details: dict` (from YAML: `service_provider`, `service_model_id`).
    *   `LLMInfo(BaseModelInfo)`: Parses LLM-specific details (e.g., `context_window`) from its `capabilities` and `costs` dicts.
    *   `ImageModelInfo(BaseModelInfo)`: Parses image-specific details (e.g., `supported_resolutions`) from its `capabilities` and `costs` dicts. (And similar for `AudioModelInfo` etc. in the future).
*   **`schemas/providers/` (New Subdirectory Structure for Pydantic Schemas):**
    *   Example: `schemas/providers/openai.py`, `schemas/providers/flux.py`.
    *   These schemas define the **complete set of parameters accepted by the provider's API** for a given model.
*   **`schemas/config.py` (`ConfigSchema` components):**
    *   Modify `ModelMappingSchema` (for `models:` entries in YAML):
        *   Add `modality: str | None = None` (the tag).
        *   Add `schema_key: str | None = None` (explicit key to a registered Pydantic schema).
*   **`data/`:**
    *   Retain `default_models.json`. Introduce `default_image_models.json`, `default_audio_models.json` etc., primarily for human organization.
    *   All metadata JSON files (default and user-provided) will be loaded and merged by `RegistryConfig`.
    *   Each model ID within these JSON files (e.g., "openai/gpt-4o", "stabilityai/stable-diffusion-3") must be unique across all loaded metadata.

**2.2. Schema Management (e.g., `langgate.schemas.manager.py`)**

*   Internal registry: `_SCHEMA_REGISTRY: dict[str, Type[BaseModel]] = {}`.
    *   Keys are typically like `"openai/dall-e-3"` (matching `service_provider/service_model_id`).
*   Public function: `register_model_schema(key: str, schema_class: Type[BaseModel])`.
*   Auto-populate `_SCHEMA_REGISTRY` with LangGate's built-in schemas (from `langgate.core.schemas.providers.*`).

**2.3. Registry (`packages/registry/src/langgate/registry`)**

*   **`config.py` (`RegistryConfig`):**
    *   Load *all* specified metadata JSON files (e.g., `default_models.json`, `default_image_models.json`, user paths) and merge them into a single collection of metadata, keyed by the model IDs found within those JSONs.
    *   Ensure the JSON entries contain a new `modality` tag. Alternatively, we cold just infer this from the JSON file that the model ID is found in (e.g., `default_image_models.json` implies `modality: "image"`). The latter is likely a better approach to avoid redundancy.
    *   Ensure docs are updated to inform users that they need to update `model_mappings` (from `langgate_config.yaml`) to include any `schema_key` if using a schema that differs from the default supplied for a given service mrovide's model ID. Adding this key to yaml entries should typically not be nececessary, as users can override the default args in passed to these schemas, as per the existing LangGate functionality.
*   **`models.py` (`ModelRegistry`):**
    *   `_build_model_cache`: For each entry in `langgate_config.yaml` (`model_mappings`):
        1.  Use the `service_details.service_provider` and `service_details.service_model_id` (or a conventional mapping from the LangGate `id`) to look up the pre-loaded metadata (costs, capabilities) from the merged collection.
        2.  Instantiate `LLMInfo`, `ImageModelInfo` (or a generic `ModelInfo` if specific parsing isn't immediately needed beyond `BaseModelInfo`) based on the available metadata or potentially a hint if required for parsing, but *not* based on the `modality` tag for core data association. The `modality` tag from YAML is simply stored.
        *   Cache type: `dict[str, BaseModelInfo]`.
    *   `get_model_info` return type: `BaseModelInfo`.

**2.4. Transform (`packages/transform/src/langgate/transform`)**

*   **`local.py` (`LocalTransformerClient`):**
    *   `get_params(model_id: str, input_params: dict[str, Any]) -> dict[str, Any]`:
        1.  Get model mapping config for `model_id` (this is the LangGate unique ID from YAML). This includes `service_details` (`service_provider`, `service_model_id`) and `schema_key`.
        2.  Determine `schema_lookup_key`:
            *   If `schema_key` is present, use it.
            *   Else, construct a default key from `service_details.service_provider` and `service_details.service_model_id` (e.g., `"openai/dall-e-3"`).
        3.  Attempt to retrieve `schema_class` from `_SCHEMA_REGISTRY` using `schema_lookup_key`.
        4.  Perform generic parameter transformations (defaults, overrides, renames from `langgate_config.yaml`) on `input_params`.
        5.  If `schema_class` was found: `validated_model = schema_class.model_validate(transformed_params_from_langgate)`. Return `validated_model.model_dump(exclude_none=True, by_alias=True)`.
        6.  Else (no schema found/registered for this provider model), return `transformed_params_from_langgate` as is.
        7.  Ensure the `service_model_id` (from `langgate_config.yaml`'s `service.model_id`) is set as the `"model"` key in the final returned dictionary.

**2.5. Client (`packages/client/src/langgate/client`)**

*   **`protocol.py`:** `LLMInfoT` becomes `ModelInfoT` bounded by `BaseModelInfo`.
*   **`local.py`:** Adapt for `BaseModelInfo`.

**3. Action Items**

*   [ ] **Core:** Implement `BaseModelInfo`, `LLMInfo`, `ImageModelInfo` (parsing from `capabilities` and `costs` dicts etc).
*   [ ] **Core:** Update `ModelMappingSchema` in `schemas/config.py` (add `modality` tag, `schema_key`).
*   [ ] **Core:** Create `schemas/providers/` and add initial Pydantic schemas for provider APIs.
*   [ ] **Core:** Structure `data/` for multiple `default_*.json` files (e.g., `default_image_models.json`); ensure `capabilities` objects are comprehensive.
*   [ ] **Schema Management:** Implement `_SCHEMA_REGISTRY`, `register_model_schema()`, auto-registration.
*   [ ] **Registry:** Update `RegistryConfig` to merge all metadata JSONs. Update `ModelRegistry._build_model_cache` for unified metadata lookup.
*   [ ] **Transform:** Update `LocalTransformerClient.get_params` for schema lookup (independent of `modality` tag) and validation against full provider API schemas.
*   [ ] **Client:** Update protocols and local clients.
*   [ ] **Documentation:** Clarify `modality` as a tag, schema scope, and metadata organization.
*   [ ] **Testing:** Add comprehensive tests.
