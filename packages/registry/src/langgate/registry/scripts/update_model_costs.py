#!/usr/bin/env python
# src/langgate/registry/scripts/update_model_costs.py
import importlib.resources
import os
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol

import httpx
import simplejson as json
from pydantic import BaseModel

from langgate.core.logging import get_logger
from langgate.core.utils.config_utils import resolve_path
from langgate.registry.models import (
    ContextWindow,
    ModelCapabilities,
    ModelCost,
)

logger = get_logger(__name__)

registry_resources = importlib.resources.files("langgate.registry")
default_models_path = Path(
    str(registry_resources.joinpath("data", "default_models.json"))
)
models_cwd_path = Path.cwd() / "langgate_models.json"
# Path to our model mappings JSON
models_data_path = resolve_path(
    "LANGGATE_MODELS",
    None,
    models_cwd_path if models_cwd_path.exists() else default_models_path,
    "models_data_path",
    logger=logger,
)

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"

# TODO: Add programmatic solution for updating Gemini costs from `curl https://generativelanguage.googleapis.com/v1beta/models?key=$GEMINI_API_KEY``
# TODO: Add programmatic solution for updating Fireworks AI costs from https://fireworks.ai/pricing#serverless-pricing
# This may require using an LLM to parse the HTML and reason about the data.
# TODO: Add programmatic solution for updating xAI costs fby querying `https://api.x.ai/v1/language-models`
# with header Authorization: Bearer <XAI_API_KEY>.


class ModelMatcher(Protocol):
    """Protocol for model matching strategies."""

    async def find_match(
        self,
        model_id: str,
        our_data: dict[str, Any],
        provider_models: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Find a matching model from provider data and return the model and matched ID."""
        ...


class ModelUpdatePolicy(Protocol):
    """Protocol for model update policy."""

    async def should_update(self, model_id: str, our_data: dict[str, Any]) -> bool:
        """Determine if a model should be updated."""
        ...


class OpenRouterData(BaseModel):
    """Schema for mapped OpenRouter data."""

    costs: ModelCost
    context: ContextWindow
    capabilities: ModelCapabilities
    description: str | None = None


class DefaultModelMatcher:
    """Generic model matching strategy."""

    async def find_match(
        self,
        model_id: str,
        our_data: dict[str, Any],
        provider_models: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Find a matching model using various strategies."""
        # Strategy 1: Check for explicit mapping
        if "openrouter_model_id" in our_data:
            openrouter_id: str = our_data["openrouter_model_id"]

            # 1a: Direct match with openrouter_model_id
            for model in provider_models:
                if model["id"] == openrouter_id:
                    return model, model["id"]

            # 1b: Handle -latest suffix in openrouter_model_id
            if "-latest" in openrouter_id:
                match_result = await self._match_latest_version(
                    openrouter_id, provider_models
                )
                if match_result:
                    return match_result[0], match_result[1]

        # Strategy 2: Direct ID match with our model_id
        for model in provider_models:
            if model["id"] == model_id:
                return model, model["id"]

        # Strategy 3: For provider/name format models, try normalized matching
        if "/" in model_id:
            provider, model_name = model_id.split("/", 1)

            # Handle provider name variations (underscores vs hyphens)
            normalized_provider = provider.replace("_", "-")

            # Try normalized model name (without version specifics)
            base_name = self._get_base_model_name(model_name)

            # Special handling for -latest suffix in our model_id
            if "-latest" in model_name:
                match_result = await self._match_latest_version(
                    model_id, provider_models
                )
                if match_result:
                    return match_result[0], match_result[1]

            # General normalized matching
            for model in provider_models:
                or_id: str = model["id"]
                # Try both provider variations for matching
                for prov in [provider, normalized_provider]:
                    if (
                        or_id.startswith(f"{prov}/")
                        and base_name
                        and self._is_specific_match(base_name, or_id)
                        and self._is_version_compatible(model_id, or_id)
                    ):
                        return model, model["id"]

        # No match found
        return None, None

    async def _match_latest_version(
        self, model_id: str, provider_models: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], str] | None:
        """Match a model with -latest suffix to the most recent version."""
        if "/" not in model_id:
            return None

        provider, model_name = model_id.split("/", 1)
        # Normalize provider name
        provider = provider.replace("_", "-")

        # Remove -latest suffix and any version numbers
        base_pattern = model_name.replace("-latest", "")

        # Clean up remaining parts to get core model name
        base_parts = []
        for part in base_pattern.split("-"):
            # Keep only non-version, non-date parts
            if (
                not part.isdigit()
                and not (len(part) == 8 and part.isdigit())
                and not (len(part) == 1 and part.isalpha())
            ):
                base_parts.append(part)

        base_pattern = "-".join(base_parts)

        # Find all matching models
        candidates: list[tuple[dict[str, Any], str]] = []
        for model in provider_models:
            or_id: str = model["id"]
            or_provider, or_model = or_id.split("/", 1) if "/" in or_id else ("", or_id)

            # Check if provider matches and if base pattern is specifically contained
            if or_provider == provider and self._is_specific_match(
                base_pattern, or_model
            ):
                candidates.append((model, or_id))

        # Return the newest model by created timestamp
        if candidates:
            return max(candidates, key=lambda x: x[0].get("created", 0))

        return None

    def _is_specific_match(self, base_pattern: str, full_id: str) -> bool:
        """Check if base pattern matches specifically within full_id.
        Avoids false positives like 'gpt-4o' matching 'gpt-4o-mini'.
        """
        # Split both strings into parts
        pattern_parts = base_pattern.lower().split("-")
        full_parts = (
            full_id.lower().split("-")
            if "/" not in full_id
            else full_id.lower().split("/", 1)[1].split("-")
        )

        # If pattern has multiple parts, all must be found in sequence
        if len(pattern_parts) > 1:
            # Try to find a starting position for the first part
            for i, part in enumerate(full_parts):
                if pattern_parts[0] in part:
                    # Check if remaining parts match in sequence
                    matches = True
                    for j, pattern_part in enumerate(pattern_parts[1:], 1):
                        # Ensure we don't go out of bounds
                        if i + j >= len(full_parts):
                            matches = False
                            break
                        # Check if this part contains the pattern part
                        if pattern_part not in full_parts[i + j]:
                            matches = False
                            break
                    if matches:
                        return True
            return False

        # For single-part patterns, be more strict to avoid false positives
        for part in full_parts:
            if base_pattern == part:  # Exact match
                return True
            if base_pattern in part:
                # Check if it's a partial match that might be a false positive
                # Make sure it's not followed directly by more characters
                if part.startswith(base_pattern + "-") or part == base_pattern:
                    return True
                # Check word boundaries
                if base_pattern == part.rstrip("0123456789-"):
                    return True

        return False

    def _get_base_model_name(self, model_name: str) -> str:
        """Extract base model name without version specifics."""
        # Remove version numbers and suffixes
        parts = model_name.lower().split("-")
        # Extract core model name (without numbers and dates)
        core_parts = []
        for part in parts:
            # Skip version numbers, dates, and special suffixes
            if (
                part.isdigit()
                or "preview" in part
                or "latest" in part
                or (len(part) == 8 and part.isdigit())  # Date format
            ):
                continue
            # Keep single letters that are part of model names (like 'v' in claude-3-5-sonnet)
            # but skip standalone version indicators
            core_parts.append(part)

        return "-".join(core_parts)

    def _is_version_compatible(self, our_model_id: str, openrouter_id: str) -> bool:
        """Check if the models are version-compatible to prevent cross-version contamination."""
        # Extract major version numbers from both model IDs
        our_version = self._extract_major_version(our_model_id)
        or_version = self._extract_major_version(openrouter_id)

        # If both have version numbers, they must match
        if our_version is not None and or_version is not None:
            return our_version == or_version

        # If neither has a version number, they're compatible
        # If only one has a version number, be more cautious
        # Allow matching only if we're confident it's not a version conflict
        return our_version is None and or_version is None

    def _extract_major_version(self, model_id: str) -> int | None:
        """Extract major version number from model ID using generalized patterns."""
        import re

        # General pattern to find version numbers in model IDs
        # Matches: word-number-, word-number$, or number at start/middle of segments
        version_patterns = [
            r"-(\d+)-",  # -3-, -4- (most common)
            r"-(\d+)$",  # -3, -4 at end
            r"^(\d+)-",  # 3-, 4- at start
            r"(\d+)\.(\d+)",  # 3.5, 4.1 (take first number)
        ]

        for pattern in version_patterns:
            matches = re.findall(pattern, model_id.lower())
            if matches:
                # For tuple matches (like version 3.5), take the first number
                if isinstance(matches[0], tuple):
                    return int(matches[0][0])
                # For single matches, take the first one found
                return int(matches[0])

        return None


class DefaultUpdatePolicy:
    """Default policy for determining if a model should be updated."""

    async def should_update(self, model_id: str, our_data: dict[str, Any]) -> bool:
        """Check if model should be updated based on its metadata."""
        # Skip models with a non-openrouter source
        if (
            (source := our_data.get("source", ""))
            and isinstance(source, str)
            and "openrouter" not in source.lower()
        ):
            await logger.ainfo(
                "skipping_model_with_custom_source",
                model_id=model_id,
                source=source,
            )
            return False
        return True


class OpenRouterDataMapper:
    """Maps OpenRouter data to our schema formats."""

    PRICE_MAPPINGS = {
        "prompt": "input_cost_per_token",
        "completion": "output_cost_per_token",
        "image": "input_cost_per_image",
        "input_cache_read": "cache_read_input_token_cost",
    }

    def map_to_our_schema(self, openrouter_model: dict[str, Any]) -> OpenRouterData:
        """Convert OpenRouter model data to our schemas."""
        # Map costs
        costs_data = {}
        pricing = openrouter_model.get("pricing", {})

        for their_field, our_field in self.PRICE_MAPPINGS.items():
            if (
                their_field in pricing
                and pricing[their_field]
                and pricing[their_field] != "0"
            ):
                with suppress(ValueError, TypeError):  # Invalid cost value, skip
                    costs_data[our_field] = Decimal(pricing[their_field])

        costs = ModelCost.model_validate(costs_data)

        # Map context window
        context_data = {}
        if context_length := openrouter_model.get("context_length"):
            context_data["max_input_tokens"] = context_length

        if max_completion_tokens := openrouter_model.get("top_provider", {}).get(
            "max_completion_tokens"
        ):
            context_data["max_output_tokens"] = max_completion_tokens

        context = ContextWindow.model_validate(context_data)

        # Map capabilities
        capabilities_data = {}
        if (
            "architecture" in openrouter_model
            and "modality" in openrouter_model["architecture"]
        ):
            modality: str = openrouter_model["architecture"]["modality"]
            if "image" in modality.split("->")[0]:
                capabilities_data["supports_vision"] = True

        capabilities = ModelCapabilities.model_validate(capabilities_data)

        # Create complete OpenRouterData
        return OpenRouterData(
            costs=costs,
            context=context,
            capabilities=capabilities,
            description=openrouter_model.get("description"),
        )


class ModelUpdater:
    """Service for updating model costs and capabilities from data providers."""

    def __init__(
        self,
        model_path: Path = models_data_path,
        matcher: ModelMatcher | None = None,
        update_policy: ModelUpdatePolicy | None = None,
        data_mapper: OpenRouterDataMapper | None = None,
    ):
        self.model_path = model_path
        self.our_models: dict[str, dict[str, Any]] = {}
        self.provider_models: list[dict[str, Any]] = []
        self.updated_models: set[str] = set()
        self.skipped_models: set[str] = set()
        self.unmapped_models: set[str] = set()

        # Use provided components or create defaults
        self.matcher = matcher or DefaultModelMatcher()
        self.update_policy = update_policy or DefaultUpdatePolicy()
        self.data_mapper = data_mapper or OpenRouterDataMapper()

    async def load_our_models(self) -> None:
        """Load our current model data from JSON file."""
        try:
            with open(self.model_path) as f:
                self.our_models = json.load(f, parse_float=Decimal)
            await logger.ainfo("loaded_model_data", model_count=len(self.our_models))
        except Exception as exc:
            await logger.aexception("failed_to_load_model_data", error=str(exc))
            raise

    async def fetch_provider_data(self) -> None:
        """Fetch model data from provider API."""
        try:
            headers = {}
            # Add OpenRouter API key if available in environment
            if "OPENROUTER_API_KEY" in os.environ:
                headers["Authorization"] = f"Bearer {os.environ['OPENROUTER_API_KEY']}"

            async with httpx.AsyncClient() as client:
                response = await client.get(OPENROUTER_API_URL, headers=headers or None)
                response.raise_for_status()
                data = response.json()
                self.provider_models = data.get("data", [])

            await logger.ainfo(
                "fetched_provider_data", model_count=len(self.provider_models)
            )
        except Exception as exc:
            await logger.aexception("failed_to_fetch_provider_data", error=str(exc))
            raise

    async def _update_model(
        self, model_id: str, provider_model: dict[str, Any], matched_id: str
    ) -> None:
        """Update our model data with information from provider model."""
        our_data = self.our_models[model_id]

        # Convert provider data to our schema format
        mapped_data = self.data_mapper.map_to_our_schema(provider_model)

        # Update costs if any valid values exist
        if not mapped_data.costs.model_dump(
            exclude_defaults=True, exclude_none=True, mode="json"
        ):
            await logger.adebug("no_valid_cost_data", model_id=model_id)
        else:
            our_data["costs"] = ModelCost(**our_data.get("costs", {})).model_dump(
                exclude_none=True, mode="json"
            )
            our_data["costs"].update(
                mapped_data.costs.model_dump(
                    exclude_defaults=True, exclude_none=True, mode="json"
                )
            )

        # Update context window if any valid values exist
        if not mapped_data.context.model_dump(
            exclude_defaults=True, exclude_none=True, mode="json"
        ):
            await logger.adebug("no_valid_context_data", model_id=model_id)
        else:
            our_data["context"] = ContextWindow(
                **our_data.get("context", {})
            ).model_dump(exclude_none=True, mode="json")

            our_data["context"].update(
                mapped_data.context.model_dump(
                    exclude_defaults=True, exclude_none=True, mode="json"
                )
            )

        # Update capabilities if any valid values exist
        if not mapped_data.capabilities.model_dump(
            exclude_defaults=True, exclude_none=True, mode="json"
        ):
            await logger.adebug("no_valid_capabilities_data", model_id=model_id)
        else:
            our_data["capabilities"] = ModelCapabilities(
                **our_data.get("capabilities", {})
            ).model_dump(exclude_none=True, mode="json")

            our_data["capabilities"].update(
                mapped_data.capabilities.model_dump(
                    exclude_defaults=True, exclude_none=True, mode="json"
                )
            )

        # Add description if missing but available
        if not our_data.get("description") and mapped_data.description:
            our_data["description"] = mapped_data.description

        # Add metadata about update
        our_data["_last_updated"] = datetime.now(UTC).isoformat()
        our_data["_data_source"] = "openrouter"
        our_data["_last_updated_from_id"] = matched_id
        self.updated_models.add(model_id)

    async def update_models(self) -> None:
        """Update model data with information from provider."""
        if not self.our_models or not self.provider_models:
            await logger.aerror("missing_data_for_update")
            return

        for model_id, model_data in self.our_models.items():
            # Check if model should be updated according to policy
            should_update = await self.update_policy.should_update(model_id, model_data)

            if not should_update:
                self.skipped_models.add(model_id)
                continue

            # Find matching provider model
            provider_model, matched_id = await self.matcher.find_match(
                model_id, model_data, self.provider_models
            )

            if provider_model and matched_id:
                await self._update_model(model_id, provider_model, matched_id)
                await logger.adebug(
                    "model_updated", model_id=model_id, matched_with=matched_id
                )
            else:
                self.unmapped_models.add(model_id)
                await logger.adebug("no_matching_provider_model", model_id=model_id)

        await logger.ainfo(
            "model_update_complete",
            total_models=len(self.our_models),
            updated=len(self.updated_models),
            skipped=len(self.skipped_models),
            unmapped=len(self.unmapped_models),
        )

    async def save_updated_models(self) -> None:
        """Save updated model data to JSON file."""
        try:
            with open(self.model_path, "w") as f:
                json.dump(self.our_models, f, indent=2, use_decimal=True)
            await logger.ainfo("saved_updated_models", path=str(self.model_path))
        except Exception as exc:
            await logger.aexception("failed_to_save_updated_models", error=str(exc))
            raise

    async def run(self) -> None:
        """Run the complete update process."""
        try:
            await self.load_our_models()
            await self.fetch_provider_data()
            await self.update_models()

            if self.updated_models:
                await self.save_updated_models()
            else:
                await logger.ainfo("no_models_updated")

            # Provide guidance for unmapped models
            if self.unmapped_models:
                await logger.awarning(
                    "models_without_provider_mapping",
                    models=list(self.unmapped_models),
                    help="Add 'openrouter_model_id' field to these models or 'source' field to skip updates",
                )

            await logger.ainfo(
                "update_process_completed",
                updated_count=len(self.updated_models),
                skipped_count=len(self.skipped_models),
                unmapped_count=len(self.unmapped_models),
            )
        except Exception as exc:
            await logger.aexception("update_process_failed", error=str(exc))
            raise


async def update_model_costs() -> None:
    """Update model costs and context window info from OpenRouter."""
    updater = ModelUpdater()
    await updater.run()


def main() -> None:
    import asyncio

    asyncio.run(update_model_costs())


if __name__ == "__main__":
    main()
