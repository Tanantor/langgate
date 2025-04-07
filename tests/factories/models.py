"""Factories for model objects."""

from datetime import UTC
from decimal import Decimal

import factory.declarations as factory
from factory.faker import Faker

from langgate.core.models import (
    MODEL_PROVIDER_ANTHROPIC,
    MODEL_PROVIDER_GOOGLE,
    MODEL_PROVIDER_META,
    MODEL_PROVIDER_OPENAI,
    ContextWindow,
    LLMInfo,
    ModelCapabilities,
    ModelCost,
    ModelProvider,
)
from tests.factories.base import BasePydanticFactory


class ContextWindowFactory(BasePydanticFactory[ContextWindow]):
    """Factory for ContextWindow objects."""

    max_input_tokens = Faker("pyint", min_value=1024, max_value=200000)
    max_output_tokens = Faker("pyint", min_value=1024, max_value=200000)


class ModelCapabilitiesFactory(BasePydanticFactory[ModelCapabilities]):
    """Factory for ModelCapabilities objects."""

    supports_vision = Faker("pybool")
    supports_tools = Faker("pybool")
    supports_parallel_tool_calls = Faker("pybool")
    supports_response_schema = Faker("pybool")


class ModelCostFactory(BasePydanticFactory[ModelCost]):
    """Factory for ModelCost objects."""

    input_cost_per_token = factory.LazyFunction(lambda: Decimal("0.00001"))
    output_cost_per_token = factory.LazyFunction(lambda: Decimal("0.00002"))


class ModelProviderFactory(BasePydanticFactory[ModelProvider]):
    """Factory for ModelProvider objects."""

    id = factory.Iterator(
        [
            MODEL_PROVIDER_OPENAI,
            MODEL_PROVIDER_ANTHROPIC,
            MODEL_PROVIDER_META,
            MODEL_PROVIDER_GOOGLE,
        ]
    )
    name = factory.LazyAttribute(lambda o: o.id.title())
    description = Faker("sentence")


class LLMInfoFactory(BasePydanticFactory[LLMInfo]):
    """Factory for LLMInfo objects."""

    id = factory.Sequence(lambda n: f"model-{n}")
    name = factory.Sequence(lambda n: f"Test Model {n}")
    description = Faker("sentence")
    provider = factory.SubFactory(ModelProviderFactory)
    costs = factory.SubFactory(ModelCostFactory)
    capabilities = factory.SubFactory(ModelCapabilitiesFactory)
    context_window = factory.SubFactory(ContextWindowFactory)
    updated_dt = Faker("date_time_this_year", tzinfo=UTC)
