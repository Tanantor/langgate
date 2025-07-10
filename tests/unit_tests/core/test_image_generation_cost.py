"""Unit tests for ImageGenerationCost model validation."""

from decimal import Decimal

import pytest

from langgate.core.models import ImageGenerationCost


class TestImageGenerationCostValidation:
    """Test ImageGenerationCost pricing model validation logic."""

    def test_flat_rate_valid(self):
        """Test valid flat rate pricing model."""
        cost = ImageGenerationCost(flat_rate=Decimal("0.05"))
        assert cost.flat_rate == Decimal("0.05")
        assert cost.quality_tiers is None
        assert cost.cost_per_megapixel is None
        assert cost.cost_per_second is None

    def test_quality_tiers_valid(self):
        """Test valid quality tiers pricing model."""
        tiers = {
            "standard": {"input_cost_per_image": Decimal("0.04")},
            "hd": {"input_cost_per_image": Decimal("0.08")},
        }
        cost = ImageGenerationCost(quality_tiers=tiers)
        assert cost.quality_tiers == tiers
        assert cost.flat_rate is None
        assert cost.cost_per_megapixel is None
        assert cost.cost_per_second is None

    def test_cost_per_megapixel_valid(self):
        """Test valid cost per megapixel pricing model."""
        cost = ImageGenerationCost(cost_per_megapixel=Decimal("0.002"))
        assert cost.cost_per_megapixel == Decimal("0.002")
        assert cost.flat_rate is None
        assert cost.quality_tiers is None
        assert cost.cost_per_second is None

    def test_cost_per_second_valid(self):
        """Test valid cost per second pricing model."""
        cost = ImageGenerationCost(cost_per_second=Decimal("0.00025"))
        assert cost.cost_per_second == Decimal("0.00025")
        assert cost.flat_rate is None
        assert cost.quality_tiers is None
        assert cost.cost_per_megapixel is None

    def test_no_pricing_model_raises(self):
        """Test that no pricing model raises ValueError."""
        with pytest.raises(ValueError, match="Exactly one pricing model must be set"):
            ImageGenerationCost()

    def test_multiple_pricing_models_raises(self):
        """Test that multiple pricing models raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one pricing model must be set"):
            ImageGenerationCost(
                flat_rate=Decimal("0.05"), cost_per_second=Decimal("0.01")
            )

    @pytest.mark.parametrize(
        "pricing_data",
        [
            # Two pricing models
            {"flat_rate": Decimal("0.05"), "cost_per_megapixel": Decimal("0.002")},
            # Three pricing models
            {
                "flat_rate": Decimal("0.05"),
                "cost_per_second": Decimal("0.01"),
                "cost_per_megapixel": Decimal("0.002"),
            },
            # All four pricing models
            {
                "flat_rate": Decimal("0.05"),
                "quality_tiers": {"standard": {"cost": Decimal("0.04")}},
                "cost_per_megapixel": Decimal("0.002"),
                "cost_per_second": Decimal("0.01"),
            },
        ],
    )
    def test_multiple_pricing_model_combinations_raise(self, pricing_data):
        """Test various combinations of multiple pricing models all raise ValueError."""
        with pytest.raises(ValueError, match="Exactly one pricing model must be set"):
            ImageGenerationCost(**pricing_data)

    def test_zero_values_valid(self):
        """Test that zero values are valid for pricing models."""
        cost = ImageGenerationCost(flat_rate=Decimal("0"))
        assert cost.flat_rate == Decimal("0")

    def test_empty_quality_tiers_valid(self):
        """Test that empty quality tiers dict is valid."""
        cost = ImageGenerationCost(quality_tiers={})
        assert cost.quality_tiers == {}
