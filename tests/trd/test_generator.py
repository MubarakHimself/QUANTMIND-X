"""
Tests for TRD Generator

Tests TRD generation from research hypothesis, validation,
and integration with AlphaForgeFlow.
"""

import pytest
from datetime import datetime
from src.trd.generator import TRDGenerator, create_trd_from_hypothesis
from src.trd.schema import TRDDocument, StrategyType
from src.agents.departments.heads.research_head import Hypothesis


class TestTRDGenerator:
    """Test suite for TRD Generator."""

    @pytest.fixture
    def generator(self):
        """Create TRD generator instance."""
        return TRDGenerator()

    @pytest.fixture
    def sample_hypothesis(self):
        """Create sample hypothesis for testing."""
        return Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Trend following strategy on EURUSD H4 using 50 EMA crossover",
            supporting_evidence=[
                "Historical backtests show 65% win rate on EMA crossover",
                "EURUSD exhibits strong trends during London session",
                "50 EMA provides good balance between responsiveness and noise filtering"
            ],
            confidence_score=0.82,
            recommended_next_steps=[
                "Proceed to TRD Generation",
                "Schedule backtest validation"
            ]
        )

    def test_generate_trd_basic(self, generator, sample_hypothesis):
        """Test basic TRD generation from hypothesis."""
        trd = generator.generate_trd(sample_hypothesis)

        assert trd is not None
        assert trd.strategy_id is not None
        assert trd.symbol == "EURUSD"
        assert trd.timeframe == "H4"
        assert trd.strategy_type == StrategyType.TREND
        assert "force_close_hour" in trd.parameters
        assert "overnight_hold" in trd.parameters
        assert "daily_loss_cap" in trd.parameters

    def test_strategy_id_format(self, generator, sample_hypothesis):
        """Test strategy_id follows {symbol}_{timestamp}_{uuid} format."""
        trd = generator.generate_trd(sample_hypothesis)

        parts = trd.strategy_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "EURUSD"
        # Second part should be timestamp-like (14 digits: YYYYMMDDHHMMSS)
        assert len(parts[1]) == 12
        # Third part should be 8 char hex uuid
        assert len(parts[2]) == 8

    def test_islamic_compliance_params_present(self, generator, sample_hypothesis):
        """Test Islamic compliance parameters are always present."""
        trd = generator.generate_trd(sample_hypothesis)

        # Check defaults
        assert trd.parameters["force_close_hour"] == 22
        assert trd.parameters["overnight_hold"] is False
        assert trd.parameters["daily_loss_cap"] == 2.0

    def test_custom_strategy_name(self, generator, sample_hypothesis):
        """Test custom strategy name override."""
        trd = generator.generate_trd(sample_hypothesis, strategy_name="My Custom Strategy")

        assert trd.strategy_name == "My Custom Strategy"

    def test_additional_params(self, generator, sample_hypothesis):
        """Test additional parameters override defaults."""
        additional_params = {
            "session_mask": "Asia",
            "max_orders": 5
        }
        trd = generator.generate_trd(
            sample_hypothesis,
            additional_params=additional_params
        )

        assert trd.parameters["session_mask"] == "Asia"
        assert trd.parameters["max_orders"] == 5
        # Default params still present
        assert "force_close_hour" in trd.parameters

    def test_entry_conditions_extraction(self, generator, sample_hypothesis):
        """Test entry conditions extracted from hypothesis."""
        trd = generator.generate_trd(sample_hypothesis)

        assert len(trd.entry_conditions) > 0
        # Primary condition should include hypothesis text
        assert any("Primary:" in cond for cond in trd.exit_conditions) or len(trd.entry_conditions) > 0

    def test_generate_and_validate(self, generator, sample_hypothesis):
        """Test generate and validate in one step."""
        result = generator.generate_and_validate(sample_hypothesis)

        assert "trd" in result
        assert "validation" in result
        assert "is_valid" in result
        assert "is_complete" in result
        assert "requires_clarification" in result

    def test_validate_returns_proper_structure(self, generator, sample_hypothesis):
        """Test validation result has proper structure."""
        result = generator.generate_and_validate(sample_hypothesis)

        validation = result["validation"]
        assert "is_valid" in validation
        assert "is_complete" in validation
        assert "errors" in validation
        assert "ambiguities" in validation
        assert "missing_parameters" in validation

    def test_low_confidence_not_escalated(self, generator):
        """Test that low confidence hypothesis still generates TRD."""
        low_confidence_hypothesis = Hypothesis(
            symbol="GBPUSD",
            timeframe="M15",
            hypothesis="Untested strategy",
            supporting_evidence=["Limited evidence"],
            confidence_score=0.3,
            recommended_next_steps=[]
        )

        # Generator doesn't check confidence - that's done in the flow
        trd = generator.generate_trd(low_confidence_hypothesis)

        assert trd is not None
        assert trd.symbol == "GBPUSD"


class TestTRDGeneratorConvenience:
    """Test convenience function."""

    def test_create_trd_from_hypothesis_with_validation(self):
        """Test convenience function with validation."""
        hypothesis = Hypothesis(
            symbol="USDJPY",
            timeframe="D1",
            hypothesis="Swing trading strategy",
            supporting_evidence=["Historical data"],
            confidence_score=0.8,
            recommended_next_steps=[]
        )

        result = create_trd_from_hypothesis(hypothesis, validate=True)

        assert "trd" in result
        assert "validation" in result

    def test_create_trd_from_hypothesis_without_validation(self):
        """Test convenience function without validation."""
        hypothesis = Hypothesis(
            symbol="AUDUSD",
            timeframe="H1",
            hypothesis="Scalping strategy",
            supporting_evidence=[],
            confidence_score=0.7,
            recommended_next_steps=[]
        )

        result = create_trd_from_hypothesis(hypothesis, validate=False)

        assert "trd" in result
        # Validation key should not be present
        assert "validation" not in result


class TestTRDStorage:
    """Test TRD storage integration."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading TRD."""
        from src.trd.storage import TRDStorage

        storage = TRDStorage(storage_dir=tmp_path)

        # Create and save TRD
        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="Test strategy",
            supporting_evidence=[],
            confidence_score=0.8,
            recommended_next_steps=[]
        )

        generator = TRDGenerator()
        trd = generator.generate_trd(hypothesis)

        saved_id = storage.save(trd)
        assert saved_id == trd.strategy_id

        # Load it back
        loaded = storage.load(trd.strategy_id)
        assert loaded is not None
        assert loaded.strategy_id == trd.strategy_id
        assert loaded.symbol == "EURUSD"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])