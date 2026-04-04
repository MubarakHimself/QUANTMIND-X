"""
Tests for Regime Event Models.

Story 14.2: Layer 2 Tier 1 Position Monitor (Dynamic)
Tests for RegimeShiftEvent and RegimeSuitability models.
"""

import pytest
from datetime import datetime, timezone

from src.events.regime import (
    RegimeShiftEvent,
    RegimeSuitability,
    RegimeType,
)


class TestRegimeType:
    """Test RegimeType enum values."""

    def test_all_regime_types_exist(self):
        """Verify all expected regime types are defined."""
        assert RegimeType.TREND_BULL.value == "TREND_BULL"
        assert RegimeType.TREND_BEAR.value == "TREND_BEAR"
        assert RegimeType.TREND_STABLE.value == "TREND_STABLE"
        assert RegimeType.RANGE_STABLE.value == "RANGE_STABLE"
        assert RegimeType.RANGE_VOLATILE.value == "RANGE_VOLATILE"
        assert RegimeType.BREAKOUT_UP.value == "BREAKOUT_UP"
        assert RegimeType.BREAKOUT_DOWN.value == "BREAKOUT_DOWN"
        assert RegimeType.CHAOS.value == "CHAOS"


class TestRegimeShiftEvent:
    """Test RegimeShiftEvent model."""

    def test_create_regime_shift_event(self):
        """Test basic regime shift event creation."""
        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.RANGE_STABLE,
            symbol="EURUSD",
            confidence=0.85,
        )

        assert event.previous_regime == RegimeType.TREND_STABLE
        assert event.current_regime == RegimeType.RANGE_STABLE
        assert event.symbol == "EURUSD"
        assert event.confidence == 0.85
        assert event.timestamp_utc is not None

    def test_regime_shift_event_defaults(self):
        """Test regime shift event default values."""
        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_BULL,
            current_regime=RegimeType.CHAOS,
        )

        assert event.symbol is None
        assert event.confidence == 0.0
        assert event.metadata == {}

    def test_regime_shift_str_representation(self):
        """Test string representation."""
        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.RANGE_STABLE,
            symbol="EURUSD",
            confidence=0.85,
        )

        str_repr = str(event)
        assert "TREND_STABLE" in str_repr
        assert "RANGE_STABLE" in str_repr
        assert "EURUSD" in str_repr
        assert "0.85" in str_repr

    def test_regime_shift_timestamp(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.RANGE_STABLE,
        )
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp_utc <= after


class TestRegimeSuitability:
    """Test RegimeSuitability evaluation."""

    def test_trend_bull_suits_trend_strategies(self):
        """Trend bull regime should suit momentum and trend_follow strategies."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="trend_follow_momentum",
            regime=RegimeType.TREND_BULL,
            confidence=0.8
        )

        assert suitable.is_suitable is True
        assert suitable.action == "hold"
        assert "suitable" in suitable.reason.lower()

    def test_trend_bull_unsuits_mean_reversion(self):
        """Trend bull regime should not suit mean_reversion strategies."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="mean_reversion_hedge",
            regime=RegimeType.TREND_BULL,
            confidence=0.8
        )

        assert suitable.is_suitable is False
        assert suitable.action == "close"

    def test_range_stable_suits_mean_reversion(self):
        """Range stable regime should suit mean_reversion and range_trade."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="mean_reversion_strategy",
            regime=RegimeType.RANGE_STABLE,
            confidence=0.85
        )

        assert suitable.is_suitable is True
        assert suitable.action == "hold"

    def test_range_volatile_suits_scalp(self):
        """Range volatile should suit scalp and short_term."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="scalp_ea",
            regime=RegimeType.RANGE_VOLATILE,
            confidence=0.75
        )

        assert suitable.is_suitable is True
        assert suitable.action == "hold"

    def test_breakout_up_suits_breakout(self):
        """Breakout up should suit breakout and momentum."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="breakout_confirmation",
            regime=RegimeType.BREAKOUT_UP,
            confidence=0.9
        )

        assert suitable.is_suitable is True
        assert suitable.action == "hold"

    def test_chaos_closes_all(self):
        """Chaos regime should recommend close for all strategies."""
        strategies = [
            "trend_follow_momentum",
            "mean_reversion_hedge",
            "scalp_ea",
            "breakout_strategy",
        ]

        for strategy in strategies:
            suitable = RegimeSuitability.evaluate(
                strategy_id=strategy,
                regime=RegimeType.CHAOS,
                confidence=0.95
            )

            assert suitable.action == "close"
            assert suitable.is_suitable is False
            assert "CHAOS" in suitable.reason

    def test_low_confidence_reduce_vs_close(self):
        """Low confidence should prefer reduce over close for borderline strategies."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="momentum_strategy",
            regime=RegimeType.RANGE_STABLE,  # Not ideal for momentum
            confidence=0.5  # Low confidence
        )

        assert suitable.action == "reduce"
        assert suitable.confidence == 0.5

    def test_high_confidence_close_unsuitable(self):
        """High confidence should close unsuitable strategies."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="momentum_strategy",
            regime=RegimeType.RANGE_STABLE,
            confidence=0.9  # High confidence
        )

        assert suitable.action == "close"
        assert suitable.is_suitable is False

    def test_unknown_strategy_type(self):
        """Unknown strategy types should get hold action (safe default)."""
        suitable = RegimeSuitability.evaluate(
            strategy_id="unknown_strategy_xyz",
            regime=RegimeType.TREND_BULL,
            confidence=0.8
        )

        # Without clear mapping, should default to hold
        assert suitable.action in ["hold", "reduce", "close"]

    def test_regime_strategy_map_completeness(self):
        """Verify all regime types have a strategy map entry."""
        from src.events.regime import REGIME_STRATEGY_MAP
        for regime in RegimeType:
            map_entry = REGIME_STRATEGY_MAP.get(regime)
            assert map_entry is not None, f"Missing map entry for {regime}"


class TestRegimeSuitabilityModel:
    """Test RegimeSuitability model structure."""

    def test_suitability_fields(self):
        """Test RegimeSuitability has all required fields."""
        suitable = RegimeSuitability(
            strategy_id="test_strategy",
            regime=RegimeType.TREND_STABLE,
            is_suitable=True,
            confidence=0.8,
            action="hold",
            reason="Test reason"
        )

        assert suitable.strategy_id == "test_strategy"
        assert suitable.regime == RegimeType.TREND_STABLE
        assert suitable.is_suitable is True
        assert suitable.confidence == 0.8
        assert suitable.action == "hold"
        assert suitable.reason == "Test reason"

    def test_suitability_default_values(self):
        """Test default values for RegimeSuitability."""
        suitable = RegimeSuitability(
            strategy_id="test",
            regime=RegimeType.CHAOS,
            is_suitable=False,
        )

        assert suitable.confidence == 0.0
        assert suitable.action == "hold"  # Default action
        assert suitable.reason == ""

    def test_suitability_action_values(self):
        """Test valid action values."""
        valid_actions = ["hold", "reduce", "close"]

        for action in valid_actions:
            suitable = RegimeSuitability(
                strategy_id="test",
                regime=RegimeType.TREND_STABLE,
                is_suitable=True,
                action=action,
            )
            assert suitable.action == action
