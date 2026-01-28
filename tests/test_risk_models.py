"""
Tests for Risk Management Data Models

Focused tests for Pydantic models covering:
- StrategyPerformance validation (win_rate edge cases, negative values)
- MarketPhysics timestamp validation and staleness detection
- SizingRecommendation constraint tracking
- PositionSizingResult lot calculation accuracy

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError

from src.risk.models import (
    StrategyPerformance,
    MarketPhysics,
    SizingRecommendation,
    PositionSizingResult,
    RiskLevel,
)


class TestStrategyPerformance:
    """Test StrategyPerformance model validation and calculations."""

    def test_valid_strategy_performance(self):
        """Test creating a valid StrategyPerformance instance."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100
        )
        assert perf.win_rate == 0.55
        assert perf.avg_win == 100.0
        assert perf.avg_loss == 50.0
        assert perf.total_trades == 100

    def test_win_rate_boundary_values(self):
        """Test win_rate validation at boundaries (0 and 1 should fail)."""
        # win_rate = 0 should fail
        with pytest.raises(ValidationError) as exc_info:
            StrategyPerformance(
                win_rate=0.0,
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )
        assert "win_rate must be strictly between 0 and 1" in str(exc_info.value).lower()

        # win_rate = 1 should fail
        with pytest.raises(ValidationError) as exc_info:
            StrategyPerformance(
                win_rate=1.0,
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )
        assert "win_rate must be strictly between 0 and 1" in str(exc_info.value).lower()

    def test_win_rate_out_of_range(self):
        """Test win_rate validation for values outside [0, 1]."""
        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=1.5,  # > 1
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )

        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=-0.1,  # < 0
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=100
            )

    def test_negative_avg_win_fails(self):
        """Test that negative avg_win raises validation error."""
        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=0.55,
                avg_win=-100.0,  # Negative
                avg_loss=50.0,
                total_trades=100
            )

    def test_negative_avg_loss_fails(self):
        """Test that negative avg_loss raises validation error."""
        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=0.55,
                avg_win=100.0,
                avg_loss=-50.0,  # Negative
                total_trades=100
            )

    def test_zero_avg_win_fails(self):
        """Test that zero avg_win raises validation error."""
        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=0.55,
                avg_win=0.0,  # Must be > 0
                avg_loss=50.0,
                total_trades=100
            )

    def test_payoff_ratio_calculation(self):
        """Test payoff_ratio property calculation."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100
        )
        assert perf.payoff_ratio == 2.0  # 100 / 50

    def test_payoff_ratio_too_low_fails(self):
        """Test that very low payoff ratio (avg_win << avg_loss) fails validation."""
        with pytest.raises(ValidationError) as exc_info:
            StrategyPerformance(
                win_rate=0.55,
                avg_win=1.0,   # Very small win
                avg_loss=50.0,  # Large loss
                total_trades=100
            )
        assert "payoff ratio" in str(exc_info.value).lower()

    def test_expectancy_calculation(self):
        """Test expectancy calculation formula."""
        # Expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100
        )
        expected_expectancy = (0.55 * 100.0) - (0.45 * 50.0)  # 55 - 22.5 = 32.5
        assert abs(perf.expectancy() - expected_expectancy) < 0.001

    def test_negative_expectancy(self):
        """Test strategy with negative expectancy."""
        perf = StrategyPerformance(
            win_rate=0.40,
            avg_win=50.0,
            avg_loss=100.0,
            total_trades=100
        )
        assert perf.expectancy() < 0
        assert not perf.is_profitable()

    def test_positive_expectancy(self):
        """Test strategy with positive expectancy."""
        perf = StrategyPerformance(
            win_rate=0.60,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100
        )
        assert perf.expectancy() > 0
        assert perf.is_profitable()

    def test_kelly_criterion_calculation(self):
        """Test Kelly criterion formula: f = ((B + 1) * P - 1) / B."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100
        )
        # B = 2.0, P = 0.55
        # f = ((2 + 1) * 0.55 - 1) / 2 = (1.65 - 1) / 2 = 0.325
        expected_kelly = ((2.0 + 1) * 0.55 - 1) / 2.0
        assert abs(perf.kelly_criterion() - expected_kelly) < 0.001

    def test_adjusted_kelly_with_fraction(self):
        """Test adjusted Kelly with k_fraction multiplier."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=100,
            k_fraction=0.5  # Half-Kelly
        )
        full_kelly = perf.kelly_criterion()
        expected_adjusted = full_kelly * 0.5
        assert abs(perf.adjusted_kelly() - expected_adjusted) < 0.001

    def test_confidence_level_low_sample(self):
        """Test confidence_level returns 'low' for small samples."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=20  # < 30
        )
        assert perf.confidence_level() == "low"

    def test_confidence_level_medium_sample(self):
        """Test confidence_level returns 'medium' for medium samples."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=50  # 30-99
        )
        assert perf.confidence_level() == "medium"

    def test_confidence_level_high_sample(self):
        """Test confidence_level returns 'high' for large samples."""
        perf = StrategyPerformance(
            win_rate=0.55,
            avg_win=100.0,
            avg_loss=50.0,
            total_trades=150  # >= 100
        )
        assert perf.confidence_level() == "high"

    def test_total_trades_minimum(self):
        """Test that total_trades must be at least 1."""
        with pytest.raises(ValidationError):
            StrategyPerformance(
                win_rate=0.55,
                avg_win=100.0,
                avg_loss=50.0,
                total_trades=0  # Must be >= 1
            )


class TestMarketPhysics:
    """Test MarketPhysics model validation and risk calculations."""

    def test_valid_market_physics(self):
        """Test creating a valid MarketPhysics instance."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics.lyapunov_exponent == 0.5
        assert physics.ising_susceptibility == 1.0
        assert physics.ising_magnetization == 0.3

    def test_ising_magnetization_bounds(self):
        """Test Ising magnetization is constrained to [-1, 1]."""
        # Magnetization > 1 should fail
        with pytest.raises(ValidationError):
            MarketPhysics(
                lyapunov_exponent=0.5,
                ising_susceptibility=1.0,
                ising_magnetization=1.5  # > 1
            )

        # Magnetization < -1 should fail
        with pytest.raises(ValidationError):
            MarketPhysics(
                lyapunov_exponent=0.5,
                ising_susceptibility=1.0,
                ising_magnetization=-1.5  # < -1
            )

    def test_negative_susceptibility_fails(self):
        """Test that negative susceptibility raises validation error."""
        with pytest.raises(ValidationError):
            MarketPhysics(
                lyapunov_exponent=0.5,
                ising_susceptibility=-1.0,  # Must be >= 0
                ising_magnetization=0.3
            )

    def test_is_fresh_with_recent_timestamp(self):
        """Test is_fresh returns True for recent data."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            calculated_at=datetime.utcnow()
        )
        assert physics.is_fresh(max_age_seconds=3600)  # 1 hour

    def test_is_fresh_with_old_timestamp(self):
        """Test is_fresh returns False for old data."""
        old_time = datetime.utcnow() - timedelta(hours=2)
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            calculated_at=old_time
        )
        assert not physics.is_fresh(max_age_seconds=3600)  # 1 hour threshold

    def test_mark_stale(self):
        """Test marking physics data as stale."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            is_stale=False
        )
        assert not physics.is_stale
        physics.mark_stale()
        assert physics.is_stale

    def test_mark_fresh(self):
        """Test marking physics data as fresh."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            is_stale=True
        )
        assert physics.is_stale
        physics.mark_fresh()
        assert not physics.is_stale

    def test_risk_level_extreme(self):
        """Test risk_level returns EXTREME for very high Lyapunov."""
        physics = MarketPhysics(
            lyapunov_exponent=2.5,  # > 2.0
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics.risk_level() == RiskLevel.EXTREME

    def test_risk_level_high_chaotic(self):
        """Test risk_level returns HIGH for positive Lyapunov."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,  # Positive = chaotic
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics.risk_level() == RiskLevel.HIGH

    def test_risk_level_high_susceptibility(self):
        """Test risk_level returns HIGH for high susceptibility."""
        physics = MarketPhysics(
            lyapunov_exponent=-0.5,  # Stable
            ising_susceptibility=2.5,  # > 2.0
            ising_magnetization=0.3
        )
        assert physics.risk_level() == RiskLevel.HIGH

    def test_risk_level_moderate(self):
        """Test risk_level returns MODERATE for moderate susceptibility."""
        physics = MarketPhysics(
            lyapunov_exponent=-0.5,  # Stable
            ising_susceptibility=1.5,  # 1.0 < x < 2.0
            ising_magnetization=0.3
        )
        assert physics.risk_level() == RiskLevel.MODERATE

    def test_risk_level_low(self):
        """Test risk_level returns LOW for stable conditions."""
        physics = MarketPhysics(
            lyapunov_exponent=-0.5,  # Stable
            ising_susceptibility=0.5,  # < 1.0
            ising_magnetization=0.3
        )
        assert physics.risk_level() == RiskLevel.LOW

    def test_physics_multiplier_by_risk_level(self):
        """Test physics_multiplier returns correct values for each risk level."""
        # EXTREME: 0.25
        physics_extreme = MarketPhysics(
            lyapunov_exponent=3.0,
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics_extreme.physics_multiplier() == 0.25

        # HIGH: 0.5
        physics_high = MarketPhysics(
            lyapunov_exponent=1.0,
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics_high.physics_multiplier() == 0.5

        # MODERATE: 0.75
        physics_moderate = MarketPhysics(
            lyapunov_exponent=-0.5,
            ising_susceptibility=1.5,
            ising_magnetization=0.3
        )
        assert physics_moderate.physics_multiplier() == 0.75

        # LOW: 1.0
        physics_low = MarketPhysics(
            lyapunov_exponent=-0.5,
            ising_susceptibility=0.5,
            ising_magnetization=0.3
        )
        assert physics_low.physics_multiplier() == 1.0

    def test_has_signal_with_rmt(self):
        """Test has_signal returns True when max_eigenvalue > threshold."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            rmt_max_eigenvalue=3.5,
            rmt_noise_threshold=2.0
        )
        assert physics.has_signal()

    def test_has_signal_no_rmt(self):
        """Test has_signal returns True when RMT data not available."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            rmt_max_eigenvalue=None,
            rmt_noise_threshold=None
        )
        assert physics.has_signal()  # Defaults to True

    def test_has_signal_no_signal(self):
        """Test has_signal returns False when max_eigenvalue <= threshold."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3,
            rmt_max_eigenvalue=1.8,
            rmt_noise_threshold=2.0
        )
        assert not physics.has_signal()

    def test_trend_direction_bullish(self):
        """Test trend_direction returns 'bullish' for positive magnetization."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.5  # > 0.2
        )
        assert physics.trend_direction() == "bullish"

    def test_trend_direction_bearish(self):
        """Test trend_direction returns 'bearish' for negative magnetization."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=-0.5  # < -0.2
        )
        assert physics.trend_direction() == "bearish"

    def test_trend_direction_neutral(self):
        """Test trend_direction returns 'neutral' for small magnetization."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.1  # Between -0.2 and 0.2
        )
        assert physics.trend_direction() == "neutral"

    def test_is_chaotic_true(self):
        """Test is_chaotic returns True for positive Lyapunov."""
        physics = MarketPhysics(
            lyapunov_exponent=0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert physics.is_chaotic()

    def test_is_chaotic_false(self):
        """Test is_chaotic returns False for negative Lyapunov."""
        physics = MarketPhysics(
            lyapunov_exponent=-0.5,
            ising_susceptibility=1.0,
            ising_magnetization=0.3
        )
        assert not physics.is_chaotic()


class TestSizingRecommendation:
    """Test SizingRecommendation model constraint tracking."""

    def test_valid_sizing_recommendation(self):
        """Test creating a valid SizingRecommendation instance."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.02,
            position_size_lots=0.15
        )
        assert rec.raw_kelly == 0.15
        assert rec.final_risk_pct == 0.02
        assert rec.position_size_lots == 0.15

    def test_raw_kelly_bounds(self):
        """Test raw_kelly is constrained to [0, 1]."""
        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=-0.1,  # Negative
                final_risk_pct=0.02,
                position_size_lots=0.15
            )

        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=1.5,  # > 1
                final_risk_pct=0.02,
                position_size_lots=0.15
            )

    def test_final_risk_pct_exceeds_maximum(self):
        """Test final_risk_pct > 0.10 (10%) raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SizingRecommendation(
                raw_kelly=0.15,
                final_risk_pct=0.15,  # > 0.10
                position_size_lots=0.15
            )
        assert "final_risk_pct" in str(exc_info.value).lower()

    def test_final_risk_pct_negative_fails(self):
        """Test negative final_risk_pct raises validation error."""
        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=0.15,
                final_risk_pct=-0.01,  # Negative
                position_size_lots=0.15
            )

    def test_physics_multiplier_bounds(self):
        """Test physics_multiplier is constrained to [0, 1]."""
        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=0.15,
                physics_multiplier=-0.5,  # Negative
                final_risk_pct=0.02,
                position_size_lots=0.15
            )

        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=0.15,
                physics_multiplier=1.5,  # > 1
                final_risk_pct=0.02,
                position_size_lots=0.15
            )

    def test_negative_position_size_fails(self):
        """Test negative position_size_lots raises validation error."""
        with pytest.raises(ValidationError):
            SizingRecommendation(
                raw_kelly=0.15,
                final_risk_pct=0.02,
                position_size_lots=-0.1  # Negative
            )

    def test_is_constrained_true(self):
        """Test is_constrained returns True when final < adjusted raw."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            physics_multiplier=0.5,  # Adjusted raw = 0.075
            final_risk_pct=0.02,  # < 0.075
            position_size_lots=0.15
        )
        assert rec.is_constrained()

    def test_is_constrained_false(self):
        """Test is_constrained returns False when final = adjusted raw."""
        rec = SizingRecommendation(
            raw_kelly=0.04,
            physics_multiplier=0.5,  # Adjusted raw = 0.02
            final_risk_pct=0.02,  # = adjusted raw
            position_size_lots=0.15
        )
        assert not rec.is_constrained()

    def test_constraint_reason(self):
        """Test constraint_reason returns the constraint source."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            physics_multiplier=0.5,
            final_risk_pct=0.02,
            position_size_lots=0.15,
            constraint_source="risk_cap"
        )
        assert rec.constraint_reason() == "risk_cap"

    def test_constraint_reason_none(self):
        """Test constraint_reason returns 'none' when unconstrained."""
        rec = SizingRecommendation(
            raw_kelly=0.04,
            physics_multiplier=0.5,
            final_risk_pct=0.02,
            position_size_lots=0.15
        )
        assert rec.constraint_reason() == "none"

    def test_risk_reduction_pct(self):
        """Test risk_reduction_pct calculation."""
        rec = SizingRecommendation(
            raw_kelly=0.10,
            physics_multiplier=0.5,  # Adjusted raw = 0.05
            final_risk_pct=0.02,  # 60% reduction from 0.05
            position_size_lots=0.15
        )
        # Reduction = (0.05 - 0.02) / 0.05 = 0.60
        assert abs(rec.risk_reduction_pct() - 0.60) < 0.01

    def test_is_zero_position_true(self):
        """Test is_zero_position returns True when lot_size = 0."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.0,
            position_size_lots=0.0
        )
        assert rec.is_zero_position()

    def test_is_zero_position_false(self):
        """Test is_zero_position returns False when lot_size > 0."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.02,
            position_size_lots=0.15
        )
        assert not rec.is_zero_position()

    def test_apply_penalty(self):
        """Test apply_penalty reduces position size and tracks adjustments."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.04,
            position_size_lots=2.0
        )
        rec.apply_penalty("physics_high", 0.5, "High volatility detected")

        assert rec.position_size_lots == 1.0  # 2.0 * 0.5
        assert rec.final_risk_pct == 0.02  # 0.04 * 0.5
        assert rec.constraint_source == "physics_high"
        assert len(rec.adjustments_applied) == 1

    def test_apply_penalty_invalid_multiplier(self):
        """Test apply_penalty raises error for invalid multiplier."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.02,
            position_size_lots=0.15
        )
        with pytest.raises(ValueError):
            rec.apply_penalty("test", 1.5)  # > 1

    def test_add_adjustment(self):
        """Test add_adjustment appends to adjustments list."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            final_risk_pct=0.02,
            position_size_lots=0.15
        )
        rec.add_adjustment("Applied risk cap")
        rec.add_adjustment("Reduced for high volatility")

        assert len(rec.adjustments_applied) == 2

    def test_to_dict(self):
        """Test to_dict serialization."""
        rec = SizingRecommendation(
            raw_kelly=0.15,
            physics_multiplier=0.5,
            final_risk_pct=0.02,
            position_size_lots=0.15,
            constraint_source="risk_cap"
        )
        result_dict = rec.to_dict()

        assert result_dict["raw_kelly"] == 0.15
        assert result_dict["physics_multiplier"] == 0.5
        assert result_dict["final_risk_pct"] == 0.02
        assert result_dict["position_size_lots"] == 0.15
        assert result_dict["constraint_source"] == "risk_cap"
        assert "is_constrained" in result_dict
        assert "risk_reduction_pct" in result_dict


class TestPositionSizingResult:
    """Test PositionSizingResult model margin calculations."""

    def test_valid_position_sizing_result(self):
        """Test creating a valid PositionSizingResult instance."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15
        )
        assert result.account_balance == 10000.0
        assert result.risk_amount == 200.0
        assert result.lot_size == 0.15

    def test_account_balance_too_low(self):
        """Test account_balance < $100 raises validation error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=50.0,  # Too low
                risk_amount=200.0,
                stop_loss_pips=20.0,
                pip_value=10.0,
                lot_size=0.15
            )

    def test_account_balance_unrealistically_high(self):
        """Test unrealistically high account_balance raises error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=2e9,  # 2 billion
                risk_amount=200.0,
                stop_loss_pips=20.0,
                pip_value=10.0,
                lot_size=0.15
            )

    def test_negative_risk_amount_fails(self):
        """Test negative risk_amount raises validation error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=10000.0,
                risk_amount=-100.0,  # Negative
                stop_loss_pips=20.0,
                pip_value=10.0,
                lot_size=0.15
            )

    def test_negative_lot_size_fails(self):
        """Test negative lot_size raises validation error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=10000.0,
                risk_amount=200.0,
                stop_loss_pips=20.0,
                pip_value=10.0,
                lot_size=-0.1  # Negative
            )

    def test_negative_stop_loss_fails(self):
        """Test negative stop_loss_pips raises validation error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=10000.0,
                risk_amount=200.0,
                stop_loss_pips=-20.0,  # Negative
                pip_value=10.0,
                lot_size=0.15
            )

    def test_negative_pip_value_fails(self):
        """Test negative pip_value raises validation error."""
        with pytest.raises(ValidationError):
            PositionSizingResult(
                account_balance=10000.0,
                risk_amount=200.0,
                stop_loss_pips=20.0,
                pip_value=-10.0,  # Negative
                lot_size=0.15
            )

    def test_risk_percentage_calculation(self):
        """Test risk_percentage calculation."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15
        )
        assert result.risk_percentage() == 0.02  # 200 / 10000

    def test_risk_per_lot_calculation(self):
        """Test risk_per_lot calculation."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15
        )
        assert result.risk_per_lot() == 200.0  # 20 * 10

    def test_lot_calculation_accuracy(self):
        """Test lot size is calculated accurately from risk parameters."""
        # Position size = risk_amount / (stop_loss_pips * pip_value)
        # For $200 risk, 20 pip stop, $10/pip: 200 / (20 * 10) = 1.0 lot
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=1.0
        )
        expected_lots = 200.0 / (20.0 * 10.0)
        assert abs(result.lot_size - expected_lots) < 0.001

    def test_add_step(self):
        """Test add_step appends to calculation_steps."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15
        )
        result.add_step("Calculated raw Kelly")
        result.add_step("Applied physics multiplier")

        assert len(result.calculation_steps) == 2

    def test_is_zero_position_true(self):
        """Test is_zero_position returns True when lot_size = 0."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=0.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.0
        )
        assert result.is_zero_position()

    def test_is_zero_position_false(self):
        """Test is_zero_position returns False when lot_size > 0."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15
        )
        assert not result.is_zero_position()

    def test_margin_utilization_pct(self):
        """Test margin_utilization_pct calculation."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=1.0,
            estimated_margin=1000.0  # 10% of balance
        )
        assert result.margin_utilization_pct() == 0.1

    def test_margin_utilization_pct_none(self):
        """Test margin_utilization_pct returns None when margin not calculated."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15,
            estimated_margin=None
        )
        assert result.margin_utilization_pct() is None

    def test_validate_margin_sufficiency_true(self):
        """Test validate_margin_sufficiency returns True when margin sufficient."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=1.0,
            estimated_margin=1000.0,
            remaining_margin=9000.0  # Positive
        )
        assert result.validate_margin_sufficiency()

    def test_validate_margin_sufficiency_false(self):
        """Test validate_margin_sufficiency returns False when margin insufficient."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=1.0,
            estimated_margin=10000.0,
            remaining_margin=0.0  # Exactly zero (insufficient)
        )
        assert not result.validate_margin_sufficiency()

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = PositionSizingResult(
            account_balance=10000.0,
            risk_amount=200.0,
            stop_loss_pips=20.0,
            pip_value=10.0,
            lot_size=0.15,
            estimated_margin=500.0
        )
        result_dict = result.to_dict()

        assert result_dict["account_balance"] == 10000.0
        assert result_dict["risk_amount"] == 200.0
        assert result_dict["risk_percentage"] == 0.02
        assert result_dict["lot_size"] == 0.15
        assert result_dict["estimated_margin"] == 500.0
