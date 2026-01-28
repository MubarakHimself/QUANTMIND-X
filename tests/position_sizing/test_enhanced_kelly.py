"""
Comprehensive tests for Enhanced Kelly Position Sizing Calculator.

Tests cover:
- Kelly fraction calculation
- Risk cap enforcement
- Dynamic volatility adjustment
- Edge case handling
- Configuration validation
- Prop firm presets
"""

import pytest
import numpy as np
from decimal import Decimal

from src.position_sizing.enhanced_kelly import (
    EnhancedKellyCalculator,
    KellyResult,
    enhanced_kelly_position_size
)
from src.position_sizing.kelly_config import (
    EnhancedKellyConfig,
    PropFirmPresets
)


@pytest.mark.kelly
class TestEnhancedKellyCalculator:
    """Test suite for Enhanced Kelly Calculator."""

    def test_positive_expectancy_calculation(
        self, kelly_calculator, account_balance, stop_loss_pips, pip_value
    ):
        """Test Kelly calculation with positive expectancy."""
        # Strategy: 55% win rate, 2:1 reward-risk ratio
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0
        current_atr = 0.0012
        average_atr = 0.0010

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Verify calculation
        assert result.position_size > 0
        assert result.kelly_f > 0
        assert result.kelly_f < kelly_calculator.config.max_risk_pct
        assert result.status == 'calculated'

        # Check base Kelly formula: f = ((B+1)*P - 1) / B
        # where B = avg_win/avg_loss = 2, P = 0.55
        expected_base_kelly = ((2 + 1) * 0.55 - 1) / 2
        assert abs(result.base_kelly_f - expected_base_kelly) < 0.01

        # Check Kelly fraction (50%) and risk cap (2%)
        expected_kelly = min(expected_base_kelly * 0.5, 0.02)
        assert abs(result.kelly_f - expected_kelly) < 0.01

    def test_high_win_rate_low_reward(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test Kelly with high win rate but low reward (1:1)."""
        win_rate = 0.65
        avg_win = 200.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # f = ((1+1)*0.65 - 1) / 1 = 0.30
        # Half-Kelly = 0.15 (15%)
        # Capped at 2%
        assert result.kelly_f == 0.02

    def test_low_win_rate_high_reward(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test Kelly with low win rate but high reward (3:1)."""
        win_rate = 0.40
        avg_win = 600.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # f = ((3+1)*0.40 - 1) / 3 = 0.133
        # Half-Kelly = 0.067 (6.7%)
        # Capped at 2%
        assert result.kelly_f == 0.02

    def test_negative_expectancy(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test Kelly calculation with negative expectancy."""
        win_rate = 0.40
        avg_win = 150.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Should return zero position for negative expectancy
        assert result.position_size == 0.0
        assert result.kelly_f == 0.0
        assert result.base_kelly_f <= 0
        assert result.status == 'zero'

    def test_breakeven_strategy(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test Kelly calculation with breakeven strategy (zero expectancy)."""
        win_rate = 0.50
        avg_win = 200.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # f = 0 for breakeven
        assert result.kelly_f == 0.0
        assert result.status == 'zero'


@pytest.mark.kelly
class TestVolatilityAdjustment:
    """Test suite for Dynamic Volatility Adjustment (Layer 3)."""

    def test_high_volatility_penalty(
        self, kelly_calculator, account_balance, stop_loss_pips, pip_value, high_volatility_state
    ):
        """Test position size reduction during high volatility."""
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=high_volatility_state["current_atr"],
            average_atr=high_volatility_state["average_atr"],
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # ATR ratio = 2.0, should apply penalty: kelly_f / atr_ratio
        # Check that high volatility penalty was applied
        adjustments_str = " ".join(result.adjustments_applied)
        assert "high vol" in adjustments_str.lower() or "Layer 3" in adjustments_str

    def test_low_volatility_boost(
        self, kelly_calculator, account_balance, stop_loss_pips, pip_value, low_volatility_state
    ):
        """Test position size increase during low volatility."""
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=low_volatility_state["current_atr"],
            average_atr=low_volatility_state["average_atr"],
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # ATR ratio = 0.5, should apply boost: kelly_f * 1.2
        # Check that low volatility boost was applied
        adjustments_str = " ".join(result.adjustments_applied)
        assert "low vol" in adjustments_str.lower() or "Layer 3" in adjustments_str

    def test_normal_volatility(
        self, kelly_calculator, account_balance, stop_loss_pips, pip_value, sample_market_state
    ):
        """Test position size during normal volatility."""
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=sample_market_state["current_atr"],
            average_atr=sample_market_state["average_atr"],
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # ATR ratio = 1.2, should not apply extreme adjustments
        assert result.status == 'calculated'


@pytest.mark.kelly
class TestPositionSizeCalculation:
    """Test suite for position size calculation and lot rounding."""

    def test_lot_calculation_accuracy(self, kelly_calculator, account_balance):
        """Test end-to-end lot calculation formula."""
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010
        stop_loss_pips = 20.0
        pip_value = 10.0

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Verify lot calculation: lots = (balance * kelly_f) / (sl_pips * pip_value)
        # For 10k balance, 2% risk, 20 pip SL, $10/pip:
        # risk_amount = 10000 * 0.02 = $200
        # risk_per_lot = 20 * 10 = $200
        # position_size = 200 / 200 = 1.0 lot
        expected_risk = account_balance * result.kelly_f
        expected_position = expected_risk / (stop_loss_pips * pip_value)
        assert abs(result.position_size - expected_position) < 0.1

    def test_minimum_lot_enforcement(self, standard_config, account_balance):
        """Test position size below minimum lot."""
        config = standard_config
        config.min_lot_size = 0.01
        config.allow_zero_position = False

        calculator = EnhancedKellyCalculator(config)

        # Small account and wide stop loss to get < 0.01 lots
        result = calculator.calculate(
            account_balance=100.0,  # Small account
            win_rate=0.55,
            avg_win=40.0,
            avg_loss=20.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=100.0,  # Wide stop
            pip_value=10.0
        )

        # Should return min_lot_size, not zero
        assert result.position_size >= config.min_lot_size

    def test_maximum_lot_enforcement(self, standard_config, account_balance):
        """Test position size above maximum lot."""
        config = standard_config
        config.max_lot_size = 10.0

        calculator = EnhancedKellyCalculator(config)

        # Large account to trigger max lot cap
        result = calculator.calculate(
            account_balance=1000000.0,  # Large account
            win_rate=0.60,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        # Should cap at max_lot_size
        assert result.position_size <= config.max_lot_size

    def test_lot_step_rounding(self, kelly_calculator, account_balance):
        """Test rounding to broker lot step."""
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0
        current_atr = 0.0010
        average_atr = 0.0010
        stop_loss_pips = 17.0  # Odd number for non-round result
        pip_value = 10.0

        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Should be rounded to 0.01 step
        step_multiplier = result.position_size / kelly_calculator.config.lot_step
        assert abs(step_multiplier - round(step_multiplier)) < 0.01


@pytest.mark.edge_case
class TestEdgeCases:
    """Test suite for edge case handling."""

    def test_zero_win_rate(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with zero win rate (all losses)."""
        with pytest.raises(ValueError):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=0.0,
                avg_win=100.0,
                avg_loss=200.0,
                current_atr=0.0010,
                average_atr=0.0010,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )

    def test_perfect_win_rate(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with 100% win rate."""
        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=1.0,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Should cap at max_risk_pct
        assert result.kelly_f <= kelly_calculator.config.max_risk_pct

    def test_zero_average_loss(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with zero average loss (division by zero protection)."""
        with pytest.raises(ValueError, match="avg_loss"):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=0.0,
                current_atr=0.0010,
                average_atr=0.0010,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )

    def test_invalid_win_rate_negative(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with negative win rate."""
        with pytest.raises(ValueError, match="win_rate"):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=-0.1,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0010,
                average_atr=0.0010,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )

    def test_invalid_win_rate_above_one(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with win rate above 100%."""
        with pytest.raises(ValueError, match="win_rate"):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=1.5,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0010,
                average_atr=0.0010,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )

    def test_zero_stop_loss(self, kelly_calculator, account_balance):
        """Test with zero stop loss."""
        with pytest.raises(ValueError, match="stop_loss"):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0010,
                average_atr=0.0010,
                stop_loss_pips=0.0,
                pip_value=10.0
            )

    def test_zero_average_atr(self, kelly_calculator, account_balance, stop_loss_pips, pip_value):
        """Test with zero average ATR."""
        with pytest.raises(ValueError, match="average_atr"):
            kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0010,
                average_atr=0.0,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value
            )


@pytest.mark.kelly
class TestPropFirmPresets:
    """Test suite for prop firm preset configurations."""

    def test_ftmo_challenge_preset(self, ftmo_config, account_balance, stop_loss_pips, pip_value):
        """Test FTMO Challenge preset (ultra conservative)."""
        calculator = EnhancedKellyCalculator(ftmo_config)

        result = calculator.calculate(
            account_balance=account_balance,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # FTMO Challenge: max 1% risk, 40% Kelly fraction
        assert calculator.config.max_risk_pct == 0.01
        assert calculator.config.kelly_fraction == 0.40
        assert result.kelly_f <= 0.01

    def test_ftmo_funded_preset(self, account_balance, stop_loss_pips, pip_value):
        """Test FTMO Funded preset."""
        config = PropFirmPresets.ftmo_funded()
        calculator = EnhancedKellyCalculator(config)

        result = calculator.calculate(
            account_balance=account_balance,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # FTMO Funded: max 1.5% risk, 55% Kelly fraction
        assert calculator.config.max_risk_pct == 0.015
        assert calculator.config.kelly_fraction == 0.55
        assert result.kelly_f <= 0.015

    def test_the5ers_preset(self, the5ers_config, account_balance, stop_loss_pips, pip_value):
        """Test The5%ers preset."""
        calculator = EnhancedKellyCalculator(the5ers_config)

        result = calculator.calculate(
            account_balance=account_balance,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # The5%ers: max 2% risk, 50% Kelly fraction
        assert calculator.config.max_risk_pct == 0.02
        assert calculator.config.kelly_fraction == 0.50

    def test_personal_aggressive_preset(self, account_balance, stop_loss_pips, pip_value):
        """Test personal aggressive preset."""
        config = PropFirmPresets.personal_aggressive()
        calculator = EnhancedKellyCalculator(config)

        result = calculator.calculate(
            account_balance=account_balance,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Personal: max 2.5% risk, 60% Kelly fraction
        assert calculator.config.max_risk_pct == 0.025
        assert calculator.config.kelly_fraction == 0.60


@pytest.mark.kelly
class TestConvenienceFunction:
    """Test suite for convenience function."""

    def test_enhanced_kelly_position_size_function(self, account_balance, stop_loss_pips, pip_value):
        """Test the convenience function returns float."""
        position_size = enhanced_kelly_position_size(
            account_balance=account_balance,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value
        )

        # Should return float
        assert isinstance(position_size, float)
        assert position_size > 0


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for realistic scenarios."""

    def test_profitable_strategy_normal_market(
        self, kelly_calculator, account_balance, sample_market_state
    ):
        """Test profitable strategy in normal market conditions."""
        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=0.58,
            avg_win=450.0,
            avg_loss=200.0,
            current_atr=sample_market_state["current_atr"],
            average_atr=sample_market_state["average_atr"],
            stop_loss_pips=25.0,
            pip_value=10.0
        )

        # Should recommend position size
        assert result.position_size > 0
        assert result.status == 'calculated'
        # Risk should be reasonable (0.5-2%)
        assert 0.005 <= result.kelly_f <= 0.02

    def test_conservative_strategy_high_volatility(
        self, kelly_calculator, account_balance, high_volatility_state
    ):
        """Test conservative strategy during high volatility."""
        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=0.52,
            avg_win=300.0,
            avg_loss=200.0,
            current_atr=high_volatility_state["current_atr"],
            average_atr=high_volatility_state["average_atr"],
            stop_loss_pips=30.0,
            pip_value=10.0
        )

        # Should reduce position due to high volatility
        assert result.position_size > 0
        # Check volatility adjustment was applied
        adjustments_str = " ".join(result.adjustments_applied)
        has_vol_adjustment = any(term in adjustments_str for term in ["vol", "ATR", "Layer 3"])
        assert has_vol_adjustment

    def test_aggressive_strategy_low_volatility(
        self, kelly_calculator, account_balance, low_volatility_state
    ):
        """Test aggressive strategy during low volatility."""
        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=0.62,
            avg_win=500.0,
            avg_loss=200.0,
            current_atr=low_volatility_state["current_atr"],
            average_atr=low_volatility_state["average_atr"],
            stop_loss_pips=15.0,
            pip_value=10.0
        )

        # Should allow larger position due to low volatility
        assert result.position_size > 0
        assert result.status == 'calculated'
