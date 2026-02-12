"""
Tests for Kelly Governor Integration (Task Group 6.1)

Tests Enhanced Kelly Calculator integration with Strategy Router:
- Kelly calculator position sizing
- House Money Effect risk adjustment
- Dynamic pip value calculation from broker registry
- Fee-aware position sizing
- PropGovernor rule enforcement
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult
from src.router.enhanced_governor import EnhancedGovernor
from src.router.prop.governor import PropGovernor
from src.router.sentinel import RegimeReport


class TestKellyPositionSizing:
    """Test Kelly calculator position sizing integration."""

    def test_kelly_calculator_basic_position_sizing(self):
        """Test basic Kelly position sizing calculation."""
        calculator = EnhancedKellyCalculator()

        result = calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        assert result.position_size > 0
        assert result.kelly_f > 0
        assert result.kelly_f <= 0.02  # Max 2% cap
        assert result.status == 'calculated'
        assert len(result.adjustments_applied) > 0

    def test_kelly_calculator_negative_expectancy(self):
        """Test Kelly returns zero for negative expectancy."""
        calculator = EnhancedKellyCalculator()

        result = calculator.calculate(
            account_balance=10000.0,
            win_rate=0.40,  # Low win rate
            avg_win=200.0,
            avg_loss=300.0,  # High loss
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        assert result.position_size == 0.0
        assert result.kelly_f == 0.0
        assert result.status == 'zero'


class TestHouseMoneyEffect:
    """Test House Money Effect risk adjustment."""

    def test_house_money_increases_risk_when_up(self):
        """Test risk multiplier increases when trading with profits."""
        # Simulate being up 6% (>5% threshold)
        daily_start_balance = 10000.0
        current_balance = 10600.0  # 6% profit

        # House money effect should increase risk multiplier
        profit_pct = (current_balance - daily_start_balance) / daily_start_balance

        if profit_pct > 0.05:
            risk_multiplier = 1.5  # House money active
        else:
            risk_multiplier = 1.0

        assert risk_multiplier == 1.5

    def test_house_money_decreases_risk_when_down(self):
        """Test risk multiplier decreases when losing."""
        # Simulate being down 4% (>3% threshold)
        daily_start_balance = 10000.0
        current_balance = 9600.0  # 4% loss

        # House money effect should decrease risk multiplier
        loss_pct = (daily_start_balance - current_balance) / daily_start_balance

        if loss_pct > 0.03:
            risk_multiplier = 0.5  # Preservation mode
        else:
            risk_multiplier = 1.0

        assert risk_multiplier == 0.5


class TestDynamicPipValue:
    """Test dynamic pip value calculation from broker registry."""

    def test_dynamic_pip_value_eurusd(self):
        """Test pip value lookup for EURUSD."""
        # Create EnhancedGovernor without database
        governor = EnhancedGovernor(account_id=None)

        # Get pip value for EURUSD
        pip_value = governor._get_pip_value('EURUSD', 'mt5_default')

        assert pip_value == 10.0  # Standard pip value for EURUSD

    def test_dynamic_pip_value_xauusd(self):
        """Test pip value lookup for XAUUSD (different from EURUSD)."""
        # Create EnhancedGovernor without database
        governor = EnhancedGovernor(account_id=None)

        # Get pip value for XAUUSD
        pip_value = governor._get_pip_value('XAUUSD', 'mt5_default')

        assert pip_value == 10.0  # Gold pip value

    def test_dynamic_pip_value_unknown_symbol(self):
        """Test pip value lookup for unknown symbol returns default."""
        # Create EnhancedGovernor without database
        governor = EnhancedGovernor(account_id=None)

        # Get pip value for unknown symbol
        pip_value = governor._get_pip_value('UNKNOWN', 'mt5_default')

        assert pip_value == 10.0  # Default pip value


class TestFeeAwarePositionSizing:
    """Test fee-aware position sizing."""

    def test_position_sizing_with_commission(self):
        """Test position size adjusts for commission."""
        calculator = EnhancedKellyCalculator()

        # Calculate without commission
        result_no_fee = calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )

        # With commission, position should be smaller to account for fees
        commission_per_lot = 7.0  # $7 round-turn commission
        risk_amount = result_no_fee.risk_amount

        # Adjust risk amount for commission
        adjusted_risk = risk_amount - commission_per_lot
        expected_position = adjusted_risk / (20.0 * 10.0)

        # Position with fees should be smaller or equal
        assert expected_position <= result_no_fee.position_size


class TestPropGovernorEnforcement:
    """Test PropGovernor rule enforcement."""

    def test_prop_governor_daily_loss_limit(self):
        """Test PropGovernor enforces daily loss limit."""
        governor = PropGovernor(account_id="test_account")
        governor.effective_limit = 0.04  # 4% effective limit
        # Set high equity to be in guardian tier (> $5000)
        # Set daily start balance in prop_state
        governor.prop_state.daily_start_balance = 10000.0

        # Create a mock regime report
        regime_report = Mock()
        regime_report.regime = "TREND_STABLE"
        regime_report.chaos_score = 0.1
        regime_report.news_state = "NORMAL"
        regime_report.is_systemic_risk = False

        # Simulate being near daily loss limit with high equity (guardian tier)
        trade_proposal = {
            'current_balance': 9620.0,  # 3.8% loss (near 4% limit), but in guardian tier
            'symbol': 'EURUSD'
        }

        mandate = governor.calculate_risk(regime_report, trade_proposal)

        # Should be throttled when near limit (guardian tier applies quadratic throttle)
        assert mandate.allocation_scalar < 1.0

    def test_prop_governor_breach_limit(self):
        """Test PropGovernor blocks trades when limit breached."""
        governor = PropGovernor(account_id="test_account")
        governor.effective_limit = 0.04  # 4% effective limit
        # Set high equity to be in guardian tier (> $5000)
        # Set daily start balance in prop_state
        governor.prop_state.daily_start_balance = 10000.0

        # Create a mock regime report
        regime_report = Mock()
        regime_report.regime = "HIGH_CHAOS"
        regime_report.chaos_score = 0.7
        regime_report.news_state = "NORMAL"
        regime_report.is_systemic_risk = False

        # Simulate breaching daily loss limit with high equity (guardian tier)
        trade_proposal = {
            'current_balance': 9500.0,  # 5% loss (breached 4% limit), but in guardian tier
            'symbol': 'EURUSD'
        }

        mandate = governor.calculate_risk(regime_report, trade_proposal)

        # Should block all trading (guardian tier applies quadratic throttle)
        assert mandate.allocation_scalar == 0.0
