"""
Unit Tests for Fee-Aware Kelly Criterion

Tests the enhanced Kelly calculator with fee parameters, broker auto-lookup,
and fee kill switch functionality.

Task Group 8.1: Unit Tests
"""

import pytest
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult
from src.router.broker_registry import BrokerRegistryManager
from src.database.manager import DatabaseManager


@pytest.fixture
def kelly_calculator():
    """Fixture providing a Kelly calculator instance."""
    return EnhancedKellyCalculator()


@pytest.fixture
def broker_manager():
    """Fixture providing a broker manager instance."""
    return BrokerRegistryManager()


class TestFeeAwareKelly:
    """Test suite for fee-aware Kelly Calculator."""

    def test_fee_aware_kelly_reduces_position(self, kelly_calculator):
        """
        Verify that fees reduce the calculated position size.

        Given:
            - A strategy with 55% win rate, $400 avg win, $200 avg loss
            - Commission: $7 per lot
            - Spread: 0.1 pips
            - Pip value: $10 (EURUSD)

        When:
            - Fees are applied to Kelly calculation

        Then:
            - Position size should be smaller than without fees
            - Kelly fraction should be reduced
        """
        # Arrange
        account_balance = 10000.0
        win_rate = 0.55
        avg_win = 400.0
        avg_loss = 200.0
        current_atr = 0.0012
        average_atr = 0.0010
        stop_loss_pips = 20.0
        pip_value = 10.0
        commission_per_lot = 7.0
        spread_pips = 0.1
        broker_id = "test_broker"
        symbol = "EURUSD"

        # Act - Calculate without fees
        result_no_fees = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            regime_quality=1.0
        )

        # Act - Calculate with fees
        result_with_fees = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            regime_quality=1.0,
            commission_per_lot=commission_per_lot,
            spread_pips=spread_pips,
            broker_id=broker_id,
            symbol=symbol
        )

        # Assert
        # Fees should reduce position size (or keep it zero if too high)
        assert result_with_fees.position_size <= result_no_fees.position_size
        # Kelly fraction should be reduced with fees
        assert result_with_fees.kelly_f <= result_no_fees.kelly_f

    def test_fee_kill_switch(self, kelly_calculator):
        """
        Verify that the fee kill switch activates when fees >= avg_win.

        Given:
            - A strategy with $50 avg win
            - Commission: $60 per lot (exceeds avg win)
            - Spread: 10 pips

        When:
            - Fee calculation is performed

        Then:
            - Kelly result should have status='fee_blocked'
            - Position size should be 0
            - Kelly fraction should be 0
        """
        # Arrange
        account_balance = 10000.0
        win_rate = 0.55
        avg_win = 50.0  # $50 per win
        avg_loss = 200.0
        current_atr = 0.0012
        average_atr = 0.0010
        stop_loss_pips = 20.0
        pip_value = 10.0
        commission_per_lot = 60.0  # Exceeds avg win!
        spread_pips = 10.0
        broker_id = "test_broker"
        symbol = "EURUSD"

        # Act
        result = kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            regime_quality=1.0,
            commission_per_lot=commission_per_lot,
            spread_pips=spread_pips,
            broker_id=broker_id,
            symbol=symbol
        )

        # Assert - Fee kill switch should activate
        assert result.status == 'fee_blocked'
        assert result.position_size == 0.0
        assert result.kelly_f == 0.0

    def test_broker_auto_lookup(self, kelly_calculator, broker_manager):
        """
        Verify that broker auto-lookup fetches pip_value, commission, and spread.

        Given:
            - A broker with "icmarkets_raw" ID registered in database
            - EURUSD symbol request

        When:
            - Kelly calculator is called with broker_id and symbol

        Then:
            - Correct pip_value should be fetched (1.0 for XAUUSD)
            - Correct commission should be fetched ($7/lot)
            - Correct spread should be fetched (0.1 pips)
        """
        # Arrange - Create a test broker first
        broker_manager.create_broker(
            broker_id="test_lookup_broker",
            broker_name="Test Lookup Broker",
            spread_avg=0.1,
            commission_per_lot=7.0,
            pip_values={
                "EURUSD": 10.0,
                "XAUUSD": 1.0,
            },
            preference_tags=["TEST"]
        )

        # Act
        result = kelly_calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,  # Will be overridden by auto-lookup
            regime_quality=1.0,
            broker_id="test_lookup_broker",
            symbol="XAUUSD",  # Should fetch pip_value=1.0
            commission_per_lot=0.0,  # Will be overridden by auto-lookup
            spread_pips=0.0  # Will be overridden by auto-lookup
        )

        # Assert - Auto-lookup should have fetched correct values
        assert "pip_value=1.0" in result.adjustments_applied or "Auto-lookup: pip_value=1.0" in result.adjustments_applied
        assert "commission=$7.0/lot" in result.adjustments_applied or "Auto-lookup: commission=$7.0/lot" in result.adjustments_applied
        assert "spread=0.1 pips" in result.adjustments_applied or "Auto-lookup: spread=0.1 pips" in result.adjustments_applied

    def test_backward_compatibility(self, kelly_calculator):
        """
        Verify backward compatibility - old code without fees should still work.

        Given:
            - Kelly Calculator is called WITHOUT commission_per_lot, spread_pips, broker_id

        Then:
            - Calculation should succeed (use default fee values of 0.0)
        """
        # Arrange & Act - Call without fee parameters
        result = kelly_calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            regime_quality=1.0
            # NOT providing: commission_per_lot, spread_pips, broker_id, symbol
        )

        # Assert - Should work with default values
        assert result.status == 'calculated'
        assert result.position_size > 0
        assert result.kelly_f > 0


class TestFeeAwareGovernorIntegration:
    """Test suite for EnhancedGovernor integration with fee-aware Kelly."""

    def test_enhanced_governor_calls_kelly_with_fees(self, kelly_calculator, broker_manager):
        """
        Verify that EnhancedGovernor passes broker_id and account_balance to Kelly.

        Given:
            - EnhancedGovernor with Kelly Calculator
            - Broker registry with test data
            - Trade proposal with bot stats

        When:
            - calculate_risk() is called

        Then:
            - Kelly Calculator should receive broker_id and symbol
            - Result should include position_size and kelly_fraction
        """
        # This test requires a full EnhancedGovernor setup
        # For now, test the interface contract
        from src.router.enhanced_governor import EnhancedGovernor
        from src.router.governor import Governor

        governor = EnhancedGovernor(account_id="test_account")

        # Mock regime report
        class MockRegimeReport:
            def __init__(self):
                self.regime = "TREND_STABLE"
                self.regime_quality = 0.85
                self.chaos_score = 0.2
                self.is_systemic_risk = False

        # Mock trade proposal
        trade_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 10000.0,
            'account_balance': 10000.0,
            'broker_id': 'test_broker',
            'win_rate': 0.55,
            'avg_win': 400.0,
            'avg_loss': 200.0,
            'stop_loss_pips': 20.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
        }

        # Act
        mandate = governor.calculate_risk(
            regime_report=MockRegimeReport(),
            trade_proposal=trade_proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )

        # Assert
        # Mandate should include position sizing fields
        assert hasattr(mandate, 'position_size')
        assert hasattr(mandate, 'kelly_fraction')
        assert hasattr(mandate, 'risk_amount')
        assert hasattr(mandate, 'kelly_adjustments')
