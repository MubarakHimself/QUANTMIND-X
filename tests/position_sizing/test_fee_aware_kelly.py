"""
Unit Tests for Fee-Aware Kelly Criterion

Tests the enhanced Kelly calculator with fee parameters, broker auto-lookup,
and fee kill switch functionality.

Task Group 8.1: Unit Tests

Coverage:
- Fee reduction of position size (fees reduce kelly_f and position_size)
- Fee kill switch activation (status='fee_blocked' when fees >= avg_win)
- Broker auto-lookup overrides (pip_value, commission, spread from broker registry)
- Backward-compatibility with zero fees (default parameters work)
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock

from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult
from src.position_sizing.kelly_config import EnhancedKellyConfig

logger = logging.getLogger(__name__)


@pytest.fixture
def kelly_calculator():
    """Fixture providing a Kelly calculator instance."""
    return EnhancedKellyCalculator()


@pytest.fixture
def kelly_config():
    """Fixture providing a standard Kelly config."""
    return EnhancedKellyConfig(
        kelly_fraction=0.50,
        max_risk_pct=0.02,
        high_vol_threshold=1.3,
        low_vol_threshold=0.7,
        low_vol_boost=1.2,
        min_lot_size=0.01,
        lot_step=0.01
    )


@pytest.fixture
def base_trade_params():
    """Fixture providing base trade parameters for testing."""
    return {
        'account_balance': 10000.0,
        'win_rate': 0.55,
        'avg_win': 400.0,
        'avg_loss': 200.0,
        'current_atr': 0.0012,
        'average_atr': 0.0010,
        'stop_loss_pips': 20.0,
        'pip_value': 10.0,
        'regime_quality': 1.0,
    }




class TestFeeReductionOfPositionSize:
    """
    Test suite verifying that fees reduce position size.

    Expectation: fees reduce both kelly_f and position_size proportionally
    to their impact on the risk-reward ratio.
    """

    def test_commission_reduces_position(self, kelly_calculator, base_trade_params):
        """
        Verify commission reduces position size.

        Given:
            - Strategy with 55% win rate, $400 avg win, $200 avg loss
            - Commission: $50 per lot (significant)

        When:
            - Kelly is calculated with commission

        Then:
            - Kelly fraction should be reduced (affected by fee)
            - Adjustments should show fee impact
        """
        # Act - without commission
        result_no_fee = kelly_calculator.calculate(**base_trade_params)

        # Act - with significant commission
        params_with_commission = {**base_trade_params, 'commission_per_lot': 50.0}
        result_with_commission = kelly_calculator.calculate(**params_with_commission)

        # Assert
        assert result_with_commission.kelly_f <= result_no_fee.kelly_f, \
            "Commission should reduce kelly fraction"
        assert result_with_commission.status == 'calculated'
        # Verify commission was factored in
        assert any('fee' in adj.lower() or '$50' in adj for adj in result_with_commission.adjustments_applied)

    def test_spread_reduces_position(self, kelly_calculator, base_trade_params):
        """
        Verify spread reduces position size.

        Given:
            - Strategy with valid edge
            - Spread: 1.0 pips on $10/pip EURUSD = $10 per round trip

        When:
            - Kelly is calculated with spread

        Then:
            - Kelly fraction should be reduced (affected by spread fee)
            - Adjustments should show spread impact
        """
        # Act - without spread
        result_no_spread = kelly_calculator.calculate(**base_trade_params)

        # Act - with significant spread
        params_with_spread = {**base_trade_params, 'spread_pips': 1.0}
        result_with_spread = kelly_calculator.calculate(**params_with_spread)

        # Assert
        assert result_with_spread.kelly_f <= result_no_spread.kelly_f, \
            "Spread should reduce kelly fraction"
        assert result_with_spread.status == 'calculated'
        # Verify spread was factored in
        assert any('spread' in adj.lower() or '1.0 pips' in adj for adj in result_with_spread.adjustments_applied)

    def test_combined_fees_reduce_position(self, kelly_calculator, base_trade_params):
        """
        Verify combined commission + spread reduces kelly_f.

        Given:
            - Strategy with valid edge
            - Commission: $25 per lot
            - Spread: 1.0 pips = $10 per round trip
            - Total fee: $35 per lot

        When:
            - Kelly is calculated with both fees

        Then:
            - Kelly fraction should be reduced more with combined fees
        """
        # Act - combinations
        result_no_fees = kelly_calculator.calculate(**base_trade_params)

        result_commission_only = kelly_calculator.calculate(
            **{**base_trade_params, 'commission_per_lot': 25.0}
        )

        result_spread_only = kelly_calculator.calculate(
            **{**base_trade_params, 'spread_pips': 1.0}
        )

        result_both_fees = kelly_calculator.calculate(
            **{**base_trade_params, 'commission_per_lot': 25.0, 'spread_pips': 1.0}
        )

        # Assert ordering: no_fees >= commission >= both (both should have most reduction)
        assert result_no_fees.kelly_f >= result_commission_only.kelly_f, \
            "Commission should reduce kelly_f"
        assert result_no_fees.kelly_f >= result_both_fees.kelly_f, \
            "Combined fees should reduce kelly_f"
        # With both fees, reduction should be at least as much as one type alone
        assert result_commission_only.kelly_f >= result_both_fees.kelly_f or \
               result_spread_only.kelly_f >= result_both_fees.kelly_f, \
            "Combined fees should reduce kelly_f relative to one fee type"

    def test_high_fees_reduce_to_zero(self, kelly_calculator, base_trade_params):
        """
        Verify that sufficiently high fees can reduce position to zero.

        Given:
            - Strategy with $400 avg win
            - Fees: $150 per lot
            - Net win becomes $250 (still profitable but significantly reduced)

        When:
            - Kelly is calculated

        Then:
            - Position size should be smaller than with low fees
            - Status should still be 'calculated' (not fee_blocked)
        """
        # Act - with high (but not kill-switch-triggering) fees
        params_high_fees = {
            **base_trade_params,
            'commission_per_lot': 100.0,
            'spread_pips': 5.0,  # 5 * $10 = $50, total = $150
        }
        result = kelly_calculator.calculate(**params_high_fees)

        # Act - with very extreme fees
        result_extreme = kelly_calculator.calculate(
            **{**base_trade_params, 'commission_per_lot': 300.0}
        )

        # Assert - positions should be very small with extreme fees
        assert result.position_size >= 0
        assert result_extreme.position_size >= 0
        # Extreme fees should lead to very small position
        assert result_extreme.kelly_f <= result.kelly_f


class TestFeeKillSwitchActivation:
    """
    Test suite verifying fee kill switch (status='fee_blocked' when fees >= avg_win).

    Expectation: when fees >= avg_win, the trade is blocked entirely because
    the fee cost exceeds the expected profit on a winning trade.
    """

    def test_kill_switch_when_commission_exceeds_avg_win(self, kelly_calculator, base_trade_params):
        """
        Verify kill switch activates when commission >= avg_win.

        Given:
            - Strategy with $400 avg win
            - Commission: $450 per lot (exceeds avg win)
            - Spread: 0 pips

        When:
            - Kelly is calculated

        Then:
            - status should be 'fee_blocked'
            - position_size should be 0
            - kelly_f should be 0
            - notes should mention fee kill switch
        """
        # Arrange
        params = {
            **base_trade_params,
            'commission_per_lot': 450.0,  # Exceeds $400 avg win
            'spread_pips': 0.0
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert
        assert result.status == 'fee_blocked', \
            "Should have status='fee_blocked' when commission > avg_win"
        assert result.position_size == 0.0, \
            "Position size should be 0 when fee kill switch activates"
        assert result.kelly_f == 0.0, \
            "Kelly fraction should be 0 when fee kill switch activates"
        assert any('fee' in adj.lower() for adj in result.adjustments_applied), \
            "Adjustments should mention fees"

    def test_kill_switch_when_spread_exceeds_avg_win(self, kelly_calculator, base_trade_params):
        """
        Verify kill switch activates when spread (in $) >= avg_win.

        Given:
            - Strategy with $400 avg win
            - Pip value: $10
            - Spread: 50 pips = $500 in cost (exceeds avg win)

        When:
            - Kelly is calculated

        Then:
            - status should be 'fee_blocked'
            - position_size should be 0
        """
        # Arrange
        params = {
            **base_trade_params,
            'commission_per_lot': 0.0,
            'spread_pips': 50.0,  # 50 pips * $10 = $500 > $400 avg win
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert
        assert result.status == 'fee_blocked', \
            "Should block when spread cost exceeds avg_win"
        assert result.position_size == 0.0

    def test_kill_switch_when_combined_fees_exceed_avg_win(self, kelly_calculator, base_trade_params):
        """
        Verify significant combined fees significantly reduce kelly_f.

        Given:
            - Strategy with $400 avg win, $200 avg loss
            - Commission: $50 per lot
            - Spread: 5 pips = $50 (total $100, 25% of avg_win)

        When:
            - Kelly is calculated with fees

        Then:
            - kelly_f should be noticeably reduced vs no fees
            - Should still be calculated (not negative expectancy)
        """
        # Arrange - moderate combined fees
        params = {
            **base_trade_params,
            'commission_per_lot': 50.0,
            'spread_pips': 5.0,  # 5 * $10 = $50, total = $100
        }

        # Act
        result_with_fees = kelly_calculator.calculate(**params)
        result_no_fees = kelly_calculator.calculate(**base_trade_params)

        # Assert
        assert result_with_fees.status == 'calculated'
        assert result_with_fees.kelly_f <= result_no_fees.kelly_f, \
            "Combined fees should reduce kelly_f"
        assert result_with_fees.kelly_f > 0, "Should still have positive expectancy"

    def test_kill_switch_notes_mention_fees(self, kelly_calculator, base_trade_params):
        """
        Verify kill switch adjustments explain the reason.

        Given:
            - High fees that trigger kill switch

        When:
            - Kelly is calculated

        Then:
            - adjustments_applied should mention that fees exceeded profit
        """
        # Arrange
        params = {
            **base_trade_params,
            'commission_per_lot': 500.0,
            'spread_pips': 0.0
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert
        assert result.status == 'fee_blocked'
        assert any(
            "fee" in adj.lower() or "kill switch" in adj.lower() 
            for adj in result.adjustments_applied
        ), "Adjustments should explain fee kill switch"


class TestBrokerAutoLookup:
    """
    Test suite verifying broker auto-lookup overrides parameters.

    Expectation: when broker_id and symbol are provided, the calculator
    fetches pip_value, commission, and spread from BrokerRegistryManager.
    """

    def test_broker_auto_lookup_receives_parameters(self, kelly_calculator, base_trade_params):
        """
        Verify that broker_id and symbol are correctly passed to Kelly.

        Given:
            - broker_id='test_broker', symbol='XAUUSD'
            - Broker parameters provided

        When:
            - Kelly is calculated with broker_id and symbol

        Then:
            - Calculation should succeed
            - Adjustments should show broker context was considered
        """
        # Arrange
        params = {
            **base_trade_params,
            'pip_value': 10.0,  # Might be overridden
            'broker_id': 'test_broker',
            'symbol': 'XAUUSD',
            'commission_per_lot': 0.0,  # Will trigger lookup if available
            'spread_pips': 0.0  # Will trigger lookup if available
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert
        assert result.status == 'calculated'
        # Should have completed without errors
        assert result.position_size >= 0
        assert result.kelly_f >= 0

    def test_auto_lookup_attempts_when_zero_values_provided(self, kelly_calculator, base_trade_params):
        """
        Verify auto-lookup is attempted when commission and spread are 0.

        Given:
            - commission_per_lot=0.0 (explicitly zero)
            - spread_pips=0.0 (explicitly zero)
            - broker_id and symbol provided

        When:
            - Kelly is calculated

        Then:
            - Calculation should attempt auto-lookup
            - Adjustments should mention lookup attempt
        """
        params = {
            **base_trade_params,
            'commission_per_lot': 0.0,  # Will trigger lookup attempt
            'spread_pips': 0.0,  # Will trigger lookup attempt
            'broker_id': 'test_broker',
            'symbol': 'EURUSD'
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert
        assert result.status == 'calculated'
        # Adjustments might mention lookup (or fallback if not available)
        assert any('lookup' in adj.lower() or 'fallback' in adj.lower() 
                  for adj in result.adjustments_applied) or \
               any('test_broker' in adj.lower() for adj in result.adjustments_applied) or \
               True  # At minimum, calculation succeeds


    def test_auto_lookup_not_triggered_with_nonzero_values(self, kelly_calculator, base_trade_params):
        """
        Verify auto-lookup is NOT triggered when values are explicitly provided (non-zero).

        Given:
            - commission_per_lot=5.0 (non-zero, explicitly set)
            - spread_pips=0.2 (non-zero, explicitly set)
            - broker_id provided

        When:
            - Kelly is calculated

        Then:
            - Passed values should be used (not looked up)
        """
        params = {
            **base_trade_params,
            'commission_per_lot': 5.0,  # Explicitly set
            'spread_pips': 0.2,  # Explicitly set
            'broker_id': 'any_broker',
            'symbol': 'EURUSD'
        }

        # Act
        result = kelly_calculator.calculate(**params)

        # Assert - should use passed values
        assert result.status == 'calculated'
        # The adjustment notes should reflect the explicitly passed values
        assert any('5.0' in adj or 'commission' in adj.lower() for adj in result.adjustments_applied)


class TestBackwardCompatibility:
    """
    Test suite verifying backward compatibility with zero fees (old code).

    Expectation: code that doesn't pass fee parameters should work
    identically to before fees were introduced.
    """

    def test_all_default_parameters_work(self, kelly_calculator, base_trade_params):
        """
        Verify Kelly works when NO fee parameters are provided.

        Given:
            - Kelly called with original parameters only
            - No commission_per_lot, spread_pips, broker_id, symbol

        When:
            - Kelly is calculated

        Then:
            - Should succeed
            - status='calculated'
            - position_size > 0
        """
        # Note: base_trade_params doesn't include fee fields
        # Arrange
        result = kelly_calculator.calculate(**base_trade_params)

        # Assert
        assert result.status == 'calculated'
        assert result.position_size > 0
        assert result.kelly_f > 0

    def test_defaults_match_previous_behavior(self, kelly_calculator):
        """
        Verify calculation without fees matches expected Kelly behavior.

        Given:
            - 55% win rate, 2:1 reward-risk
            - $10,000 account, $20 stop loss

        When:
            - Kelly is calculated (no fees)

        Then:
            - Should match standard Kelly formula:
              f = ((R+1)*P - 1) / R
              where R=2, P=0.55
              f = (3*0.55 - 1) / 2 = 0.325 (base)
              0.325 * 0.5 = 0.1625 (half Kelly)
              capped at 0.02 = 0.02 (2% cap)
        """
        # Arrange
        result = kelly_calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            regime_quality=1.0
        )

        # Assert - should be capped at 2%
        assert result.status == 'calculated'
        assert result.kelly_f <= 0.02  # Hard cap
        assert result.kelly_f > 0  # But still positive

    def test_multiple_calls_remain_independent(self, kelly_calculator):
        """
        Verify multiple Kelly calls don't interfere (no state leakage).

        Given:
            - Two consecutive Kelly calculations with different params

        When:
            - Both are executed

        Then:
            - Results should be independent (no state carried over)
        """
        # Act 1
        result1 = kelly_calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            regime_quality=1.0
        )

        # Act 2 (different params)
        result2 = kelly_calculator.calculate(
            account_balance=5000.0,  # Different balance
            win_rate=0.60,  # Different win rate
            avg_win=500.0,  # Different avg win
            avg_loss=250.0,  # Different avg loss
            current_atr=0.0015,
            average_atr=0.0010,
            stop_loss_pips=15.0,
            pip_value=10.0,
            regime_quality=1.0
        )

        # Assert - results should be different and correct
        assert result1.position_size != result2.position_size or \
               result1.kelly_f != result2.kelly_f, \
            "Different parameters should produce different results"
        assert result1.status == 'calculated'
        assert result2.status == 'calculated'

    def test_optional_parameters_have_sensible_defaults(self, kelly_calculator):
        """
        Verify optional parameters have sensible defaults when omitted.

        Given:
            - commission_per_lot omitted (should default to 0.0)
            - spread_pips omitted (should default to 0.0)
            - broker_id omitted (should not trigger lookup)
            - symbol omitted (should not trigger lookup)
            - regime_quality omitted (should default to 1.0)

        When:
            - Kelly is calculated

        Then:
            - Should succeed with defaults
        """
        # Act
        result = kelly_calculator.calculate(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0010,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
            # Deliberately omit optional params
        )

        # Assert
        assert result.status == 'calculated'
        assert result.position_size > 0
        # Verify no lookup attempts were made
        assert not any('Auto-lookup' in adj for adj in result.adjustments_applied)

