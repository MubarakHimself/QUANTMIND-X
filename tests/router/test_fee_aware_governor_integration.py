# Tests Router Package

import pytest
from src.router.enhanced_governor import EnhancedGovernor
from src.router.governor import Governor, RiskMandate


class TestEnhancedGovernorWithFees:
    """Test suite for EnhancedGovernor fee-aware Kelly integration."""

    @pytest.fixture
    def enhanced_governor(self):
        """Fixture providing EnhancedGovernor instance."""
        return EnhancedGovernor(account_id="test_account")

    @pytest.fixture
    def mock_regime_report(self):
        """Fixture providing mock regime report."""
        class MockRegimeReport:
            def __init__(self):
                self.regime = "TREND_STABLE"
                self.regime_quality = 0.85
                self.chaos_score = 0.2
                self.is_systemic_risk = False
        return MockRegimeReport()

    def test_enhanced_governor_accepts_account_balance_and_broker_id(self, enhanced_governor, mock_regime_report):
        """
        Verify that EnhancedGovernor.calculate_risk() accepts account_balance and broker_id.

        Given:
            - EnhancedGovernor instance
            - Mock regime report

        When:
            - calculate_risk() is called with account_balance and broker_id

        Then:
            - Method should execute without error
            - Return type should be RiskMandate
        """
        # Arrange
        trade_proposal = {
            'symbol': 'XAUUSD',
            'current_balance': 50000.0,
            'win_rate': 0.60,
            'avg_win': 500.0,
            'avg_loss': 250.0,
            'stop_loss_pips': 30.0,
            'current_atr': 0.0020,
            'average_atr': 0.0015,
        }

        # Act & Assert
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=trade_proposal,
            account_balance=50000.0,
            broker_id='icmarkets_raw'
        )

        assert isinstance(mandate, RiskMandate)
        assert mandate.risk_mode in ["STANDARD", "CLAMPED", "HALTED"]

    def test_governor_returns_extended_riskmandate_fields(self, enhanced_governor, mock_regime_report):
        """
        Verify that Governor returns RiskMandate with Kelly fields populated.

        Given:
            - EnhancedGovernor with Kelly Calculator
            - Valid trade proposal

        When:
            - calculate_risk() is called

        Then:
            - RiskMandate should have position_size, kelly_fraction, risk_amount, kelly_adjustments
        """
        # Arrange
        trade_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 10000.0,
            'account_balance': 10000.0,
            'broker_id': 'mt5_default',
            'win_rate': 0.55,
            'avg_win': 400.0,
            'avg_loss': 200.0,
            'stop_loss_pips': 20.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
        }

        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=trade_proposal,
            account_balance=10000.0,
            broker_id='mt5_default'
        )

        # Assert - Extended fields should be present
        assert hasattr(mandate, 'position_size'), "RiskMandate should have position_size field"
        assert hasattr(mandate, 'kelly_fraction'), "RiskMandate should have kelly_fraction field"
        assert hasattr(mandate, 'risk_amount'), "RiskMandate should have risk_amount field"
        assert hasattr(mandate, 'kelly_adjustments'), "RiskMandate should have kelly_adjustments field"

    def test_fee_kill_switch_returns_halted_mandate(self, enhanced_governor, mock_regime_report):
        """
        Verify that fee kill switch returns HALTED mandate.

        Given:
            - EnhancedGovernor with Kelly Calculator
            - Trade proposal where fees exceed avg win

        When:
            - calculate_risk() is called

        Then:
            - Mandate should have risk_mode="HALTED"
            - Position size should be 0
            - Note should mention fee kill switch
        """
        # Arrange - Trade with excessive fees
        trade_proposal = {
            'symbol': 'EURUSD',
            'current_balance': 10000.0,
            'account_balance': 10000.0,
            'broker_id': 'test_broker',
            'win_rate': 0.55,
            'avg_win': 10.0,  # Very small avg win
            'avg_loss': 200.0,
            'stop_loss_pips': 20.0,
            'current_atr': 0.0012,
            'average_atr': 0.0010,
        }

        # Act
        mandate = enhanced_governor.calculate_risk(
            regime_report=mock_regime_report,
            trade_proposal=trade_proposal,
            account_balance=10000.0,
            broker_id='test_broker'
        )

        # Assert - Fee kill switch should trigger HALT
        assert mandate.risk_mode == "HALTED", "Should halt when fees > profit"
        assert mandate.position_size == 0.0, "Position size should be zero"
        assert "fee kill switch" in mandate.notes.lower() or "kill switch" in mandate.notes.lower()


class TestFeeAwareTradingE2E:
    """End-to-end tests for fee-aware trading system."""

    @pytest.fixture
    def router_with_governor(self):
        """Fixture providing StrategyRouter with EnhancedGovernor."""
        from src.router.engine import StrategyRouter
        router = StrategyRouter(use_kelly_governor=True)
        # Inject governor for testing
        router.commander._governor = router.governor
        return router

    @pytest.fixture
    def mock_account_data(self):
        """Fixture providing mock account data."""
        return {
            'account_balance': 10000.0,
            'broker_id': 'icmarkets_raw'
        }

    def test_commander_auction_with_fee_aware_kelly(self, router_with_governor, mock_account_data):
        """
        Full flow test: Commander auction with fee-aware Kelly position sizing.

        Given:
            - StrategyRouter with EnhancedGovernor
            - Mock account data with balance and broker_id
            - Mock regime report

        When:
            - process_tick() is called

        Then:
            - Commander should receive account_balance and broker_id
            - Bots should have position_size, kelly_fraction, risk_amount
            - Only bots with positive position_size should be returned
        """
        # Arrange
        from src.router.sentinel import Sentinel, RegimeReport

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            regime_quality=0.9,
            chaos_score=0.2,
            is_systemic_risk=False
        )

        # Act
        result = router_with_governor.process_tick(
            symbol='EURUSD',
            price=1.1000,
            account_data=self.mock_account_data()
        )

        # Assert
        assert 'regime' in result
        assert 'dispatches' in result
        assert isinstance(result['dispatches'], list)

        # Verify position sizing fields in dispatches
        for dispatch in result['dispatches']:
            if 'position_size' in dispatch:
                assert dispatch['position_size'] >= 0, "Position size should be non-negative"
                assert dispatch['kelly_fraction'] >= 0, "Kelly fraction should be non-negative"
                assert dispatch['risk_amount'] >= 0, "Risk amount should be non-negative"

    def test_engine_extracts_account_balance_and_broker_id(self, router_with_governor, mock_account_data):
        """
        Verify that StrategyRouter extracts account_balance and broker_id from account_data.

        Given:
            - StrategyRouter with EnhancedGovernor
            - Mock account data with both fields

        When:
            - process_tick() is called

        Then:
            - Router should extract both fields correctly
            - Both should be passed to Commander
        """
        # Arrange
        from src.router.sentinel import RegimeReport
        account_data = {
            'account_balance': 50000.0,
            'broker_id': 'mt5_default'
        }
        regime_report = RegimeReport(
            regime="TREND_STABLE",
            regime_quality=0.9,
            chaos_score=0.2,
            is_systemic_risk=False
        )

        # Act
        # Mock the commander to verify what it receives
        from unittest.mock import MagicMock
        router_with_governor.commander = MagicMock()

        # Spy on run_auction to capture arguments
        router_with_governor.commander.run_auction = MagicMock(return_value=[])

        result = router_with_governor.process_tick(
            symbol='XAUUSD',
            price=1950.0,
            account_data=account_data
        )

        # Assert
        # Verify run_auction was called with correct arguments
        router_with_governor.commander.run_auction.assert_called_once()
        call_args = router_with_governor.commander.run_auction.call_args

        # First positional arg should be regime_report
        assert call_args[0][0] == regime_report  # or equivalent

        # Second positional arg should be account_balance
        assert call_args[0][1] == 50000.0

        # Third positional arg should be broker_id
        assert call_args[0][2] == 'mt5_default'

    def test_backtesting_with_broker_id(self):
        """
        Verify that backtesting functions accept broker_id parameter.

        Given:
            - Backtest runner functions

        When:
            - Functions are called

        Then:
            - broker_id parameter should be accepted
            - Default value should be 'icmarkets_raw'
        """
        # Verify function signatures include broker_id
        from src.backtesting.mode_runner import (
            run_vanilla_backtest,
            run_spiced_backtest,
            run_full_system_backtest,
            run_multi_symbol_backtest
        )
        import inspect

        # Check run_vanilla_backtest
        sig = inspect.signature(run_vanilla_backtest)
        assert 'broker_id' in sig.parameters, "run_vanilla_backtest should accept broker_id"
        assert sig.parameters['broker_id'].default == 'icmarkets_raw'

        # Check run_spiced_backtest
        sig = inspect.signature(run_spiced_backtest)
        assert 'broker_id' in sig.parameters, "run_spiced_backtest should accept broker_id"
        assert sig.parameters['broker_id'].default == 'icmarkets_raw'

        # Check run_full_system_backtest
        sig = inspect.signature(run_full_system_backtest)
        assert 'broker_id' in sig.parameters, "run_full_system_backtest should accept broker_id"
        assert sig.parameters['broker_id'].default == 'icmarkets_raw'

        # Check run_multi_symbol_backtest
        sig = inspect.signature(run_multi_symbol_backtest)
        assert 'broker_id' in sig.parameters, "run_multi_symbol_backtest should accept broker_id"
        assert sig.parameters['broker_id'].default == 'icmarkets_raw'
