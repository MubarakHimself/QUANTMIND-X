"""
Tests for EA Backtest Integration

Task Group 4: Connect EA variants to all backtest modes

This test module validates:
- EA variants (vanilla, spiced) connected to all backtest modes
- Normal backtest mode accepts both vanilla and spiced EA variants
- Monte accepts both vanilla and spiced EA variants
- Walk Forward Carlo backtest mode backtest mode accepts both vanilla and spiced EA variants
- BACKTEST_REQUEST message type in department mail system
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from backtesting.mode_runner import (
    EAVariant,
    BacktestType,
    EABacktestRequest,
    EABacktestResult,
    run_ea_backtest,
    run_ea_backtest_all_modes,
    BacktestMode
)
from agents.departments.department_mail import MessageType, DepartmentMailService, Priority


@pytest.mark.unit
class TestEAVariants:
    """Test EA variant enum values."""

    def test_ea_variant_vanilla_value(self):
        """Test vanilla EA variant has correct value."""
        assert EAVariant.VANILLA.value == "vanilla"

    def test_ea_variant_spiced_value(self):
        """Test spiced EA variant has correct value."""
        assert EAVariant.SPICED.value == "spiced"


@pytest.mark.unit
class TestBacktestTypes:
    """Test backtest type enum values."""

    def test_backtest_type_normal_value(self):
        """Test normal backtest type has correct value."""
        assert BacktestType.NORMAL.value == "normal"

    def test_backtest_type_monte_carlo_value(self):
        """Test monte carlo backtest type has correct value."""
        assert BacktestType.MONTE_CARLO.value == "monte_carlo"

    def test_backtest_type_walk_forward_value(self):
        """Test walk forward backtest type has correct value."""
        assert BacktestType.WALK_FORWARD.value == "walk_forward"


@pytest.mark.unit
class TestEABacktestRequest:
    """Test EA backtest request dataclass."""

    def test_create_ea_backtest_request_vanilla_normal(self):
        """Test creating EA backtest request for vanilla + normal."""
        request = EABacktestRequest(
            ea_variant=EAVariant.VANILLA,
            backtest_type=BacktestType.NORMAL,
            symbol="EURUSD",
            timeframe=1,  # PERIOD_H1
            strategy_code="def on_bar(tester): pass"
        )

        assert request.ea_variant == EAVariant.VANILLA
        assert request.backtest_type == BacktestType.NORMAL
        assert request.symbol == "EURUSD"
        assert request.mc_simulations == 1000  # default
        assert request.wf_train_pct == 0.5  # default

    def test_create_ea_backtest_request_spiced_monte_carlo(self):
        """Test creating EA backtest request for spiced + monte carlo."""
        request = EABacktestRequest(
            ea_variant=EAVariant.SPICED,
            backtest_type=BacktestType.MONTE_CARLO,
            symbol="XAUUSD",
            timeframe=4,  # PERIOD_H4
            strategy_code="def on_bar(tester): pass",
            mc_simulations=500
        )

        assert request.ea_variant == EAVariant.SPICED
        assert request.backtest_type == BacktestType.MONTE_CARLO
        assert request.mc_simulations == 500

    def test_create_ea_backtest_request_spiced_walk_forward(self):
        """Test creating EA backtest request for spiced + walk forward."""
        request = EABacktestRequest(
            ea_variant=EAVariant.SPICED,
            backtest_type=BacktestType.WALK_FORWARD,
            symbol="GBPUSD",
            timeframe=2,  # PERIOD_H2
            strategy_code="def on_bar(tester): pass",
            wf_train_pct=0.6,
            wf_test_pct=0.2,
            wf_gap_pct=0.1
        )

        assert request.ea_variant == EAVariant.SPICED
        assert request.backtest_type == BacktestType.WALK_FORWARD
        assert request.wf_train_pct == 0.6


@pytest.mark.unit
class TestEABacktestResult:
    """Test EA backtest result dataclass."""

    def test_create_ea_backtest_result_success(self):
        """Test creating successful EA backtest result."""
        mock_result = Mock()
        result = EABacktestResult(
            ea_variant="vanilla",
            backtest_type="normal",
            result=mock_result,
            success=True
        )

        assert result.ea_variant == "vanilla"
        assert result.backtest_type == "normal"
        assert result.success is True
        assert result.error is None

    def test_create_ea_backtest_result_failure(self):
        """Test creating failed EA backtest result."""
        result = EABacktestResult(
            ea_variant="spiced",
            backtest_type="monte_carlo",
            result=None,
            success=False,
            error="Insufficient data"
        )

        assert result.ea_variant == "spiced"
        assert result.success is False
        assert result.error == "Insufficient data"


@pytest.mark.unit
class TestEAVariantsConnectedToBacktestModes:
    """Test that EA variants are connected to all backtest modes (Task 4)."""

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_vanilla_connected_to_normal_backtest(self, mock_wf, mock_mc):
        """Test vanilla EA variant connected to normal backtest mode."""
        # Setup mock - patch the internal functions
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            mock_run.return_value = Mock(sharpe=1.5, return_pct=10.0, drawdown=5.0, trades=10)

            request = EABacktestRequest(
                ea_variant=EAVariant.VANILLA,
                backtest_type=BacktestType.NORMAL,
                symbol="EURUSD",
                timeframe=1,
                strategy_code="def on_bar(tester): pass"
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "vanilla"
            assert result.backtest_type == "normal"
            assert result.success is True
            mock_run.assert_called_once()

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_spiced_connected_to_normal_backtest(self, mock_wf, mock_mc):
        """Test spiced EA variant connected to normal backtest mode."""
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            mock_run.return_value = Mock(sharpe=1.8, return_pct=12.0, drawdown=4.0, trades=8)

            request = EABacktestRequest(
                ea_variant=EAVariant.SPICED,
                backtest_type=BacktestType.NORMAL,
                symbol="EURUSD",
                timeframe=1,
                strategy_code="def on_bar(tester): pass"
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "spiced"
            assert result.backtest_type == "normal"
            assert result.success is True
            mock_run.assert_called_once()

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_vanilla_connected_to_monte_carlo_backtest(self, mock_wf, mock_mc):
        """Test vanilla EA variant connected to Monte Carlo backtest mode."""
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            # Setup mocks
            mock_base_result = Mock(sharpe=1.5, return_pct=10.0, trades=[Mock() for _ in range(10)])
            mock_run.return_value = mock_base_result

            mock_mc_instance = Mock()
            mock_mc_result = Mock(
                num_simulations=1000,
                mean_return=10.5,
                confidence_interval_95th=15.0
            )
            mock_mc_instance.simulate.return_value = mock_mc_result
            mock_mc.return_value = mock_mc_instance

            request = EABacktestRequest(
                ea_variant=EAVariant.VANILLA,
                backtest_type=BacktestType.MONTE_CARLO,
                symbol="EURUSD",
                timeframe=1,
                strategy_code="def on_bar(tester): pass",
                mc_simulations=1000
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "vanilla"
            assert result.backtest_type == "monte_carlo"
            assert result.success is True
            mock_run.assert_called_once()
            mock_mc_instance.simulate.assert_called_once()

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_spiced_connected_to_monte_carlo_backtest(self, mock_wf, mock_mc):
        """Test spiced EA variant connected to Monte Carlo backtest mode."""
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            # Setup mocks
            mock_base_result = Mock(sharpe=1.8, return_pct=12.0, trades=[Mock() for _ in range(8)])
            mock_run.return_value = mock_base_result

            mock_mc_instance = Mock()
            mock_mc_result = Mock(
                num_simulations=1000,
                mean_return=12.5,
                confidence_interval_95th=18.0
            )
            mock_mc_instance.simulate.return_value = mock_mc_result
            mock_mc.return_value = mock_mc_instance

            request = EABacktestRequest(
                ea_variant=EAVariant.SPICED,
                backtest_type=BacktestType.MONTE_CARLO,
                symbol="XAUUSD",
                timeframe=4,
                strategy_code="def on_bar(tester): pass",
                mc_simulations=1000
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "spiced"
            assert result.backtest_type == "monte_carlo"
            assert result.success is True

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_vanilla_connected_to_walk_forward_backtest(self, mock_wf, mock_mc):
        """Test vanilla EA variant connected to Walk Forward backtest mode."""
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            # Setup mock
            mock_wf_instance = Mock()
            mock_wf_result = Mock(
                total_windows=5,
                win_rate=0.6,
                aggregate_metrics={'sharpe_mean': 1.5, 'return_pct_mean': 10.0}
            )
            mock_wf_instance.optimize.return_value = mock_wf_result
            mock_wf.return_value = mock_wf_instance

            request = EABacktestRequest(
                ea_variant=EAVariant.VANILLA,
                backtest_type=BacktestType.WALK_FORWARD,
                symbol="EURUSD",
                timeframe=1,
                strategy_code="def on_bar(tester): pass"
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "vanilla"
            assert result.backtest_type == "walk_forward"
            assert result.success is True
            mock_wf_instance.optimize.assert_called_once()

    @patch('backtesting.monte_carlo.MonteCarloSimulator')
    @patch('backtesting.walk_forward.WalkForwardOptimizer')
    def test_spiced_connected_to_walk_forward_backtest(self, mock_wf, mock_mc):
        """Test spiced EA variant connected to Walk Forward backtest mode."""
        with patch('backtesting.mode_runner.run_full_system_backtest') as mock_run:
            # Setup mock
            mock_wf_instance = Mock()
            mock_wf_result = Mock(
                total_windows=5,
                win_rate=0.7,
                aggregate_metrics={'sharpe_mean': 1.8, 'return_pct_mean': 12.0}
            )
            mock_wf_instance.optimize.return_value = mock_wf_result
            mock_wf.return_value = mock_wf_instance

            request = EABacktestRequest(
                ea_variant=EAVariant.SPICED,
                backtest_type=BacktestType.WALK_FORWARD,
                symbol="XAUUSD",
                timeframe=4,
                strategy_code="def on_bar(tester): pass"
            )

            result = run_ea_backtest(request)

            assert result.ea_variant == "spiced"
            assert result.backtest_type == "walk_forward"
            assert result.success is True


@pytest.mark.unit
class TestRunEABacktestAllModes:
    """Test run_ea_backtest_all_modes convenience function."""

    @patch('backtesting.mode_runner.run_ea_backtest')
    def test_run_all_modes_with_default_params(self, mock_run_ea):
        """Test running all EA variants with all backtest types."""
        # Setup mock to return success
        mock_run_ea.return_value = EABacktestResult(
            ea_variant="vanilla",
            backtest_type="normal",
            result=Mock(),
            success=True
        )

        results = run_ea_backtest_all_modes(
            strategy_code="def on_bar(tester): pass",
            symbol="EURUSD",
            timeframe=1
        )

        # Should have 6 combinations (2 variants x 3 types)
        assert len(results) == 6
        assert "vanilla_normal" in results
        assert "vanilla_monte_carlo" in results
        assert "vanilla_walk_forward" in results
        assert "spiced_normal" in results
        assert "spiced_monte_carlo" in results
        assert "spiced_walk_forward" in results

        # Should have called run_ea_backtest 6 times
        assert mock_run_ea.call_count == 6

    @patch('backtesting.mode_runner.run_ea_backtest')
    def test_run_specific_variants_only(self, mock_run_ea):
        """Test running only specific EA variants."""
        mock_run_ea.return_value = EABacktestResult(
            ea_variant="vanilla",
            backtest_type="normal",
            result=Mock(),
            success=True
        )

        results = run_ea_backtest_all_modes(
            strategy_code="def on_bar(tester): pass",
            symbol="EURUSD",
            timeframe=1,
            ea_variants=[EAVariant.VANILLA],
            backtest_types=[BacktestType.NORMAL]
        )

        assert len(results) == 1
        assert "vanilla_normal" in results


@pytest.mark.unit
class TestBACKTESTREQUESTMessageType:
    """Test BACKTEST_REQUEST message type in department mail (Task 5)."""

    def test_backtest_request_message_type_exists(self):
        """Test BACKTEST_REQUEST message type is defined."""
        assert hasattr(MessageType, 'BACKTEST_REQUEST')
        assert MessageType.BACKTEST_REQUEST.value == "backtest_request"

    def test_message_type_values(self):
        """Test all message type values are correct."""
        assert MessageType.STATUS.value == "status"
        assert MessageType.QUESTION.value == "question"
        assert MessageType.RESULT.value == "result"
        assert MessageType.ERROR.value == "error"
        assert MessageType.DISPATCH.value == "dispatch"
        assert MessageType.STRATEGY_DISPATCH.value == "strategy_dispatch"
        assert MessageType.BACKTEST_REQUEST.value == "backtest_request"

    def test_can_create_message_with_backtest_request_type(self):
        """Test creating a message with BACKTEST_REQUEST type."""
        from datetime import datetime, timezone
        from agents.departments.department_mail import DepartmentMessage

        message = DepartmentMessage(
            id="test-123",
            from_dept="trading",
            to_dept="backtesting",
            type=MessageType.BACKTEST_REQUEST,
            subject="Run EURUSD backtest",
            body='{"ea_variant": "vanilla", "backtest_type": "normal"}',
            priority=Priority.HIGH,
            timestamp=datetime.now(timezone.utc),
            read=False
        )

        assert message.type == MessageType.BACKTEST_REQUEST
        assert message.from_dept == "trading"
        assert message.to_dept == "backtesting"


@pytest.mark.unit
class TestDepartmentMailWithBacktestRequest:
    """Test department mail service with BACKTEST_REQUEST messages."""

    def test_send_backtest_request_message(self, tmp_path):
        """Test sending a BACKTEST_REQUEST message."""
        db_path = tmp_path / "test_mail.db"
        mail_service = DepartmentMailService(db_path=str(db_path))

        try:
            message = mail_service.send(
                from_dept="trading",
                to_dept="backtesting",
                type=MessageType.BACKTEST_REQUEST,
                subject="Run EURUSD backtest",
                body='{"ea_variant": "spiced", "backtest_type": "monte_carlo", "symbol": "EURUSD"}',
                priority=Priority.HIGH
            )

            assert message.type == MessageType.BACKTEST_REQUEST
            assert message.subject == "Run EURUSD backtest"
            assert "spiced" in message.body
        finally:
            mail_service.close()

    def test_check_inbox_for_backtest_requests(self, tmp_path):
        """Test checking inbox for BACKTEST_REQUEST messages."""
        db_path = tmp_path / "test_mail.db"
        mail_service = DepartmentMailService(db_path=str(db_path))

        try:
            # Send backtest request
            mail_service.send(
                from_dept="trading",
                to_dept="backtesting",
                type=MessageType.BACKTEST_REQUEST,
                subject="Run EURUSD backtest",
                body='{"ea_variant": "vanilla", "backtest_type": "normal"}',
                priority=Priority.NORMAL
            )

            # Check inbox
            inbox = mail_service.check_inbox(
                dept="backtesting",
                unread_only=True,
                limit=10
            )

            assert len(inbox) == 1
            assert inbox[0].type == MessageType.BACKTEST_REQUEST
        finally:
            mail_service.close()


# --- Fixtures ---

@pytest.fixture
def sample_backtest_data():
    """Create sample backtest data for testing."""
    return pd.DataFrame({
        'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
        'open': np.linspace(1.1000, 1.1100, 100),
        'high': np.linspace(1.1050, 1.1150, 100),
        'low': np.linspace(1.0950, 1.1050, 100),
        'close': np.linspace(1.1000, 1.1100, 100),
        'tick_volume': np.random.randint(1000, 3000, 100)
    })
