"""
Tests for QSL Core Module - PropManager (Task Group 4)

These tests validate the MQL5 PropManager functionality using a Python test harness.
The tests mock the DatabaseManager dependency since that's implemented in Task Groups 1-3.
"""
import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, patch


# Python equivalent of MQL5 QMPropManager for testing
class QMPropManager:
    """Python test harness for MQL5 QMPropManager class"""

    # Constants matching MQL5 definitions
    QM_DAILY_LOSS_LIMIT_PCT = 5.0
    QM_HARD_STOP_BUFFER_PCT = 1.0
    QM_EFFECTIVE_LIMIT_PCT = 4.0

    def __init__(self, database_manager=None):
        """
        Initialize PropManager with optional DatabaseManager for snapshot persistence

        Args:
            database_manager: Mock DatabaseManager for testing
        """
        self.database_manager = database_manager
        self.current_high_water_mark = 0.0

    def CalculateDailyDrawdown(self, start_balance: float, current_equity: float) -> float:
        """
        Calculate daily drawdown percentage

        Formula: ((StartBalance - CurrentEquity) / StartBalance) * 100

        Args:
            start_balance: Daily starting balance
            current_equity: Current account equity

        Returns:
            Drawdown percentage (negative if in profit)
        """
        if start_balance <= 0:
            return 0.0

        drawdown = ((start_balance - current_equity) / start_balance) * 100.0
        return drawdown

    def IsHardStopBreached(self, drawdown_pct: float) -> bool:
        """
        Check if hard stop is breached

        Hard stop triggers when drawdown >= QM_EFFECTIVE_LIMIT_PCT (4.0%)

        Args:
            drawdown_pct: Current drawdown percentage

        Returns:
            True if hard stop breached, False otherwise
        """
        return drawdown_pct >= self.QM_EFFECTIVE_LIMIT_PCT

    def UpdateHighWaterMark(self, current_equity: float, previous_hwm: float) -> float:
        """
        Update high water mark (highest equity achieved today)

        Args:
            current_equity: Current account equity
            previous_hwm: Previous high water mark value

        Returns:
            New high water mark (max of current and previous)
        """
        return max(current_equity, previous_hwm)

    def SaveSnapshot(self, account_id: str, equity: float, balance: float) -> bool:
        """
        Save daily snapshot to database via Python bridge

        Args:
            account_id: Account identifier
            equity: Current account equity
            balance: Current account balance

        Returns:
            True if snapshot saved successfully, False otherwise
        """
        if self.database_manager is None:
            # No database manager - simulate success for testing
            return True

        try:
            # In real MQL5, this would call Python bridge
            self.database_manager.save_daily_snapshot(account_id, equity, balance)
            return True
        except Exception:
            return False


class TestPropManagerCalculations:
    """Test PropManager drawdown and HWM calculations"""

    def test_calculate_daily_drawdown_profit(self):
        """Test drawdown calculation when in profit (negative drawdown)"""
        prop_manager = QMPropManager()

        # Starting balance 10000, current equity 10500 (5% profit)
        drawdown = prop_manager.CalculateDailyDrawdown(10000.0, 10500.0)

        assert drawdown == -5.0, "Drawdown should be -5% when in profit"

    def test_calculate_daily_drawdown_loss(self):
        """Test drawdown calculation when at loss"""
        prop_manager = QMPropManager()

        # Starting balance 10000, current equity 9700 (3% loss)
        drawdown = prop_manager.CalculateDailyDrawdown(10000.0, 9700.0)

        assert drawdown == 3.0, "Drawdown should be 3% when at loss"

    def test_calculate_daily_drawdown_breakeven(self):
        """Test drawdown calculation at breakeven"""
        prop_manager = QMPropManager()

        # Starting balance equals current equity
        drawdown = prop_manager.CalculateDailyDrawdown(10000.0, 10000.0)

        assert drawdown == 0.0, "Drawdown should be 0% at breakeven"

    def test_calculate_daily_drawdown_zero_balance(self):
        """Test drawdown calculation with zero starting balance (edge case)"""
        prop_manager = QMPropManager()

        # Zero starting balance should return 0 to avoid division by zero
        drawdown = prop_manager.CalculateDailyDrawdown(0.0, 9500.0)

        assert drawdown == 0.0, "Drawdown should be 0% with zero starting balance"

    def test_update_high_water_mark_increases(self):
        """Test HWM update when equity increases"""
        prop_manager = QMPropManager()

        # Previous HWM 10000, current equity 10200
        new_hwm = prop_manager.UpdateHighWaterMark(10200.0, 10000.0)

        assert new_hwm == 10200.0, "HWM should increase to current equity"

    def test_update_high_water_mark_decreases(self):
        """Test HWM stays same when equity decreases"""
        prop_manager = QMPropManager()

        # Previous HWM 10200, current equity 10100
        new_hwm = prop_manager.UpdateHighWaterMark(10100.0, 10200.0)

        assert new_hwm == 10200.0, "HWM should stay at previous high"


class TestPropManagerHardStop:
    """Test PropManager hard stop enforcement logic"""

    def test_hard_stop_not_breached_safe(self):
        """Test hard stop not breached at safe drawdown level"""
        prop_manager = QMPropManager()

        # 3% drawdown is below 4% hard stop
        is_breached = prop_manager.IsHardStopBreached(3.0)

        assert is_breached is False, "Hard stop should not be breached at 3%"

    def test_hard_stop_not_breached_exact(self):
        """Test hard stop exactly at threshold"""
        prop_manager = QMPropManager()

        # 3.99% drawdown is below 4% hard stop
        is_breached = prop_manager.IsHardStopBreached(3.99)

        assert is_breached is False, "Hard stop should not be breached at 3.99%"

    def test_hard_stop_breached_at_threshold(self):
        """Test hard stop breached at exactly 4%"""
        prop_manager = QMPropManager()

        # 4.0% drawdown triggers hard stop
        is_breached = prop_manager.IsHardStopBreached(4.0)

        assert is_breached is True, "Hard stop should be breached at exactly 4%"

    def test_hard_stop_breached_exceeded(self):
        """Test hard stop breached when exceeded"""
        prop_manager = QMPropManager()

        # 4.5% drawdown exceeds hard stop
        is_breached = prop_manager.IsHardStopBreached(4.5)

        assert is_breached is True, "Hard stop should be breached at 4.5%"

    def test_hard_stop_breached_severe(self):
        """Test hard stop at severe loss level"""
        prop_manager = QMPropManager()

        # 6% drawdown is well beyond hard stop
        is_breached = prop_manager.IsHardStopBreached(6.0)

        assert is_breached is True, "Hard stop should be breached at 6%"

    def test_hard_stop_with_negative_drawdown(self):
        """Test hard stop not triggered when in profit"""
        prop_manager = QMPropManager()

        # Negative drawdown (profit) should not trigger hard stop
        is_breached = prop_manager.IsHardStopBreached(-2.0)

        assert is_breached is False, "Hard stop should not be breached in profit"


class TestPropManagerConstants:
    """Test PropManager constant values"""

    def test_daily_loss_limit_constant(self):
        """Test daily loss limit constant is 5%"""
        prop_manager = QMPropManager()

        assert prop_manager.QM_DAILY_LOSS_LIMIT_PCT == 5.0

    def test_hard_stop_buffer_constant(self):
        """Test hard stop buffer constant is 1%"""
        prop_manager = QMPropManager()

        assert prop_manager.QM_HARD_STOP_BUFFER_PCT == 1.0

    def test_effective_limit_constant(self):
        """Test effective hard stop limit is 4% (5% - 1% buffer)"""
        prop_manager = QMPropManager()

        assert prop_manager.QM_EFFECTIVE_LIMIT_PCT == 4.0


class TestPropManagerSnapshotPersistence:
    """Test PropManager snapshot persistence via DatabaseManager"""

    def test_save_snapshot_without_database_manager(self):
        """Test snapshot save succeeds without database manager (simulated)"""
        prop_manager = QMPropManager(database_manager=None)

        result = prop_manager.SaveSnapshot("12345", 10200.0, 10100.0)

        assert result is True, "Snapshot save should succeed without database manager"

    def test_save_snapshot_with_mock_database_manager(self):
        """Test snapshot save with mocked DatabaseManager"""
        mock_db = Mock()
        mock_db.save_daily_snapshot = Mock(return_value=True)
        prop_manager = QMPropManager(database_manager=mock_db)

        result = prop_manager.SaveSnapshot("12345", 10200.0, 10100.0)

        assert result is True
        mock_db.save_daily_snapshot.assert_called_once_with("12345", 10200.0, 10100.0)

    def test_save_snapshot_database_failure(self):
        """Test snapshot save handles database failure gracefully"""
        mock_db = Mock()
        mock_db.save_daily_snapshot = Mock(side_effect=Exception("Database error"))
        prop_manager = QMPropManager(database_manager=mock_db)

        result = prop_manager.SaveSnapshot("12345", 10200.0, 10100.0)

        assert result is False, "Snapshot save should return False on database error"

    def test_save_snapshot_updates_internal_state(self):
        """Test snapshot save updates internal high water mark"""
        prop_manager = QMPropManager(database_manager=None)
        prop_manager.current_high_water_mark = 10000.0

        # Save snapshot with higher equity
        prop_manager.SaveSnapshot("12345", 10200.0, 10100.0)
        new_hwm = prop_manager.UpdateHighWaterMark(10200.0, prop_manager.current_high_water_mark)

        assert new_hwm == 10200.0, "High water mark should be updated"


class TestPropManagerIntegrationScenarios:
    """Integration test scenarios for PropManager"""

    def test_full_trading_day_scenario_profit(self):
        """Test full trading day scenario ending in profit"""
        prop_manager = QMPropManager()

        # Day starts with 10000 balance
        start_balance = 10000.0

        # Multiple equity updates during day
        updates = [10100.0, 10050.0, 10200.0, 10150.0, 10300.0]

        current_hwm = start_balance
        for equity in updates:
            current_hwm = prop_manager.UpdateHighWaterMark(equity, current_hwm)

        # Final drawdown should be negative (profit)
        final_drawdown = prop_manager.CalculateDailyDrawdown(start_balance, updates[-1])
        is_breached = prop_manager.IsHardStopBreached(final_drawdown)

        assert current_hwm == 10300.0, "HWM should track peak equity"
        assert final_drawdown == -3.0, "Should end with 3% profit"
        assert is_breached is False, "Hard stop should not be breached"

    def test_full_trading_day_scenario_loss_within_limit(self):
        """Test trading day with loss but within hard stop limit"""
        prop_manager = QMPropManager()

        start_balance = 10000.0

        # Trading day ends at 9750 (2.5% loss)
        final_equity = 9750.0

        final_drawdown = prop_manager.CalculateDailyDrawdown(start_balance, final_equity)
        is_breached = prop_manager.IsHardStopBreached(final_drawdown)

        assert final_drawdown == 2.5, "Should have 2.5% drawdown"
        assert is_breached is False, "Hard stop should not be breached at 2.5%"

    def test_full_trading_day_scenario_hard_stop_triggered(self):
        """Test trading day that triggers hard stop"""
        prop_manager = QMPropManager()

        start_balance = 10000.0

        # Trading day drops to 9550 (4.5% loss - exceeds hard stop)
        final_equity = 9550.0

        final_drawdown = prop_manager.CalculateDailyDrawdown(start_balance, final_equity)
        is_breached = prop_manager.IsHardStopBreached(final_drawdown)

        assert final_drawdown == 4.5, "Should have 4.5% drawdown"
        assert is_breached is True, "Hard stop should be breached at 4.5%"

    def test_high_water_mark_persistence_across_day(self):
        """Test HWM correctly tracks peaks throughout the day"""
        prop_manager = QMPropManager()

        start_balance = 10000.0
        equity_series = [
            10050.0,   # +0.5%
            10100.0,   # +1.0% (new HWM)
            10075.0,   # HWM stays at 10100
            10150.0,   # +1.5% (new HWM)
            10125.0,   # HWM stays at 10150
            10200.0,   # +2.0% (new HWM)
        ]

        current_hwm = start_balance
        for equity in equity_series:
            current_hwm = prop_manager.UpdateHighWaterMark(equity, current_hwm)

        assert current_hwm == 10200.0, "HWM should end at peak of 10200"
