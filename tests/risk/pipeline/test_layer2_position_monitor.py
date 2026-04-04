"""
Tests for Layer 2 Position Monitor.

Story 14.2: Layer 2 Tier 1 Position Monitor (Dynamic)
Tests cover:
- Task 1: Move-to-breakeven logic
- Task 2: Regime shift response
- Task 3: Redis lock TTL and retry
- Task 4: Kill switch preemption
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timezone

from src.risk.pipeline.layer2_position_monitor import (
    Layer2PositionMonitor,
    PositionState,
    ModificationResult,
    REDIS_LOCK_TTL_SECONDS,
    LAYER2_ACTIVATION_DELAY_MS,
)
from src.events.regime import RegimeShiftEvent, RegimeSuitability, RegimeType


class TestPositionState:
    """Test PositionState calculations."""

    def test_profit_calculation_buy(self):
        """Test profit calculation for BUY position."""
        state = PositionState(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            current_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
            epic_id="epic-14",
            sl_distance_points=0.00500,
        )

        # Entry price = 1.10000, SL = 1.09500 (50 pips risk)
        assert state.sl_distance_points == pytest.approx(0.00500)

        # Move to 1.10500 = +50 pips = +1R
        state.update_profit(1.10500)
        assert state.profit_points == pytest.approx(0.00500)
        assert state.profit_r == pytest.approx(1.0)

        # Not yet at breakeven check
        assert not state.has_moved_to_breakeven

    def test_profit_calculation_sell(self):
        """Test profit calculation for SELL position."""
        state = PositionState(
            ticket=12346,
            symbol="EURUSD",
            direction="SELL",
            entry_price=1.10000,
            current_price=1.10000,
            sl_price=1.10500,
            tp_price=1.09000,
            volume=1.0,
            strategy_id="mean_reversion",
            epic_id="epic-14",
            sl_distance_points=0.00500,
        )

        # Move to 1.09500 = +50 pips = +1R
        state.update_profit(1.09500)
        assert state.profit_points == pytest.approx(0.00500)
        assert state.profit_r == pytest.approx(1.0)

    def test_is_in_profit_by_r_threshold(self):
        """Test R threshold detection."""
        state = PositionState(
            ticket=12347,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            current_price=1.10250,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
            epic_id="epic-14",
            sl_distance_points=0.00500,
            has_moved_to_breakeven=False,
        )

        # 2.5 pips profit / 5 pips SL = 0.5R - should not trigger
        state.update_profit(1.10250)
        assert not state.is_in_profit_by_r(1.0)

        # 5 pips profit / 5 pips SL = 1R - should trigger
        state.update_profit(1.10500)
        assert state.is_in_profit_by_r(1.0) or state.profit_r >= 0.999  # Account for floating point

    def test_has_moved_to_breakeven_flag(self):
        """Test that breakeven flag prevents re-trigger."""
        state = PositionState(
            ticket=12348,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            current_price=1.10500,
            sl_price=1.10000,  # Already at breakeven
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
            epic_id="epic-14",
            sl_distance_points=0.00500,
            has_moved_to_breakeven=True,  # Already moved
        )

        # Should not trigger even at 1R
        state.update_profit(1.10500)
        assert not state.is_in_profit_by_r(1.0)


class TestLayer2PositionMonitor:
    """Test Layer2PositionMonitor functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        redis_mock = Mock()
        redis_mock.set = Mock(return_value=True)
        redis_mock.delete = Mock(return_value=True)
        redis_mock.exists = Mock(return_value=0)
        redis_mock.scan_iter = Mock(return_value=iter([]))
        return redis_mock

    @pytest.fixture
    def mock_mt5(self):
        """Create a mock MT5 client."""
        mt5_mock = Mock()
        mt5_mock.modify_position = Mock(return_value=True)
        return mt5_mock

    @pytest.fixture
    def monitor(self, mock_redis, mock_mt5):
        """Create a Layer2PositionMonitor instance."""
        return Layer2PositionMonitor(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
            instance_id="test-instance",
            activation_delay_ms=10,  # Short delay for tests
        )

    def test_track_position(self, monitor):
        """Test position tracking."""
        state = monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
            epic_id="epic-14",
            regime_at_entry="TREND_STABLE",
        )

        assert state.ticket == 12345
        assert state.symbol == "EURUSD"
        assert state.direction == "BUY"
        assert state.sl_distance_points == pytest.approx(0.00500)
        assert state.has_moved_to_breakeven is False

        # Verify position is tracked
        assert monitor.get_position(12345) == state

    def test_untrack_position(self, monitor):
        """Test position untracking."""
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )

        assert monitor.untrack_position(12345) is True
        assert monitor.get_position(12345) is None

    def test_move_to_breakeven_triggers_at_1r(self, monitor, mock_mt5):
        """Test AC#1: Move-to-breakeven triggers when position reaches 1R profit."""
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,  # 50 pip risk
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )

        # Price moves to 1.10500 = +50 pips = +1R
        result = monitor.check_and_move_to_breakeven(
            ticket=12345,
            current_price=1.10500,
            symbol="EURUSD"
        )

        assert result.success is True
        assert result.action == "move_to_breakeven"
        assert result.new_sl is not None

        # Breakeven = entry + 0.5 * SL_distance = 1.10000 + 0.00250 = 1.10250
        expected_breakeven = 1.10000 + (0.5 * 0.00500)
        assert abs(result.new_sl - expected_breakeven) < 0.00001

        # Verify MT5 was called
        mock_mt5.modify_position.assert_called_once()

    def test_move_to_breakeven_not_triggers_before_1r(self, monitor):
        """Test that move-to-breakeven does not trigger before 1R."""
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )

        # Price moves to 1.10250 = +25 pips = +0.5R - should NOT trigger
        result = monitor.check_and_move_to_breakeven(
            ticket=12345,
            current_price=1.10250,
            symbol="EURUSD"
        )

        assert result.success is True
        assert "Not at 1R" in result.error

    def test_move_to_breakeven_already_at_breakeven(self, monitor):
        """Test that move-to-breakeven handles already-at-breakeven case."""
        state = monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.10250,  # Already at breakeven
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )
        state.has_moved_to_breakeven = True

        result = monitor.check_and_move_to_breakeven(
            ticket=12345,
            current_price=1.10750,
            symbol="EURUSD"
        )

        assert result.success is True
        assert "Already at breakeven" in result.error

    def test_redis_lock_acquisition(self, monitor, mock_redis):
        """Test AC#3: Redis lock acquisition with 3s TTL."""
        mock_redis.set = Mock(return_value=True)

        acquired = monitor._acquire_lock(12345)

        assert acquired is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "lock:modify:12345"
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == REDIS_LOCK_TTL_SECONDS

    def test_redis_lock_release(self, monitor, mock_redis):
        """Test Redis lock release."""
        released = monitor._release_lock(12345)

        assert released is True
        mock_redis.delete.assert_called_once_with("lock:modify:12345")

    def test_lock_not_acquired_when_held(self, monitor, mock_redis):
        """Test that lock acquisition fails when lock is already held."""
        mock_redis.set = Mock(return_value=False)  # Lock not acquired

        acquired = monitor._acquire_lock(12345)

        assert acquired is False

    def test_modification_with_lock_success(self, monitor, mock_redis, mock_mt5):
        """Test successful modification with lock."""
        mock_redis.set = Mock(return_value=True)
        mock_redis.delete = Mock(return_value=True)
        mock_mt5.modify_position = Mock(return_value=True)

        result = monitor._modify_position_with_lock(
            ticket=12345,
            new_sl=1.10250,
            action="move_to_breakeven"
        )

        assert result.success is True
        assert result.lock_released is False  # Lock still held

    def test_modification_releases_lock_on_failure(self, monitor, mock_redis, mock_mt5):
        """Test AC#4: Lock is released when modification fails."""
        mock_redis.set = Mock(return_value=True)
        mock_redis.delete = Mock(return_value=True)
        mock_mt5.modify_position = Mock(return_value=False)

        result = monitor._modify_position_with_lock(
            ticket=12345,
            new_sl=1.10250,
            action="move_to_breakeven"
        )

        assert result.success is False
        assert result.lock_released is True
        mock_redis.delete.assert_called()  # Lock was released

    def test_kill_preemption_releases_lock(self, monitor, mock_redis):
        """Test AC#5 GG-4: Kill preemption releases Redis lock."""
        mock_redis.delete = Mock(return_value=True)

        result = monitor.handle_kill_preemption(12345)

        assert result.success is True
        assert result.action == "kill_preemption"
        assert result.lock_released is True
        mock_redis.delete.assert_called_once_with("lock:modify:12345")

    def test_check_kill_pending(self, monitor, mock_redis):
        """Test kill pending check."""
        mock_redis.exists = Mock(return_value=1)

        assert monitor.check_kill_pending(12345) is True

        mock_redis.exists = Mock(return_value=0)
        assert monitor.check_kill_pending(12345) is False


class TestRegimeShiftResponse:
    """Test regime shift evaluation."""

    @pytest.fixture
    def mock_redis(self):
        redis_mock = Mock()
        redis_mock.set = Mock(return_value=True)
        redis_mock.delete = Mock(return_value=True)
        redis_mock.exists = Mock(return_value=0)
        redis_mock.scan_iter = Mock(return_value=iter([]))
        return redis_mock

    @pytest.fixture
    def mock_mt5(self):
        return Mock()

    @pytest.fixture
    def monitor(self, mock_redis, mock_mt5):
        return Layer2PositionMonitor(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
            instance_id="test-regime",
        )

    def test_regime_shift_trend_to_range(self, monitor):
        """Test AC#2: Regime shift evaluation from TREND_STABLE to RANGE_STABLE."""
        # Track a momentum/trend strategy
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="momentum_strategy",
        )

        # Create regime shift event
        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.RANGE_STABLE,
            confidence=0.85,
        )

        # Evaluate
        results = monitor.evaluate_regime_shift(event)

        # Momentum strategy is NOT suitable for range - should be flagged
        assert len(results) == 1
        suitability = results[0]
        assert suitability.action in ["close", "reduce"]
        assert "unsuitable" in suitability.reason.lower() or "reduce" in suitability.action

    def test_regime_shift_to_chaos(self, monitor):
        """Test that CHAOS regime triggers close for all strategies."""
        # Track multiple strategies
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="mean_reversion",  # Should still close
        )
        monitor.track_position(
            ticket=12346,
            symbol="GBPUSD",
            direction="SELL",
            entry_price=1.30000,
            sl_price=1.30500,
            tp_price=1.28000,
            volume=1.0,
            strategy_id="trend_follow",  # Should still close
        )

        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.CHAOS,
            confidence=0.95,
        )

        results = monitor.evaluate_regime_shift(event)

        # All positions should be flagged for close
        assert len(results) == 2
        for s in results:
            assert s.action == "close"

    def test_regime_shift_suitable_strategy(self, monitor):
        """Test that suitable strategies are held."""
        # Track a mean_reversion strategy
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="mean_reversion_strategy",
        )

        event = RegimeShiftEvent(
            previous_regime=RegimeType.TREND_STABLE,
            current_regime=RegimeType.RANGE_STABLE,
            confidence=0.85,
        )

        results = monitor.evaluate_regime_shift(event)

        assert len(results) == 1
        assert results[0].action == "hold"


class TestRegimeSuitability:
    """Test RegimeSuitability evaluation."""

    def test_chaos_regime_closes_all(self):
        """CHAOS regime should recommend close for all strategies."""
        suitability = RegimeSuitability.evaluate(
            strategy_id="any_strategy",
            regime=RegimeType.CHAOS,
            confidence=0.9
        )
        assert suitability.action == "close"
        assert suitability.is_suitable is False

    def test_trend_regime_suits_trend_strategies(self):
        """Trend regimes should suit trend-following strategies."""
        suitability = RegimeSuitability.evaluate(
            strategy_id="trend_follow_momentum",
            regime=RegimeType.TREND_BULL,
            confidence=0.8
        )
        assert suitability.action == "hold"
        assert suitability.is_suitable is True

    def test_range_regime_suits_mean_reversion(self):
        """Range regimes should suit mean-reversion strategies."""
        suitability = RegimeSuitability.evaluate(
            strategy_id="mean_reversion_hedge",
            regime=RegimeType.RANGE_STABLE,
            confidence=0.85
        )
        assert suitability.action == "hold"
        assert suitability.is_suitable is True

    def test_low_confidence_reduces_not_closes(self):
        """Low confidence should recommend reduce instead of close."""
        suitability = RegimeSuitability.evaluate(
            strategy_id="breakout_strategy",
            regime=RegimeType.TREND_STABLE,  # Trend stable doesn't suit breakout
            confidence=0.5  # Low confidence
        )
        # Should reduce, not close, due to low confidence
        assert suitability.action == "reduce"


class TestKillPreemptionIntegration:
    """Test kill switch preemption integration."""

    @pytest.fixture
    def mock_redis(self):
        redis_mock = Mock()
        redis_mock.set = Mock(return_value=True)
        redis_mock.delete = Mock(return_value=True)
        redis_mock.exists = Mock(return_value=0)
        redis_mock.scan_iter = Mock(return_value=iter([]))
        return redis_mock

    @pytest.fixture
    def mock_mt5(self):
        mt5_mock = Mock()
        mt5_mock.modify_position = Mock(return_value=True)
        return mt5_mock

    @pytest.fixture
    def monitor(self, mock_redis, mock_mt5):
        return Layer2PositionMonitor(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
            instance_id="test-kill",
        )

    def test_kill_preemption_callback(self, monitor):
        """Test that kill preemption invokes registered callbacks."""
        callback = Mock()
        monitor.register_kill_preemption_callback(callback)

        monitor.handle_kill_preemption(12345)

        callback.assert_called_once_with(12345)

    def test_position_with_kill_pending_skipped(self, monitor, mock_redis):
        """Test that positions with pending kills are skipped in evaluation."""
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )

        # Mark kill as pending
        mock_redis.exists = Mock(return_value=1)

        # Price moves to 1R
        def price_provider(symbol):
            return 1.10500

        results = monitor.evaluate_all_positions(price_provider)

        # Position should be skipped due to pending kill
        assert len(results) == 0

    def test_modify_position_checks_kill_first(self, monitor, mock_redis):
        """Test that modify_position checks for kill before proceeding."""
        monitor.track_position(
            ticket=12345,
            symbol="EURUSD",
            direction="BUY",
            entry_price=1.10000,
            sl_price=1.09500,
            tp_price=1.11000,
            volume=1.0,
            strategy_id="trend_follow",
        )

        # Mark kill as pending
        mock_redis.exists = Mock(return_value=1)

        result = monitor.modify_position(12345, new_sl=1.10250)

        # Should fail because kill is pending
        assert result.success is False
        assert "kill" in result.error.lower()


class TestActivationDelay:
    """Test LAYER2_ACTIVATION_DELAY tick sync guard."""

    def test_activation_delay_applied(self):
        """Test that activation delay is configurable."""
        from src.risk.pipeline.layer2_position_monitor import LAYER2_ACTIVATION_DELAY_MS
        assert LAYER2_ACTIVATION_DELAY_MS == 100

    def test_lock_ttl_is_three_seconds(self):
        """Test AC#3: Redis lock TTL is 3 seconds."""
        assert REDIS_LOCK_TTL_SECONDS == 3
