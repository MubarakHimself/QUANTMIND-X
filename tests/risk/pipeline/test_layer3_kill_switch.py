"""
Tests for Layer 3 Kill Switch.

Story 14.3: Layer 3 CHAOS + Kill Switch Forced Exit
Tests for Layer3KillSwitch class.

AC #1: LYAPUNOV_EXCEEDED detection and kill:pending queue production
AC #2: Priority queue processing and forced exit execution
AC #3: SVSS RVOL early warning chain (graceful fallback)
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timezone

from src.risk.pipeline.layer3_kill_switch import (
    Layer3KillSwitch,
    KillQueueEntry,
    ForcedExitResult,
    LYAPUNOV_CHAOS_THRESHOLD,
    RVOL_WARNING_THRESHOLD,
    KILL_QUEUE_KEY,
    LAYER2_LOCK_PREFIX,
)
from src.events.chaos import (
    ChaosEvent,
    ChaosLevel,
    ForcedExitOutcome,
    KillSwitchResult,
    RVOLWarningEvent,
)


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self._data = {}
        self._sorted_sets = {}
        self._hashes = {}
        self._expiry = {}

    def exists(self, key):
        return 1 if key in self._data else 0

    def get(self, key):
        return self._data.get(key)

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._data:
            return False
        self._data[key] = value
        if ex:
            self._expiry[key] = ex
        return True

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            return 1
        if key in self._sorted_sets:
            del self._sorted_sets[key]
            return 1
        if key in self._hashes:
            del self._hashes[key]
            return 1
        return 0

    def hset(self, name, mapping=None, **kwargs):
        if name not in self._hashes:
            self._hashes[name] = {}
        mapping = mapping or kwargs
        self._hashes[name].update({k.encode() if isinstance(k, str) else k:
                                   v.encode() if isinstance(v, str) else v
                                   for k, v in mapping.items()})
        return True

    def hgetall(self, name):
        return self._hashes.get(name, {})

    def expire(self, name, time):
        self._expiry[name] = time
        return True

    def zadd(self, name, mapping):
        if name not in self._sorted_sets:
            self._sorted_sets[name] = {}
        for k, v in mapping.items():
            self._sorted_sets[name][k] = v
        return len(mapping)

    def zpopmin(self, name, num):
        if name not in self._sorted_sets or not self._sorted_sets[name]:
            return []
        items = []
        sorted_items = sorted(self._sorted_sets[name].items(), key=lambda x: x[1])
        for i, (k, v) in enumerate(sorted_items):
            if i >= num:
                break
            items.append((k, v))
            del self._sorted_sets[name][k]
        return items

    def zrange(self, name, start, end):
        if name not in self._sorted_sets:
            return []
        sorted_items = sorted(self._sorted_sets[name].items(), key=lambda x: x[1])
        if end == -1:
            return [k for k, v in sorted_items[start:]]
        return [k for k, v in sorted_items[start:end + 1]]

    def zcard(self, name):
        return len(self._sorted_sets.get(name, {}))


class TestLayer3KillSwitchConstants:
    """Test Layer3KillSwitch configuration constants."""

    def test_lyapunov_threshold(self):
        """Test LYAPUNOV_CHAOS_THRESHOLD is 0.95."""
        assert LYAPUNOV_CHAOS_THRESHOLD == 0.95

    def test_rvol_threshold(self):
        """Test RVOL_WARNING_THRESHOLD is 0.5."""
        assert RVOL_WARNING_THRESHOLD == 0.5

    def test_kill_queue_key(self):
        """Test KILL_QUEUE_KEY format."""
        assert KILL_QUEUE_KEY == "kill:pending:queue"

    def test_layer2_lock_prefix(self):
        """Test LAYER2_LOCK_PREFIX format."""
        assert LAYER2_LOCK_PREFIX == "lock:modify:"


class TestLayer3KillSwitchInit:
    """Test Layer3KillSwitch initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        assert kill_switch._redis is mock_redis
        assert kill_switch._mt5 is mock_mt5
        assert kill_switch._instance_id == "layer3-default"
        assert kill_switch._svss_available is True

    def test_init_with_custom_instance_id(self):
        """Test initialization with custom instance ID."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
            instance_id="layer3-primary",
        )

        assert kill_switch._instance_id == "layer3-primary"


class TestLayer3KillSwitchChaosDetection:
    """Test chaos detection (AC #1)."""

    def test_detect_chaos_below_threshold(self):
        """Test detect_chaos returns False when Lyapunov < 0.95."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        assert kill_switch.detect_chaos(0.90) is False
        assert kill_switch.detect_chaos(0.94) is False
        assert kill_switch.detect_chaos(0.949) is False

    def test_detect_chaos_at_threshold(self):
        """Test detect_chaos returns True when Lyapunov > 0.95 (exceeds, not equal)."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # "Exceeds" means strictly greater than (>)
        assert kill_switch.detect_chaos(0.95) is False  # Not chaos - exactly at threshold
        assert kill_switch.detect_chaos(0.951) is True  # Chaos - exceeds threshold
        assert kill_switch.detect_chaos(0.99) is True
        assert kill_switch.detect_chaos(1.05) is True

    def test_flag_positions_empty_list(self):
        """Test flag_positions_for_forced_exit with empty list."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.flag_positions_for_forced_exit([])
        assert result == []

    def test_flag_positions_for_lyapunov(self):
        """Test flagging positions for LYAPUNOV triggered kill."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        tickets = [1001, 1002, 1003]
        result = kill_switch.flag_positions_for_forced_exit(
            tickets=tickets,
            triggered_by="LYAPUNOV"
        )

        assert result == tickets
        assert len(result) == 3

        # Verify queue entries were created
        for ticket in tickets:
            kill_key = f"kill:pending:{ticket}"
            assert kill_key in mock_redis._hashes
            assert mock_redis._hashes[kill_key][b"triggered_by"] == b"LYAPUNOV"

    def test_flag_positions_for_rvol(self):
        """Test flagging positions for RVOL triggered kill."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        tickets = [2001]
        result = kill_switch.flag_positions_for_forced_exit(
            tickets=tickets,
            triggered_by="RVOL",
            symbol="EURUSD"
        )

        assert result == tickets
        kill_key = "kill:pending:2001"
        assert mock_redis._hashes[kill_key][b"triggered_by"] == b"RVOL"
        assert mock_redis._hashes[kill_key][b"symbol"] == b"EURUSD"


class TestLayer3KillSwitchQueueProcessing:
    """Test queue processing and forced exit (AC #2)."""

    def test_release_layer2_locks_not_held(self):
        """Test release_layer2_locks when no lock is held."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.release_layer2_locks(12345)
        assert result is True

    def test_release_layer2_locks_when_held(self):
        """Test release_layer2_locks releases held lock."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Simulate Layer 2 holding a lock
        lock_key = f"{LAYER2_LOCK_PREFIX}12345"
        mock_redis._data[lock_key] = "layer2:instance:value"

        result = kill_switch.release_layer2_locks(12345)
        assert result is True
        assert lock_key not in mock_redis._data

    def test_execute_forced_close_mt5_not_available(self):
        """Test execute_forced_close when MT5 client is None."""
        mock_redis = MockRedis()
        mock_mt5 = None
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.execute_forced_close(
            ticket=12345,
            triggered_by="LYAPUNOV"
        )

        assert result.outcome == ForcedExitOutcome.REJECTED
        assert "MT5 client not available" in result.error

    def test_execute_forced_close_mt5_missing_method(self):
        """Test execute_forced_close when MT5 client lacks method."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        del mock_mt5.force_close_by_ticket
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.execute_forced_close(
            ticket=12345,
            triggered_by="LYAPUNOV"
        )

        assert result.outcome == ForcedExitOutcome.REJECTED
        assert "missing force_close_by_ticket" in result.error

    def test_execute_forced_close_success(self):
        """Test execute_forced_close with successful close."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": False,
            "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.execute_forced_close(
            ticket=12345,
            triggered_by="LYAPUNOV"
        )

        assert result.outcome == ForcedExitOutcome.FILLED
        assert result.lock_released is True
        assert result.lyapunov_triggered is True
        assert result.rvol_triggered is False

    def test_execute_forced_close_partial(self):
        """Test execute_forced_close with partial close."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": True,
            "volume": 0.50
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.execute_forced_close(
            ticket=12345,
            triggered_by="LYAPUNOV"
        )

        assert result.outcome == ForcedExitOutcome.PARTIAL
        assert result.partial_volume == 0.50

    def test_execute_forced_close_rejected(self):
        """Test execute_forced_close when MT5 rejects."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": False,
            "error": "insufficient margin"
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.execute_forced_close(
            ticket=12345,
            triggered_by="LYAPUNOV"
        )

        assert result.outcome == ForcedExitOutcome.REJECTED
        assert "insufficient margin" in result.error

    def test_process_kill_queue_empty(self):
        """Test process_kill_queue with empty queue."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        results = kill_switch.process_kill_queue()
        assert results == []

    def test_process_kill_queue_with_items(self):
        """Test process_kill_queue with items in queue."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": False,
            "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Pre-populate queue
        tickets = [1001, 1002]
        for ticket in tickets:
            kill_key = f"kill:pending:{ticket}"
            mock_redis.zadd(KILL_QUEUE_KEY, {kill_key: 1000 + ticket})
            mock_redis.hset(kill_key, mapping={
                "ticket": str(ticket),
                "triggered_by": "LYAPUNOV",
                "symbol": "",
                "timestamp": "1000",
                "instance_id": "test"
            })

        results = kill_switch.process_kill_queue(max_items=10)
        assert len(results) == 2
        assert all(r.outcome == ForcedExitOutcome.FILLED for r in results)


class TestLayer3KillSwitchSVSS:
    """Test SVSS early warning chain (AC #3 - graceful fallback)."""

    def test_svss_initially_available(self):
        """Test SVSS is initially available."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        assert kill_switch.is_svss_available() is True

    def test_handle_rvol_warning_below_threshold(self):
        """Test handle_rvol_warning when RVOL >= 0.5."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.handle_rvol_warning(
            symbol="EURUSD",
            rvol=0.55,
            has_open_positions=True
        )

        assert result is None

    def test_handle_rvol_warning_above_threshold(self):
        """Test handle_rvol_warning when RVOL < 0.5."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        result = kill_switch.handle_rvol_warning(
            symbol="EURUSD",
            rvol=0.42,
            has_open_positions=True
        )

        assert result is not None
        assert isinstance(result, RVOLWarningEvent)
        assert result.symbol == "EURUSD"
        assert result.rvol == 0.42
        assert result.blocked_entries is True

    def test_handle_rvol_fallback(self):
        """Test handle_rvol_fallback switches to LYAPUNOV-only mode."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        assert kill_switch.is_svss_available() is True

        kill_switch.handle_rvol_fallback()

        assert kill_switch.is_svss_available() is False

    def test_handle_rvol_warning_svss_unavailable(self):
        """Test handle_rvol_warning when SVSS is unavailable."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)
        kill_switch.handle_rvol_fallback()

        result = kill_switch.handle_rvol_warning(
            symbol="EURUSD",
            rvol=0.42,
            has_open_positions=True
        )

        assert result is None


class TestLayer3KillSwitchEventHandling:
    """Test event handling entry points."""

    def test_on_lyapunov_event_below_threshold(self):
        """Test on_lyapunov_event when Lyapunov < 0.95."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        event = kill_switch.on_lyapunov_event(
            lyapunov_value=0.90,
            tickets=[1001, 1002]
        )

        assert event.chaos_level == ChaosLevel.NORMAL
        assert len(event.tickets) == 0  # No tickets enqueued

    def test_on_lyapunov_event_above_threshold(self):
        """Test on_lyapunov_event when Lyapunov >= 0.95."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        event = kill_switch.on_lyapunov_event(
            lyapunov_value=0.96,
            tickets=[1001, 1002]
        )

        assert event.chaos_level == ChaosLevel.WARNING
        assert len(event.tickets) == 2
        assert event.tickets == [1001, 1002]


class TestLayer3KillSwitchQueueInspection:
    """Test queue inspection methods."""

    def test_get_queue_size_empty(self):
        """Test get_queue_size when queue is empty."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        assert kill_switch.get_queue_size() == 0

    def test_get_queue_size_with_items(self):
        """Test get_queue_size with items in queue."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        mock_redis.zadd(KILL_QUEUE_KEY, {"kill:pending:1": 1000})
        mock_redis.zadd(KILL_QUEUE_KEY, {"kill:pending:2": 1001})

        assert kill_switch.get_queue_size() == 2

    def test_get_pending_tickets_empty(self):
        """Test get_pending_tickets when queue is empty."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        assert kill_switch.get_pending_tickets() == []

    def test_get_pending_tickets_with_items(self):
        """Test get_pending_tickets with items in queue."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        mock_redis.zadd(KILL_QUEUE_KEY, {"kill:pending:1001": 1000})
        mock_redis.zadd(KILL_QUEUE_KEY, {"kill:pending:1002": 1001})

        tickets = kill_switch.get_pending_tickets()
        assert len(tickets) == 2
        assert 1001 in tickets
        assert 1002 in tickets


class TestLayer3KillSwitchShutdown:
    """Test shutdown behavior."""

    def test_shutdown_completes(self):
        """Test shutdown completes without error."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Should not raise
        kill_switch.shutdown()


class TestLayer3KillSwitchIntegration:
    """Integration tests for full kill switch workflow."""

    def test_full_chaos_to_forced_exit_workflow(self):
        """Test full workflow: chaos detection -> queue -> forced exit."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": False,
            "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Step 1: Lyapunov event
        tickets = [1001, 1002]
        event = kill_switch.on_lyapunov_event(
            lyapunov_value=0.97,
            tickets=tickets
        )

        assert event.chaos_level == ChaosLevel.WARNING
        assert len(event.tickets) == 2

        # Step 2: Verify queue has entries
        queue_size = kill_switch.get_queue_size()
        assert queue_size == 2

        # Step 3: Process queue
        results = kill_switch.process_kill_queue()

        assert len(results) == 2
        assert all(r.outcome == ForcedExitOutcome.FILLED for r in results)
        assert all(r.lyapunov_triggered for r in results)

        # Step 4: Queue should be empty
        assert kill_switch.get_queue_size() == 0

    def test_kill_preempts_layer2_lock(self):
        """Test that kill switch releases Layer 2 lock before forced exit."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": False,
            "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Simulate Layer 2 holding a lock
        lock_key = f"{LAYER2_LOCK_PREFIX}9999"
        mock_redis._data[lock_key] = "layer2:instance:value"

        # Execute forced close
        result = kill_switch.execute_forced_close(
            ticket=9999,
            triggered_by="LYAPUNOV"
        )

        # Lock should be released
        assert lock_key not in mock_redis._data
        assert result.lock_released is True

    def test_svss_graceful_fallback_workflow(self):
        """Test SVSS unavailable: fallback to LYAPUNOV-only mode."""
        mock_redis = MockRedis()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True,
            "partial": False,
            "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(mock_redis, mock_mt5)

        # Simulate SVSS becoming unavailable
        kill_switch.handle_rvol_fallback()

        # SVSS should not be available
        assert kill_switch.is_svss_available() is False

        # RVOL warnings should be ignored
        rvol_result = kill_switch.handle_rvol_warning(
            symbol="EURUSD",
            rvol=0.42,
            has_open_positions=True
        )
        assert rvol_result is None

        # LYAPUNOV path should still work
        event = kill_switch.on_lyapunov_event(
            lyapunov_value=0.97,
            tickets=[1001]
        )
        assert event.chaos_level == ChaosLevel.WARNING
