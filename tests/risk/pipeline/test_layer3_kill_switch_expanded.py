"""
Expanded Tests for Layer 3 Kill Switch — Story 14.3 Layer 3 CHAOS Kill Switch Forced Exit

P0 critical-path expansion covering:
- get_open_scalping_tickets MT5 unavailable + _orders fallback (P0)
- subscribe_rvol_channel Redis error sets SVSS unavailable (P0)
- process_rvol_messages full implementation paths (P0)
- _get_tickets_for_symbol MT5 unavailable + exception (P0)
- _block_symbol_entries Redis error handling (P1)
- _log_forced_exit DB not-found + DB error graceful handling (P0)
- process_kill_queue Redis error paths (P0)
- process_kill_queue invalid key / non-numeric ticket handling (P0)
- execute_forced_close exception path (P0)
- release_layer2_locks Redis error returns False (P0)
- on_rvol_event full event handling (P1)
- get_open_positions returns empty (P1)

Epic 14.3 | P0-P1 | 22 new tests
"""

import pytest
from unittest.mock import MagicMock, patch
import redis as redis_lib

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


class MockRedisWithErrors:
    """Mock Redis that can simulate Redis errors."""

    def __init__(self, fail_on=None):
        self._data = {}
        self._sorted_sets = {}
        self._hashes = {}
        self._expiry = {}
        self._fail_on = fail_on or set()

    def exists(self, key):
        if "exists" in self._fail_on:
            raise redis_lib.RedisError("Simulated exists error")
        return 1 if key in self._data else 0

    def set(self, key, value, nx=False, ex=None):
        if "set" in self._fail_on:
            raise redis_lib.RedisError("Simulated set error")
        self._data[key] = value
        if ex:
            self._expiry[key] = ex
        return True

    def delete(self, key):
        if "delete" in self._fail_on:
            raise redis_lib.RedisError("Simulated delete error")
        for d in [self._data, self._sorted_sets, self._hashes]:
            if key in d:
                del d[key]
                return 1
        return 0

    def hset(self, name, mapping=None, **kwargs):
        if "hset" in self._fail_on:
            raise redis_lib.RedisError("Simulated hset error")
        if name not in self._hashes:
            self._hashes[name] = {}
        mapping = mapping or kwargs
        self._hashes[name].update({k.encode() if isinstance(k, str) else k:
                                   v.encode() if isinstance(v, str) else v
                                   for k, v in mapping.items()})
        return True

    def hgetall(self, name):
        if "hgetall" in self._fail_on:
            raise redis_lib.RedisError("Simulated hgetall error")
        return self._hashes.get(name, {})

    def expire(self, name, time):
        self._expiry[name] = time
        return True

    def zadd(self, name, mapping):
        if "zadd" in self._fail_on:
            raise redis_lib.RedisError("Simulated zadd error")
        if name not in self._sorted_sets:
            self._sorted_sets[name] = {}
        for k, v in mapping.items():
            self._sorted_sets[name][k] = v
        return len(mapping)

    def zpopmin(self, name, num):
        if "zpopmin" in self._fail_on:
            raise redis_lib.RedisError("Simulated zpopmin error")
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
        if "zcard" in self._fail_on:
            raise redis_lib.RedisError("Simulated zcard error")
        return len(self._sorted_sets.get(name, {}))

    def pubsub(self):
        if "pubsub" in self._fail_on:
            raise redis_lib.RedisError("Simulated pubsub error")
        mock_pubsub = MagicMock()
        return mock_pubsub


# =============================================================================
# get_open_scalping_tickets — MT5 Unavailable + _orders Fallback (P0)
# =============================================================================

class TestGetOpenScalpingTickets:
    """P0: get_open_scalping_tickets handles MT5 unavailable and _orders fallback."""

    def test_get_open_scalping_tickets_mt5_none(self):
        """MT5 client is None — returns empty list."""
        mock_redis = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=None,
        )

        tickets = kill_switch.get_open_scalping_tickets()
        assert tickets == []

    def test_get_open_scalping_tickets_mt5_get_open_positions(self):
        """MT5 with get_open_positions — returns filtered scalping tickets."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 1001, "symbol": "EURUSD", "strategy_id": "scalp-01"},
            {"ticket": 1002, "symbol": "GBPUSD", "strategy_id": "trend-follow"},
            {"ticket": 1003, "symbol": "USDJPY", "strategy_id": "scalping-fast"},
        ]

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch.get_open_scalping_tickets()
        assert 1001 in tickets
        assert 1003 in tickets
        assert 1002 not in tickets

    def test_get_open_scalping_tickets_mt5_orders_fallback(self):
        """MT5 without get_open_positions but with _orders — uses fallback."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        del mock_mt5.get_open_positions  # Remove the method

        # Create mock order objects with strategy_id
        order1 = MagicMock()
        order1.ticket = 2001
        order1.strategy_id = "scalp-alpha"

        order2 = MagicMock()
        order2.ticket = 2002
        order2.strategy_id = "swing-trade"

        order3 = MagicMock()
        order3.ticket = 2003
        order3.strategy_id = "SCALPING-BETA"

        mock_mt5._orders = [order1, order2, order3]

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch.get_open_scalping_tickets()
        assert 2001 in tickets
        assert 2003 in tickets
        assert 2002 not in tickets

    def test_get_open_scalping_tickets_mt5_exception(self):
        """MT5 get_open_positions raises exception — returns empty list."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.side_effect = RuntimeError("MT5 disconnected")

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch.get_open_scalping_tickets()
        assert tickets == []


# =============================================================================
# subscribe_rvol_channel — Redis Error Handling (P0)
# =============================================================================

class TestSubscribeRvolChannel:
    """P0: subscribe_rvol_channel sets SVSS unavailable on Redis error."""

    def test_subscribe_rvol_channel_redis_error(self):
        """Redis error on subscribe — sets _svss_available to False."""
        mock_redis = MockRedisWithErrors(fail_on={"pubsub"})
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )

        assert kill_switch.is_svss_available() is True  # Initially available

        kill_switch.subscribe_rvol_channel("EURUSD")

        # Should handle error gracefully and set SVSS unavailable
        assert kill_switch.is_svss_available() is False


# =============================================================================
# process_rvol_messages — Full Implementation Paths (P0)
# =============================================================================

class TestProcessRvolMessages:
    """P0: process_rvol_messages handles all SVSS message processing paths."""

    def test_process_rvol_messages_no_pubsub(self):
        """pubsub not set — returns empty list."""
        mock_redis = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )
        # No _pubsub attribute set

        events = kill_switch.process_rvol_messages()
        assert events == []

    def test_process_rvol_messages_svss_unavailable(self):
        """SVSS unavailable — returns empty list."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )
        kill_switch._svss_available = False
        kill_switch._pubsub = MagicMock()

        events = kill_switch.process_rvol_messages()
        assert events == []

    def test_process_rvol_messages_normal_message(self):
        """Valid RVOL warning message — returns event and blocks symbol."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 3001, "symbol": "EURUSD"},
        ]

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )
        kill_switch._svss_available = True
        kill_switch._pubsub = MagicMock()
        kill_switch._pubsub.get_message.return_value = {
            "type": "message",
            "data": '{"symbol": "EURUSD", "rvvol": 0.42}',
        }

        events = kill_switch.process_rvol_messages()

        assert len(events) == 1
        assert events[0].symbol == "EURUSD"
        assert events[0].rvol == 0.42

    def test_process_rvol_messages_rvol_above_threshold(self):
        """RVOL >= 0.5 — no warning event returned."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 3001, "symbol": "EURUSD"},
        ]

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )
        kill_switch._svss_available = True
        kill_switch._pubsub = MagicMock()
        kill_switch._pubsub.get_message.return_value = {
            "type": "message",
            "data": '{"symbol": "EURUSD", "rvvol": 0.55}',
        }

        events = kill_switch.process_rvol_messages()
        # RVOL >= 0.5, no event returned
        assert len(events) == 0


# =============================================================================
# _get_tickets_for_symbol — MT5 Unavailable + Exception (P0)
# =============================================================================

class TestGetTicketsForSymbol:
    """P0: _get_tickets_for_symbol handles MT5 unavailable and exception paths."""

    def test_get_tickets_for_symbol_mt5_none(self):
        """MT5 is None — returns empty list."""
        mock_redis = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=None,
        )

        tickets = kill_switch._get_tickets_for_symbol("EURUSD")
        assert tickets == []

    def test_get_tickets_for_symbol_get_open_positions(self):
        """MT5 with get_open_positions — returns matching tickets."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = [
            {"ticket": 4001, "symbol": "EURUSD"},
            {"ticket": 4002, "symbol": "GBPUSD"},
            {"ticket": 4003, "symbol": "EURUSD"},
        ]

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch._get_tickets_for_symbol("EURUSD")
        assert 4001 in tickets
        assert 4003 in tickets
        assert 4002 not in tickets

    def test_get_tickets_for_symbol_mt5_exception(self):
        """MT5 get_open_positions raises exception — returns empty list."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.side_effect = RuntimeError("MT5 error")

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch._get_tickets_for_symbol("EURUSD")
        assert tickets == []


# =============================================================================
# _block_symbol_entries — Redis Error Handling (P1)
# =============================================================================

class TestBlockSymbolEntries:
    """P1: _block_symbol_entries handles Redis errors gracefully."""

    def test_block_symbol_entries_sets_redis_key(self):
        """Block key is set in Redis with 24h TTL."""
        mock_redis = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )

        kill_switch._block_symbol_entries("EURUSD")

        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == "symbol:blocked:EURUSD"
        assert call_args[1]["ex"] == 86400  # 24h TTL

    def test_block_symbol_entries_redis_error(self):
        """Redis error on set — does not crash, logs error."""
        mock_redis = MockRedisWithErrors(fail_on={"set"})
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )

        # Should not raise — error is caught and logged
        kill_switch._block_symbol_entries("EURUSD")


# =============================================================================
# _log_forced_exit — DB Error Graceful Handling (P0)
# =============================================================================

class TestLogForcedExit:
    """P0: _log_forced_exit handles DB not-found and DB errors gracefully."""

    @patch("src.risk.pipeline.layer3_kill_switch.SessionLocal")
    def test_log_forced_exit_no_record_found(self, mock_session_local):
        """No trade record found — logs warning, does not crash."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Should not raise — logs warning for no record
        kill_switch._log_forced_exit(
            ticket=5001,
            outcome=ForcedExitOutcome.FILLED,
            triggered_by="LYAPUNOV",
            symbol="EURUSD",
        )

        mock_session.close.assert_called_once()

    @patch("src.risk.pipeline.layer3_kill_switch.SessionLocal")
    def test_log_forced_exit_db_exception(self, mock_session_local):
        """DB exception during logging — logs error, does not crash."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session
        mock_session.query.side_effect = Exception("DB connection refused")

        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Should not raise — error is caught and logged
        kill_switch._log_forced_exit(
            ticket=5002,
            outcome=ForcedExitOutcome.REJECTED,
            triggered_by="RVOL",
            error="insufficient margin",
        )


# =============================================================================
# process_kill_queue — Redis Error Paths (P0)
# =============================================================================

class TestProcessKillQueueErrorPaths:
    """P0: process_kill_queue handles Redis errors gracefully."""

    def test_process_kill_queue_redis_error_zpopmin(self):
        """Redis error on zpopmin — returns empty list."""
        mock_redis = MockRedisWithErrors(fail_on={"zpopmin"})
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        results = kill_switch.process_kill_queue()
        assert results == []

    def test_process_kill_queue_invalid_key_format(self):
        """Kill queue key does not start with kill:pending: — skipped."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.return_value = {
            "success": True, "partial": False, "volume": 1.0
        }
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Pre-populate queue with invalid key format
        mock_redis.zpopmin.return_value = [("invalid:key:format", 1000)]

        results = kill_switch.process_kill_queue()
        # Invalid key skipped, no results
        assert results == []

    def test_process_kill_queue_non_numeric_ticket(self):
        """Kill queue key has non-numeric ticket — ValueError caught, skipped."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Pre-populate queue with non-numeric ticket in key
        mock_redis.zpopmin.return_value = [("kill:pending:notanumber", 1000)]

        results = kill_switch.process_kill_queue()
        # Non-numeric ticket skipped, no results
        assert results == []

    def test_process_kill_queue_empty_metadata(self):
        """Kill queue entry has empty metadata — skipped."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Pre-populate queue
        mock_redis.zpopmin.return_value = [("kill:pending:6001", 1000)]
        mock_redis.hgetall.return_value = {}  # Empty metadata

        results = kill_switch.process_kill_queue()
        assert results == []

    def test_process_kill_queue_redis_error_hgetall(self):
        """Redis error on hgetall — skipped gracefully."""
        mock_redis = MockRedisWithErrors(fail_on={"hgetall"})
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        # Pre-populate queue - use the actual zpopmin method with test data
        kill_switch._redis.zadd(KILL_QUEUE_KEY, {"kill:pending:6002": 1000})

        results = kill_switch.process_kill_queue()
        # Error on hgetall causes skip, no results
        assert results == []


# =============================================================================
# execute_forced_close — Exception Path (P0)
# =============================================================================

class TestExecuteForcedCloseException:
    """P0: execute_forced_close handles unexpected exceptions gracefully."""

    def test_execute_forced_close_unexpected_exception(self):
        """Unexpected exception during force close — returns REJECTED."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.force_close_by_ticket.side_effect = RuntimeError("Unexpected error")

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        result = kill_switch.execute_forced_close(
            ticket=7001,
            triggered_by="LYAPUNOV",
        )

        assert result.outcome == ForcedExitOutcome.REJECTED
        assert "Unexpected error" in result.error


# =============================================================================
# release_layer2_locks — Redis Error Handling (P0)
# =============================================================================

class TestReleaseLayer2LocksRedisErrors:
    """P0: release_layer2_locks returns False on Redis error."""

    def test_release_layer2_locks_redis_error_exists(self):
        """Redis error on exists check — returns False."""
        mock_redis = MockRedisWithErrors(fail_on={"exists"})
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )

        # Pre-set a lock to trigger the exists check
        lock_key = f"{LAYER2_LOCK_PREFIX}8001"
        mock_redis._data[lock_key] = "some_value"

        result = kill_switch.release_layer2_locks(8001)
        assert result is False

    def test_release_layer2_locks_redis_error_delete(self):
        """Redis error on delete — returns False."""
        mock_redis = MockRedisWithErrors(fail_on={"delete"})
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=MagicMock(),
        )

        # Pre-set a lock to trigger the exists check
        lock_key = f"{LAYER2_LOCK_PREFIX}8002"
        mock_redis._data[lock_key] = "some_value"

        result = kill_switch.release_layer2_locks(8002)
        assert result is False


# =============================================================================
# on_rvol_event — Full Event Handling (P1)
# =============================================================================

class TestOnRvolEvent:
    """P1: on_rvol_event processes RVOL events end-to-end."""

    def test_on_rvol_event_no_warning_because_threshold(self):
        """RVOL >= 0.5 — returns None (no warning)."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        event = kill_switch.on_rvol_event(
            symbol="EURUSD",
            rvol=0.60,
            open_tickets=[1001],
        )

        assert event is None

    def test_on_rvol_event_with_warning(self):
        """RVOL < 0.5 with open tickets — returns RVOLWarningEvent."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        event = kill_switch.on_rvol_event(
            symbol="EURUSD",
            rvol=0.42,
            open_tickets=[1001, 1002],
        )

        assert event is not None
        assert isinstance(event, RVOLWarningEvent)
        assert event.symbol == "EURUSD"
        assert event.rvol == 0.42

    def test_on_rvol_event_svss_unavailable(self):
        """SVSS unavailable — returns None."""
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )
        kill_switch._svss_available = False

        event = kill_switch.on_rvol_event(
            symbol="EURUSD",
            rvol=0.42,
            open_tickets=[1001],
        )

        assert event is None


# =============================================================================
# get_open_positions — Empty State (P1)
# =============================================================================

class TestGetOpenPositions:
    """P1: get_open_positions returns empty when no positions."""

    def test_get_open_positions_no_positions(self):
        """MT5 returns empty list — get_open_scalping_tickets returns []. """
        mock_redis = MagicMock()
        mock_mt5 = MagicMock()
        mock_mt5.get_open_positions.return_value = []

        kill_switch = Layer3KillSwitch(
            redis_client=mock_redis,
            mt5_client=mock_mt5,
        )

        tickets = kill_switch.get_open_scalping_tickets()
        assert tickets == []
