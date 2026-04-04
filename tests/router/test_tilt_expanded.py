"""
Expanded Tests for TiltStateMachine.

Story 16.1: Tilt — Universal Session Boundary Mechanism
[P1] Coverage gaps from existing tests

These tests fill gaps identified in the existing test suite:
- on_regime_confirmed() ignores non-WAIT states
- get_audit_log_entries() filtered audit log
- Redis publish error handling
- Layer callback error paths
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

from src.events.tilt import (
    REGIME_PERSISTENCE_SECONDS,
    TiltPhase,
    TiltPhaseEvent,
    TiltState,
)
from src.events.chaos import ChaosEvent, ChaosLevel
from src.router.tilt import TiltStateMachine


class MockRedis:
    """Mock Redis client for testing."""

    def __init__(self, publish_raises=False):
        self._data = {}
        self._channels = {}
        self._publish_raises = publish_raises

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self._data:
            return False
        self._data[key] = value
        return True

    def get(self, key):
        return self._data.get(key)

    def delete(self, key):
        if key in self._data:
            del self._data[key]
            return 1
        return 0

    def exists(self, key):
        return 1 if key in self._data else 0

    def publish(self, channel, message):
        if self._publish_raises:
            raise RedisPublishError("Simulated publish failure")
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(message)

    def subscribe(self, channel):
        pass


class RedisPublishError(Exception):
    """Simulated Redis publish error."""
    pass


class TestOnRegimeConfirmedEdgeCases:
    """Test on_regime_confirmed() ignores events when not in WAIT state."""

    def test_regime_confirmation_ignored_in_lock_state(self):
        """Test regime confirmation is ignored when in LOCK state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        with patch.object(tilt, '_publish_phase_event'):
            tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0

    def test_regime_confirmation_ignored_in_signal_state(self):
        """Test regime confirmation is ignored when in SIGNAL state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SIGNAL
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0

    def test_regime_confirmation_ignored_in_re_rank_state(self):
        """Test regime confirmation is ignored when in RE_RANK state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.RE_RANK
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0

    def test_regime_confirmation_ignored_in_activate_state(self):
        """Test regime confirmation is ignored when in ACTIVATE state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.ACTIVATE
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0

    def test_regime_confirmation_ignored_in_idle_state(self):
        """Test regime confirmation is ignored when in IDLE state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.IDLE
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0

    def test_regime_confirmation_ignored_in_suspended_state(self):
        """Test regime confirmation is ignored when in SUSPENDED state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain 0
        assert tilt.regime_persistence_timer == 0


class TestGetAuditLogEntries:
    """Test get_audit_log_entries() filtered audit log retrieval."""

    def test_filter_by_closing_session(self):
        """Test filtering audit log by closing session."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        # Add some audit entries
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)
        tilt._transition_to(TiltState.SIGNAL)

        tilt._closing_session = "NY"
        tilt._incoming_session = "ASIAN"
        tilt._transition_to(TiltState.LOCK)

        # Filter by LONDON closing session
        entries = tilt.get_audit_log_entries(closing_session="LONDON")

        assert len(entries) == 2
        for entry in entries:
            assert entry.closing_session == "LONDON"

    def test_filter_by_incoming_session(self):
        """Test filtering audit log by incoming session."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        # Add some audit entries
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)

        tilt._closing_session = "NY"
        tilt._incoming_session = "ASIAN"
        tilt._transition_to(TiltState.LOCK)

        # Filter by ASIAN incoming session
        entries = tilt.get_audit_log_entries(incoming_session="ASIAN")

        assert len(entries) == 1
        assert entries[0].incoming_session == "ASIAN"

    def test_filter_by_both_sessions(self):
        """Test filtering audit log by both sessions."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)
        tilt._transition_to(TiltState.SIGNAL)

        tilt._closing_session = "NY"
        tilt._incoming_session = "ASIAN"
        tilt._transition_to(TiltState.LOCK)

        # Filter by LONDON->NY
        entries = tilt.get_audit_log_entries(
            closing_session="LONDON",
            incoming_session="NY"
        )

        assert len(entries) == 2
        for entry in entries:
            assert entry.closing_session == "LONDON"
            assert entry.incoming_session == "NY"

    def test_no_filter_returns_all(self):
        """Test that no filter returns all entries."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)
        tilt._transition_to(TiltState.SIGNAL)

        tilt._closing_session = "NY"
        tilt._incoming_session = "ASIAN"
        tilt._transition_to(TiltState.LOCK)

        entries = tilt.get_audit_log_entries()

        assert len(entries) == 3

    def test_no_matching_entries(self):
        """Test filtering returns empty when no matches."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)

        entries = tilt.get_audit_log_entries(
            closing_session="TOKYO"
        )

        assert len(entries) == 0


class TestRedisPublishErrorHandling:
    """Test Redis publish error handling in Tilt."""

    def test_publish_phase_event_handles_redis_error(self):
        """Test _publish_phase_event gracefully handles Redis errors."""
        redis = MockRedis(publish_raises=True)
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        # Should not raise, should log error
        tilt._publish_phase_event()

        # State should be unchanged
        assert tilt._state == TiltState.LOCK

    def test_publish_session_close_handles_redis_error(self):
        """Test _publish_session_close gracefully handles Redis errors."""
        redis = MockRedis(publish_raises=True)
        tilt = TiltStateMachine(redis_client=redis)

        # Should not raise
        tilt._publish_session_close("LONDON")

    def test_publish_session_open_handles_redis_error(self):
        """Test _publish_session_open gracefully handles Redis errors."""
        redis = MockRedis(publish_raises=True)
        tilt = TiltStateMachine(redis_client=redis)

        # Should not raise
        tilt._publish_session_open("NY")

    def test_publish_chaos_suspend_handles_redis_error(self):
        """Test _publish_chaos_suspend gracefully handles Redis errors."""
        from src.events.tilt import TiltChaosSuspendEvent

        redis = MockRedis(publish_raises=True)
        tilt = TiltStateMachine(redis_client=redis)

        event = TiltChaosSuspendEvent(
            previous_state=TiltState.WAIT,
            chaos_lyapunov=0.97,
        )

        # Should not raise
        tilt._publish_chaos_suspend(event)

    def test_publish_chaos_resume_handles_redis_error(self):
        """Test _publish_chaos_resume gracefully handles Redis errors."""
        from src.events.tilt import TiltChaosResumeEvent

        redis = MockRedis(publish_raises=True)
        tilt = TiltStateMachine(redis_client=redis)

        event = TiltChaosResumeEvent(
            resuming_to_state=TiltState.WAIT,
            chaos_resolved=True,
        )

        # Should not raise
        tilt._publish_chaos_resume(event)


class TestLayerCallbackErrorHandling:
    """Test Layer callback error handling in Tilt."""

    def test_layer1_hold_callback_error_does_not_crash(self):
        """Test that Layer1 hold callback error is caught and logged."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        # Set callback that raises
        def failing_callback(session):
            raise RuntimeError("Layer1 hold failed")

        tilt.set_layer1_hold_callback(failing_callback)

        # _execute_lock should not raise
        with patch.object(tilt, '_execute_signal', return_value=True):
            tilt._execute_lock()

    def test_layer2_lock_callback_error_does_not_crash(self):
        """Test that Layer2 lock callback error is caught and logged."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        # Set callback that raises
        def failing_callback(session):
            raise RuntimeError("Layer2 lock failed")

        tilt.set_layer2_lock_callback(failing_callback)

        # _execute_lock should not raise
        with patch.object(tilt, '_execute_signal', return_value=True):
            tilt._execute_lock()

    def test_dpr_re_rank_callback_error_is_logged(self):
        """Test that DPR re-rank callback error is caught and logged."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._incoming_session = "NY"

        # Set callback that raises
        def failing_callback(session):
            raise RuntimeError("DPR re-rank failed")

        tilt.set_dpr_re_rank_callback(failing_callback)

        # _execute_re_rank should proceed even with callback error
        with patch.object(tilt, '_execute_activate', return_value=True):
            tilt._execute_re_rank()

    def test_session_activate_callback_error_does_not_crash(self):
        """Test that session activate callback error is caught and logged."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._incoming_session = "NY"

        # Set callback that raises
        def failing_callback(session):
            raise RuntimeError("Session activate failed")

        tilt.set_session_activate_callback(failing_callback)

        # _execute_activate should not raise
        tilt._execute_activate()


class TestOnChaosEvent:
    """Test on_chaos_event() CHAOS handling."""

    def test_chaos_event_ignored_when_level_normal(self):
        """Test CHAOS events with NORMAL level are ignored."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK

        # ChaosLevel.NORMAL should be ignored
        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.NORMAL

        tilt.on_chaos_event(chaos_event)

        # State should remain LOCK (not suspended)
        assert tilt._state == TiltState.LOCK

    def test_chaos_event_suspends_on_warning(self):
        """Test CHAOS WARNING level suspends Tilt."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.WAIT

        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.WARNING
        chaos_event.lyapunov_value = 0.97

        tilt.on_chaos_event(chaos_event)

        assert tilt._state == TiltState.SUSPENDED

    def test_chaos_event_suspends_on_critical(self):
        """Test CHAOS CRITICAL level suspends Tilt."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK

        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.CRITICAL
        chaos_event.lyapunov_value = 1.02

        tilt.on_chaos_event(chaos_event)

        assert tilt._state == TiltState.SUSPENDED


class TestResumeTiltEdgeCases:
    """Test resume_tilt() edge cases."""

    def test_resume_tilt_from_lock_state(self):
        """Test resuming Tilt when suspended from LOCK state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = TiltState.LOCK
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        with patch.object(tilt, '_execute_lock', return_value=True) as mock_lock:
            tilt.resume_tilt()

        mock_lock.assert_called_once()

    def test_resume_tilt_from_signal_state(self):
        """Test resuming Tilt when suspended from SIGNAL state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = TiltState.SIGNAL
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        with patch.object(tilt, '_execute_signal', return_value=True) as mock_signal:
            tilt.resume_tilt()

        mock_signal.assert_called_once()

    def test_resume_tilt_from_re_rank_state(self):
        """Test resuming Tilt when suspended from RE_RANK state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = TiltState.RE_RANK
        tilt._incoming_session = "NY"

        with patch.object(tilt, '_execute_re_rank', return_value=True) as mock_rerank:
            tilt.resume_tilt()

        mock_rerank.assert_called_once()

    def test_resume_tilt_from_activate_state(self):
        """Test resuming Tilt when suspended from ACTIVATE state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = TiltState.ACTIVATE
        tilt._incoming_session = "NY"

        with patch.object(tilt, '_execute_activate', return_value=True) as mock_activate:
            tilt.resume_tilt()

        mock_activate.assert_called_once()

    def test_resume_tilt_fails_when_suspended_from_none(self):
        """Test resume fails when suspended_from_state is None."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = None
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        result = tilt.resume_tilt()

        assert result is False
        # Should reset to IDLE
        assert tilt._state == TiltState.IDLE


class TestStartTransitionEdgeCases:
    """Test start_transition() edge cases."""

    def test_start_transition_resets_previous_state(self):
        """Test that starting a new transition resets session variables properly."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        # First transition completes
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)
        tilt._transition_to(TiltState.SIGNAL)
        tilt._transition_to(TiltState.IDLE)

        # Now start a new transition
        with patch.object(tilt, '_execute_lock', return_value=True):
            result = tilt.start_transition("NY", "ASIAN")

        # Verify start_transition succeeded
        assert result is True
        # Verify session variables are updated
        assert tilt._closing_session == "NY"
        assert tilt._incoming_session == "ASIAN"
