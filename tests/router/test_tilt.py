"""
Tests for TiltStateMachine.

Story 16.1: Tilt — Universal Session Boundary Mechanism
Tests for the TiltStateMachine class implementing LOCK→SIGNAL→WAIT→RE-RANK→ACTIVATE sequence.
"""

import asyncio
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

    def __init__(self):
        self._data = {}
        self._channels = {}

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
        if channel not in self._channels:
            self._channels[channel] = []
        self._channels[channel].append(message)

    def subscribe(self, channel):
        pass


class TestTiltStateMachineInit:
    """Test TiltStateMachine initialization."""

    def test_initial_state_is_idle(self):
        """Verify Tilt starts in IDLE state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        assert tilt.state == TiltState.IDLE
        assert tilt.is_active is False
        assert tilt.is_suspended is False

    def test_initial_sessions_are_none(self):
        """Verify initial session names are None."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        assert tilt.current_session is None
        assert tilt.incoming_session is None

    def test_initial_timer_is_zero(self):
        """Verify initial regime persistence timer is 0."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        assert tilt.regime_persistence_timer == 0

    def test_instance_id_is_set(self):
        """Verify instance_id is stored correctly."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis, instance_id="tilt-test")

        # Access internal state to verify
        assert tilt._instance_id == "tilt-test"


class TestTiltStartTransition:
    """Test TiltStateMachine.start_transition()."""

    def test_start_transition_from_idle(self):
        """Test starting a transition from IDLE state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        # Note: _execute_lock calls async methods, so we need to mock
        with patch.object(tilt, '_execute_lock', return_value=True):
            result = tilt.start_transition("LONDON", "NY")

        assert result is True

    def test_start_transition_rejected_when_active(self):
        """Test that start_transition is rejected when already in transition."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        result = tilt.start_transition("NY", "ASIAN")

        assert result is False

    def test_start_transition_rejected_when_suspended(self):
        """Test that start_transition is rejected when suspended."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED

        result = tilt.start_transition("LONDON", "NY")

        assert result is False

    def test_start_transition_sets_sessions(self):
        """Test that start_transition stores session names."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        with patch.object(tilt, '_execute_lock', return_value=True):
            tilt.start_transition("LONDON", "NY")

        assert tilt.current_session == "LONDON"
        assert tilt.incoming_session == "NY"


class TestTiltChaosPreemption:
    """Test TiltStateMachine CHAOS preemption."""

    def test_suspend_when_active(self):
        """Test suspending Tilt when active."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.WAIT
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.WARNING
        chaos_event.lyapunov_value = 0.97

        with patch.object(tilt, '_cancel_wait_timer'):
            result = tilt.suspend_tilt(chaos_event)

        assert result is True
        assert tilt.state == TiltState.SUSPENDED
        assert tilt._suspended_from_state == TiltState.WAIT

    def test_suspend_ignored_when_idle(self):
        """Test that CHAOS is ignored when Tilt is IDLE."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.WARNING
        chaos_event.lyapunov_value = 0.97

        result = tilt.suspend_tilt(chaos_event)

        assert result is False
        assert tilt.state == TiltState.IDLE

    def test_resume_tilt(self):
        """Test resuming Tilt after CHAOS resolution."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.SUSPENDED
        tilt._suspended_from_state = TiltState.WAIT
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        with patch.object(tilt, '_execute_wait', return_value=True):
            result = tilt.resume_tilt()

        assert result is True

    def test_resume_tilt_rejected_when_not_suspended(self):
        """Test that resume is rejected when not suspended."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.IDLE

        result = tilt.resume_tilt()

        assert result is False

    def test_on_chaos_event_suspends_active_tilt(self):
        """Test that on_chaos_event suspends active Tilt."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK

        chaos_event = MagicMock()
        chaos_event.chaos_level = ChaosLevel.CRITICAL
        chaos_event.lyapunov_value = 1.02

        tilt.on_chaos_event(chaos_event)

        assert tilt.state == TiltState.SUSPENDED


class TestTiltRegimeConfirmation:
    """Test TiltStateMachine regime confirmation handling."""

    def test_regime_confirmation_resets_timer_during_wait(self):
        """Test that regime confirmation resets timer during WAIT."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.WAIT
        tilt._regime_persistence_timer = 1500  # 5 minutes remaining

        regime_event = MagicMock()

        with patch.object(tilt, '_publish_phase_event'):
            tilt.on_regime_confirmed(regime_event)

        assert tilt.regime_persistence_timer == REGIME_PERSISTENCE_SECONDS

    def test_regime_confirmation_ignored_when_not_wait(self):
        """Test that regime confirmation is ignored when not in WAIT."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._regime_persistence_timer = 0

        regime_event = MagicMock()

        tilt.on_regime_confirmed(regime_event)

        # Timer should remain unchanged (0)
        assert tilt.regime_persistence_timer == 0


class TestTiltPhaseEvent:
    """Test TiltStateMachine.get_current_phase_event()."""

    def test_phase_event_in_idle(self):
        """Test phase event when IDLE."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        event = tilt.get_current_phase_event()

        assert event.state == TiltState.IDLE
        assert event.closing_session == ""
        assert event.incoming_session == ""

    def test_phase_event_in_wait_with_timer(self):
        """Test phase event during WAIT with timer."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.WAIT
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._regime_persistence_timer = 1500

        event = tilt.get_current_phase_event()

        assert event.state == TiltState.WAIT
        assert event.phase == TiltPhase.WAIT
        assert event.regime_persistence_timer == 1500


class TestTiltLayerCallbacks:
    """Test TiltStateMachine layer callbacks."""

    def test_set_layer1_hold_callback(self):
        """Test setting Layer 1 hold callback."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        callback = MagicMock()
        tilt.set_layer1_hold_callback(callback)

        assert tilt._layer1_hold_callback is callback

    def test_set_layer2_lock_callback(self):
        """Test setting Layer 2 lock callback."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        callback = MagicMock()
        tilt.set_layer2_lock_callback(callback)

        assert tilt._layer2_lock_callback is callback

    def test_set_dpr_re_rank_callback(self):
        """Test setting DPR re-rank callback."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        callback = MagicMock(return_value=True)
        tilt.set_dpr_re_rank_callback(callback)

        assert tilt._dpr_re_rank_callback is callback

    def test_set_session_activate_callback(self):
        """Test setting session activation callback."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        callback = MagicMock()
        tilt.set_session_activate_callback(callback)

        assert tilt._session_activate_callback is callback


class TestTiltAuditLog:
    """Test TiltStateMachine audit log."""

    def test_get_audit_log_empty_initially(self):
        """Test that audit log is empty initially."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        log = tilt.get_audit_log()

        assert log == []

    def test_transition_creates_audit_entry(self):
        """Test that state transition creates audit entry."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._transition_to(TiltState.LOCK)

        log = tilt.get_audit_log()

        assert len(log) == 1
        assert log[0].from_state == TiltState.IDLE
        assert log[0].to_state == TiltState.LOCK
        assert log[0].closing_session == "LONDON"
        assert log[0].incoming_session == "NY"


class TestTiltSessionLocking:
    """Test Tilt session locking via Redis."""

    def test_publish_phase_event(self):
        """Test phase event publishing to Redis."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        tilt._publish_phase_event()

        assert "tilt:phase" in redis._channels
        assert len(redis._channels["tilt:phase"]) == 1

    def test_publish_session_close(self):
        """Test session close publishing to Redis."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._publish_session_close("LONDON")

        assert "tilt:session:close" in redis._channels
        assert len(redis._channels["tilt:session:close"]) == 1

    def test_publish_session_open(self):
        """Test session open publishing to Redis."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._publish_session_open("NY")

        assert "tilt:session:open" in redis._channels
        assert len(redis._channels["tilt:session:open"]) == 1


class TestTiltStateTransitions:
    """Test Tilt state transition logic."""

    def test_state_to_phase_mapping(self):
        """Test that state-to-phase mapping is correct."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        # Verify the mapping produces correct phases
        tilt._state = TiltState.LOCK
        event = tilt.get_current_phase_event()
        assert event.phase == TiltPhase.LOCK

        tilt._state = TiltState.SIGNAL
        event = tilt.get_current_phase_event()
        assert event.phase == TiltPhase.SIGNAL

        tilt._state = TiltState.WAIT
        event = tilt.get_current_phase_event()
        assert event.phase == TiltPhase.WAIT

        tilt._state = TiltState.RE_RANK
        event = tilt.get_current_phase_event()
        assert event.phase == TiltPhase.RE_RANK

        tilt._state = TiltState.ACTIVATE
        event = tilt.get_current_phase_event()
        assert event.phase == TiltPhase.ACTIVATE


class TestTiltEdgeCases:
    """Test Tilt edge cases and error conditions."""

    def test_start_transition_stores_both_sessions(self):
        """Test that both sessions are stored on start."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        with patch.object(tilt, '_execute_lock', return_value=True):
            tilt.start_transition("ASIAN", "LONDON")

        assert tilt._closing_session == "ASIAN"
        assert tilt._incoming_session == "LONDON"

    def test_reset_clears_state(self):
        """Test that _reset clears all state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._state = TiltState.LOCK
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"
        tilt._regime_persistence_timer = 1500
        tilt._suspended_from_state = TiltState.SIGNAL

        tilt._reset()

        assert tilt._state == TiltState.IDLE
        assert tilt._closing_session is None
        assert tilt._incoming_session is None
        assert tilt._regime_persistence_timer == 0
        assert tilt._suspended_from_state is None

    def test_is_active_when_not_idle(self):
        """Test is_active returns True when not IDLE."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        for state in [TiltState.LOCK, TiltState.SIGNAL, TiltState.WAIT,
                      TiltState.RE_RANK, TiltState.ACTIVATE, TiltState.SUSPENDED]:
            tilt._state = state
            assert tilt.is_active is True

        tilt._state = TiltState.IDLE
        assert tilt.is_active is False

    def test_is_suspended_only_when_suspended(self):
        """Test is_suspended returns True only for SUSPENDED state."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._state = TiltState.SUSPENDED
        assert tilt.is_suspended is True

        for state in [TiltState.IDLE, TiltState.LOCK, TiltState.SIGNAL,
                      TiltState.WAIT, TiltState.RE_RANK, TiltState.ACTIVATE]:
            tilt._state = state
            assert tilt.is_suspended is False


class TestTiltInternalMethods:
    """Test Tilt internal methods."""

    def test_log_regime_transition(self):
        """Test _log_regime_transition creates audit log entry per AC #3."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        # Should not raise and should create audit log entry
        tilt._log_regime_transition()

        # Verify audit log entry was created
        log = tilt.get_audit_log()
        assert len(log) == 1
        assert log[0].from_state == TiltState.ACTIVATE
        assert log[0].to_state == TiltState.IDLE
        assert log[0].closing_session == "LONDON"
        assert log[0].incoming_session == "NY"
        assert "regime_state" in log[0].metadata

    def test_transition_to_updates_state(self):
        """Test _transition_to updates state and logs."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)
        tilt._closing_session = "LONDON"
        tilt._incoming_session = "NY"

        tilt._transition_to(TiltState.LOCK)

        assert tilt._state == TiltState.LOCK
        assert len(tilt._audit_log) == 1
        assert tilt._audit_log[0].to_state == TiltState.LOCK

    def test_publish_chaos_suspend(self):
        """Test _publish_chaos_suspend publishes to Redis."""
        from src.events.tilt import TiltChaosSuspendEvent

        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        event = TiltChaosSuspendEvent(
            previous_state=TiltState.WAIT,
            chaos_lyapunov=0.97,
        )

        tilt._publish_chaos_suspend(event)

        assert "tilt:phase" in redis._channels

    def test_publish_chaos_resume(self):
        """Test _publish_chaos_resume publishes to Redis."""
        from src.events.tilt import TiltChaosResumeEvent

        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        event = TiltChaosResumeEvent(
            resuming_to_state=TiltState.WAIT,
            chaos_resolved=True,
        )

        tilt._publish_chaos_resume(event)

        assert "tilt:phase" in redis._channels


class TestTiltAsyncWaitTimer:
    """Test Tilt async WAIT timer functionality."""

    def test_wait_timer_cancelled_flag(self):
        """Test wait timer cancellation flag."""
        redis = MockRedis()
        tilt = TiltStateMachine(redis_client=redis)

        tilt._wait_cancelled = False
        tilt._cancel_wait_timer()

        assert tilt._wait_cancelled is True
