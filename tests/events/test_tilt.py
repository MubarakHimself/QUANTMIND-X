"""
Tests for Tilt Event Models.

Story 16.1: Tilt — Universal Session Boundary Mechanism
Tests for TiltState, TiltPhase, TiltPhaseEvent, and related models.
"""

import pytest
from datetime import datetime, timezone

from src.events.tilt import (
    REGIME_PERSISTENCE_SECONDS,
    TiltChaosResumeEvent,
    TiltChaosSuspendEvent,
    TiltPhase,
    TiltPhaseEvent,
    TiltSessionBoundaryEvent,
    TiltState,
    TiltTransitionAuditLog,
)


class TestTiltState:
    """Test TiltState enum values."""

    def test_all_states_exist(self):
        """Verify all expected Tilt states are defined."""
        assert TiltState.IDLE.value == "idle"
        assert TiltState.LOCK.value == "lock"
        assert TiltState.SIGNAL.value == "signal"
        assert TiltState.WAIT.value == "wait"
        assert TiltState.RE_RANK.value == "re_rank"
        assert TiltState.ACTIVATE.value == "activate"
        assert TiltState.SUSPENDED.value == "suspended"

    def test_state_is_string_enum(self):
        """Verify TiltState is a string enum for serialization."""
        assert isinstance(TiltState.LOCK.value, str)


class TestTiltPhase:
    """Test TiltPhase enum values for UI display."""

    def test_all_phases_exist(self):
        """Verify all expected Tilt phases are defined."""
        assert TiltPhase.LOCK.value == "LOCK"
        assert TiltPhase.SIGNAL.value == "SIGNAL"
        assert TiltPhase.WAIT.value == "WAIT"
        assert TiltPhase.RE_RANK.value == "RE_RANK"
        assert TiltPhase.ACTIVATE.value == "ACTIVATE"

    def test_phase_is_string_enum(self):
        """Verify TiltPhase is a string enum for serialization."""
        assert isinstance(TiltPhase.LOCK.value, str)


class TestRegimePersistenceSeconds:
    """Test regime persistence timer constant."""

    def test_regime_persistence_is_30_minutes(self):
        """Verify REGIME_PERSISTENCE_SECONDS is 1800 (30 minutes)."""
        assert REGIME_PERSISTENCE_SECONDS == 1800


class TestTiltPhaseEvent:
    """Test TiltPhaseEvent model."""

    def test_create_phase_event(self):
        """Test TiltPhaseEvent creation with all fields."""
        event = TiltPhaseEvent(
            phase=TiltPhase.LOCK,
            state=TiltState.LOCK,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1800,
        )

        assert event.phase == TiltPhase.LOCK
        assert event.state == TiltState.LOCK
        assert event.closing_session == "LONDON"
        assert event.incoming_session == "NY"
        assert event.regime_persistence_timer == 1800

    def test_phase_event_timer_zero_when_not_wait(self):
        """Test that regime_persistence_timer defaults to 0."""
        event = TiltPhaseEvent(
            phase=TiltPhase.LOCK,
            state=TiltState.LOCK,
            closing_session="LONDON",
            incoming_session="NY",
        )

        assert event.regime_persistence_timer == 0

    def test_phase_event_timestamp_set(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        event = TiltPhaseEvent(
            phase=TiltPhase.SIGNAL,
            state=TiltState.SIGNAL,
            closing_session="LONDON",
            incoming_session="NY",
        )
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp_utc <= after

    def test_phase_event_metadata(self):
        """Test metadata is stored correctly."""
        event = TiltPhaseEvent(
            phase=TiltPhase.WAIT,
            state=TiltState.WAIT,
            closing_session="LONDON",
            incoming_session="NY",
            metadata={"instance_id": "tilt-1", "timer_reset_count": 2},
        )

        assert event.metadata["instance_id"] == "tilt-1"
        assert event.metadata["timer_reset_count"] == 2

    def test_phase_event_str_representation(self):
        """Test string representation."""
        event = TiltPhaseEvent(
            phase=TiltPhase.WAIT,
            state=TiltState.WAIT,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1500,
        )

        str_repr = str(event)
        assert "WAIT" in str_repr
        assert "LONDON" in str_repr
        assert "NY" in str_repr
        assert "1500s" in str_repr

    def test_phase_event_to_redis_message(self):
        """Test Redis message serialization."""
        event = TiltPhaseEvent(
            phase=TiltPhase.LOCK,
            state=TiltState.LOCK,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1800,
        )

        msg = event.to_redis_message()
        assert msg["phase"] == "LOCK"
        assert msg["state"] == "lock"
        assert msg["closing_session"] == "LONDON"
        assert msg["incoming_session"] == "NY"
        assert msg["regime_persistence_timer"] == 1800
        assert "timestamp_utc" in msg

    def test_phase_event_from_redis_message(self):
        """Test creation from Redis message."""
        data = {
            "phase": "WAIT",
            "state": "wait",
            "closing_session": "ASIAN",
            "incoming_session": "LONDON",
            "regime_persistence_timer": 900,
            "timestamp_utc": "2026-03-25T10:00:00+00:00",
            "metadata": {"instance_id": "tilt-1"},
        }

        event = TiltPhaseEvent.from_redis_message(data)
        assert event.phase == TiltPhase.WAIT
        assert event.state == TiltState.WAIT
        assert event.closing_session == "ASIAN"
        assert event.incoming_session == "LONDON"
        assert event.regime_persistence_timer == 900


class TestTiltSessionBoundaryEvent:
    """Test TiltSessionBoundaryEvent model."""

    def test_create_boundary_event(self):
        """Test TiltSessionBoundaryEvent creation."""
        event = TiltSessionBoundaryEvent(
            closing_session="LONDON",
            incoming_session="NY",
        )

        assert event.closing_session == "LONDON"
        assert event.incoming_session == "NY"

    def test_boundary_event_timestamp(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        event = TiltSessionBoundaryEvent(
            closing_session="LONDON",
            incoming_session="NY",
        )
        after = datetime.now(timezone.utc)

        assert before <= event.boundary_time_utc <= after

    def test_boundary_event_str_representation(self):
        """Test string representation."""
        event = TiltSessionBoundaryEvent(
            closing_session="LONDON",
            incoming_session="NY",
        )

        str_repr = str(event)
        assert "LONDON" in str_repr
        assert "NY" in str_repr


class TestTiltChaosSuspendEvent:
    """Test TiltChaosSuspendEvent model."""

    def test_create_chaos_suspend_event(self):
        """Test TiltChaosSuspendEvent creation."""
        event = TiltChaosSuspendEvent(
            previous_state=TiltState.WAIT,
            chaos_lyapunov=0.97,
        )

        assert event.previous_state == TiltState.WAIT
        assert event.chaos_lyapunov == 0.97

    def test_chaos_suspend_event_metadata(self):
        """Test metadata is stored correctly."""
        event = TiltChaosSuspendEvent(
            previous_state=TiltState.SIGNAL,
            chaos_lyapunov=1.02,
            metadata={"instance_id": "tilt-1"},
        )

        assert event.metadata["instance_id"] == "tilt-1"

    def test_chaos_suspend_event_str(self):
        """Test string representation."""
        event = TiltChaosSuspendEvent(
            previous_state=TiltState.WAIT,
            chaos_lyapunov=0.97,
        )

        str_repr = str(event)
        assert "wait" in str_repr
        assert "0.9700" in str_repr


class TestTiltChaosResumeEvent:
    """Test TiltChaosResumeEvent model."""

    def test_create_chaos_resume_event(self):
        """Test TiltChaosResumeEvent creation."""
        event = TiltChaosResumeEvent(
            resuming_to_state=TiltState.WAIT,
            chaos_resolved=True,
        )

        assert event.resuming_to_state == TiltState.WAIT
        assert event.chaos_resolved is True

    def test_chaos_resume_event_defaults(self):
        """Test default values."""
        event = TiltChaosResumeEvent(resuming_to_state=TiltState.LOCK)

        assert event.chaos_resolved is True

    def test_chaos_resume_event_str(self):
        """Test string representation."""
        event = TiltChaosResumeEvent(
            resuming_to_state=TiltState.WAIT,
            chaos_resolved=True,
        )

        str_repr = str(event)
        assert "wait" in str_repr
        assert "True" in str_repr


class TestTiltTransitionAuditLog:
    """Test TiltTransitionAuditLog model."""

    def test_create_audit_log_entry(self):
        """Test TiltTransitionAuditLog creation."""
        entry = TiltTransitionAuditLog(
            entry_id="audit-123",
            from_state=TiltState.LOCK,
            to_state=TiltState.SIGNAL,
            closing_session="LONDON",
            incoming_session="NY",
        )

        assert entry.entry_id == "audit-123"
        assert entry.from_state == TiltState.LOCK
        assert entry.to_state == TiltState.SIGNAL
        assert entry.closing_session == "LONDON"
        assert entry.incoming_session == "NY"

    def test_audit_log_initial_transition(self):
        """Test audit log for initial transition (no from_state)."""
        entry = TiltTransitionAuditLog(
            entry_id="audit-456",
            from_state=None,
            to_state=TiltState.LOCK,
            closing_session="ASIAN",
            incoming_session="LONDON",
        )

        assert entry.from_state is None
        assert entry.to_state == TiltState.LOCK

    def test_audit_log_str(self):
        """Test string representation."""
        entry = TiltTransitionAuditLog(
            entry_id="audit-789",
            from_state=TiltState.LOCK,
            to_state=TiltState.SIGNAL,
            closing_session="LONDON",
            incoming_session="NY",
        )

        str_repr = str(entry)
        assert "lock" in str_repr
        assert "signal" in str_repr
        assert "LONDON" in str_repr
        assert "NY" in str_repr


class TestTiltEventIntegration:
    """Integration tests for Tilt events with realistic scenarios."""

    def test_full_tilt_sequence_phases(self):
        """Test that all Tilt phases are correctly defined."""
        phases = [TiltPhase.LOCK, TiltPhase.SIGNAL, TiltPhase.WAIT, TiltPhase.RE_RANK, TiltPhase.ACTIVATE]
        states = [TiltState.LOCK, TiltState.SIGNAL, TiltState.WAIT, TiltState.RE_RANK, TiltState.ACTIVATE]

        for phase, state in zip(phases, states):
            event = TiltPhaseEvent(
                phase=phase,
                state=state,
                closing_session="LONDON",
                incoming_session="NY",
                regime_persistence_timer=REGIME_PERSISTENCE_SECONDS,
            )
            assert event.phase == phase
            assert event.state == state

    def test_chaos_suspend_during_wait(self):
        """Test chaos suspend event during WAIT phase."""
        # Simulate chaos during WAIT
        suspend = TiltChaosSuspendEvent(
            previous_state=TiltState.WAIT,
            chaos_lyapunov=0.97,
        )

        assert suspend.previous_state == TiltState.WAIT
        assert suspend.chaos_lyapunov == 0.97

    def test_chaos_resume_after_resolution(self):
        """Test chaos resume after CHAOS resolution."""
        # Simulate resume after chaos
        resume = TiltChaosResumeEvent(
            resuming_to_state=TiltState.WAIT,
            chaos_resolved=True,
        )

        assert resume.resuming_to_state == TiltState.WAIT
        assert resume.chaos_resolved is True

    def test_timer_countdown_during_wait(self):
        """Test timer countdown values during WAIT phase."""
        # Initial WAIT
        event_start = TiltPhaseEvent(
            phase=TiltPhase.WAIT,
            state=TiltState.WAIT,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1800,
        )
        assert event_start.regime_persistence_timer == 1800

        # After 5 minutes (300 seconds)
        event_5min = TiltPhaseEvent(
            phase=TiltPhase.WAIT,
            state=TiltState.WAIT,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1500,
        )
        assert event_5min.regime_persistence_timer == 1500

        # After regime confirmation (reset to 1800)
        event_reset = TiltPhaseEvent(
            phase=TiltPhase.WAIT,
            state=TiltState.WAIT,
            closing_session="LONDON",
            incoming_session="NY",
            regime_persistence_timer=1800,
        )
        assert event_reset.regime_persistence_timer == 1800

    def test_session_boundary_transitions(self):
        """Test various session boundary scenarios."""
        # London -> NY
        boundary1 = TiltSessionBoundaryEvent(
            closing_session="LONDON",
            incoming_session="NY",
        )
        assert boundary1.closing_session == "LONDON"
        assert boundary1.incoming_session == "NY"

        # ASIAN -> LONDON
        boundary2 = TiltSessionBoundaryEvent(
            closing_session="ASIAN",
            incoming_session="LONDON",
        )
        assert boundary2.closing_session == "ASIAN"
        assert boundary2.incoming_session == "LONDON"

        # NY -> ASIAN (next day)
        boundary3 = TiltSessionBoundaryEvent(
            closing_session="NY",
            incoming_session="ASIAN",
        )
        assert boundary3.closing_session == "NY"
        assert boundary3.incoming_session == "ASIAN"
