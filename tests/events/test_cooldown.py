"""
Tests for Cooldown Event Models.

Story 16.3: Inter-Session Cooldown Window (10:00–13:00 GMT)

Tests for:
- CooldownState enum
- CooldownPhase enum
- NYQueueCandidate model
- InterSessionCooldownStateEvent model
- CooldownPhaseEvent model
- InterSessionCooldownCompletionEvent model
- CooldownAuditLog model
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from src.events.cooldown import (
    COOLDOWN_STATE_TO_PHASE,
    CooldownAuditLog,
    CooldownPhase,
    CooldownPhaseEvent,
    CooldownState,
    InterSessionCooldownCompletionEvent,
    InterSessionCooldownStateEvent,
    NYQueueCandidate,
    STEP_WINDOWS,
)


class TestCooldownState:
    """Test CooldownState enum."""

    def test_all_states_defined(self):
        """Verify all cooldown states are defined."""
        assert CooldownState.PENDING == "pending"
        assert CooldownState.STEP_1_SCORING == "step_1_scoring"
        assert CooldownState.STEP_2_PAPER_RECOVERY == "step_2_paper_recovery"
        assert CooldownState.STEP_3_QUEUE_BUILD == "step_3_queue_build"
        assert CooldownState.STEP_4_HEALTH_CHECK == "step_4_health_check"
        assert CooldownState.COMPLETED == "completed"

    def test_state_values_are_unique(self):
        """Verify all state values are unique strings."""
        values = [s.value for s in CooldownState]
        assert len(values) == len(set(values))


class TestCooldownPhase:
    """Test CooldownPhase enum."""

    def test_all_phases_defined(self):
        """Verify all cooldown phases are defined."""
        assert CooldownPhase.STEP_1 == "STEP_1"
        assert CooldownPhase.STEP_2 == "STEP_2"
        assert CooldownPhase.STEP_3 == "STEP_3"
        assert CooldownPhase.STEP_4 == "STEP_4"
        assert CooldownPhase.COMPLETED == "COMPLETED"


class TestCooldownStateToPhaseMapping:
    """Test COOLDOWN_STATE_TO_PHASE mapping."""

    def test_mapping_complete(self):
        """Verify all states map to phases."""
        assert len(COOLDOWN_STATE_TO_PHASE) == 6
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.PENDING] == CooldownPhase.STEP_1
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.STEP_1_SCORING] == CooldownPhase.STEP_1
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.STEP_2_PAPER_RECOVERY] == CooldownPhase.STEP_2
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.STEP_3_QUEUE_BUILD] == CooldownPhase.STEP_3
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.STEP_4_HEALTH_CHECK] == CooldownPhase.STEP_4
        assert COOLDOWN_STATE_TO_PHASE[CooldownState.COMPLETED] == CooldownPhase.COMPLETED


class TestStepWindows:
    """Test STEP_WINDOWS constant."""

    def test_step_1_window(self):
        """Verify Step 1 window (30 minutes)."""
        window = STEP_WINDOWS[CooldownState.STEP_1_SCORING]
        assert window == (0, 30)  # 10:00-10:30

    def test_step_2_window(self):
        """Verify Step 2 window (60 minutes)."""
        window = STEP_WINDOWS[CooldownState.STEP_2_PAPER_RECOVERY]
        assert window == (30, 90)  # 10:30-11:30

    def test_step_3_window(self):
        """Verify Step 3 window (70 minutes)."""
        window = STEP_WINDOWS[CooldownState.STEP_3_QUEUE_BUILD]
        assert window == (90, 160)  # 11:30-12:40

    def test_step_4_window(self):
        """Verify Step 4 window (20 minutes)."""
        window = STEP_WINDOWS[CooldownState.STEP_4_HEALTH_CHECK]
        assert window == (160, 180)  # 12:40-13:00


class TestNYQueueCandidate:
    """Test NYQueueCandidate model."""

    def test_valid_candidate(self):
        """Test creating a valid NYQueueCandidate."""
        candidate = NYQueueCandidate(
            bot_id="bot-123",
            queue_position=1,
            source="london_performer",
            tier=1,
            dpr_score=0.85,
            session_specialist=True,
            consecutive_paper_wins=3,
        )

        assert candidate.bot_id == "bot-123"
        assert candidate.queue_position == 1
        assert candidate.source == "london_performer"
        assert candidate.tier == 1
        assert candidate.dpr_score == 0.85
        assert candidate.session_specialist is True
        assert candidate.consecutive_paper_wins == 3

    def test_minimal_candidate(self):
        """Test creating NYQueueCandidate with minimal fields."""
        candidate = NYQueueCandidate(
            bot_id="bot-456",
            queue_position=2,
            source="tier_3_dpr",
            tier=3,
        )

        assert candidate.bot_id == "bot-456"
        assert candidate.queue_position == 2
        assert candidate.source == "tier_3_dpr"
        assert candidate.tier == 3
        assert candidate.dpr_score is None
        assert candidate.session_specialist is False
        assert candidate.consecutive_paper_wins == 0

    def test_tier_validation(self):
        """Test tier must be 1, 2, or 3."""
        with pytest.raises(ValidationError):
            NYQueueCandidate(
                bot_id="bot-789",
                queue_position=1,
                source="test",
                tier=4,  # Invalid
            )

    def test_str_representation(self):
        """Test string representation."""
        candidate = NYQueueCandidate(
            bot_id="bot-123",
            queue_position=1,
            source="london_performer",
            tier=1,
        )
        str_repr = str(candidate)
        assert "bot-123" in str_repr
        assert "pos=1" in str_repr
        assert "london_performer" in str_repr
        assert "tier=1" in str_repr


class TestInterSessionCooldownStateEvent:
    """Test InterSessionCooldownStateEvent model."""

    def test_valid_event(self):
        """Test creating a valid state event."""
        now = datetime.now(timezone.utc)
        event = InterSessionCooldownStateEvent(
            state=CooldownState.STEP_1_SCORING,
            current_step=1,
            step_name="London Session Scoring",
            window_start=now,
            window_end=now,
            ny_roster_locked=False,
        )

        assert event.state == CooldownState.STEP_1_SCORING
        assert event.current_step == 1
        assert event.step_name == "London Session Scoring"
        assert event.ny_roster_locked is False

    def test_default_values(self):
        """Test default values are set correctly."""
        event = InterSessionCooldownStateEvent(
            state=CooldownState.PENDING,
            current_step=0,
            step_name="Pending",
        )

        assert event.ny_roster_locked is False
        assert event.window_start is not None
        assert event.window_end is not None
        assert event.timestamp_utc is not None

    def test_str_representation(self):
        """Test string representation."""
        event = InterSessionCooldownStateEvent(
            state=CooldownState.STEP_1_SCORING,
            current_step=1,
            step_name="London Session Scoring",
        )
        str_repr = str(event)
        assert "step_1_scoring" in str_repr
        assert "step=1" in str_repr


class TestCooldownPhaseEvent:
    """Test CooldownPhaseEvent model."""

    def test_valid_event(self):
        """Test creating a valid phase event."""
        now = datetime.now(timezone.utc)
        event = CooldownPhaseEvent(
            phase=CooldownPhase.STEP_1,
            step_name="London Session Scoring",
            started_at=now,
            completed_at=None,
            results={"score": 0.85},
        )

        assert event.phase == CooldownPhase.STEP_1
        assert event.step_name == "London Session Scoring"
        assert event.results == {"score": 0.85}

    def test_to_redis_message(self):
        """Test Redis message serialization."""
        now = datetime.now(timezone.utc)
        event = CooldownPhaseEvent(
            phase=CooldownPhase.STEP_2,
            step_name="Paper Recovery Review",
            started_at=now,
            results={"bots_reviewed": 5},
        )

        msg = event.to_redis_message()
        assert msg["phase"] == "STEP_2"
        assert msg["step_name"] == "Paper Recovery Review"
        assert msg["results"] == {"bots_reviewed": 5}
        assert "timestamp_utc" in msg

    def test_from_redis_message(self):
        """Test creating event from Redis message."""
        now = datetime.now(timezone.utc)
        data = {
            "phase": "STEP_3",
            "step_name": "NY Queue Order",
            "started_at": now.isoformat(),
            "completed_at": None,
            "results": {"candidates": 10},
            "timestamp_utc": now.isoformat(),
        }

        event = CooldownPhaseEvent.from_redis_message(data)
        assert event.phase == CooldownPhase.STEP_3
        assert event.step_name == "NY Queue Order"
        assert event.results == {"candidates": 10}


class TestInterSessionCooldownCompletionEvent:
    """Test InterSessionCooldownCompletionEvent model."""

    def test_valid_event(self):
        """Test creating a valid completion event."""
        now = datetime.now(timezone.utc)
        event = InterSessionCooldownCompletionEvent(
            completed_at=now,
            ny_roster_locked=True,
            roster_summary={"bots": 5, "tier_1": 2},
            tilt_activate_triggered=True,
        )

        assert event.ny_roster_locked is True
        assert event.roster_summary == {"bots": 5, "tier_1": 2}
        assert event.tilt_activate_triggered is True

    def test_default_values(self):
        """Test default values."""
        event = InterSessionCooldownCompletionEvent()

        assert event.ny_roster_locked is True  # Default True
        assert event.roster_summary == {}
        assert event.tilt_activate_triggered is False

    def test_str_representation(self):
        """Test string representation."""
        event = InterSessionCooldownCompletionEvent(
            ny_roster_locked=True,
            tilt_activate_triggered=True,
        )
        str_repr = str(event)
        assert "roster_locked=True" in str_repr
        assert "tilt_activate=True" in str_repr


class TestCooldownAuditLog:
    """Test CooldownAuditLog model."""

    def test_valid_audit_log(self):
        """Test creating a valid audit log entry."""
        now = datetime.now(timezone.utc)
        audit = CooldownAuditLog(
            entry_id="audit-123",
            from_state=CooldownState.PENDING,
            to_state=CooldownState.STEP_1_SCORING,
            step_name="London Session Scoring",
            timestamp_utc=now,
        )

        assert audit.entry_id == "audit-123"
        assert audit.from_state == CooldownState.PENDING
        assert audit.to_state == CooldownState.STEP_1_SCORING
        assert audit.step_name == "London Session Scoring"

    def test_from_state_none_allowed(self):
        """Test from_state can be None for initial entry."""
        audit = CooldownAuditLog(
            entry_id="audit-456",
            from_state=None,
            to_state=CooldownState.PENDING,
            step_name="Pending",
        )

        assert audit.from_state is None
        assert audit.to_state == CooldownState.PENDING

    def test_str_representation(self):
        """Test string representation."""
        audit = CooldownAuditLog(
            entry_id="audit-789",
            from_state=CooldownState.PENDING,
            to_state=CooldownState.STEP_1_SCORING,
            step_name="London Session Scoring",
        )
        str_repr = str(audit)
        assert "pending" in str_repr  # from_state is PENDING
        assert "step_1_scoring" in str_repr
