"""
Cooldown Event Models for Inter-Session Cooldown Window.

Story 16.3: Inter-Session Cooldown Window (10:00–13:00 GMT)

Contains:
- CooldownState: State machine states for the 4-step cooldown sequence
- InterSessionCooldownStateEvent: Current state event for UI subscription
- CooldownPhaseEvent: Phase change event published to Redis pub/sub
- NYQueueCandidate: Data structure for hybrid NY queue composition
- InterSessionCooldownCompletionEvent: Completion event with roster summary

Per NFR-M2: InterSessionCooldownOrchestrator is a synchronous workflow orchestrator
— NO LLM calls in hot path.
Per NFR-D1: All cooldown state transitions logged with timestamps for audit.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CooldownState(str, Enum):
    """
    Inter-Session Cooldown state machine states.

    State transitions:
        PENDING → STEP_1_SCORING → STEP_2_PAPER_RECOVERY → STEP_3_QUEUE_BUILD → STEP_4_HEALTH_CHECK → COMPLETED
    """
    PENDING = "pending"
    STEP_1_SCORING = "step_1_scoring"
    STEP_2_PAPER_RECOVERY = "step_2_paper_recovery"
    STEP_3_QUEUE_BUILD = "step_3_queue_build"
    STEP_4_HEALTH_CHECK = "step_4_health_check"
    COMPLETED = "completed"


class CooldownPhase(str, Enum):
    """
    Cooldown phases for UI display.

    STEP_1: 10:00-10:30 London session scoring
    STEP_2: 10:30-11:30 Paper recovery review
    STEP_3: 11:30-12:40 NY queue order via DPR + Tier remix
    STEP_4: 12:40-13:00 System health/SQS/Sentinel pre-check
    COMPLETED: Cooldown sequence finished
    """
    STEP_1 = "STEP_1"
    STEP_2 = "STEP_2"
    STEP_3 = "STEP_3"
    STEP_4 = "STEP_4"
    COMPLETED = "COMPLETED"


# Mapping from CooldownState to CooldownPhase for UI
COOLDOWN_STATE_TO_PHASE: Dict[CooldownState, CooldownPhase] = {
    CooldownState.PENDING: CooldownPhase.STEP_1,
    CooldownState.STEP_1_SCORING: CooldownPhase.STEP_1,
    CooldownState.STEP_2_PAPER_RECOVERY: CooldownPhase.STEP_2,
    CooldownState.STEP_3_QUEUE_BUILD: CooldownPhase.STEP_3,
    CooldownState.STEP_4_HEALTH_CHECK: CooldownPhase.STEP_4,
    CooldownState.COMPLETED: CooldownPhase.COMPLETED,
}


# Step window definitions (in minutes from start)
STEP_WINDOWS = {
    CooldownState.STEP_1_SCORING: (0, 30),      # 10:00-10:30 (30 min)
    CooldownState.STEP_2_PAPER_RECOVERY: (30, 90),  # 10:30-11:30 (60 min)
    CooldownState.STEP_3_QUEUE_BUILD: (90, 160),    # 11:30-12:40 (70 min)
    CooldownState.STEP_4_HEALTH_CHECK: (160, 180),  # 12:40-13:00 (20 min)
}


class NYQueueCandidate(BaseModel):
    """
    Candidate in the hybrid NY queue.

    Attributes:
        bot_id: Unique identifier for the bot
        queue_position: Position in the NY queue (1-indexed)
        source: Source of this candidate (london_performer, recovery_candidate, tier_3_dpr, tier_2_fresh)
        tier: TIER level (1, 2, or 3)
        dpr_score: DPR composite score (optional)
        session_specialist: Whether bot is a session specialist
        consecutive_paper_wins: Number of consecutive paper trading wins (for recovery candidates)
    """
    bot_id: str = Field(..., description="Unique bot identifier")
    queue_position: int = Field(..., description="Queue position (1-indexed)")
    source: str = Field(
        ...,
        description="Source: london_performer | recovery_candidate | tier_3_dpr | tier_2_fresh"
    )
    tier: int = Field(..., ge=1, le=3, description="TIER level (1, 2, or 3)")
    dpr_score: Optional[float] = Field(None, description="DPR composite score")
    session_specialist: bool = Field(default=False, description="Session specialist flag")
    consecutive_paper_wins: int = Field(default=0, description="Consecutive paper wins for recovery")

    def __str__(self) -> str:
        return (
            f"NYQueueCandidate(bot={self.bot_id}, pos={self.queue_position}, "
            f"source={self.source}, tier={self.tier})"
        )


class InterSessionCooldownStateEvent(BaseModel):
    """
    Current state event for Inter-Session Cooldown.

    Published to cooldown:phase channel for UI subscription.

    Attributes:
        state: Current CooldownState
        current_step: Step number (1-4)
        step_name: Human-readable step name
        window_start: When the cooldown window started (UTC)
        window_end: When the cooldown window ends (UTC)
        ny_roster_locked: Whether NY roster has been locked
        timestamp_utc: When the event was generated
    """
    state: CooldownState = Field(..., description="Current cooldown state")
    current_step: int = Field(..., description="Current step number (1-4)")
    step_name: str = Field(..., description="Human-readable step name")
    window_start: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Cooldown window start time"
    )
    window_end: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Cooldown window end time"
    )
    ny_roster_locked: bool = Field(
        default=False,
        description="Whether NY roster has been locked"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )

    def __str__(self) -> str:
        return (
            f"InterSessionCooldownStateEvent(state={self.state.value}, "
            f"step={self.current_step}: {self.step_name})"
        )


class CooldownPhaseEvent(BaseModel):
    """
    Phase change event for Inter-Session Cooldown.

    Published to Redis cooldown:phase channel on each step transition.

    Attributes:
        phase: Current CooldownPhase
        step_name: Human-readable step name
        started_at: When the phase started
        completed_at: When the phase completed (None if still running)
        results: Step-specific results dictionary
        timestamp_utc: Event timestamp
    """
    phase: CooldownPhase = Field(..., description="Current cooldown phase")
    step_name: str = Field(..., description="Human-readable step name")
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the phase started"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="When the phase completed (None if still running)"
    )
    results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Step-specific results"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )

    def __str__(self) -> str:
        return (
            f"CooldownPhaseEvent(phase={self.phase.value}, "
            f"step={self.step_name})"
        )

    def to_redis_message(self) -> dict:
        """
        Convert to dictionary for Redis pub/sub serialization.

        Returns:
            Dictionary suitable for Redis publish()
        """
        return {
            "phase": self.phase.value,
            "step_name": self.step_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "results": self.results,
            "timestamp_utc": self.timestamp_utc.isoformat(),
        }

    @classmethod
    def from_redis_message(cls, data: dict) -> "CooldownPhaseEvent":
        """
        Create CooldownPhaseEvent from Redis message data.

        Args:
            data: Dictionary from Redis subscribe message

        Returns:
            CooldownPhaseEvent instance
        """
        started_at = data.get("started_at")
        if started_at and isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)

        completed_at = data.get("completed_at")
        if completed_at and isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        timestamp_utc = data.get("timestamp_utc")
        if timestamp_utc and isinstance(timestamp_utc, str):
            timestamp_utc = datetime.fromisoformat(timestamp_utc)

        return cls(
            phase=CooldownPhase(data["phase"]),
            step_name=data["step_name"],
            started_at=started_at or datetime.now(timezone.utc),
            completed_at=completed_at,
            results=data.get("results", {}),
            timestamp_utc=timestamp_utc or datetime.now(timezone.utc),
        )


class InterSessionCooldownCompletionEvent(BaseModel):
    """
    Completion event for Inter-Session Cooldown.

    Published when the cooldown sequence completes at 13:00 GMT.

    Attributes:
        completed_at: When the cooldown completed
        ny_roster_locked: Whether NY roster was locked
        roster_summary: Summary of the NY roster
        tilt_activate_triggered: Whether Tilt ACTIVATE was triggered
        timestamp_utc: Event timestamp
    """
    completed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When cooldown completed"
    )
    ny_roster_locked: bool = Field(
        default=True,
        description="NY roster was locked"
    )
    roster_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of NY roster composition"
    )
    tilt_activate_triggered: bool = Field(
        default=False,
        description="Whether Tilt ACTIVATE was triggered for NY open"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )

    def __str__(self) -> str:
        return (
            f"InterSessionCooldownCompletionEvent("
            f"roster_locked={self.ny_roster_locked}, "
            f"tilt_activate={self.tilt_activate_triggered})"
        )


class CooldownAuditLog(BaseModel):
    """
    Immutable audit log entry for Inter-Session Cooldown state transitions.

    Per NFR-D1: Trade records and position data must be persisted before
    any system acknowledgment. All cooldown state transitions are logged.

    Attributes:
        entry_id: Unique identifier for this audit entry
        from_state: Previous state (None for initial)
        to_state: New state
        step_name: Name of the step
        timestamp_utc: When transition occurred
        metadata: Additional context
    """
    entry_id: str = Field(..., description="Unique audit entry ID")
    from_state: Optional[CooldownState] = Field(None, description="Previous state")
    to_state: CooldownState = Field(..., description="New state")
    step_name: str = Field(..., description="Human-readable step name")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When transition occurred"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        from_str = self.from_state.value if self.from_state else "NONE"
        return (
            f"CooldownAudit({from_str}→{self.to_state.value}, "
            f"step={self.step_name})"
        )
