"""
Tilt Event Models for Universal Session Boundary Mechanism.

Story 16.1: Tilt — Universal Session Boundary Mechanism

Contains:
- TiltState: State machine states (IDLE, LOCK, SIGNAL, WAIT, RE_RANK, ACTIVATE, SUSPENDED)
- TiltPhase: Phases for UI display (LOCK, SIGNAL, WAIT, RE_RANK, ACTIVATE)
- TiltPhaseEvent: Event published on each phase transition for UI subscription

Per NFR-M2: Tilt is a synchronous state machine — NO LLM calls in hot path.
Per NFR-D1: All Tilt phase transitions logged with timestamps for audit.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class TiltState(str, Enum):
    """
    Tilt state machine states.

    State transitions:
        IDLE → LOCK → SIGNAL → WAIT → RE_RANK → ACTIVATE → IDLE
                  ↑
                  └── SUSPENDED (when CHAOS fires during any phase)
                      Resume from SUSPENDED → previous phase when CHAOS resolves
    """
    IDLE = "idle"
    LOCK = "lock"
    SIGNAL = "signal"
    WAIT = "wait"
    RE_RANK = "re_rank"
    ACTIVATE = "activate"
    SUSPENDED = "suspended"


class TiltPhase(str, Enum):
    """
    Tilt phases for UI display.

    These are the human-readable phases shown in the Tilt overlay.
    """
    LOCK = "LOCK"
    SIGNAL = "SIGNAL"
    WAIT = "WAIT"
    RE_RANK = "RE_RANK"
    ACTIVATE = "ACTIVATE"


# Mapping from TiltState to TiltPhase for UI
TILT_STATE_TO_PHASE = {
    TiltState.LOCK: TiltPhase.LOCK,
    TiltState.SIGNAL: TiltPhase.SIGNAL,
    TiltState.WAIT: TiltPhase.WAIT,
    TiltState.RE_RANK: TiltPhase.RE_RANK,
    TiltState.ACTIVATE: TiltPhase.ACTIVATE,
}


# Regime persistence timer constant (30 minutes = 1800 seconds)
REGIME_PERSISTENCE_SECONDS = 1800


class TiltPhaseEvent(BaseModel):
    """
    Tilt phase change event for UI and internal subscription.

    Published to Redis 'tilt:phase' channel on each state transition.

    Attributes:
        phase: Current Tilt phase (LOCK, SIGNAL, WAIT, RE_RANK, ACTIVATE)
        state: Internal TiltState for state machine tracking
        closing_session: Name of the session that is closing
        incoming_session: Name of the session that is opening
        regime_persistence_timer: Seconds remaining in WAIT phase (0 otherwise)
        timestamp_utc: When the phase transition occurred
        metadata: Additional context (previous state, chaos info, etc.)
    """
    phase: TiltPhase = Field(..., description="Current Tilt phase for UI")
    state: TiltState = Field(..., description="Internal state machine state")
    closing_session: str = Field(..., description="Session that is closing")
    incoming_session: str = Field(..., description="Session that is opening")
    regime_persistence_timer: int = Field(
        default=0,
        description="Seconds remaining in WAIT phase (0 when not in WAIT)"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When phase transition occurred"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional context (previous_state, chaos_event, etc.)"
    )

    def __str__(self) -> str:
        return (
            f"TiltPhaseEvent(phase={self.phase.value}, "
            f"session={self.closing_session}→{self.incoming_session}, "
            f"timer={self.regime_persistence_timer}s)"
        )

    def to_redis_message(self) -> dict:
        """
        Convert to dictionary for Redis pub/sub serialization.

        Returns:
            Dictionary suitable for Redis publish()
        """
        return {
            "phase": self.phase.value,
            "state": self.state.value,
            "closing_session": self.closing_session,
            "incoming_session": self.incoming_session,
            "regime_persistence_timer": self.regime_persistence_timer,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_redis_message(cls, data: dict) -> "TiltPhaseEvent":
        """
        Create TiltPhaseEvent from Redis message data.

        Args:
            data: Dictionary from Redis subscribe message

        Returns:
            TiltPhaseEvent instance
        """
        return cls(
            phase=TiltPhase(data["phase"]),
            state=TiltState(data["state"]),
            closing_session=data["closing_session"],
            incoming_session=data["incoming_session"],
            regime_persistence_timer=data.get("regime_persistence_timer", 0),
            timestamp_utc=datetime.fromisoformat(data["timestamp_utc"]),
            metadata=data.get("metadata", {}),
        )


class TiltSessionBoundaryEvent(BaseModel):
    """
    Event indicating a session boundary has been detected.

    This event is produced by SessionDetector when a canonical window
    transition is detected. Tilt subscribes to these events to trigger
    the session transition sequence.

    Attributes:
        closing_session: Session that is about to close
        incoming_session: Session that is about to open
        boundary_time_utc: When the boundary occurs (UTC)
        metadata: Additional context (reason, etc.)
    """
    closing_session: str = Field(..., description="Session that is closing")
    incoming_session: str = Field(..., description="Session that is opening")
    boundary_time_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the boundary occurs"
    )
    metadata: dict = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        return (
            f"TiltSessionBoundaryEvent({self.closing_session}→{self.incoming_session} "
            f"at {self.boundary_time_utc.isoformat()})"
        )


class TiltChaosSuspendEvent(BaseModel):
    """
    Event indicating Tilt has been suspended due to CHAOS.

    When Layer 3 fires a CHAOS event, Tilt transitions to SUSPENDED state
    and publishes this event. Used for UI to show Tilt is paused.

    Attributes:
        previous_state: State Tilt was in before suspension
        chaos_lyapunov: Lyapunov value that triggered CHAOS
        timestamp_utc: When suspension occurred
        metadata: Additional context
    """
    previous_state: TiltState = Field(..., description="State before suspension")
    chaos_lyapunov: float = Field(..., description="Lyapunov value that triggered CHAOS")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When suspension occurred"
    )
    metadata: dict = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        return (
            f"TiltChaosSuspendEvent(previous={self.previous_state.value}, "
            f"lyapunov={self.chaos_lyapunov:.4f})"
        )


class TiltChaosResumeEvent(BaseModel):
    """
    Event indicating Tilt has resumed after CHAOS resolution.

    When Layer 3 CHAOS is resolved and session transition is still valid,
    Tilt resumes from the previous state.

    Attributes:
        resuming_to_state: State Tilt is resuming to
        chaos_resolved: Whether CHAOS has been fully resolved
        timestamp_utc: When resumption occurred
        metadata: Additional context
    """
    resuming_to_state: TiltState = Field(..., description="State Tilt is resuming to")
    chaos_resolved: bool = Field(default=True, description="Whether CHAOS is resolved")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When resumption occurred"
    )
    metadata: dict = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        return (
            f"TiltChaosResumeEvent(resuming_to={self.resuming_to_state.value}, "
            f"resolved={self.chaos_resolved})"
        )


class TiltTransitionAuditLog(BaseModel):
    """
    Immutable audit log entry for Tilt state transitions.

    Per NFR-D1: Trade records and position data must be persisted before
    any system acknowledgment. All Tilt phase transitions are logged.

    Attributes:
        entry_id: Unique identifier for this audit entry
        from_state: Previous state (None for initial)
        to_state: New state
        closing_session: Session that closed
        incoming_session: Session that opened
        timestamp_utc: When transition occurred
        metadata: Additional context
    """
    entry_id: str = Field(..., description="Unique audit entry ID")
    from_state: Optional[TiltState] = Field(None, description="Previous state")
    to_state: TiltState = Field(..., description="New state")
    closing_session: str = Field(..., description="Session that closed")
    incoming_session: str = Field(..., description="Session that opened")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When transition occurred"
    )
    metadata: dict = Field(default_factory=dict, description="Additional context")

    def __str__(self) -> str:
        from_str = self.from_state.value if self.from_state else "NONE"
        return (
            f"TiltAudit({from_str}→{self.to_state.value}, "
            f"{self.closing_session}→{self.incoming_session})"
        )
