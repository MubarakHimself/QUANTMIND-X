"""
Chaos Event Models for Layer 3 Kill Switch.

Story 14.3: Layer 3 CHAOS + Kill Switch Forced Exit

Contains:
- ChaosEvent: Triggered when Lyapunov Exponent exceeds 0.95 threshold
- RVOLWarningEvent: Triggered when RVOL < 0.5 on a symbol with open positions

Per NFR-M2: Layer 3 is a synchronous, event-driven system — NO LLM calls in hot path.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ChaosLevel(str, Enum):
    """Chaos levels based on Lyapunov Exponent threshold."""
    NORMAL = "NORMAL"  # Lyapunov < 0.95
    WARNING = "WARNING"  # Lyapunov 0.95-0.99
    CRITICAL = "CRITICAL"  # Lyapunov >= 0.99


class ForcedExitOutcome(str, Enum):
    """Outcome of a forced exit attempt."""
    FILLED = "filled"  # Fully closed at market
    PARTIAL = "partial"  # Partially closed
    REJECTED = "rejected"  # Broker rejected the close
    LOCK_CONFLICT = "lock_conflict"  # Could not release Layer 2 lock
    SVSS_UNAVAILABLE = "svss_unavailable"  # SVSS not available, using fallback


class ChaosEvent(BaseModel):
    """
    Chaos event triggered when Lyapunov Exponent exceeds threshold.

    AC #1: Given Lyapunov Exponent exceeds 0.95 (CHAOS threshold),
    When Layer 3 evaluates, Then all scalping positions are flagged for forced exit.

    Attributes:
        lyapunov_value: Current Lyapunov Exponent value
        chaos_level: Classification of chaos severity
        threshold: Threshold that was breached (default 0.95)
        tickets: List of open ticket numbers flagged for forced exit
        timestamp_utc: When the chaos event was detected
        metadata: Additional context (source sensor, etc.)
    """
    lyapunov_value: float = Field(..., description="Current Lyapunov Exponent value")
    chaos_level: ChaosLevel = Field(..., description="Chaos severity level")
    threshold: float = Field(default=0.95, description="CHAOS threshold that was breached")
    tickets: List[int] = Field(default_factory=list, description="Tickets flagged for forced exit")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When chaos event was detected"
    )
    metadata: dict = Field(default_factory=dict, description="Additional sensor context")

    def __str__(self) -> str:
        return (
            f"ChaosEvent(level={self.chaos_level.value}, lyapunov={self.lyapunov_value:.4f}, "
            f"tickets={len(self.tickets)}, threshold={self.threshold})"
        )

    @classmethod
    def create(
        cls,
        lyapunov_value: float,
        tickets: List[int],
        threshold: float = 0.95,
        metadata: Optional[dict] = None
    ) -> "ChaosEvent":
        """
        Factory method to create a ChaosEvent with automatic level classification.

        Args:
            lyapunov_value: Current Lyapunov Exponent value
            tickets: List of tickets to flag for forced exit
            threshold: CHAOS threshold (default 0.95)
            metadata: Additional context

        Returns:
            ChaosEvent instance
        """
        if lyapunov_value >= 0.99:
            level = ChaosLevel.CRITICAL
        elif lyapunov_value >= 0.95:
            level = ChaosLevel.WARNING
        else:
            level = ChaosLevel.NORMAL

        return cls(
            lyapunov_value=lyapunov_value,
            chaos_level=level,
            threshold=threshold,
            tickets=tickets,
            metadata=metadata or {}
        )


class RVOLWarningEvent(BaseModel):
    """
    RVOL warning event for SVSS→Layer3 early warning chain.

    AC #3: Given SVSS reports RVOL < 0.5 on a symbol with open positions,
    When Layer 3 evaluates, Then new entries on that symbol are blocked.

    Note: This AC requires Epic 15 (SVSS) to be complete first.
    Layer 3 implements graceful fallback: if SVSS unavailable, rely on LYAPUNOV_EXCEEDED path.

    Attributes:
        symbol: Trading symbol with low RVOL
        rvol: Relative Volume value
        threshold: Threshold that was breached (< 0.5)
        has_open_positions: Whether the symbol has open positions
        timestamp_utc: When the warning was detected
        metadata: Additional context (SVSS sensor info, correlation data)
    """
    symbol: str = Field(..., description="Symbol with low RVOL")
    rvol: float = Field(..., description="Relative Volume value")
    threshold: float = Field(default=0.5, description="RVOL threshold")
    has_open_positions: bool = Field(default=False, description="Whether symbol has open positions")
    blocked_entries: bool = Field(default=True, description="Whether new entries are blocked")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When RVOL warning was detected"
    )
    metadata: dict = Field(default_factory=dict, description="Additional SVSS context")

    def __str__(self) -> str:
        return (
            f"RVOLWarningEvent(symbol={self.symbol}, rvol={self.rvol:.2f}, "
            f"has_positions={self.has_open_positions}, blocked={self.blocked_entries})"
        )

    @classmethod
    def create(
        cls,
        symbol: str,
        rvol: float,
        has_open_positions: bool = False,
        threshold: float = 0.5,
        metadata: Optional[dict] = None
    ) -> "RVOLWarningEvent":
        """
        Factory method to create an RVOLWarningEvent.

        Args:
            symbol: Trading symbol
            rvol: Relative Volume value
            has_open_positions: Whether symbol has open positions
            threshold: RVOL threshold (default 0.5)
            metadata: Additional context

        Returns:
            RVOLWarningEvent instance
        """
        return cls(
            symbol=symbol,
            rvol=rvol,
            threshold=threshold,
            has_open_positions=has_open_positions,
            blocked_entries=rvol < threshold,
            metadata=metadata or {}
        )


class KillSwitchResult(BaseModel):
    """
    Result of a kill switch forced exit operation.

    Used by Layer3KillSwitch.process_kill_queue() to record outcomes.

    Attributes:
        ticket: MT5 ticket number
        outcome: Outcome of the forced exit attempt
        lyapunov_triggered: Whether Lyapunov threshold triggered this exit
        rvol_triggered: Whether RVOL warning triggered this exit
        lock_released: Whether Layer 2 Redis lock was released before exit
        partial_volume: Volume that was partially closed (if partial)
        error: Error message if rejected
        timestamp_utc: When the result was recorded
    """
    ticket: int = Field(..., description="MT5 ticket number")
    outcome: ForcedExitOutcome = Field(..., description="Outcome of forced exit")
    lyapunov_triggered: bool = Field(default=False, description="CHAOS/Lyapunov triggered exit")
    rvol_triggered: bool = Field(default=False, description="RVOL warning triggered exit")
    lock_released: bool = Field(default=False, description="Layer 2 lock was released")
    partial_volume: Optional[float] = Field(None, description="Volume partially closed")
    error: Optional[str] = Field(None, description="Error message if rejected")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When result was recorded"
    )

    def __str__(self) -> str:
        return (
            f"KillSwitchResult(ticket={self.ticket}, outcome={self.outcome.value}, "
            f"lyapunov={self.lyapunov_triggered}, rvol={self.rvol_triggered})"
        )
