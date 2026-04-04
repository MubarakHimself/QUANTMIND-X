"""
SSL Event Models for Survivorship Selection Loop.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Contains:
- SSLCircuitBreakerEvent: Event emitted on all SSL state transitions
- SSLState: Enum for SSL state machine states
- SSLEventType: Enum for SSL event types

Per NFR-M2: SSL is a synchronous state machine — NO LLM calls in hot path.
Per NFR-D1: All SSL events logged immutably to bot_circuit_breaker before Redis publish.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SSLState(str, Enum):
    """SSL state machine states."""
    LIVE = "live"       # Bot is actively trading live
    PAPER = "paper"     # Bot is in paper trading (quarantined from live)
    RECOVERY = "recovery"  # Bot is recovering from paper (2 consecutive wins)
    RETIRED = "retired"  # Bot has failed to recover and is retired


class SSLEventType(str, Enum):
    """SSL event types for state transitions."""
    MOVE_TO_PAPER = "move_to_paper"
    RECOVERY_STEP_1 = "recovery_step_1"
    RECOVERY_CONFIRMED = "recovery_confirmed"
    RETIRED = "retired"
    # Story 18.3: 3-loss-in-a-day halt
    HALT_FOR_DAY = "halt_for_day"  # FAIL-S02: 3 losses in a day
    QUARANTINE_FOR_WEEK = "quarantine_for_week"  # FAIL-S03: 3 days with 3-loss trigger in week


class BotTier(str, Enum):
    """Bot paper trading tier definitions."""
    TIER_1 = "TIER_1"  # Quarantined live (already passed paper before going live)
    TIER_2 = "TIER_2"  # Fresh from AlphaForge (2-week minimum before live eligibility)


class TradeOutcome(BaseModel):
    """
    Trade outcome data for SSL event emission (AC#1).

    Attributes:
        pnl: Net profit/loss of the trade
        win_rate: Session win rate at time of trade
        ev_per_trade: Expected value per trade
        session_id: Session identifier
    """
    pnl: float = Field(..., description="Net profit/loss of the trade")
    win_rate: float = Field(..., description="Session win rate at trade time")
    ev_per_trade: float = Field(..., description="Expected value per trade")
    session_id: str = Field(..., description="Session identifier")


class SSLCircuitBreakerEvent(BaseModel):
    """
    SSL Circuit Breaker event emitted on all state transitions.

    Published to Redis `ssl:events` channel after database commit per NFR-D1.

    Attributes:
        bot_id: Bot identifier
        magic_number: MT5 magic number (strategy version identifier)
        event_type: Type of SSL event
        consecutive_losses: Current consecutive loss count at event time
        tier: Assigned tier (TIER_1 or TIER_2) for move_to_paper events
        previous_state: State before transition
        new_state: State after transition
        recovery_win_count: Recovery win count (for recovery events)
        timestamp_utc: When event occurred
        metadata: Additional context
        trade_outcome: Trade outcome data for AC#1 (optional)
    """
    bot_id: str = Field(..., description="Bot identifier")
    magic_number: str = Field(..., description="MT5 magic number")
    event_type: SSLEventType = Field(..., description="Type of SSL event")
    consecutive_losses: int = Field(..., description="Consecutive loss count at event time")
    tier: Optional[Literal["TIER_1", "TIER_2"]] = Field(
        None, description="Assigned tier for move_to_paper events"
    )
    previous_state: Optional[SSLState] = Field(
        None, description="State before transition"
    )
    new_state: SSLState = Field(..., description="State after transition")
    recovery_win_count: Optional[int] = Field(
        None, description="Recovery win count (for recovery events)"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When event occurred"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional context (bot_type, threshold_breached, etc.)"
    )
    trade_outcome: Optional[TradeOutcome] = Field(
        None,
        description="Trade outcome data for AC#1 (PnL, WR, EV, session_id)"
    )

    def __str__(self) -> str:
        tier_str = f", tier={self.tier}" if self.tier else ""
        recovery_str = f", recovery_wins={self.recovery_win_count}" if self.recovery_win_count is not None else ""
        return (
            f"SSLCircuitBreakerEvent(bot={self.bot_id}, type={self.event_type.value}, "
            f"state={self.new_state.value}{tier_str}, losses={self.consecutive_losses}{recovery_str})"
        )

    def to_redis_message(self) -> str:
        """Serialize to JSON for Redis publishing."""
        return self.model_dump_json()
