"""
DPR Event Models for Daily Performance Ranking System.

Story 17.1: DPR Composite Score Calculation

Contains:
- DPRComponentScores: Component scores with weights for composite calculation
- DPRScoreEvent: Audit log event for score calculations
- DPRConcernEvent: Event emitted when SESSION_CONCERN flag is triggered

Per NFR-M2: DPR is a synchronous scoring engine — NO LLM calls in scoring path.
Per NFR-D1: All DPR score calculations logged with timestamps before API response.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Tuple, Literal
from pydantic import BaseModel, Field


# Weights for composite score calculation
DPR_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.30, 0.20, 0.25)
"""Weights for (win_rate, pnl, consistency, ev_per_trade) components."""


class DPRComponentScores(BaseModel):
    """
    Component scores for DPR composite calculation.

    Each component is normalized to 0-100 scale before weighting.

    Attributes:
        win_rate: Session win rate normalized score (0-100)
        pnl: Net PnL normalized score (0-100)
        consistency: Consistency normalized score (0-100, inverse of variance)
        ev_per_trade: Expected value per trade normalized score (0-100)
        weights: Tuple of weights summing to 1.0 (default: 25%, 30%, 20%, 25%)
    """
    win_rate: float = Field(..., description="Win rate normalized score (0-100)")
    pnl: float = Field(..., description="Net PnL normalized score (0-100)")
    consistency: float = Field(..., description="Consistency normalized score (0-100)")
    ev_per_trade: float = Field(..., description="EV per trade normalized score (0-100)")
    weights: Tuple[float, float, float, float] = Field(
        default=DPR_WEIGHTS,
        description="Component weights (win_rate, pnl, consistency, ev_per_trade)"
    )

    def composite_score(self) -> int:
        """
        Calculate weighted composite score.

        Returns:
            Integer composite score (0-100)
        """
        composite = (
            self.win_rate * self.weights[0] +
            self.pnl * self.weights[1] +
            self.consistency * self.weights[2] +
            self.ev_per_trade * self.weights[3]
        )
        return int(round(composite))

    def __str__(self) -> str:
        return (
            f"DPRComponentScores(WR={self.win_rate:.1f}, PnL={self.pnl:.1f}, "
            f"Cons={self.consistency:.1f}, EV={self.ev_per_trade:.1f})"
        )


class DPRScoreEvent(BaseModel):
    """
    DPR score calculation audit log event.

    Persisted to SQLite before any system acknowledgment per NFR-D1.

    Attributes:
        bot_id: Bot identifier
        session_id: Session identifier (e.g., "LONDON", "NY_AM")
        scoring_window: Time window for scoring ("session", "fortnight")
        component_scores: Individual component scores
        composite_score: Final composite score (0-100)
        is_tied: Whether this score resulted in a tie
        tie_break_winner: Bot ID of tie-break winner if applicable
        specialist_boost_applied: Whether SESSION_SPECIALIST boost was applied
        session_concern_flag: Whether SESSION_CONCERN flag was set
        timestamp_utc: When calculation occurred
        metadata: Additional context (trade count, magic number, etc.)
    """
    bot_id: str = Field(..., description="Bot identifier")
    session_id: str = Field(..., description="Session identifier")
    scoring_window: str = Field(
        default="session",
        description="Scoring window (session or fortnight)"
    )
    component_scores: DPRComponentScores = Field(
        ..., description="Individual component scores"
    )
    composite_score: int = Field(..., description="Final composite score (0-100)")
    is_tied: bool = Field(default=False, description="Whether score resulted in tie")
    tie_break_winner: Optional[str] = Field(
        None, description="Bot ID of tie-break winner if applicable"
    )
    specialist_boost_applied: bool = Field(
        default=False, description="Whether SESSION_SPECIALIST boost was applied"
    )
    session_concern_flag: bool = Field(
        default=False, description="Whether SESSION_CONCERN flag was set"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When calculation occurred"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional context (trade_count, magic_number, max_drawdown, etc.)"
    )

    def __str__(self) -> str:
        boost_str = " +SPECIALIST" if self.specialist_boost_applied else ""
        concern_str = " +CONCERN" if self.session_concern_flag else ""
        return (
            f"DPRScoreEvent(bot={self.bot_id}, session={self.session_id}, "
            f"score={self.composite_score}{boost_str}{concern_str})"
        )


class DPRConcernEvent(BaseModel):
    """
    Event emitted when a bot's DPR score drops >20 points week-over-week.

    AC #4: Given a bot's DPR score drops >20 points week-over-week,
    When the fortnight accumulation completes,
    Then a SESSION_CONCERN flag is set on that bot,
    And Copilot surfaces the flag in the next Dead Zone briefing.

    Attributes:
        bot_id: Bot identifier
        previous_score: Score from previous fortnight
        current_score: Score from current fortnight
        score_delta: Change in score (negative when dropping)
        threshold: Threshold that was breached (default -20)
        timestamp_utc: When the concern was detected
        metadata: Additional context
    """
    bot_id: str = Field(..., description="Bot identifier")
    previous_score: int = Field(..., description="Score from previous fortnight")
    current_score: int = Field(..., description="Score from current fortnight")
    score_delta: int = Field(..., description="Change in score (negative = dropped)")
    threshold: int = Field(default=-20, description="Threshold that was breached")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When concern was detected"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional context (session_id, consecutive_sessions_declining, etc.)"
    )

    def __str__(self) -> str:
        return (
            f"DPRConcernEvent(bot={self.bot_id}, delta={self.score_delta} "
            f"({self.previous_score}→{self.current_score}))"
        )


class SSLEventType(str, Enum):
    """SSL event types that can affect DPR queue."""
    MOVE_TO_PAPER = "MOVE_TO_PAPER"
    RECOVERY_STEP_1 = "RECOVERY_STEP_1"
    RECOVERY_CONFIRMED = "RECOVERY_CONFIRMED"
    RETIRED = "RETIRED"


class SSLEvent(BaseModel):
    """
    SSL mid-session event that affects DPR queue state.

    Events are queued mid-session and applied at next Dead Zone per AC #6.

    Attributes:
        event_type: Type of SSL event
        bot_id: Bot identifier
        magic_number: MT5 magic number for the bot
        session_id: Session during which event occurred
        tier: Paper tier if applicable (TIER_1 or TIER_2) - AC#6
        timestamp_utc: When event occurred
    """
    event_type: SSLEventType = Field(..., description="Type of SSL event")
    bot_id: str = Field(..., description="Bot identifier")
    magic_number: str = Field(..., description="MT5 magic number")
    session_id: str = Field(..., description="Session during which event occurred")
    tier: Optional[Literal["TIER_1", "TIER_2"]] = Field(
        None, description="Paper tier for move_to_paper events"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When event occurred"
    )

    def __str__(self) -> str:
        tier_str = f", tier={self.tier}" if self.tier else ""
        return (
            f"SSLEvent(type={self.event_type.value}, bot={self.bot_id}, "
            f"session={self.session_id}{tier_str}, time={self.timestamp_utc.isoformat()})"
        )
