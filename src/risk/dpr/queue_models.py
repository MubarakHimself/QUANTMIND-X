"""
DPR Queue Models — Daily Performance Ranking Queue Output Models.

Story 17.2: DPR Queue Tier Remix

Contains:
- QueueEntry: Individual bot entry in the queue
- DPRQueueOutput: Full queue output for a session
- DPRQueueAuditRecord: Audit log record for queue decisions
- SSLEvent: SSL mid-session event for queue

Per NFR-M2: DPR is a synchronous queue manager — NO LLM calls in remix path.
Per NFR-D1: All queue outputs logged with timestamps before session lock.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class Tier(str, Enum):
    """Bot tier classification."""
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class QueueEntry(BaseModel):
    """
    Individual bot entry in the DPR queue.

    Attributes:
        bot_id: Bot identifier
        queue_position: 1-indexed position in queue
        dpr_composite_score: DPR composite score (0-100)
        tier: Bot tier (TIER_1, TIER_2, TIER_3)
        specialist_session: Session if SESSION_SPECIALIST, else None
        specialist_boost_applied: Whether +5 boost was applied
        concern_flag: Whether SESSION_CONCERN flag is set
        recovery_step: 0=not in recovery, 1=first win, 2=eligible
        in_concern_subqueue: Whether in concern sub-queue within tier
        ssl_state: SSL state (live, paper, recovery, retired) - AC#7
        ssl_tier: Paper tier if in paper (TIER_1 or TIER_2) - AC#7
        is_paper_only: True if bot is in paper-only mode - AC#8
        paper_entry_timestamp: When bot entered paper tier - AC#7
    """
    bot_id: str = Field(..., description="Bot identifier")
    queue_position: int = Field(..., description="1-indexed queue position")
    dpr_composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: Tier = Field(..., description="Bot tier classification")
    specialist_session: Optional[str] = Field(
        None, description="Session if SESSION_SPECIALIST tag"
    )
    specialist_boost_applied: bool = Field(
        default=False, description="Whether +5 specialist boost was applied"
    )
    concern_flag: bool = Field(
        default=False, description="Whether SESSION_CONCERN flag is set"
    )
    recovery_step: int = Field(
        default=0,
        description="Recovery step: 0=not in recovery, 1=first win, 2=eligible"
    )
    in_concern_subqueue: bool = Field(
        default=False, description="Whether in concern sub-queue"
    )
    # AC#7: SSL state integration
    ssl_state: str = Field(
        default="live",
        description="SSL state: live, paper, recovery, retired"
    )
    ssl_tier: Optional[str] = Field(
        default=None,
        description="Paper tier if in paper: TIER_1 or TIER_2"
    )
    is_paper_only: bool = Field(
        default=False,
        description="True if bot is in paper-only mode"
    )
    paper_entry_timestamp: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when bot entered paper tier"
    )

    def __str__(self) -> str:
        specialist = f" [SPECIALIST:{self.specialist_session}]" if self.specialist_session else ""
        concern = " [CONCERN]" if self.concern_flag else ""
        recovery = f" [RECOVERY:{self.recovery_step}]" if self.recovery_step > 0 else ""
        return (
            f"QueueEntry(pos={self.queue_position}, bot={self.bot_id}, "
            f"tier={self.tier.value}, score={self.dpr_composite_score}"
            f"{specialist}{concern}{recovery})"
        )


class DPRQueueOutput(BaseModel):
    """
    Full DPR queue output for a session.

    Attributes:
        session_id: Session identifier (e.g., "LONDON", "NY")
        queue_timestamp: When queue was assembled (ISO8601 UTC)
        locked: Whether queue is locked for session
        bots: Ordered list of queue entries
        ny_hybrid_override: True when NY hybrid queue is active
    """
    session_id: str = Field(..., description="Session identifier")
    queue_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When queue was assembled"
    )
    locked: bool = Field(default=False, description="Whether queue is locked")
    bots: List[QueueEntry] = Field(
        default_factory=list,
        description="Ordered list of queue entries"
    )
    ny_hybrid_override: bool = Field(
        default=False,
        description="True when NY hybrid queue is active"
    )

    def __str__(self) -> str:
        return (
            f"DPRQueueOutput(session={self.session_id}, "
            f"locked={self.locked}, ny_hybrid={self.ny_hybrid_override}, "
            f"bots={len(self.bots)})"
        )


class DPRQueueAuditRecord(BaseModel):
    """
    Audit log record for a single queue decision.

    Persisted to SQLite for full audit trail per AC #7.

    Attributes:
        session_id: Session identifier
        bot_id: Bot identifier
        queue_position: Assigned queue position
        dpr_composite_score: DPR composite score at time of queue
        tier: Bot tier
        specialist_flag: Whether SESSION_SPECIALIST tag was present
        concern_flag: Whether SESSION_CONCERN flag was present
        timestamp_utc: When decision was made
    """
    session_id: str = Field(..., description="Session identifier")
    bot_id: str = Field(..., description="Bot identifier")
    queue_position: int = Field(..., description="Assigned queue position")
    dpr_composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: str = Field(..., description="Bot tier")
    specialist_flag: bool = Field(
        default=False,
        description="Whether SESSION_SPECIALIST tag was present"
    )
    concern_flag: bool = Field(
        default=False,
        description="Whether SESSION_CONCERN flag was set"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When decision was made"
    )

    def __str__(self) -> str:
        return (
            f"DPRQueueAuditRecord(session={self.session_id}, "
            f"bot={self.bot_id}, pos={self.queue_position}, "
            f"tier={self.tier}, score={self.dpr_composite_score})"
        )


