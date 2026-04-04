"""
UI Event Models for SSL State Change Events.

Story 18.2: SSL → DPR Integration

Contains:
- SSLUIStateChangeEvent: Event emitted when SSL state changes for UI rendering

Per NFR-M2: UI events are fire-and-forget — no blocking on UI update.
"""

from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field


class SSLUIStateChangeEvent(BaseModel):
    """
    Event emitted when SSL state changes for UI rendering.

    AC#8: When bot transitions to paper-only:
    - SSL emits UI state change event
    - Trading Canvas and Portfolio Canvas update bot's tile rendering
    - Desaturated tile, dashed border, "PAPER" badge (UX Surface F-13)
    - Live Trading Canvas removes bot from live queue count

    Attributes:
        bot_id: Bot identifier
        new_state: SSL state after transition (paper, live, recovery, retired)
        tier: Paper trading tier if applicable (TIER_1 or TIER_2)
        timestamp_utc: When state change occurred
        metadata: Additional UI rendering hints
    """
    bot_id: str = Field(..., description="Bot identifier")
    new_state: str = Field(..., description="SSL state: paper, live, recovery, retired")
    tier: Optional[Literal["TIER_1", "TIER_2"]] = Field(
        None, description="Paper tier if in paper"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When state change occurred"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="UI rendering hints: paper_badge_text, tile_style, etc."
    )

    def __str__(self) -> str:
        tier_str = f", tier={self.tier}" if self.tier else ""
        return f"SSLUIStateChangeEvent(bot={self.bot_id}, state={self.new_state}{tier_str})"
