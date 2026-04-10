"""QuantMindLib V1 — SessionContext Schema"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SessionContext(BaseModel):
    """Current trading session state.

    Represents whether the market is in a canonical session (e.g. London,
    New York, overlap), whether a news blackout is active, and which
    currencies are exposed during this session.

    Emitted by the SessionBridge from SessionDetector and NewsBlackout.
    Consumed by FeatureEvaluator, BotStateManager, and SafetyHooks.
    """

    session_id: str
    is_active: bool = Field(default=False)
    is_blackout: bool = Field(default=False)
    blackout_reason: str | None = None
    minutes_to_event: int | None = Field(default=None, ge=0)
    currencies: list[str]

    __all__ = ["SessionContext"]
