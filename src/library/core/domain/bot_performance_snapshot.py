"""QuantMindLib V1 — BotPerformanceSnapshot Schema"""
from __future__ import annotations

from pydantic import BaseModel, Field


class BotPerformanceSnapshot(BaseModel):
    """Periodic performance snapshot for DPR and registry updates.

    Emitted by the DPRBridge on a configurable cadence (e.g. every 60s
    during an active trading session). Downstream consumers include the
    BotRegistry, LifecycleBridge, and the DPR Redis publisher.
    """

    bot_id: str
    snapshot_id: str
    snapshot_at_ms: int
    current_metrics: dict[str, float]
    daily_pnl: float
    total_pnl: float
    active_days: int = Field(ge=0)
    current_session: str
    is_in_session: bool = Field(default=False)

    __all__ = ["BotPerformanceSnapshot"]
