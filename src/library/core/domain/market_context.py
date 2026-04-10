"""
QuantMindLib V1 — MarketContext Schema
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.library.core.types.enums import NewsState, RegimeType


class RegimeReport(BaseModel):
    """Structured input from SentinelBridge — used to construct MarketContext."""
    model_config = BaseModel.model_config

    regime: RegimeType
    regime_confidence: float
    news_state: NewsState
    trend_strength: Optional[float] = None
    volatility_regime: Optional[str] = None
    session_id: Optional[str] = None
    timestamp_ms: int


class MarketContext(BaseModel):
    """Market regime and health state — mutable runtime object."""

    regime: RegimeType
    news_state: NewsState
    regime_confidence: float = Field(ge=0.0, le=1.0)
    session_id: Optional[str] = None
    is_stale: bool = False
    spread_state: Optional[str] = None
    depth_snapshot: Optional[Dict[str, int]] = None
    last_update_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    trend_strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    volatility_regime: Optional[str] = None

    def is_fresh(self, threshold_ms: int = 5000) -> bool:
        """Returns True if last_update_ms is within threshold_ms of current time."""
        now_ms = int(time.time() * 1000)
        return (now_ms - self.last_update_ms) <= threshold_ms


__all__ = [
    "MarketContext",
    "RegimeReport",
]
