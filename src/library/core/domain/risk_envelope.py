"""
QuantMindLib V1 — RiskEnvelope Schema
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from src.library.core.types.enums import RiskMode


class RiskEnvelope(BaseModel):
    """Risk system authorization output — the approved risk mandate for a trade."""

    bot_id: str
    max_position_size: float = Field(ge=0)
    max_daily_loss: float = Field(ge=0)
    current_drawdown: float = Field(ge=0.0, le=1.0)
    risk_mode: RiskMode
    daily_loss_used: float = Field(ge=0)
    max_slippage_ticks: int = Field(ge=0)
    stop_ticks: int = Field(ge=0)
    position_size: float = Field(ge=0)
    open_risk: float = Field(ge=0)
    last_check_ms: int

    model_config = BaseModel.model_config

    __all__ = ["RiskEnvelope"]
