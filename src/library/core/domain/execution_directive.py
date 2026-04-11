"""
QuantMindLib V1 — ExecutionDirective Schema
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from src.library.core.types.enums import RiskMode, TradeDirection


class ExecutionDirective(BaseModel):
    """Governor's authorized execution decision — consumed by cTraderExecutionAdapter."""

    bot_id: str
    direction: TradeDirection
    symbol: str
    quantity: float = Field(gt=0)
    risk_mode: RiskMode
    max_slippage_ticks: int = Field(ge=0)
    stop_ticks: int = Field(gt=0)
    limit_ticks: Optional[int] = None
    timestamp_ms: int
    authorization: str
    approved: bool = True
    rejection_reason: Optional[str] = None

    model_config = BaseModel.model_config

    __all__ = ["ExecutionDirective"]
