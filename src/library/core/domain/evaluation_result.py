"""
QuantMindLib V1 — EvaluationResult Schema
"""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

class EvaluationResult(BaseModel):
    """Standardized evaluation output across all evaluation modes."""

    bot_id: str
    mode: str  # "BACKTEST" | "PAPER" | "LIVE" — plain str for bridge compatibility
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float = Field(ge=0.0, le=1.0)
    profit_factor: float
    expectancy: float = 0.0
    total_trades: int = Field(ge=0)
    return_pct: float
    kelly_score: float
    passes_gate: bool
    regime_distribution: Optional[Dict[str, int]] = None
    filtered_trades: Optional[int] = None

    model_config = BaseModel.model_config

    __all__ = ["EvaluationResult"]
