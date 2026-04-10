"""
QuantMindLib V1 — TradeIntent Schema
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field

from src.library.core.types.enums import TradeDirection


class TradeIntent(BaseModel):
    """Bot's trade intent — emitted at decision time, consumed by risk layer."""

    bot_id: str
    direction: TradeDirection
    confidence: int = Field(ge=0, le=100)
    urgency: Literal["IMMEDIATE", "HIGH", "NORMAL", "LOW"]
    reason: str
    timestamp_ms: int
    symbol: str

    model_config = BaseModel.model_config

    __all__ = [
        "TradeIntent",
        "TradeIntentBatch",
    ]


class TradeIntentBatch(BaseModel):
    """Batch of TradeIntent emitted in a single decision cycle."""

    intents: List[TradeIntent]
    batch_id: str
    timestamp_ms: int

    model_config = BaseModel.model_config
