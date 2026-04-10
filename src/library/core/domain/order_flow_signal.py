"""
QuantMindLib V1 — OrderFlowSignal Schema
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field

from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import OrderFlowSource, SignalDirection


class OrderFlowSignal(BaseModel):
    """Order flow proxy signal from tick/price/volume data.

    Quality-aware: the ``confidence`` field carries feed quality tagging
    so bots can degrade gracefully when order flow data is proxied or
    approximated rather than natively sourced.
    """

    bot_id: str
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    strength: float = Field(ge=0.0, le=1.0)
    source: OrderFlowSource
    confidence: FeatureConfidence
    supporting_evidence: Dict[str, Any]

    model_config = BaseModel.model_config

    __all__ = ["OrderFlowSignal"]
