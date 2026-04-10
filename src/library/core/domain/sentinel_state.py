"""
QuantMindLib V1 — SentinelState Schema
"""
from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.library.core.domain.market_context import RegimeReport


class SensorState(BaseModel):
    """Per-sensor health and reading state."""

    sensor_id: str
    state: str
    confidence: float = Field(ge=0.0, le=1.0)
    updated_at_ms: int
    is_active: Optional[bool] = None
    last_reading: Optional[float] = None
    status: Optional[str] = Field(default="NOMINAL")


class HMMState(BaseModel):
    """Hidden Markov Model regime state snapshot."""

    regime: str
    confidence: float = Field(ge=0.0, le=1.0)
    features: Dict[str, float] = Field(default_factory=dict)
    updated_at_ms: int
    state_id: Optional[int] = None
    state_label: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None


class SentinelState(BaseModel):
    """Full sentinel/market intelligence state snapshot."""

    regime_report: RegimeReport
    sensor_states: Dict[str, SensorState] = Field(default_factory=dict)
    hmm_state: Optional[HMMState] = None
    ensemble_vote: str


__all__ = [
    "SentinelState",
    "SensorState",
    "HMMState",
]
