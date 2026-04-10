"""
QuantMindLib V1 — PatternSignal Schema (Placeholder)

V1 placeholder — Q-3 Decision deferred.
Internal pattern recognition engine is OUT-OF-SCOPE for V1.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.library.core.types.enums import SignalDirection


class PatternSignal(BaseModel):
    """Placeholder chart pattern detection output.

    V1 does not implement pattern recognition (Q-3 Decision deferred).
    This schema exists as a marker so downstream consumers have a valid type.
    The ``is_defined`` field is always ``False`` until Q-3 resolves this schema.
    """

    bot_id: str
    pattern_type: str  # e.g. "DOJI", "ENGULFING", "HEAD_AND_SHOULDERS" — no enum yet
    direction: SignalDirection  # BULLISH / BEARISH / NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0–1.0 placeholder confidence
    timestamp: datetime
    is_defined: bool = False  # Always False until Q-3 Decision resolves schema
    note: str = "V1 placeholder — Q-3 Decision deferred"
