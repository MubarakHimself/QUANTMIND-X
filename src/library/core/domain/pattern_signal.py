"""
QuantMindLib V1 — PatternSignal Schema

V1 placeholder refined — Q-3 Decision deferred.
Internal pattern recognition engine is OUT-OF-SCOPE for V1.
Schema enhanced with PatternConfidence and PatternMetadata breakdowns.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from src.library.core.types.enums import SignalDirection


class PatternConfidence(BaseModel):
    """
    Confidence breakdown for a pattern signal.

    Provides multi-dimensional confidence scoring so downstream consumers
    can weight structural, temporal, and volume confirmation independently.
    """

    structural: float = Field(ge=0.0, le=1.0, default=0.0)
    temporal: float = Field(ge=0.0, le=1.0, default=0.0)
    volume_confirm: float = Field(ge=0.0, le=1.0, default=0.0)
    overall: float = Field(ge=0.0, le=1.0, default=0.0)

    model_config = BaseModel.model_config


class PatternMetadata(BaseModel):
    """
    Metadata about the detected pattern.

    Provides context about the pattern's nature, timeframe, and lifecycle
    so downstream consumers can make contextual decisions.
    """

    pattern_family: str = ""  # "REVERSAL" | "CONTINUATION" | "NEUTRAL"
    time_frame: str = ""      # "INTRA" | "SWING" | "POSITIONAL"
    detected_at_idx: int = 0  # Bar index where pattern was detected
    bars_in_pattern: int = 0  # Number of bars forming the pattern
    age_bars: int = 0        # Bars since pattern completion

    model_config = BaseModel.model_config


class PatternSignal(BaseModel):
    """Chart pattern detection output.

    V1 does not implement pattern recognition (Q-3 Decision deferred).
    This schema exists as a marker so downstream consumers have a valid type.
    The ``is_defined`` field is True only when confidence_breakdown
    and metadata are both provided — otherwise it remains False as a
    placeholder marker.
    """

    bot_id: str = ""
    pattern_type: str = ""   # e.g. "DOJI", "ENGULFING", "HEAD_AND_SHOULDERS" — no enum yet
    direction: SignalDirection = SignalDirection.NEUTRAL
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)  # legacy single-value confidence
    timestamp: datetime = Field(default_factory=datetime.now)
    is_defined: bool = False
    note: str = "V1 placeholder — Q-3 Decision deferred"

    # V1 schema refinement: optional confidence breakdown and metadata
    confidence_breakdown: PatternConfidence | None = None
    metadata: PatternMetadata | None = None

    model_config = BaseModel.model_config

    def model_post_init(self, __context: Any) -> None:
        """Set is_defined=True when both confidence_breakdown and metadata are present."""
        if self.confidence_breakdown is not None and self.metadata is not None:
            object.__setattr__(self, "is_defined", True)
