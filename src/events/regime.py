"""
Regime Shift Event Model for Layer 2 Position Monitor.

Story 14.2: Layer 2 Tier 1 Position Monitor (Dynamic)
- RegimeShiftEvent: Pydantic model for Sentinel regime shift events
- Consumed by Layer2PositionMonitor.evaluate_regime_shift()
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RegimeType(str, Enum):
    """Market regime types from Sentinel HMM."""
    TREND_BULL = "TREND_BULL"
    TREND_BEAR = "TREND_BEAR"
    TREND_STABLE = "TREND_STABLE"
    RANGE_STABLE = "RANGE_STABLE"
    RANGE_VOLATILE = "RANGE_VOLATILE"
    BREAKOUT_UP = "BREAKOUT_UP"
    BREAKOUT_DOWN = "BREAKOUT_DOWN"
    CHAOS = "CHAOS"


# Regime-to-strategy suitability mapping (module-level constant)
REGIME_STRATEGY_MAP = {
    # Trend-following strategies suit trending markets
    RegimeType.TREND_BULL: ["momentum", "trend_follow", "breakout"],
    RegimeType.TREND_BEAR: ["momentum", "trend_follow", "breakout"],
    RegimeType.TREND_STABLE: ["mean_reversion", "range_trade"],
    # Range-bound strategies suit ranging markets
    RegimeType.RANGE_STABLE: ["mean_reversion", "range_trade", "scalp"],
    RegimeType.RANGE_VOLATILE: ["scalp", "short_term"],
    # Breakout strategies suit breakout conditions
    RegimeType.BREAKOUT_UP: ["breakout", "momentum"],
    RegimeType.BREAKOUT_DOWN: ["breakout", "momentum"],
    # No strategies suit chaos - all should reduce/close
    RegimeType.CHAOS: [],
}


class RegimeShiftEvent(BaseModel):
    """
    Regime shift event from Sentinel.

    When Sentinel detects a regime change (e.g., TREND_STABLE → RANGE_STABLE),
    it publishes this event to be consumed by Layer 2 Position Monitor.

    Attributes:
        previous_regime: The regime before the shift
        current_regime: The regime after the shift
        symbol: Symbol the regime shift applies to (None = all symbols)
        confidence: Confidence score of regime detection (0.0-1.0)
        timestamp_utc: When the regime shift was detected
        metadata: Additional context from Sentinel
    """
    previous_regime: RegimeType = Field(..., description="Regime before shift")
    current_regime: RegimeType = Field(..., description="Regime after shift")
    symbol: Optional[str] = Field(None, description="Symbol (None = all symbols)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Detection confidence")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When regime shift was detected"
    )
    metadata: dict = Field(default_factory=dict, description="Additional Sentinel context")

    def __str__(self) -> str:
        return (
            f"RegimeShiftEvent({self.previous_regime.value} → {self.current_regime.value}, "
            f"symbol={self.symbol}, confidence={self.confidence:.2f})"
        )


class RegimeSuitability(BaseModel):
    """
    Regime suitability assessment for a strategy.

    Used by Layer2PositionMonitor to determine if a position's strategy
    is suitable for the current regime.
    """
    strategy_id: str = Field(..., description="Strategy being evaluated")
    regime: RegimeType = Field(..., description="Current regime")
    is_suitable: bool = Field(..., description="Whether strategy fits current regime")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    action: str = Field(
        default="hold",
        description="Recommended action: hold, reduce, close"
    )
    reason: str = Field(default="", description="Reason for the assessment")

    @classmethod
    def evaluate(cls, strategy_id: str, regime: RegimeType, confidence: float = 0.8) -> "RegimeSuitability":
        """
        Evaluate whether a strategy is suitable for a given regime.

        Args:
            strategy_id: Strategy identifier to evaluate
            regime: Current market regime
            confidence: Detection confidence (affects action certainty)

        Returns:
            RegimeSuitability with recommended action
        """
        suitable_strategies = REGIME_STRATEGY_MAP.get(regime, [])
        strategy_type = strategy_id.lower()

        is_suitable = any(
            suitable in strategy_type
            for suitable in suitable_strategies
        )

        # In chaos, nothing is suitable - everything should close
        if regime == RegimeType.CHAOS:
            action = "close"
            reason = "CHAOS regime - no strategies suitable, force close all"
        elif not suitable_strategies:
            action = "hold"
            reason = f"No suitability mapping for regime {regime.value}"
        elif is_suitable:
            action = "hold"
            reason = f"Strategy {strategy_id} suitable for {regime.value}"
        else:
            # Low confidence = recommend reduce instead of close
            if confidence < 0.7:
                action = "reduce"
                reason = f"Strategy {strategy_id} may not suit {regime.value}, reducing exposure"
            else:
                action = "close"
                reason = f"Strategy {strategy_id} unsuitable for {regime.value}"

        return cls(
            strategy_id=strategy_id,
            regime=regime,
            is_suitable=is_suitable,
            confidence=confidence,
            action=action,
            reason=reason
        )
