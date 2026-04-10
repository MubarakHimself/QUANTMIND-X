"""
QuantMindLib V1 -- IntentEmitter
Emits TradeIntent from FeatureVector + MarketContext + BotSpec.
"""
from __future__ import annotations

import time
from typing import Literal, Optional

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.types.enums import RegimeType, TradeDirection


# Regime type aliases used in archetype signal logic.
# Maps archetype signal names to actual RegimeType enum values.
_REGIME_MAP = {
    "BREAKOUT_TRENDING": RegimeType.BREAKOUT_PRIME,
    "RANGE_BOUND": RegimeType.RANGE_STABLE,
    "BREAKOUT_TRENDING_REVERSE": RegimeType.HIGH_CHAOS,
}


def _map_regime(regime: RegimeType) -> str:
    """Return canonical signal name for a RegimeType (for archetype signal logic)."""
    for alias, rt in _REGIME_MAP.items():
        if rt == regime:
            return alias
    # Default: uppercase enum value name without prefix
    return regime.name


class IntentEmitter:
    """
    Converts feature evaluations into trade decisions.
    Applies archetype-specific signal logic to decide direction, confidence, and urgency.
    """

    def __init__(
        self,
        bot_spec: BotSpec,
        regime_confidence_threshold: float = 0.5,
        min_feature_confidence: float = 0.3,
    ) -> None:
        self.bot_spec = bot_spec
        self.regime_confidence_threshold = regime_confidence_threshold
        self.min_feature_confidence = min_feature_confidence

    def emit(
        self,
        fv: FeatureVector,
        market_ctx: MarketContext,
        symbol: str,
    ) -> Optional[TradeIntent]:
        """
        Evaluate feature vector + market context and emit a TradeIntent.
        Returns None if signal does not meet thresholds.

        Args:
            fv: Computed FeatureVector from FeatureEvaluator
            market_ctx: Current MarketContext from SentinelBridge
            symbol: Trading symbol (e.g. 'EURUSD')

        Returns:
            TradeIntent if signal is valid, None if conditions not met
        """
        # 1. Check regime confidence threshold
        if market_ctx.regime_confidence < self.regime_confidence_threshold:
            return None

        # 2. Check news state -- reject during non-clear events
        if market_ctx.news_state.value != "CLEAR":
            return None

        # 3. Check market context freshness
        if market_ctx.is_stale or not market_ctx.is_fresh():
            return None

        # 4. Get archetype-specific signal
        archetype = self.bot_spec.archetype
        direction, confidence = self._compute_signal(archetype, fv, market_ctx)

        if direction is None:
            return None

        # 5. Validate confidence threshold
        if confidence < self.min_feature_confidence:
            return None

        # 6. Determine urgency based on market regime
        urgency = self._determine_urgency(market_ctx.regime, confidence)

        # 7. Build reason string
        reason = self._build_reason(archetype, fv, market_ctx, direction)

        return TradeIntent(
            bot_id=self.bot_spec.id,
            direction=direction,
            confidence=int(confidence * 100),
            urgency=urgency,
            reason=reason,
            timestamp_ms=int(time.time() * 1000),
            symbol=symbol,
        )

    def _compute_signal(
        self,
        archetype: str,
        fv: FeatureVector,
        market_ctx: MarketContext,
    ) -> tuple[Optional[TradeDirection], float]:
        """
        Archetype-specific signal computation.
        Returns (direction, confidence) where confidence is 0.0-1.0.
        """
        archetype_lower = archetype.lower()

        if archetype_lower in ("opening_range_breakout", "london_orb", "ny_orb"):
            return self._orb_signal(fv, market_ctx)
        elif archetype_lower in ("breakout_scalper", "scalper_m1"):
            return self._scalper_signal(fv, market_ctx)
        elif archetype_lower in ("pullback_scalper",):
            return self._pullback_signal(fv, market_ctx)
        elif archetype_lower in ("mean_reversion",):
            return self._mean_reversion_signal(fv, market_ctx)
        else:
            return self._default_signal(fv, market_ctx)

    def _orb_signal(
        self, fv: FeatureVector, market_ctx: MarketContext
    ) -> tuple[Optional[TradeDirection], float]:
        """ORB signal: breakout direction from opening range."""
        rsi_val = fv.get("rsi_14", 50.0)
        regime_signal = _map_regime(market_ctx.regime)
        regime_confidence = market_ctx.regime_confidence

        if regime_signal == "BREAKOUT_TRENDING":
            return (TradeDirection.LONG, regime_confidence)
        elif regime_signal == "RANGE_BOUND":
            # In range, look for RSI extremes
            if rsi_val < 30:
                return (TradeDirection.LONG, 0.5)
            elif rsi_val > 70:
                return (TradeDirection.SHORT, 0.5)
            return (None, 0.0)
        elif regime_signal == "BREAKOUT_TRENDING_REVERSE":
            return (TradeDirection.SHORT, regime_confidence)
        return (None, 0.0)

    def _scalper_signal(
        self, fv: FeatureVector, market_ctx: MarketContext
    ) -> tuple[Optional[TradeDirection], float]:
        """Scalper signal: tight momentum from orderflow proxies."""
        aggression = fv.get("aggression_proxy", 0.5)
        spread_state = market_ctx.spread_state

        if spread_state == "WIDE":
            return (None, 0.0)

        if aggression > 0.65:
            return (TradeDirection.LONG, aggression)
        elif aggression < 0.35:
            return (TradeDirection.SHORT, 1.0 - aggression)
        return (None, 0.0)

    def _pullback_signal(
        self, fv: FeatureVector, market_ctx: MarketContext
    ) -> tuple[Optional[TradeDirection], float]:
        """Pullback signal: trade with trend on RSI pullback."""
        rsi_val = fv.get("rsi_14", 50.0)
        regime_signal = _map_regime(market_ctx.regime)

        if regime_signal != "BREAKOUT_TRENDING":
            return (None, 0.0)

        # Pullback: RSI pulled back to ~50 from overbought
        if 45 <= rsi_val <= 55:
            return (TradeDirection.LONG, 0.6)
        return (None, 0.0)

    def _mean_reversion_signal(
        self, fv: FeatureVector, market_ctx: MarketContext
    ) -> tuple[Optional[TradeDirection], float]:
        """Mean reversion signal: extreme deviation from VWAP."""
        vwap_dist = fv.get("vwap_distance", 0.0)

        if abs(vwap_dist) < 0.005:  # Less than 0.5% deviation
            return (None, 0.0)

        direction = TradeDirection.LONG if vwap_dist < 0 else TradeDirection.SHORT
        # Larger deviation = higher confidence
        confidence = min(1.0, abs(vwap_dist) / 0.02)  # 2% = max confidence
        return (direction, confidence)

    def _default_signal(
        self, fv: FeatureVector, market_ctx: MarketContext
    ) -> tuple[Optional[TradeDirection], float]:
        """Default: regime-based with RSI confirmation."""
        regime_signal = _map_regime(market_ctx.regime)
        rsi_val = fv.get("rsi_14", 50.0)
        confidence = market_ctx.regime_confidence

        if regime_signal == "BREAKOUT_TRENDING" and rsi_val < 50:
            return (TradeDirection.LONG, confidence)
        elif regime_signal == "BREAKOUT_TRENDING" and rsi_val > 50:
            return (TradeDirection.SHORT, confidence)
        return (None, 0.0)

    def _determine_urgency(
        self, regime: RegimeType, confidence: float
    ) -> Literal["IMMEDIATE", "HIGH", "NORMAL", "LOW"]:
        """Determine urgency tier from confidence and regime."""
        if regime == RegimeType.BREAKOUT_PRIME and confidence > 0.8:
            return "IMMEDIATE"
        elif confidence > 0.7:
            return "HIGH"
        elif confidence > 0.5:
            return "NORMAL"
        return "LOW"

    def _build_reason(
        self,
        archetype: str,
        fv: FeatureVector,
        market_ctx: MarketContext,
        direction: TradeDirection,
    ) -> str:
        """Build human-readable reason string."""
        parts = [
            f"archetype={archetype}",
            f"regime={market_ctx.regime.value}",
            f"confidence={market_ctx.regime_confidence:.2f}",
            f"direction={direction.value}",
        ]
        rsi = fv.get("rsi_14", None)
        if rsi is not None:
            parts.append(f"rsi={rsi:.1f}")
        return "; ".join(parts)
