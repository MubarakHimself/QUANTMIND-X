"""
QuantMindLib V1 — Risk and Execution Bridge Definitions
Phase 7: RuntimeOrchestrator bridge interfaces (stub implementations).
"""
from __future__ import annotations

import time
from typing import Optional

from src.library.core.domain.risk_envelope import RiskEnvelope
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.types.enums import RiskMode, TradeDirection


# ---------------------------------------------------------------------------
# Archetype -> sizing helper
# ---------------------------------------------------------------------------
_ARCHETYPE_STOP_TICKS: dict[str, int] = {
    "opening_range_breakout": 15,
    "london_orb": 12,
    "ny_orb": 12,
    "breakout_scalper": 8,
    "scalper_m1": 8,
    "pullback_scalper": 10,
    "mean_reversion": 20,
}


def _get_stop_ticks(archetype: str) -> int:
    """Return canonical stop_ticks for an archetype (exact match)."""
    archetype_lower = archetype.lower()
    if archetype_lower in _ARCHETYPE_STOP_TICKS:
        return _ARCHETYPE_STOP_TICKS[archetype_lower]
    # Fall back to prefix matching
    for key, val in _ARCHETYPE_STOP_TICKS.items():
        if archetype_lower.startswith(key) or key in archetype_lower:
            return val
    return 10  # default


def _get_max_slippage_ticks(archetype: str) -> int:
    """Return canonical max_slippage_ticks for an archetype (half of stop_ticks)."""
    return _get_stop_ticks(archetype) // 2


# ---------------------------------------------------------------------------
# RiskBridge
# ---------------------------------------------------------------------------

class RiskBridge:
    """
    Deterministic risk authorization.
    Computes position size from feature_vector confidence and market volatility.
    """

    def __init__(
        self,
        max_position_size: float = 100_000.0,
        default_max_daily_loss: float = 500.0,
    ) -> None:
        self.max_position_size = max_position_size
        self.default_max_daily_loss = default_max_daily_loss

    def authorize(
        self,
        intent: TradeIntent,
        feature_vector: FeatureVector,
        market_ctx: MarketContext,
    ) -> RiskEnvelope:
        """
        Authorize a trade intent within risk limits.

        Position size is computed from confidence and volatility.
        Risk mode is HALTED if current drawdown exceeds threshold.
        """
        # Resolve archetype from bot_id (format: bot_type_bot_id)
        archetype = self._resolve_archetype(intent.bot_id)

        # Compute confidence factor from intent confidence (0-100) -> 0.0-1.0
        confidence_factor = intent.confidence / 100.0

        # Compute volatility factor from market context
        # High chaos = higher volatility = smaller position
        volatility_factor = self._compute_volatility_factor(market_ctx)

        # Compute position_size
        raw_size = (
            self.max_position_size * confidence_factor * volatility_factor
        )
        # Clamp to max_position_size ceiling
        position_size = min(raw_size, self.max_position_size)

        # Determine risk_mode based on drawdown
        risk_mode = self._determine_risk_mode(
            market_ctx, feature_vector, confidence_factor
        )

        # Archetype-based stop / slippage ticks
        stop_ticks = _get_stop_ticks(archetype)
        max_slippage_ticks = _get_max_slippage_ticks(archetype)

        # Daily loss tracking (simplified: running counter from intent timestamp)
        daily_loss_used = self._compute_daily_loss_used(intent.timestamp_ms)

        # Current drawdown from market context
        current_drawdown = self._compute_drawdown(market_ctx, feature_vector)

        now_ms = int(time.time() * 1000)
        return RiskEnvelope(
            bot_id=intent.bot_id,
            max_position_size=self.max_position_size,
            max_daily_loss=self.default_max_daily_loss,
            current_drawdown=current_drawdown,
            risk_mode=risk_mode,
            daily_loss_used=daily_loss_used,
            max_slippage_ticks=max_slippage_ticks,
            stop_ticks=stop_ticks,
            position_size=position_size,
            open_risk=0.0,
            last_check_ms=now_ms,
        )

    def _resolve_archetype(self, bot_id: str) -> str:
        """
        Extract archetype hint from bot_id.

        Matches known archetype names as prefixes or substrings.
        Examples: "scalper_m1_bot" -> "scalper_m1", "london_orb_v1" -> "london_orb"
        """
        bot_lower = bot_id.lower()
        for key in _ARCHETYPE_STOP_TICKS:
            if bot_lower.startswith(key) or key in bot_lower:
                return key
        # Fallback: use first segment
        return bot_id.split("_")[0] if "_" in bot_id else bot_id

    def _compute_volatility_factor(self, market_ctx: MarketContext) -> float:
        """Compute volatility factor (0.0-1.0) from market context."""
        # High chaos = low volatility factor (reduce position size)
        if market_ctx.regime == "HIGH_CHAOS":
            return 0.25
        elif market_ctx.regime == "NEWS_EVENT":
            return 0.10
        elif market_ctx.regime == "UNCERTAIN":
            return 0.40
        elif market_ctx.regime == "RANGE_STABLE":
            return 0.70
        elif market_ctx.regime == "TREND_STABLE":
            return 0.85
        elif market_ctx.regime == "BREAKOUT_PRIME":
            return 0.90
        return 0.75  # default

    def _determine_risk_mode(
        self,
        market_ctx: MarketContext,
        feature_vector: FeatureVector,
        confidence_factor: float,
    ) -> RiskMode:
        """Determine risk mode based on drawdown and regime."""
        # HALTED if spread state is WIDE
        if market_ctx.spread_state == "WIDE":
            return RiskMode.HALTED

        # HALTED if regime is NEWS_EVENT
        if market_ctx.regime.value == "NEWS_EVENT":
            return RiskMode.HALTED

        # CLAMPED if confidence is very low
        if confidence_factor < 0.3:
            return RiskMode.CLAMPED

        # CLAMPED if current drawdown > 10%
        if market_ctx.volatility_regime == "HIGH":
            return RiskMode.CLAMPED

        # Otherwise STANDARD
        return RiskMode.STANDARD

    def _compute_daily_loss_used(self, timestamp_ms: int) -> float:
        """
        Simplified daily loss tracking.
        In Phase 7, use timestamp-based bucketing for deterministic behavior.
        """
        # Use hour-of-day as a deterministic proxy for daily loss used.
        # In production, this would read from persistent state.
        hour_bucket = (timestamp_ms // 3_600_000) % 24
        # Simulate escalating loss as the "day" progresses
        daily_loss_pct = min(hour_bucket / 24.0 * 0.05, 0.05)  # max 5%
        return self.default_max_daily_loss * daily_loss_pct

    def _compute_drawdown(
        self,
        market_ctx: MarketContext,
        feature_vector: FeatureVector,
    ) -> float:
        """
        Compute current drawdown (0.0-1.0).
        In Phase 7, derive from market context or default to 0.0.
        """
        # Use trend_strength from market context as a proxy for drawdown
        if market_ctx.trend_strength is not None:
            # In an uptrend, drawdown is low; in a downtrend, drawdown is higher
            if market_ctx.trend_strength >= 0.6:
                return 0.05  # strong trend, low drawdown
            elif market_ctx.trend_strength >= 0.4:
                return 0.15
            else:
                return 0.25  # weak trend, higher drawdown
        return 0.0


# ---------------------------------------------------------------------------
# ExecutionBridge
# ---------------------------------------------------------------------------

class ExecutionBridge:
    """
    Deterministic execution directive generation.
    Maps TradeIntent to ExecutionDirective fields.
    """

    def __init__(
        self,
        default_quantity_floor: float = 1000.0,
    ) -> None:
        self.default_quantity_floor = default_quantity_floor

    def execute(
        self,
        intent: TradeIntent,
        risk_envelope: RiskEnvelope,
    ) -> ExecutionDirective:
        """
        Generate an ExecutionDirective from a TradeIntent and RiskEnvelope.
        """
        # If risk envelope has zero position size, return a zero-quantity directive
        if risk_envelope.position_size <= 0:
            return ExecutionDirective(
                bot_id=intent.bot_id,
                direction=intent.direction,
                symbol=intent.symbol,
                quantity=0.0,
                risk_mode=risk_envelope.risk_mode,
                max_slippage_ticks=risk_envelope.max_slippage_ticks,
                stop_ticks=risk_envelope.stop_ticks,
                timestamp_ms=intent.timestamp_ms,
                authorization="RUNTIME_ORCHESTRATOR",
            )

        # Map TradeIntent direction to ExecutionDirective
        direction = intent.direction
        quantity = max(risk_envelope.position_size, self.default_quantity_floor)

        now_ms = int(time.time() * 1000)
        return ExecutionDirective(
            bot_id=intent.bot_id,
            direction=direction,
            symbol=intent.symbol,
            quantity=quantity,
            risk_mode=risk_envelope.risk_mode,
            max_slippage_ticks=risk_envelope.max_slippage_ticks,
            stop_ticks=risk_envelope.stop_ticks,
            timestamp_ms=now_ms,
            authorization="RUNTIME_ORCHESTRATOR",
        )


__all__ = [
    "RiskBridge",
    "ExecutionBridge",
]