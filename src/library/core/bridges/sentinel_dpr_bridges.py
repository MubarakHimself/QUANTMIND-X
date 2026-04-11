"""
QuantMindLib V1 — SentinelBridge + DPRBridge

Phase 3 (Bridge Definitions) of QuantMindLib V1 packet delivery.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.library.core.errors import BridgeError
from src.library.core.types.enums import RegimeType

from src.library.core.domain.market_context import MarketContext, RegimeReport
from src.library.core.domain.sentinel_state import HMMState, SensorState, SentinelState
from src.library.core.domain.bot_performance_snapshot import BotPerformanceSnapshot
from src.library.core.domain.bot_spec import BotRuntimeProfile


class SentinelBridgeUnavailableError(BridgeError):
    """Raised when SentinelBridge cannot reach the sentinel/market intelligence system."""

    pass


class SentinelBridge(BaseModel):
    """
    Transforms market data adapter outputs into Sentinel domain state.
    Routes regime, sensor, and HMM information into the SentinelState for the Governor.
    """

    last_regime: Optional[RegimeType] = None
    last_hmm_state: Optional["HMMState"] = None
    last_sentinel_state: Optional["SentinelState"] = None
    regime_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    sensor_readings: Dict[str, "SensorState"] = Field(default_factory=dict)
    ensemble_vote: Optional[str] = None
    last_update_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def to_sentinel_state(
        self,
        market_context: "MarketContext",
        order_flow: Any,
        pattern: Any,
    ) -> "SentinelState":
        """
        Build SentinelState from market context + order flow + pattern signals.
        Derives regime from market_context.regime.
        Maps sensor readings from the sensor_readings dict.
        Sets ensemble_vote from the computed regime.
        """
        regime_report = RegimeReport(
            regime=market_context.regime,
            regime_confidence=market_context.regime_confidence,
            news_state=market_context.news_state,
            trend_strength=market_context.trend_strength,
            volatility_regime=market_context.volatility_regime,
            session_id=market_context.session_id,
            timestamp_ms=int(time.time() * 1000),
        )
        return SentinelState(
            regime_report=regime_report,
            sensor_states=self.sensor_readings,
            hmm_state=self.last_hmm_state,
            ensemble_vote=self.ensemble_vote or market_context.regime.value,
        )

    def update_hmm_state(self, hmm_state: "HMMState") -> None:
        """Update the stored HMM state."""
        self.last_hmm_state = hmm_state
        self.last_update_ms = int(time.time() * 1000)

    def update_sensor(self, sensor_id: str, state: "SensorState") -> None:
        """Update a single sensor reading."""
        self.sensor_readings[sensor_id] = state
        self.last_update_ms = int(time.time() * 1000)

    def get_current_regime(self) -> Optional[RegimeType]:
        """Return the last known regime."""
        return self.last_regime

    def is_fresh(self, threshold_ms: int = 5000) -> bool:
        """True if bridge state was updated within threshold."""
        now_ms = int(time.time() * 1000)
        return (now_ms - self.last_update_ms) <= threshold_ms


class DPRScore(BaseModel):
    """Individual bot DPR score result."""

    bot_id: str
    dpr_score: float = Field(ge=0.0, le=1.0)
    sharpe_today: float
    win_rate_today: float
    daily_pnl: float
    rank: int
    tier: str  # "ELITE" | "PERFORMING" | "STANDARD" | "AT_RISK" | "CIRCUIT_BROKEN"
    computed_at_ms: int


class DPRBridge(BaseModel):
    """
    DPR score computation bridge.
    Computes DPR scores from bot performance snapshots and runtime profiles.
    DPR scores feed into Governor's bot selection and activation logic.
    """

    scores: Dict[str, DPRScore] = Field(default_factory=dict)
    last_computed_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def compute_scores(
        self,
        snapshots: Dict[str, "BotPerformanceSnapshot"],
        runtime_profiles: Dict[str, "BotRuntimeProfile"],
    ) -> Dict[str, DPRScore]:
        """
        Compute DPR scores for all bots from their performance snapshots.
        Score = weighted combination of: sharpe_today (0.4) + win_rate (0.3) + daily_pnl_normalized (0.3)
        Tier from score: >=0.85 ELITE, >=0.70 PERFORMING, >=0.50 STANDARD, >=0.30 AT_RISK, <0.30 CIRCUIT_BROKEN
        """
        results: Dict[str, DPRScore] = {}
        pnl_values = [s.daily_pnl for s in snapshots.values()]
        pnl_min = min(pnl_values) if pnl_values else 0.0
        pnl_max = max(pnl_values) - pnl_min if pnl_values else 1.0

        for bot_id, snapshot in snapshots.items():
            # Normalize PnL 0-1
            pnl_norm = ((snapshot.daily_pnl - pnl_min) / pnl_max) if pnl_max > 0 else 0.5

            # Get today's sharpe/win_rate from current_metrics if available
            metrics = snapshot.current_metrics or {}
            sharpe = float(metrics.get("sharpe_ratio", 0.0))
            win_rate = float(metrics.get("win_rate", 0.0))

            score = (sharpe * 0.4) + (win_rate * 0.3) + (pnl_norm * 0.3)
            score = max(0.0, min(1.0, score))

            # Tier
            if score >= 0.85:
                tier = "ELITE"
            elif score >= 0.70:
                tier = "PERFORMING"
            elif score >= 0.50:
                tier = "STANDARD"
            elif score >= 0.30:
                tier = "AT_RISK"
            else:
                tier = "CIRCUIT_BROKEN"

            results[bot_id] = DPRScore(
                bot_id=bot_id,
                dpr_score=score,
                sharpe_today=sharpe,
                win_rate_today=win_rate,
                daily_pnl=snapshot.daily_pnl,
                rank=0,  # rank set after sorting
                tier=tier,
                computed_at_ms=int(time.time() * 1000),
            )

        # Rank by score descending
        ranked = sorted(results.values(), key=lambda x: x.dpr_score, reverse=True)
        for rank, s in enumerate(ranked, 1):
            s.rank = rank

        self.scores = {s.bot_id: s for s in ranked}
        self.last_computed_ms = int(time.time() * 1000)
        return self.scores

    def get_top_bots(self, n: int = 5, min_tier: str = "STANDARD") -> List[DPRScore]:
        """Return top N bots above the minimum tier threshold."""
        tier_order = {"ELITE": 0, "PERFORMING": 1, "STANDARD": 2, "AT_RISK": 3, "CIRCUIT_BROKEN": 4}
        min_rank = tier_order.get(min_tier, 2)
        eligible = [s for s in self.scores.values() if tier_order.get(s.tier, 5) <= min_rank]
        return sorted(eligible, key=lambda x: x.rank)[:n]

    def get_score(self, bot_id: str) -> Optional[DPRScore]:
        """Get DPR score for a specific bot."""
        return self.scores.get(bot_id)

    def is_fresh(self, threshold_ms: int = 5000) -> bool:
        """True if DPR scores were computed within threshold."""
        now_ms = int(time.time() * 1000)
        return (now_ms - self.last_computed_ms) <= threshold_ms


__all__ = ["SentinelBridgeUnavailableError", "SentinelBridge", "DPRScore", "DPRBridge"]
