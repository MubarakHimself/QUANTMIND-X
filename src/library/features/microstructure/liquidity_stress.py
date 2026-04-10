"""LiquidityStressProxyFeature — detects liquidity stress from spread and range anomalies."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


# Stress level constants — defined at module level to avoid Pydantic field conflicts
STRESS_NORMAL: str = "NORMAL"
STRESS_ELEVATED: str = "ELEVATED"
STRESS_HIGH: str = "HIGH"
STRESS_CRITICAL: str = "CRITICAL"


class LiquidityStressProxyFeature(MicrostructureFeature):
    """
    Detects liquidity stress from spread and range anomalies.

    Heuristic: wide spread + erratic price movement = liquidity stress.
    Uses spread (from bar if available, else estimated from range/spread_ratio)
    and normalised range to compute stress score.

    Stress score = weighted combination of:
      - Spread component: spread relative to typical
      - Range component: normalised price range (high ATR-style)
      - Volume component: very high or very low volume can indicate stress

    Quality: proxy_inferred (heuristic — not a direct exchange metric)
    """

    def __init__(
        self,
        spread_ratio: float = 0.0001,
        smoothing: float = 0.3,
        **kwargs: Any,
    ) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_spread_ratio"] = spread_ratio  # Estimated spread/range ratio
        self.__dict__["_smoothing"] = smoothing
        self.__dict__["_prev_stress"] = 0.0
        self.__dict__["_spread_history"] = []

    def compute(self, bar: Dict[str, Any]) -> float:
        """
        Compute liquidity stress score for a single bar.

        Args:
            bar: Dictionary with high, low, close, volume, spread (optional) keys.

        Returns:
            float: Stress score in range [0.0, 1.0].
                   1.0 = severe stress.
                   0.0–0.3 = NORMAL
                   0.3–0.6 = ELEVATED
                   0.6–0.8 = HIGH
                   0.8–1.0 = CRITICAL
        """
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        volume = bar.get("volume", 0.0)
        spread = bar.get("spread", 0.0)

        if volume <= 0:
            return 0.0

        mid_price = (high + low) / 2.0
        if mid_price <= 0:
            return 0.0

        price_range = high - low
        range_ratio = price_range / mid_price

        # Spread component
        if spread > 0:
            # Use actual spread
            effective_spread = spread
        else:
            # Estimate from range and spread_ratio parameter
            effective_spread = price_range * self.__dict__["_spread_ratio"]

        spread_component = min(effective_spread / (mid_price * 0.0001) if mid_price > 0 else 0.0, 1.0)

        # Range component: erratic movement = stress
        # Normalise range ratio to [0, 1] using a typical threshold
        range_component = min(range_ratio * 500, 1.0)

        # Volume component: extreme volume can indicate stress
        volume_norm = min(volume / 50000.0, 1.0)
        # Very low or very high volume relative to normal is stressful
        volume_component = volume_norm * 0.5

        # Weighted combination
        stress = spread_component * 0.5 + range_component * 0.3 + volume_component * 0.2

        # Update rolling spread history for adaptive baseline
        if spread > 0:
            hist = self.__dict__["_spread_history"]
            hist.append(spread)
            if len(hist) > 20:
                hist[:] = hist[-20:]
            self.__dict__["_spread_history"] = hist

        return max(0.0, min(1.0, stress))

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[float]:
        """
        Compute stress history over multiple bars.

        Uses EMA smoothing: smoothed = smoothing * current + (1-smoothing) * prev

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of stress scores, one per bar.
        """
        if not bars:
            return []

        results: List[float] = []
        smoothed: float = self.__dict__["_prev_stress"]
        smoothing: float = self.__dict__["_smoothing"]

        for bar in bars:
            raw = self.compute(bar)
            smoothed = smoothing * raw + (1.0 - smoothing) * smoothed
            results.append(smoothed)

        self.__dict__["_prev_stress"] = smoothed
        return results

    def get_stress_level(self, stress: float) -> str:
        """
        Classify stress score into a human-readable level.

        Args:
            stress: Stress score from compute().

        Returns:
            str: One of NORMAL, ELEVATED, HIGH, CRITICAL.
        """
        if stress < 0.3:
            return STRESS_NORMAL
        elif stress < 0.6:
            return STRESS_ELEVATED
        elif stress < 0.8:
            return STRESS_HIGH
        else:
            return STRESS_CRITICAL

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/liquidity_stress_proxy",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Heuristic: wide spread + erratic range = liquidity stress",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"high", "low", "close", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"liquidity_stress", "stress_level"}

    def capability_spec(self) -> Dict[str, Any]:
        return {
            "feature_id": "microstructure/liquidity_stress_proxy",
            "quality_class": "proxy_inferred",
            "inputs": ["high", "low", "close", "volume", "spread"],
            "outputs": ["liquidity_stress", "stress_level"],
            "lookback_required": 10,
        }

    def confidence(self, n_bars: int) -> FeatureConfidence:
        """
        Compute confidence based on number of available bars.

        Args:
            n_bars: Number of bars available for the computation.

        Returns:
            FeatureConfidence with quality score and feed tag.
        """
        if n_bars < 10:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.DISABLED.value,
            )
        elif n_bars < 20:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.35,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.LOW.value,
            )
        elif n_bars < 50:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.65,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.MEDIUM.value,
            )
        else:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.85,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.HIGH.value,
            )
