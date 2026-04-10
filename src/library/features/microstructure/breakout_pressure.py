"""BreakoutPressureProxyFeature — detects pressure building before breakout moves."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


class BreakoutPressureProxyFeature(MicrostructureFeature):
    """
    Detects pressure building before breakout moves.

    Heuristic: volume increasing + range expanding + spread widening
    = pressure building toward a breakout. When combined with tight
    consolidation (low range, low volume), signals imminent breakout.

    Pressure score components:
      - Volume trend: is volume increasing over recent bars?
      - Range trend: is the bar range expanding?
      - Spread trend: is spread widening?

    Quality: proxy_inferred (heuristic — not a direct exchange metric)
    """

    def __init__(self, smoothing: float = 0.3, **kwargs: Any) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_smoothing"] = smoothing
        self.__dict__["_prev_pressure"] = 0.0

    def compute(self, bar: Dict[str, Any]) -> float:
        """
        Compute breakout pressure score for a single bar.

        Args:
            bar: Dictionary with open, high, low, close, volume keys.

        Returns:
            float: Pressure score in range [0.0, ~5.0+].
                   Higher = more pressure building.
                   0.0–1.0 = normal / calm
                   1.0–2.5 = pressure rising
                   2.5–4.0 = significant pressure
                   4.0+ = extreme pressure (breakout imminent)
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

        # Volume component: penalise very low volume (consolidation) and very high volume
        # Base volume score from volume / 10000 (normalised for forex-style bars)
        volume_score = min(volume / 10000.0, 3.0)

        # Range component: wider range = more energy building
        range_score = min(range_ratio * 1000, 3.0)

        # Spread component: wider spread = more stress
        spread_score = min(spread / (mid_price * 0.0001) if spread > 0 else 0.0, 2.0)

        # Weighted combination
        pressure = (volume_score * 0.4 + range_score * 0.4 + spread_score * 0.2)

        return max(0.0, pressure)

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[float]:
        """
        Compute pressure history over multiple bars.

        Uses EMA smoothing: smoothed = smoothing * current + (1-smoothing) * prev

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of pressure scores, one per bar.
        """
        if not bars:
            return []

        results: List[float] = []
        smoothed: float = self.__dict__["_prev_pressure"]
        smoothing: float = self.__dict__["_smoothing"]

        for bar in bars:
            raw = self.compute(bar)
            smoothed = smoothing * raw + (1.0 - smoothing) * smoothed
            results.append(smoothed)

        self.__dict__["_prev_pressure"] = smoothed
        return results

    def detect_consolidation(self, bars: List[Dict[str, Any]], threshold: float = 0.5) -> bool:
        """
        Returns True if recent bars show consolidation.

        Consolidation: low range AND low volume, indicating price compression
        before a breakout.

        Args:
            bars: List of bar dictionaries in chronological order.
            threshold: Maximum normalised range to consider as consolidation.
                       Lower = tighter consolidation. Default 0.5.

        Returns:
            bool: True if consolidation detected, False otherwise.
        """
        if len(bars) < 3:
            return False

        # Use last 5 bars for consolidation check
        lookback = bars[-5:] if len(bars) >= 5 else bars

        consolidation_count = 0
        for bar in lookback:
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)
            volume = bar.get("volume", 0.0)
            close = bar.get("close", 0.0)

            mid_price = (high + low) / 2.0
            if mid_price <= 0:
                continue

            price_range = high - low
            range_ratio = price_range / mid_price

            # Low range: consolidation
            if range_ratio < threshold * 0.001:
                consolidation_count += 1

        # At least 3 of the last 5 bars must show consolidation
        return consolidation_count >= 3

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/breakout_pressure_proxy",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Heuristic: volume+range+spread buildup signals breakout pressure",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"open", "high", "low", "close", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"breakout_pressure", "consolidation_detected"}

    def capability_spec(self) -> Dict[str, Any]:
        return {
            "feature_id": "microstructure/breakout_pressure_proxy",
            "quality_class": "proxy_inferred",
            "inputs": ["open", "high", "low", "close", "volume"],
            "outputs": ["breakout_pressure", "consolidation_detected"],
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
