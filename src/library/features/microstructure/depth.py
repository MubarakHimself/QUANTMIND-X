"""MultiLevelDepthFeature — approximates multi-level order book depth from OHLCV."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureConfidence
from src.library.core.types.enums import FeatureConfidenceLevel

from src.library.features.microstructure.microstructure_base import MicrostructureFeature


class MultiLevelDepthFeature(MicrostructureFeature):
    """
    Approximates multi-level order book depth from OHLCV price action.

    Simulates depth levels (bid/ask 1-5) from:
      - Range: wider bars suggest deeper liquidity on both sides
      - Volume: high volume on narrow range = absorbing (thin book)
      - Spread: spread widening suggests thinner book at each level

    The approximation uses these heuristics:
      - Base depth per level decreases exponentially (level 1 thickest)
      - Wide-range bars add depth symmetrically
      - High-volume narrow-range bars reduce depth (absorption signal)
      - Spread widening reduces estimated depth at outer levels

    Quality: proxy_inferred (simulated from OHLCV, not real order book data)
    """

    def __init__(self, levels: int = 5, **kwargs: Any) -> None:
        # Let Pydantic init run first so feature_id and base fields are set
        super().__init__(**kwargs)
        # Private state via __dict__ to avoid Pydantic field conflicts
        self.__dict__["_levels"] = max(1, levels)
        self.__dict__["_spread_history"] = []

    def compute(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute estimated depth levels for a single bar.

        Args:
            bar: Dictionary with high, low, close, volume, spread (optional).

        Returns:
            Dict with keys:
              - depth_bid: List[float] estimated bid depth per level
              - depth_ask: List[float] estimated ask depth per level
              - net_depth_imbalance: float (positive = more bid depth)
              - liquidity_stress: float in [0.0, 1.0]
        """
        high = bar.get("high", 0.0)
        low = bar.get("low", 0.0)
        close = bar.get("close", 0.0)
        volume = bar.get("volume", 0.0)
        spread = bar.get("spread", 0.0)

        spread_hist = self.__dict__["_spread_history"]
        if spread > 0:
            spread_hist.append(spread)
            if len(spread_hist) > 20:
                spread_hist[:] = spread_hist[-20:]
            self.__dict__["_spread_history"] = spread_hist

        price_range = high - low
        mid_price = (high + low) / 2.0

        if mid_price == 0 or price_range < 0:
            return self._neutral_depth()

        base_depth = 1000.0
        depth_bid: List[float] = []
        depth_ask: List[float] = []

        for level in range(1, self.__dict__["_levels"] + 1):
            decay = base_depth / (level ** 1.5)

            range_factor = 1.0 + min(price_range / mid_price, 1.0)

            range_ratio = price_range / mid_price if mid_price > 0 else 0.0
            if range_ratio < 0.0001:
                absorption_penalty = 0.6
            elif range_ratio < 0.001:
                absorption_penalty = 0.85
            else:
                absorption_penalty = 1.0

            adjusted_depth = decay * range_factor * absorption_penalty
            depth_bid.append(round(adjusted_depth, 2))
            depth_ask.append(round(adjusted_depth, 2))

        net_imbalance = sum(depth_bid) - sum(depth_ask)

        spread_hist = self.__dict__["_spread_history"]
        avg_spread = (
            sum(spread_hist) / len(spread_hist)
            if spread_hist
            else spread
        )
        spread_ratio = spread / avg_spread if avg_spread > 0 else 1.0
        range_ratio = price_range / mid_price if mid_price > 0 else 0.0

        volume_factor = min(volume / 10000.0, 1.0)
        range_stress = max(0.0, 1.0 - range_ratio * 100)
        spread_stress = min(spread_ratio / 2.0, 1.0)

        liquidity_stress = round(
            volume_factor * 0.3 + range_stress * 0.4 + spread_stress * 0.3, 3
        )
        liquidity_stress = max(0.0, min(1.0, liquidity_stress))

        return {
            "depth_bid": depth_bid,
            "depth_ask": depth_ask,
            "net_depth_imbalance": round(net_imbalance, 2),
            "liquidity_stress": liquidity_stress,
        }

    def compute_batch(self, bars: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compute depth estimates over multiple bars.

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            List of depth dicts.
        """
        return [self.compute(bar) for bar in bars]

    def _neutral_depth(self) -> Dict[str, Any]:
        """Return neutral depth when bar data is invalid."""
        neutral = [0.0] * self.__dict__["_levels"]
        return {
            "depth_bid": neutral.copy(),
            "depth_ask": neutral.copy(),
            "net_depth_imbalance": 0.0,
            "liquidity_stress": 0.5,
        }

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/multi_level_depth",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Simulated depth from OHLCV range/volume heuristics; not real order book",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"high", "low", "close", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"depth_bid", "depth_ask", "net_depth_imbalance", "liquidity_stress"}

    def confidence(self, n_bars: int) -> FeatureConfidence:
        """
        Compute confidence based on number of available bars.

        Args:
            n_bars: Number of bars available for the computation.

        Returns:
            FeatureConfidence with quality score and feed tag.
        """
        if n_bars < 20:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.DISABLED.value,
            )
        elif n_bars < 50:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.3,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.LOW.value,
            )
        elif n_bars < 100:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.6,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.MEDIUM.value,
            )
        else:
            return FeatureConfidence(
                source="ctrader_native",
                quality=0.8,
                latency_ms=0.0,
                feed_quality_tag=FeatureConfidenceLevel.HIGH.value,
            )
