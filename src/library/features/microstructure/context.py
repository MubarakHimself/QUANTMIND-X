"""
QuantMindLib V1 — MicrostructureContext Aggregation (Packet 11C)

Aggregates all microstructure features into a single context object.

Combines: VolumeImbalance, TickActivity, MultiLevelDepth,
Absorption, BreakoutPressure, LiquidityStress
into a unified view for downstream consumers.

FEATURE-029
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.library.features.microstructure.absorption import AbsorptionProxyFeature
from src.library.features.microstructure.breakout_pressure import (
    BreakoutPressureProxyFeature,
)
from src.library.features.microstructure.depth import MultiLevelDepthFeature
from src.library.features.microstructure.liquidity_stress import (
    LiquidityStressProxyFeature,
)
from src.library.features.microstructure.tick_activity import TickActivityFeature
from src.library.features.microstructure.volume_imbalance import (
    VolumeImbalanceFeature,
)


class MicrostructureContext:
    """
    Aggregates all microstructure features into a single context object.

    Combines: VolumeImbalance, TickActivity, MultiLevelDepth,
    Absorption, BreakoutPressure, LiquidityStress
    into a unified view for downstream consumers.

    All features are optional — compute_all / compute_batch skip any that
    are None. This allows partial composition (e.g., only volume imbalance
    and tick activity without depth).

    Example:
        ctx = MicrostructureContext(
            volume_imbalance=VolumeImbalanceFeature(),
            tick_activity=TickActivityFeature(),
        )
        result = ctx.compute_all(bar)
        # result = {"volume_imbalance": 0.35, "tick_activity": 1.2}
    """

    def __init__(
        self,
        volume_imbalance: Optional[VolumeImbalanceFeature] = None,
        tick_activity: Optional[TickActivityFeature] = None,
        depth: Optional[MultiLevelDepthFeature] = None,
        absorption: Optional[AbsorptionProxyFeature] = None,
        breakout_pressure: Optional[BreakoutPressureProxyFeature] = None,
        liquidity_stress: Optional[LiquidityStressProxyFeature] = None,
    ) -> None:
        self.volume_imbalance = volume_imbalance
        self.tick_activity = tick_activity
        self.depth = depth
        self.absorption = absorption
        self.breakout_pressure = breakout_pressure
        self.liquidity_stress = liquidity_stress

        # Compute active feature count
        self._feature_count = sum(1 for f in [
            volume_imbalance, tick_activity, depth, absorption,
            breakout_pressure, liquidity_stress,
        ] if f is not None)

        # Rolling bar history for batch operations
        self._bar_history: List[Dict[str, Any]] = []

    def compute_all(self, bar: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all features on a single bar.

        Args:
            bar: Dictionary with OHLCV fields and optional feature-specific
                 keys (tick_count, spread, etc.)

        Returns:
            Dict mapping feature names to their computed values.
            Only features that are set (not None) are included.
            Empty keys are excluded from the result.
        """
        result: Dict[str, Any] = {}

        if self.volume_imbalance is not None:
            val = self.volume_imbalance.compute(bar)
            if val is not None and val != 0.0:
                result["volume_imbalance"] = val

        if self.tick_activity is not None:
            val = self.tick_activity.compute(bar)
            if val is not None and val != 0.0:
                result["tick_activity"] = val

        if self.depth is not None:
            val = self.depth.compute(bar)
            if val is not None:
                result["depth"] = val

        if self.absorption is not None:
            val = self.absorption.compute(bar)
            if val is not None and val != 0.0:
                result["absorption"] = val

        if self.breakout_pressure is not None:
            val = self.breakout_pressure.compute(bar)
            if val is not None and val != 0.0:
                result["breakout_pressure"] = val

        if self.liquidity_stress is not None:
            val = self.liquidity_stress.compute(bar)
            if val is not None and val != 0.0:
                result["liquidity_stress"] = val

        return result

    def compute_batch(
        self,
        bars: List[Dict[str, Any]],
    ) -> Dict[str, List[Any]]:
        """
        Run all features over a batch of bars.

        Args:
            bars: List of bar dictionaries in chronological order.

        Returns:
            Dict mapping feature names to lists of computed values,
            one per bar. Only features that are set are included.
        """
        if not bars:
            return {}

        # Accumulate bar history for features that track lookback
        self._bar_history.extend(bars)
        # Keep history bounded to 200 bars
        if len(self._bar_history) > 200:
            self._bar_history[:] = self._bar_history[-200:]

        # Compute per-bar results
        per_bar = [self.compute_all(bar) for bar in bars]

        # Transpose into feature-keyed lists
        result: Dict[str, List[Any]] = {}

        feature_keys = [
            "volume_imbalance",
            "tick_activity",
            "depth",
            "absorption",
            "breakout_pressure",
            "liquidity_stress",
        ]

        for key in feature_keys:
            values = [item.get(key) for item in per_bar if key in item]
            if values:
                result[key] = values

        return result

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a one-line summary of current microstructure state.

        Returns:
            Dict with:
              - volume_imbalance: str ("BUYING", "SELLING", "BALANCED")
              - tick_activity: str ("LOW", "NORMAL", "HIGH", "EXTREME")
              - liquidity_stress: str ("NORMAL", "ELEVATED", "HIGH", "CRITICAL")
              - breakout_pressure: float (0.0-5.0+, if available)
              - absorption: float (if available)
              - market_stressed: bool
              - active_features: int
        """
        summary: Dict[str, Any] = {
            "active_features": self._feature_count,
        }

        if self.volume_imbalance is not None and self._bar_history:
            last_bar = self._bar_history[-1]
            imb = self.volume_imbalance.compute(last_bar)
            if imb is not None:
                if imb > 0.2:
                    summary["volume_imbalance"] = "BUYING"
                elif imb < -0.2:
                    summary["volume_imbalance"] = "SELLING"
                else:
                    summary["volume_imbalance"] = "BALANCED"
                summary["volume_imbalance_value"] = round(imb, 3)

        if self.tick_activity is not None and self._bar_history:
            last_bar = self._bar_history[-1]
            act = self.tick_activity.compute(last_bar)
            if act is not None:
                level = self.tick_activity.get_activity_level(act)
                summary["tick_activity"] = level
                summary["tick_activity_value"] = round(act, 3)

        if self.liquidity_stress is not None and self._bar_history:
            last_bar = self._bar_history[-1]
            stress = self.liquidity_stress.compute(last_bar)
            if stress is not None:
                level = self.liquidity_stress.get_stress_level(stress)
                summary["liquidity_stress"] = level
                summary["liquidity_stress_value"] = round(stress, 3)

        if self.breakout_pressure is not None and self._bar_history:
            last_bar = self._bar_history[-1]
            pressure = self.breakout_pressure.compute(last_bar)
            if pressure is not None:
                summary["breakout_pressure"] = round(pressure, 3)

        if self.absorption is not None and self._bar_history:
            last_bar = self._bar_history[-1]
            absorb = self.absorption.compute(last_bar)
            if absorb is not None:
                summary["absorption"] = round(absorb, 3)

        summary["market_stressed"] = self.is_market_stressed()

        return summary

    def is_market_stressed(self) -> bool:
        """
        True if any feature indicates market stress.

        Checks:
        - LiquidityStress: score > 0.6
        - BreakoutPressure: pressure > 3.0 (extreme)
        - TickActivity: EXTREME level
        """
        if not self._bar_history:
            return False

        last_bar = self._bar_history[-1]

        # Check liquidity stress
        if self.liquidity_stress is not None:
            stress = self.liquidity_stress.compute(last_bar)
            if stress is not None and stress > 0.6:
                return True

        # Check breakout pressure (extreme = imminent breakout)
        if self.breakout_pressure is not None:
            pressure = self.breakout_pressure.compute(last_bar)
            if pressure is not None and pressure > 3.0:
                return True

        # Check tick activity (EXTREME)
        if self.tick_activity is not None:
            act = self.tick_activity.compute(last_bar)
            if act is not None and self.tick_activity.get_activity_level(act) == "EXTREME":
                return True

        return False

    def get_feature_count(self) -> int:
        """
        Return the number of active microstructure features.

        Returns:
            int: Count of features that are not None (0-6).
        """
        return self._feature_count
