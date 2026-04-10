"""SpreadBehaviorFeature — characterise spread behaviour relative to volume."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class SpreadBehaviorFeature(FeatureModule):
    """
    Characterises how the spread behaves relative to volume — a key indicator
    of execution quality and transaction cost.

    Spread efficiency = volume / spread. Higher values indicate more volume
    per unit of spread cost, meaning tighter effective execution conditions.

    Quality class: proxy_inferred (combines two raw signals into a derived metric)
    Source: ctrader_native

    Inputs required:
        - spread: float or List[float]  — current spread value
        - volume: float or List[float]  — current bar volume
        - session_id: str, optional     — session context for baseline

    Outputs:
        - spread_efficiency (float): volume / spread. Higher = more efficient.
        - spread_cost_impact (float): 0.0–1.0, normalised cost impact.
          1.0 = low cost, 0.0 = high cost.
        - spread_regime (float): 1.0=EFFICIENT, 0.0=EXPENSIVE, 0.5=MODERATE
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="orderflow/spread_behavior",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Volume-per-pip efficiency: higher volume per spread unit = better conditions",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"spread", "volume"}

    @property
    def output_keys(self) -> Set[str]:
        return {"spread_efficiency", "spread_cost_impact", "spread_regime"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        spread_raw = inputs.get("spread")
        volume_raw = inputs.get("volume")

        # Normalise spread to float
        if isinstance(spread_raw, list):
            spread_val = spread_raw[-1] if spread_raw else 1.0
        else:
            spread_val = spread_raw if spread_raw is not None else 1.0

        # Normalise volume to float
        if isinstance(volume_raw, list):
            volume_val = volume_raw[-1] if volume_raw else 0.0
        else:
            volume_val = volume_raw if volume_raw is not None else 0.0

        if spread_val <= 0:
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={
                    "spread_efficiency": 0.0,
                    "spread_cost_impact": 0.0,
                    "spread_regime": 0.5,
                },
                feature_confidence={
                    "spread_efficiency": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="NO_DATA",
                    ),
                    "spread_cost_impact": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="NO_DATA",
                    ),
                    "spread_regime": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="NO_DATA",
                    ),
                },
            )

        if volume_val < 0:
            volume_val = 0.0

        # Efficiency: higher volume per pip of spread = more efficient market
        efficiency = volume_val / spread_val

        # Cost impact normalised: efficiency / 1000 gives 0-1 range
        cost_impact = min(1.0, efficiency / 1000.0)

        # Regime encoding: 1.0=EFFICIENT, 0.0=EXPENSIVE, 0.5=MODERATE
        if efficiency >= 1.0:
            regime_val = 1.0
            tag = "HIGH"
        elif efficiency < 0.3:
            regime_val = 0.0
            tag = "LOW"
        else:
            regime_val = 0.5
            tag = "MEDIUM"

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={
                "spread_efficiency": efficiency,
                "spread_cost_impact": cost_impact,
                "spread_regime": regime_val,
            },
            feature_confidence={
                "spread_efficiency": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.8,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
                "spread_cost_impact": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.8,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
                "spread_regime": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.8,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
            },
        )
