"""SpreadStateFeature — detect spread widening/narrowing relative to rolling average."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class SpreadStateFeature(FeatureModule):
    """
    Detects spread widening or narrowing relative to a rolling average.
    Key signal for scalping decisions — tight spreads favour entry,
    wide spreads signal adverse conditions.

    Quality class: native_supported
    Source: ctrader_native (computed from spread bars)

    Inputs required:
        - spread: float or List[float]  — current spread (or last element of history)
        - spread_history: List[float], optional — rolling history of spreads

    Outputs:
        - spread_state (float): 1.0=WIDE, 0.0=NORMAL, 0.5=NARROW
        - spread_ratio (float): current_spread / avg_spread
        - spread_tightness (float): 0.0–1.0, how tight (1.0 = very tight)
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/spread_state",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"spread"}

    @property
    def output_keys(self) -> Set[str]:
        return {"spread_state", "spread_ratio", "spread_tightness"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        spread = inputs.get("spread")
        spread_history = inputs.get("spread_history", [])

        # Normalise spread to float value
        if isinstance(spread, list):
            spread_val = spread[-1] if spread else 0.0
        else:
            spread_val = spread if spread is not None else 0.0

        # Determine rolling average
        if spread_history and len(spread_history) >= 5:
            avg_spread = sum(spread_history) / len(spread_history)
        else:
            # Cannot determine state without sufficient history
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={
                    "spread_state": 0.5,
                    "spread_ratio": 1.0,
                    "spread_tightness": 0.5,
                },
                feature_confidence={
                    "spread_state": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "spread_ratio": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "spread_tightness": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                },
            )

        # Compute ratio
        ratio = spread_val / avg_spread if avg_spread > 0 else 1.0

        # Determine state
        if ratio > 1.5:
            state_val = 1.0   # WIDE
            tag = "MEDIUM"
        elif ratio < 0.5:
            state_val = 0.5  # NARROW
            tag = "HIGH"
        else:
            state_val = 0.0  # NORMAL
            tag = "HIGH"

        # Tightness: 1.0 = very tight, 0.0 = very wide
        # ratio clamped to [0, 3] before inverting
        tightness = max(0.0, 1.0 - min(ratio, 3.0) / 3.0)

        # Quality improves with more history
        quality = min(0.9, 0.5 + len(spread_history) * 0.02)
        quality_tag = "HIGH" if quality >= 0.7 else "MEDIUM"

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={
                "spread_state": state_val,
                "spread_ratio": ratio,
                "spread_tightness": tightness,
            },
            feature_confidence={
                "spread_state": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "spread_ratio": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "spread_tightness": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
            },
        )
