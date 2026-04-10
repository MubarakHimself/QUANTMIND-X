"""TopOfBookPressureFeature — measure order book imbalance at top of book."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class TopOfBookPressureFeature(FeatureModule):
    """
    Measures order book imbalance at the top of the book (best bid vs best ask).

    A positive pressure indicates buy-side dominance; negative indicates sell-side.
    Threshold at +/-0.3 to avoid noise from minor imbalances.

    Quality class: native_supported
    Source: ctrader_native (computed from TOB bid/ask sizes)

    Inputs required:
        - bid_size: float  — size at best bid
        - ask_size: float  — size at best ask

    Outputs:
        - tob_pressure (float): range [-1, 1]. Positive=buy, negative=sell.
        - tob_imbalance (float): 1.0=BUY_HEAVY, -1.0=SELL_HEAVY, 0.0=BALANCED
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/tob_pressure",
            quality_class="native_supported",
            source="ctrader_native",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"bid_size", "ask_size"}

    @property
    def output_keys(self) -> Set[str]:
        return {"tob_pressure", "tob_imbalance"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        bid_size = inputs.get("bid_size", 0.0)
        ask_size = inputs.get("ask_size", 0.0)

        # Validate non-zero positive sizes
        if bid_size <= 0 or ask_size <= 0:
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={
                    "tob_pressure": 0.0,
                    "tob_imbalance": 0.0,
                },
                feature_confidence={
                    "tob_pressure": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "tob_imbalance": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                },
            )

        total = bid_size + ask_size
        pressure = (bid_size - ask_size) / total if total > 0 else 0.0

        # Encode imbalance as float: 1=BUY_HEAVY, -1=SELL_HEAVY, 0=BALANCED
        if pressure > 0.3:
            imbalance_val = 1.0
            tag = "HIGH"
        elif pressure < -0.3:
            imbalance_val = -1.0
            tag = "HIGH"
        else:
            imbalance_val = 0.0
            tag = "MEDIUM"

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={
                "tob_pressure": pressure,
                "tob_imbalance": imbalance_val,
            },
            feature_confidence={
                "tob_pressure": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.9,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
                "tob_imbalance": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.9,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
            },
        )
