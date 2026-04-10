"""DOMPressureFeature — order book depth pressure at multiple levels."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class DOMPressureFeature(FeatureModule):
    """
    Measures order book pressure across multiple depth levels (bid/ask sizes
    at successive price levels), computing imbalance at each level and an
    aggregate depth assessment.

    Quality class: proxy_inferred (aggregates raw TOB+depth signals)
    Source: ctrader_native

    Inputs required:
        - bid_sizes: List[float]  — sizes at successive bid levels
        - ask_sizes: List[float]  — sizes at successive ask levels

    Outputs:
        - dom_pressure_1/2/3 (float): pressure at level N, [-1, 1]
        - depth_imbalance (float): 1.0=LONG_THICK, -1.0=SHORT_THICK, 0.0=BALANCED
        - total_bid_depth (float): sum of all bid_sizes
        - total_ask_depth (float): sum of all ask_sizes
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="orderflow/dom_pressure",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Multi-level order book pressure from bid/ask depth arrays",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"bid_sizes", "ask_sizes"}

    @property
    def output_keys(self) -> Set[str]:
        return {
            "dom_pressure_1",
            "dom_pressure_2",
            "dom_pressure_3",
            "depth_imbalance",
            "total_bid_depth",
            "total_ask_depth",
        }

    def _compute_pressure(self, bid: float, ask: float) -> float:
        total = bid + ask
        if total == 0:
            return 0.0
        return (bid - ask) / total

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        bid_sizes_raw = inputs.get("bid_sizes", [])
        ask_sizes_raw = inputs.get("ask_sizes", [])

        # Ensure lists
        if not isinstance(bid_sizes_raw, list):
            bid_sizes: List[float] = [bid_sizes_raw] if bid_sizes_raw else []
        else:
            bid_sizes = bid_sizes_raw

        if not isinstance(ask_sizes_raw, list):
            ask_sizes: List[float] = [ask_sizes_raw] if ask_sizes_raw else []
        else:
            ask_sizes = ask_sizes_raw

        # Check sufficient data
        has_data = len(bid_sizes) >= 1 and len(ask_sizes) >= 1
        has_positive = any(b > 0 for b in bid_sizes) and any(a > 0 for a in ask_sizes)

        features: Dict[str, float] = {}
        confidences: Dict[str, FeatureConfidence] = {}

        if not has_data:
            # Return zeroed outputs with insufficient data confidence
            for key in self.output_keys:
                features[key] = 0.0
                confidences[key] = FeatureConfidence(
                    source="ctrader_native",
                    quality=0.0,
                    latency_ms=0.0,
                    feed_quality_tag="INSUFFICIENT_DATA",
                )
            return FeatureVector(timestamp=datetime.now(), bot_id="", features=features, feature_confidence=confidences)

        # Compute pressure at each available level
        min_len = min(len(bid_sizes), len(ask_sizes))
        for i in range(min(3, min_len)):
            key = f"dom_pressure_{i + 1}"
            pressure = self._compute_pressure(bid_sizes[i], ask_sizes[i])
            features[key] = pressure
            confidences[key] = FeatureConfidence(
                source="ctrader_native",
                quality=0.85,
                latency_ms=0.0,
                feed_quality_tag="HIGH" if has_positive else "LOW",
            )

        # Pad missing levels
        for i in range(min_len, 3):
            key = f"dom_pressure_{i + 1}"
            features[key] = 0.0
            confidences[key] = FeatureConfidence(
                source="ctrader_native",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag="INSUFFICIENT_DATA",
            )

        # Total depth
        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        features["total_bid_depth"] = total_bid
        features["total_ask_depth"] = total_ask
        confidences["total_bid_depth"] = FeatureConfidence(
            source="ctrader_native", quality=0.85, latency_ms=0.0, feed_quality_tag="HIGH" if has_positive else "LOW"
        )
        confidences["total_ask_depth"] = FeatureConfidence(
            source="ctrader_native", quality=0.85, latency_ms=0.0, feed_quality_tag="HIGH" if has_positive else "LOW"
        )

        # Depth imbalance encoding: 1.0=LONG_THICK, -1.0=SHORT_THICK, 0.0=BALANCED
        if total_bid > total_ask * 1.5:
            imbalance_val = 1.0
            imbalance_tag = "HIGH"
        elif total_ask > total_bid * 1.5:
            imbalance_val = -1.0
            imbalance_tag = "HIGH"
        else:
            imbalance_val = 0.0
            imbalance_tag = "MEDIUM"

        features["depth_imbalance"] = imbalance_val
        confidences["depth_imbalance"] = FeatureConfidence(
            source="ctrader_native",
            quality=0.85,
            latency_ms=0.0,
            feed_quality_tag=imbalance_tag,
        )

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features=features,
            feature_confidence=confidences,
        )
