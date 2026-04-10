"""DepthThinningFeature — detect when order book depth is thinning."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class DepthThinningFeature(FeatureModule):
    """
    Detects when order book depth is thinning — a potential precursor to
    sharp price moves or liquidity voids.

    Compares current total depth against a fixed baseline (20000 units total,
    10000 per side). A ratio below 0.4 signals significant thinning.

    Quality class: proxy_inferred (uses fixed baseline, not live history)
    Source: ctrader_native

    Inputs required:
        - bid_sizes: List[float]  — sizes at successive bid levels
        - ask_sizes: List[float]  — sizes at successive ask levels

    Outputs:
        - depth_thinning_ratio (float): 0.0–1.0. 1.0=normal depth, 0.0=extreme thinning.
        - depth_thinning_signal (float): 1.0=THINNING, 0.0=NORMAL
        - book_vacuity (float): 0.0–1.0. 0.0=full book, 1.0=empty.
    """

    # Fixed baseline: 10000 units per side = 20000 total
    DEFAULT_BASELINE_PER_SIDE: float = 10000.0

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="orderflow/depth_thinning",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Fixed baseline comparison — no live historical tracking in library layer",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"bid_sizes", "ask_sizes"}

    @property
    def output_keys(self) -> Set[str]:
        return {"depth_thinning_ratio", "depth_thinning_signal", "book_vacuity"}

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
        has_positive = any(b > 0 for b in bid_sizes) or any(a > 0 for a in ask_sizes)

        if not has_data:
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={
                    "depth_thinning_ratio": 1.0,
                    "depth_thinning_signal": 0.0,
                    "book_vacuity": 0.0,
                },
                feature_confidence={
                    "depth_thinning_ratio": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "depth_thinning_signal": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "book_vacuity": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                },
            )

        # Current depth
        total_bid = sum(bid_sizes)
        total_ask = sum(ask_sizes)
        current_depth = total_bid + total_ask

        # Fixed baseline (both sides)
        baseline = self.DEFAULT_BASELINE_PER_SIDE * 2
        baseline = max(baseline, 0.001)  # Prevent divide-by-zero

        # Thinning ratio clamped to [0, 1]
        ratio = current_depth / baseline
        ratio = max(0.0, min(1.0, ratio))

        # Signal encoding: 1.0=THINNING, 0.0=NORMAL
        if ratio < 0.4:
            signal_val = 1.0
            tag = "MEDIUM"
        else:
            signal_val = 0.0
            tag = "HIGH"

        # Vacuity = inverse of ratio
        vacuity = 1.0 - ratio

        quality = 0.75 if has_positive else 0.3
        quality_tag = "MEDIUM" if has_positive else "LOW"

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={
                "depth_thinning_ratio": ratio,
                "depth_thinning_signal": signal_val,
                "book_vacuity": vacuity,
            },
            feature_confidence={
                "depth_thinning_ratio": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
                "depth_thinning_signal": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
                "book_vacuity": FeatureConfidence(
                    source="ctrader_native",
                    quality=quality,
                    latency_ms=0.0,
                    feed_quality_tag=quality_tag,
                ),
            },
        )
