"""AggressionProxyFeature — proxy for order flow aggression from price action + volume."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Set

from src.library.features.base import FeatureModule, FeatureConfig
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


class AggressionProxyFeature(FeatureModule):
    """
    Proxies order flow aggression based on price action and volume over the last 2 bars.

    Combines direction (price up/down) with conviction (volume increasing) to
    classify aggression as aggressive buy, aggressive sell, or neutral.

    Quality class: proxy_inferred (heuristic — not a direct exchange metric)
    Source: ctrader_native

    Inputs required:
        - close_prices: List[float]  — at least 2 elements
        - volumes: List[float]       — at least 2 elements

    Outputs:
        - aggression_proxy (float): 0.0–1.0. Higher = more aggressive buying.
          0.8=aggressive buy, 0.2=aggressive sell, 0.5=neutral.
        - aggression_signal (float): 1.0=AGGRESSIVE_BUY, -1.0=AGGRESSIVE_SELL, 0.0=NEUTRAL
    """

    @property
    def config(self) -> FeatureConfig:
        return FeatureConfig(
            feature_id="microstructure/aggression",
            quality_class="proxy_inferred",
            source="ctrader_native",
            notes="Heuristic approximation of order flow aggression from price+volume",
        )

    @property
    def required_inputs(self) -> Set[str]:
        return {"close_prices", "volumes"}

    @property
    def output_keys(self) -> Set[str]:
        return {"aggression_proxy", "aggression_signal"}

    def compute(self, inputs: Dict[str, Any]) -> FeatureVector:
        close_prices = inputs.get("close_prices", [])
        volumes = inputs.get("volumes", [])

        if len(close_prices) < 2 or len(volumes) < 2:
            return FeatureVector(
                timestamp=datetime.now(),
                bot_id="",
                features={
                    "aggression_proxy": 0.5,
                    "aggression_signal": 0.0,
                },
                feature_confidence={
                    "aggression_proxy": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                    "aggression_signal": FeatureConfidence(
                        source="ctrader_native",
                        quality=0.0,
                        latency_ms=0.0,
                        feed_quality_tag="INSUFFICIENT_DATA",
                    ),
                },
            )

        price_up = close_prices[-1] > close_prices[-2]
        volume_up = volumes[-1] > volumes[-2]

        if price_up and volume_up:
            proxy = 0.8
            signal_val = 1.0    # AGGRESSIVE_BUY
            tag = "MEDIUM"
        elif not price_up and volume_up:
            proxy = 0.2
            signal_val = -1.0   # AGGRESSIVE_SELL
            tag = "MEDIUM"
        else:
            proxy = 0.5
            signal_val = 0.0    # NEUTRAL
            tag = "LOW"

        return FeatureVector(
            timestamp=datetime.now(),
            bot_id="",
            features={
                "aggression_proxy": proxy,
                "aggression_signal": signal_val,
            },
            feature_confidence={
                "aggression_proxy": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.7,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
                "aggression_signal": FeatureConfidence(
                    source="ctrader_native",
                    quality=0.7,
                    latency_ms=0.0,
                    feed_quality_tag=tag,
                ),
            },
        )
