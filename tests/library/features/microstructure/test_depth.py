"""
Self-Verification Tests for Packet 11A: MultiLevelDepthFeature.
15 tests covering init, compute, compute_batch, depth levels, liquidity_stress, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.depth import MultiLevelDepthFeature


class TestMultiLevelDepthInit:
    def test_default_levels(self):
        feat = MultiLevelDepthFeature()
        assert feat._levels == 5

    def test_custom_levels(self):
        feat = MultiLevelDepthFeature(levels=3)
        assert feat._levels == 3

    def test_feature_id(self):
        feat = MultiLevelDepthFeature()
        assert feat.config.feature_id == "microstructure/multi_level_depth"

    def test_quality_class(self):
        feat = MultiLevelDepthFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestMultiLevelDepthCompute:
    def test_returns_dict_with_all_keys(self):
        """compute() returns dict with depth_bid, depth_ask, net_depth_imbalance, liquidity_stress."""
        feat = MultiLevelDepthFeature()
        bar = {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0, "spread": 0.0002}
        result = feat.compute(bar)
        assert "depth_bid" in result
        assert "depth_ask" in result
        assert "net_depth_imbalance" in result
        assert "liquidity_stress" in result

    def test_depth_levels_count(self):
        """depth_bid and depth_ask lists have correct length matching _levels."""
        feat5 = MultiLevelDepthFeature(levels=5)
        bar = {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0}
        result = feat5.compute(bar)
        assert len(result["depth_bid"]) == 5
        assert len(result["depth_ask"]) == 5

        feat3 = MultiLevelDepthFeature(levels=3)
        result3 = feat3.compute(bar)
        assert len(result3["depth_bid"]) == 3
        assert len(result3["depth_ask"]) == 3

    def test_bid_ask_symmetric_by_default(self):
        """Without asymmetry signals, bid and ask depth are equal."""
        feat = MultiLevelDepthFeature()
        bar = {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0, "spread": 0.0002}
        result = feat.compute(bar)
        assert result["depth_bid"] == result["depth_ask"]
        assert result["net_depth_imbalance"] == 0.0

    def test_depth_decreases_by_level(self):
        """Deeper levels have lower depth values (exponential decay)."""
        feat = MultiLevelDepthFeature()
        bar = {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0}
        result = feat.compute(bar)
        for i in range(len(result["depth_bid"]) - 1):
            assert result["depth_bid"][i] >= result["depth_bid"][i + 1]
            assert result["depth_ask"][i] >= result["depth_ask"][i + 1]

    def test_liquidity_stress_in_range(self):
        """liquidity_stress is always in [0.0, 1.0]."""
        feat = MultiLevelDepthFeature()
        bars = [
            {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0, "spread": 0.0002},
            {"high": 1.0860, "low": 1.0854, "close": 1.0854, "volume": 50000.0, "spread": 0.001},
            {"high": 1.0870, "low": 1.0830, "close": 1.0850, "volume": 1000.0, "spread": 0.0001},
        ]
        for bar in bars:
            result = feat.compute(bar)
            assert 0.0 <= result["liquidity_stress"] <= 1.0

    def test_neutral_depth_on_invalid_data(self):
        """Invalid bar (zero price) returns neutral depth."""
        feat = MultiLevelDepthFeature()
        bar = {"high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0}
        result = feat.compute(bar)
        assert result["net_depth_imbalance"] == 0.0
        assert result["liquidity_stress"] == 0.5


class TestMultiLevelDepthComputeBatch:
    def test_batch_returns_list_of_dicts(self):
        feat = MultiLevelDepthFeature()
        bars = [
            {"high": 1.0860, "low": 1.0848, "close": 1.0854, "volume": 5000.0},
            {"high": 1.0870, "low": 1.0850, "close": 1.0860, "volume": 6000.0},
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 2
        for r in results:
            assert "depth_bid" in r
            assert "depth_ask" in r
            assert "net_depth_imbalance" in r
            assert "liquidity_stress" in r

    def test_batch_empty_returns_empty(self):
        feat = MultiLevelDepthFeature()
        assert feat.compute_batch([]) == []


class TestMultiLevelDepthCapabilitySpec:
    def test_required_inputs(self):
        feat = MultiLevelDepthFeature()
        assert "high" in feat.required_inputs
        assert "low" in feat.required_inputs
        assert "close" in feat.required_inputs
        assert "volume" in feat.required_inputs

    def test_output_keys(self):
        feat = MultiLevelDepthFeature()
        assert "depth_bid" in feat.output_keys
        assert "depth_ask" in feat.output_keys
        assert "net_depth_imbalance" in feat.output_keys
        assert "liquidity_stress" in feat.output_keys


class TestMultiLevelDepthConfidence:
    def test_disabled_under_20_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(19)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_20_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_low_at_49_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(49)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_50_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(50)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_medium_at_99_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(99)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_100_bars(self):
        feat = MultiLevelDepthFeature()
        conf = feat.confidence(100)
        assert conf.quality == 0.8
        assert conf.feed_quality_tag == "HIGH"
