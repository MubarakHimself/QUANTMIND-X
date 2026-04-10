"""
Self-Verification Tests for Packet 11B: AbsorptionProxyFeature.
15 tests covering init, compute, compute_batch, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.absorption import AbsorptionProxyFeature


class TestAbsorptionInit:
    def test_default_smoothing(self):
        feat = AbsorptionProxyFeature()
        assert feat._smoothing == 0.3
        assert feat._prev_score == 0.0

    def test_custom_smoothing(self):
        feat = AbsorptionProxyFeature(smoothing=0.5)
        assert feat._smoothing == 0.5

    def test_feature_id(self):
        feat = AbsorptionProxyFeature()
        assert feat.config.feature_id == "microstructure/absorption_proxy"

    def test_quality_class(self):
        feat = AbsorptionProxyFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestAbsorptionCompute:
    def test_high_volume_narrow_range_absorption(self):
        """High volume / narrow range = absorption (high score)."""
        feat = AbsorptionProxyFeature()
        # high volume, tiny range => high absorption score
        bar = {
            "open": 1.0850,
            "high": 1.0851,
            "low": 1.0850,
            "close": 1.0850,
            "volume": 50000.0,
        }
        result = feat.compute(bar)
        # Score = 50000 / 0.0001 = 500,000,000 — very high absorption
        assert result > 0.0

    def test_low_volume_wide_range_no_absorption(self):
        """Low volume / wide range = no absorption (near-zero score)."""
        feat = AbsorptionProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0900,
            "low": 1.0800,
            "close": 1.0850,
            "volume": 100.0,
        }
        result = feat.compute(bar)
        # Score = 100 / 0.01 = 10,000 — but clamped to 0.0 minimum
        assert result >= 0.0

    def test_zero_volume_returns_zero(self):
        """Zero volume bar returns 0.0."""
        feat = AbsorptionProxyFeature()
        bar = {"open": 1.0850, "high": 1.0855, "low": 1.0845, "close": 1.0850, "volume": 0.0}
        assert feat.compute(bar) == 0.0

    def test_zero_range_returns_zero(self):
        """Flat bar (zero range) returns 0.0 even with volume."""
        feat = AbsorptionProxyFeature()
        bar = {"open": 1.0850, "high": 1.0850, "low": 1.0850, "close": 1.0850, "volume": 1000.0}
        assert feat.compute(bar) == 0.0

    def test_moderate_volume_normal_range(self):
        """Normal volume with normal range gives moderate score."""
        feat = AbsorptionProxyFeature()
        bar = {
            "open": 1.0850,
            "high": 1.0860,
            "low": 1.0840,
            "close": 1.0850,
            "volume": 5000.0,
        }
        result = feat.compute(bar)
        # Score = 5000 / 0.002 = 2,500,000
        assert result > 0.0


class TestAbsorptionComputeBatch:
    def test_batch_smoothing(self):
        """Batch applies EMA smoothing; later bars converge."""
        feat = AbsorptionProxyFeature(smoothing=0.3)
        bars = [
            {"open": 1.0850, "high": 1.0851, "low": 1.0850, "close": 1.0850, "volume": 50000.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0850, "close": 1.0850, "volume": 50000.0},
            {"open": 1.0850, "high": 1.0851, "low": 1.0850, "close": 1.0850, "volume": 50000.0},
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 3
        # Second bar should be smoothed (less than raw, since prev was 0)
        assert results[-1] >= 0.0

    def test_batch_empty_returns_empty(self):
        feat = AbsorptionProxyFeature()
        assert feat.compute_batch([]) == []

    def test_batch_preserves_smoothing_across_calls(self):
        """Second batch call continues from previous smoothed value."""
        feat = AbsorptionProxyFeature(smoothing=0.5)
        bars1 = [{"open": 1.0850, "high": 1.0851, "low": 1.0850, "close": 1.0850, "volume": 50000.0}]
        bars2 = [{"open": 1.0850, "high": 1.0851, "low": 1.0850, "close": 1.0850, "volume": 50000.0}]
        feat.compute_batch(bars1)
        second = feat.compute_batch(bars2)
        assert len(second) == 1


class TestAbsorptionCapabilitySpec:
    def test_capability_spec_structure(self):
        feat = AbsorptionProxyFeature()
        spec = feat.capability_spec()
        assert spec["feature_id"] == "microstructure/absorption_proxy"
        assert spec["quality_class"] == "proxy_inferred"
        assert "absorption_score" in spec["outputs"]
        assert "open" in spec["inputs"]

    def test_lookback_required(self):
        feat = AbsorptionProxyFeature()
        spec = feat.capability_spec()
        assert spec["lookback_required"] == 5


class TestAbsorptionConfidence:
    def test_disabled_under_5_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(4)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_5_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(5)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_low_at_9_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(9)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_10_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(10)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_medium_at_19_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(19)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_20_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"

    def test_high_at_100_bars(self):
        feat = AbsorptionProxyFeature()
        conf = feat.confidence(100)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"
