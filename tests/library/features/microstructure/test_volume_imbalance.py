"""
Self-Verification Tests for Packet 11A: VolumeImbalanceFeature.
15 tests covering init, compute, compute_batch, capability_spec, confidence.
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure.volume_imbalance import VolumeImbalanceFeature
from src.library.core.domain.feature_vector import FeatureVector


class TestVolumeImbalanceInit:
    def test_default_smoothing(self):
        feat = VolumeImbalanceFeature()
        assert feat._smoothing == 0.3
        assert feat._prev_imbalance == 0.0

    def test_custom_smoothing(self):
        feat = VolumeImbalanceFeature(smoothing=0.5)
        assert feat._smoothing == 0.5

    def test_feature_id(self):
        feat = VolumeImbalanceFeature()
        assert feat.config.feature_id == "microstructure/volume_imbalance"

    def test_quality_class(self):
        feat = VolumeImbalanceFeature()
        assert feat.config.quality_class == "proxy_inferred"


class TestVolumeImbalanceCompute:
    def test_up_bar_positive(self):
        """Up-tick bar (close > open) returns positive imbalance."""
        feat = VolumeImbalanceFeature()
        bar = {"open": 1.0850, "high": 1.0860, "low": 1.0848, "close": 1.0858, "volume": 1000.0}
        result = feat.compute(bar)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_down_bar_negative(self):
        """Down-tick bar (close < open) returns negative imbalance."""
        feat = VolumeImbalanceFeature()
        bar = {"open": 1.0858, "high": 1.0860, "low": 1.0848, "close": 1.0850, "volume": 1000.0}
        result = feat.compute(bar)
        assert result == pytest.approx(-1.0, abs=1e-9)

    def test_flat_bar_near_zero(self):
        """Flat bar (close == open) returns near-zero imbalance."""
        feat = VolumeImbalanceFeature()
        bar = {"open": 1.0850, "high": 1.0850, "low": 1.0850, "close": 1.0850, "volume": 1000.0}
        result = feat.compute(bar)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_zero_volume_returns_zero(self):
        """Zero volume bar returns 0.0 imbalance."""
        feat = VolumeImbalanceFeature()
        bar = {"open": 1.0850, "close": 1.0858, "volume": 0.0}
        result = feat.compute(bar)
        assert result == 0.0

    def test_missing_volume_returns_zero(self):
        """Missing volume field returns 0.0 imbalance."""
        feat = VolumeImbalanceFeature()
        bar = {"open": 1.0850, "close": 1.0858}
        result = feat.compute(bar)
        assert result == 0.0


class TestVolumeImbalanceComputeBatch:
    def test_batch_smoothing(self):
        """Batch applies EMA smoothing; later bars converge toward raw values."""
        feat = VolumeImbalanceFeature(smoothing=0.3)
        bars = [
            {"open": 1.0850, "close": 1.0858, "volume": 1000.0},  # +1.0
            {"open": 1.0858, "close": 1.0866, "volume": 1000.0},  # +1.0
            {"open": 1.0866, "close": 1.0874, "volume": 1000.0},  # +1.0
        ]
        results = feat.compute_batch(bars)
        assert len(results) == 3
        # First bar: smoothing * 1.0 + (1-0.3) * 0.0 = 0.3
        assert results[0] == pytest.approx(0.3, abs=1e-6)
        # Later bars should trend toward 1.0
        assert results[-1] > results[0]

    def test_batch_empty_returns_empty(self):
        feat = VolumeImbalanceFeature()
        assert feat.compute_batch([]) == []

    def test_batch_preserves_smoothing_across_calls(self):
        """Second batch call continues from previous smoothed value."""
        feat = VolumeImbalanceFeature(smoothing=0.5)
        bars1 = [{"open": 1.0850, "close": 1.0858, "volume": 1000.0}]
        bars2 = [{"open": 1.0858, "close": 1.0866, "volume": 1000.0}]
        feat.compute_batch(bars1)
        second = feat.compute_batch(bars2)
        assert len(second) == 1


class TestVolumeImbalanceCapabilitySpec:
    def test_required_inputs(self):
        feat = VolumeImbalanceFeature()
        assert "open" in feat.required_inputs
        assert "close" in feat.required_inputs
        assert "volume" in feat.required_inputs

    def test_output_keys(self):
        feat = VolumeImbalanceFeature()
        assert feat.output_keys == {"imbalance"}

    def test_capability_spec_provides(self):
        feat = VolumeImbalanceFeature()
        spec = feat.capability_spec()
        assert "imbalance" in spec.provides


class TestVolumeImbalanceConfidence:
    def test_disabled_under_5_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(4)
        assert conf.quality == 0.0
        assert conf.feed_quality_tag == "DISABLED"

    def test_low_at_5_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(5)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_low_at_9_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(9)
        assert conf.quality == 0.3
        assert conf.feed_quality_tag == "LOW"

    def test_medium_at_10_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(10)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_medium_at_19_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(19)
        assert conf.quality == 0.6
        assert conf.feed_quality_tag == "MEDIUM"

    def test_high_at_20_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(20)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"

    def test_high_at_100_bars(self):
        feat = VolumeImbalanceFeature()
        conf = feat.confidence(100)
        assert conf.quality == 0.85
        assert conf.feed_quality_tag == "HIGH"
