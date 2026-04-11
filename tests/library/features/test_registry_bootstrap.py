"""
Tests for QuantMindLib V1 Feature Registry Bootstrap

Verifies:
1. get_default_registry() returns singleton (identity check)
2. Registry contains all 24 V1 feature modules
3. Each family has at least 1 registered feature
4. FeatureEvaluator with the registry can compute RSI
5. All proxy_inferred features have correct quality_class
"""
from __future__ import annotations

import sys
from typing import Any, Dict

# Ensure the worktree's src is on the path
sys.path.insert(0, "/home/mubarkahimself/Desktop/QUANTMINDX/.claude/worktrees/elegant-turing")

import pytest

from src.library.features._registry import get_default_registry
from src.library.features.registry import DependencyMissingError, FeatureRegistry
from src.library.features.indicators.rsi import RSIFeature
from src.library.runtime.feature_evaluator import FeatureEvaluator


class TestRegistryBootstrap:
    """Tests for the feature registry singleton bootstrap."""

    def test_get_default_registry_returns_singleton(self) -> None:
        """Calling get_default_registry() twice returns the same instance."""
        reg1 = get_default_registry()
        reg2 = get_default_registry()
        assert reg1 is reg2, "get_default_registry() must return the same singleton instance"

    def test_registry_contains_expected_features(self) -> None:
        """Registry contains all 24 V1 feature modules."""
        reg = get_default_registry()
        all_features = reg.list_all()

        # Print actual list for review
        print(f"\nRegistered features ({len(all_features)}):")
        for fid in sorted(all_features):
            print(f"  - {fid}")

        # Assert count >= 24 (the verified V1 feature count)
        assert len(all_features) >= 24, (
            f"Expected >= 24 V1 features, got {len(all_features)}. "
            f"Missing features detected."
        )

        # Verify key features are present (verified from scan phase)
        expected_ids = [
            # Indicators
            "indicators/rsi_14",
            "indicators/atr_14",
            "indicators/macd",
            "indicators/vwap",
            # Volume
            "volume/rvol",
            "volume/mfi_14",
            "volume/profile",
            # OrderFlow
            "orderflow/spread_behavior",
            "orderflow/dom_pressure",
            "orderflow/depth_thinning",
            # Session
            "session/detector",
            "session/blackout",
            # Transforms
            "transforms/normalize_value",
            "transforms/rolling_value_20",
            "transforms/resample_M1_to_M5",
            # Microstructure
            "microstructure/spread_state",
            "microstructure/tob_pressure",
            "microstructure/aggression",
            "microstructure/volume_imbalance",
            "microstructure/tick_activity",
            "microstructure/multi_level_depth",
            "microstructure/absorption_proxy",
            "microstructure/breakout_pressure_proxy",
            "microstructure/liquidity_stress_proxy",
        ]

        for fid in expected_ids:
            assert reg.get(fid) is not None, f"Feature not registered: {fid}"

    def test_registry_families(self) -> None:
        """Each feature family is registered under the correct family key."""
        reg = get_default_registry()
        families = reg.families()

        # Expected families (verified from scan phase)
        expected_families = {
            "indicators",
            "volume",
            "orderflow",
            "session",
            "transforms",
            "microstructure",
        }

        for family in expected_families:
            assert family in families, f"Family not registered: {family}"
            features = reg.list_by_family(family)
            assert len(features) >= 1, f"Family '{family}' has no features registered"
            print(f"  {family}: {len(features)} features")

    def test_feature_evaluator_uses_registered_features(self) -> None:
        """FeatureEvaluator with the initialized registry can compute RSI."""
        reg = get_default_registry()
        evaluator = FeatureEvaluator(registry=reg)

        # RSI only requires close_prices — trivially satisfiable with test data
        closes = [1.0850 + i * 0.0002 for i in range(50)]  # 50 bars of price data

        inputs: Dict[str, Any] = {
            "close_prices": closes,
        }

        result = evaluator.evaluate(
            bot_id="test_bot",
            feature_ids=["indicators/rsi_14"],
            inputs=inputs,
        )

        # RSI should be in the output FeatureVector
        assert "rsi_14" in result.features, (
            f"rsi_14 not in computed features. Got: {list(result.features.keys())}"
        )

        rsi_value = result.features["rsi_14"]
        # RSI must be a valid float in [0, 100]
        assert isinstance(rsi_value, float), f"rsi_14 should be float, got {type(rsi_value)}"
        assert 0.0 <= rsi_value <= 100.0, f"rsi_14 value out of range: {rsi_value}"

        # Confidence metadata should be present
        assert result.has_confidence("rsi_14"), "rsi_14 confidence metadata missing"
        conf = result.feature_confidence["rsi_14"]
        assert conf.source == "ctrader_native"

    def test_proxy_features_have_quality_tag(self) -> None:
        """
        All Proxy-named features should have quality_class='proxy_inferred'.

        Note: MicrostructureFeature.model_post_init() only copies feature_id
        from config — it does NOT copy quality_class. So all registered
        Proxy features currently carry the FeatureModule default
        ('native_supported') instead of their declared config value
        ('proxy_inferred'). This test identifies the discrepancy by
        checking which Proxy features have mismatched quality_class.
        """
        reg = get_default_registry()

        proxy_features: list[tuple[str, str]] = []

        for fid, feature in reg._features.items():
            class_name = feature.__class__.__name__
            if "Proxy" in class_name:
                proxy_features.append((fid, class_name))

        # Confirmed from scan phase: 4 Proxy-class features are registered
        assert len(proxy_features) >= 4, (
            f"Expected at least 4 Proxy-named features, got {len(proxy_features)}: "
            f"{[n for _, n in proxy_features]}"
        )

        mismatches: list[tuple[str, str, str, str]] = []  # (fid, class_name, actual, expected)
        for fid, class_name in proxy_features:
            feature = reg.get(fid)
            assert feature is not None
            if feature.quality_class != "proxy_inferred":
                # Get the expected value from the config property
                expected = "proxy_inferred"
                mismatches.append((fid, class_name, feature.quality_class, expected))

        # Report what we found
        if mismatches:
            print(f"\nQuality class mismatches ({len(mismatches)}):")
            for fid, class_name, actual, expected in mismatches:
                print(f"  {fid} ({class_name}): actual='{actual}', config declares='{expected}'")

        # Assert: all Proxy features should have quality_class='proxy_inferred'
        # This currently FAILS due to FeatureModule.model_post_init not copying
        # quality_class from config — a pre-existing library issue in the ABC.
        assert not mismatches, (
            f"{len(mismatches)} Proxy features have incorrect quality_class. "
            "FeatureModule.model_post_init() must copy quality_class from config."
        )

        print(f"\nVerified {len(proxy_features)} proxy features have quality_class='proxy_inferred'")


class TestDependencyMissingError:
    """Tests for DependencyMissingError raised by FeatureRegistry.validate_composition."""

    def test_raises_on_missing_feature(self):
        """validate_composition raises DependencyMissingError when a feature is not registered."""
        reg = FeatureRegistry()

        with pytest.raises(DependencyMissingError) as exc_info:
            reg.validate_composition(["indicators/rsi_14"])

        assert "indicators/rsi_14" in str(exc_info.value)
        assert "Required feature not registered" in str(exc_info.value)

    def test_raises_on_first_missing_feature(self):
        """validate_composition raises on the first missing feature (not all collected)."""
        reg = FeatureRegistry()

        with pytest.raises(DependencyMissingError) as exc_info:
            # First id missing triggers immediately
            reg.validate_composition(["nonexistent_feature", "also_missing"])

        assert "nonexistent_feature" in str(exc_info.value)

    def test_returns_empty_list_when_all_present(self):
        """validate_composition returns [] when all feature_ids are registered."""
        reg = FeatureRegistry()
        # Register the real RSI feature
        rsi = RSIFeature()
        reg.register(rsi)

        # Should return empty list, not raise
        result = reg.validate_composition(["indicators/rsi_14"])
        assert result == []

    def test_error_includes_available_features(self):
        """DependencyMissingError message lists available features."""
        reg = FeatureRegistry()
        # Register the real RSI feature
        rsi = RSIFeature()
        reg.register(rsi)

        with pytest.raises(DependencyMissingError) as exc_info:
            reg.validate_composition(["test/missing"])

        assert "indicators/rsi_14" in str(exc_info.value)
