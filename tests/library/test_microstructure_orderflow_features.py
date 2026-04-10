"""
Self-Verification Tests for Packet 5E: Microstructure + OrderFlow A Feature Families.
Tests SpreadStateFeature, TopOfBookPressureFeature, AggressionProxyFeature
(SpreadBehaviorFeature, DOMPressureFeature, DepthThinningFeature).
"""
from __future__ import annotations

import pytest

from src.library.features.microstructure import SpreadStateFeature, TopOfBookPressureFeature, AggressionProxyFeature
from src.library.features.orderflow import SpreadBehaviorFeature, DOMPressureFeature, DepthThinningFeature
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence


# =============================================================================
# SpreadStateFeature Tests
# =============================================================================


class TestSpreadStateFeature:
    def test_spread_wide(self):
        """Spread > avg * 1.5 returns WIDE state (1.0)."""
        feat = SpreadStateFeature()
        history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # avg = 1.0
        inputs = {"spread": 2.0, "spread_history": history}
        fv = feat.compute(inputs)
        assert fv.features["spread_state"] == 1.0          # WIDE
        assert fv.features["spread_ratio"] == 2.0
        assert 0.0 <= fv.features["spread_tightness"] <= 1.0

    def test_spread_narrow(self):
        """Spread < avg * 0.5 returns NARROW state (0.5)."""
        feat = SpreadStateFeature()
        history = [2.0] * 10  # avg = 2.0
        inputs = {"spread": 0.5, "spread_history": history}
        fv = feat.compute(inputs)
        assert fv.features["spread_state"] == 0.5          # NARROW
        assert fv.features["spread_ratio"] == 0.25
        # tightness = 1.0 - min(0.25, 3.0) / 3.0 = 1.0 - 0.0833 = 0.9167
        assert fv.features["spread_tightness"] == pytest.approx(0.9167, abs=0.001)

    def test_spread_normal(self):
        """Spread within [0.5, 1.5] * avg returns NORMAL state (0.0)."""
        feat = SpreadStateFeature()
        history = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # avg = 1.0
        inputs = {"spread": 1.0, "spread_history": history}  # ratio = 1.0
        fv = feat.compute(inputs)
        assert fv.features["spread_state"] == 0.0          # NORMAL
        assert fv.features["spread_ratio"] == 1.0
        assert fv.features["spread_tightness"] == 1.0 - 1.0 / 3.0  # ~0.667

    def test_spread_insufficient_history(self):
        """Returns INSUFFICIENT_DATA quality when history < 5 bars."""
        feat = SpreadStateFeature()
        history = [1.0, 1.0, 1.0]  # only 3 bars
        inputs = {"spread": 2.0, "spread_history": history}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["spread_state"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"
        assert conf.quality == 0.0
        assert fv.features["spread_state"] == 0.5  # neutral

    def test_spread_no_history(self):
        """Without spread_history, returns INSUFFICIENT_DATA."""
        feat = SpreadStateFeature()
        inputs = {"spread": 1.5}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["spread_state"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"

    def test_spread_tightness_clamped(self):
        """tightness clamps ratio at 3.0 to avoid negative values."""
        feat = SpreadStateFeature()
        history = [1.0] * 10
        inputs = {"spread": 100.0, "spread_history": history}  # ratio = 100
        fv = feat.compute(inputs)
        tightness = fv.features["spread_tightness"]
        assert tightness == 0.0  # 1.0 - min(100, 3) / 3 = 0

    def test_spread_tightness_borderline(self):
        """Tightness at exact thresholds is correct."""
        feat = SpreadStateFeature()
        history = [1.0] * 10
        # ratio = 3.0 -> tightness = 1.0 - 3/3 = 0.0
        inputs = {"spread": 3.0, "spread_history": history}
        fv = feat.compute(inputs)
        assert fv.features["spread_tightness"] == 0.0

    def test_spread_boundary_wide(self):
        """Ratio exactly at 1.5 threshold."""
        feat = SpreadStateFeature()
        history = [1.0] * 10
        inputs = {"spread": 1.5, "spread_history": history}  # ratio = 1.5
        fv = feat.compute(inputs)
        assert fv.features["spread_state"] == 0.0  # NOT WIDE (>= not >)


# =============================================================================
# TopOfBookPressureFeature Tests
# =============================================================================


class TestTopOfBookPressureFeature:
    def test_tob_buy_heavy(self):
        """bid >> ask -> positive pressure, BUY_HEAVY."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 100.0, "ask_size": 20.0}
        fv = feat.compute(inputs)
        pressure = fv.features["tob_pressure"]
        # (100 - 20) / (100 + 20) = 80/120 = 0.6667
        assert 0.6 < pressure < 0.7
        assert fv.features["tob_imbalance"] == 1.0          # BUY_HEAVY

    def test_tob_sell_heavy(self):
        """ask >> bid -> negative pressure, SELL_HEAVY."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 20.0, "ask_size": 100.0}
        fv = feat.compute(inputs)
        pressure = fv.features["tob_pressure"]
        # (20 - 100) / 120 = -80/120 = -0.6667
        assert -0.7 < pressure < -0.6
        assert fv.features["tob_imbalance"] == -1.0         # SELL_HEAVY

    def test_tob_balanced(self):
        """bid ~ ask -> near-zero pressure, BALANCED."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 50.0, "ask_size": 50.0}
        fv = feat.compute(inputs)
        assert fv.features["tob_pressure"] == 0.0
        assert fv.features["tob_imbalance"] == 0.0          # BALANCED

    def test_tob_near_threshold_buy(self):
        """Pressure just above 0.3 threshold."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 65.0, "ask_size": 35.0}  # pressure = 30/100 = 0.3
        fv = feat.compute(inputs)
        # Exactly 0.3 is not > 0.3, so it's BALANCED
        assert fv.features["tob_imbalance"] == 0.0

    def test_tob_near_threshold_sell(self):
        """Pressure just below -0.3 threshold."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 35.0, "ask_size": 65.0}  # pressure = -30/100 = -0.3
        fv = feat.compute(inputs)
        assert fv.features["tob_imbalance"] == 0.0  # BALANCED

    def test_tob_insufficient_data_zero_sizes(self):
        """Zero bid or ask size returns INSUFFICIENT_DATA."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 0.0, "ask_size": 50.0}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["tob_pressure"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"
        assert conf.quality == 0.0

    def test_tob_insufficient_data_negative_sizes(self):
        """Negative sizes return INSUFFICIENT_DATA."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": -10.0, "ask_size": 50.0}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["tob_pressure"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"

    def test_tob_exact_halfway(self):
        """Pressure = 0.5 when bid=3*ask."""
        feat = TopOfBookPressureFeature()
        inputs = {"bid_size": 150.0, "ask_size": 50.0}
        fv = feat.compute(inputs)
        assert fv.features["tob_pressure"] == 0.5
        assert fv.features["tob_imbalance"] == 1.0


# =============================================================================
# AggressionProxyFeature Tests
# =============================================================================


class TestAggressionProxyFeature:
    def test_aggression_aggressive_buy(self):
        """Price up + volume up -> AGGRESSIVE_BUY."""
        feat = AggressionProxyFeature()
        inputs = {
            "close_prices": [1.0850, 1.0855],  # price up
            "volumes": [1000.0, 1500.0],        # volume up
        }
        fv = feat.compute(inputs)
        assert fv.features["aggression_proxy"] == 0.8
        assert fv.features["aggression_signal"] == 1.0   # AGGRESSIVE_BUY

    def test_aggression_aggressive_sell(self):
        """Price down + volume up -> AGGRESSIVE_SELL."""
        feat = AggressionProxyFeature()
        inputs = {
            "close_prices": [1.0855, 1.0850],  # price down
            "volumes": [1000.0, 1500.0],        # volume up
        }
        fv = feat.compute(inputs)
        assert fv.features["aggression_proxy"] == 0.2
        assert fv.features["aggression_signal"] == -1.0  # AGGRESSIVE_SELL

    def test_aggression_neutral(self):
        """Price direction without volume increase -> NEUTRAL."""
        feat = AggressionProxyFeature()
        inputs = {
            "close_prices": [1.0850, 1.0855],  # price up
            "volumes": [1500.0, 1000.0],        # volume down
        }
        fv = feat.compute(inputs)
        assert fv.features["aggression_proxy"] == 0.5
        assert fv.features["aggression_signal"] == 0.0   # NEUTRAL

    def test_aggression_neutral_price_down_no_volume(self):
        """Price down + volume down -> NEUTRAL."""
        feat = AggressionProxyFeature()
        inputs = {
            "close_prices": [1.0855, 1.0850],
            "volumes": [1500.0, 1000.0],
        }
        fv = feat.compute(inputs)
        assert fv.features["aggression_signal"] == 0.0

    def test_aggression_insufficient_data(self):
        """Fewer than 2 bars returns INSUFFICIENT_DATA."""
        feat = AggressionProxyFeature()
        inputs = {"close_prices": [1.0850], "volumes": [1000.0]}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["aggression_proxy"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"
        assert conf.quality == 0.0

    def test_aggression_quality_is_0_7(self):
        """Valid 2-bar input has quality=0.7."""
        feat = AggressionProxyFeature()
        inputs = {"close_prices": [1.0850, 1.0855], "volumes": [1000.0, 1500.0]}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["aggression_proxy"]
        assert conf.quality == 0.7


# =============================================================================
# SpreadBehaviorFeature Tests
# =============================================================================


class TestSpreadBehaviorFeature:
    def test_spread_efficient_regime(self):
        """High volume/spread ratio -> EFFICIENT regime."""
        feat = SpreadBehaviorFeature()
        inputs = {"spread": 1.0, "volume": 5000.0}  # efficiency = 5000
        fv = feat.compute(inputs)
        assert fv.features["spread_efficiency"] == 5000.0
        assert fv.features["spread_regime"] == 1.0   # EFFICIENT
        assert fv.features["spread_cost_impact"] == 1.0  # capped at 1.0

    def test_spread_expensive_regime(self):
        """Low volume/spread ratio -> EXPENSIVE regime (efficiency < 0.3)."""
        feat = SpreadBehaviorFeature()
        # efficiency = 100 / 500 = 0.2 < 0.3 -> EXPENSIVE
        inputs = {"spread": 500.0, "volume": 100.0}
        fv = feat.compute(inputs)
        assert fv.features["spread_regime"] == 0.0   # EXPENSIVE
        # cost_impact = min(1, 0.2/1000) = 0.0002
        assert fv.features["spread_cost_impact"] == pytest.approx(0.0002, abs=1e-6)

    def test_spread_moderate_regime(self):
        """Moderate volume/spread ratio -> MODERATE regime (0.3 <= efficiency < 1.0)."""
        feat = SpreadBehaviorFeature()
        # efficiency = 400 / 1000 = 0.4, in [0.3, 1.0) -> MODERATE
        inputs = {"spread": 1000.0, "volume": 400.0}
        fv = feat.compute(inputs)
        assert fv.features["spread_regime"] == 0.5   # MODERATE
        # cost_impact = min(1, 0.4/1000) = 0.0004
        assert fv.features["spread_cost_impact"] == pytest.approx(0.0004, abs=1e-6)

    def test_spread_zero_spread_no_data(self):
        """Zero or negative spread -> NO_DATA."""
        feat = SpreadBehaviorFeature()
        inputs = {"spread": 0.0, "volume": 1000.0}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["spread_efficiency"]
        assert conf.feed_quality_tag == "NO_DATA"
        assert conf.quality == 0.0

    def test_spread_list_inputs(self):
        """Spread and volume can be provided as lists."""
        feat = SpreadBehaviorFeature()
        inputs = {"spread": [1.0, 2.0], "volume": [500.0, 2000.0]}
        fv = feat.compute(inputs)
        # Should use last element: spread=2.0, volume=2000.0
        assert fv.features["spread_efficiency"] == 1000.0

    def test_spread_cost_impact_capped(self):
        """cost_impact caps at 1.0 for very high efficiency."""
        feat = SpreadBehaviorFeature()
        inputs = {"spread": 0.1, "volume": 100000.0}  # efficiency = 1_000_000
        fv = feat.compute(inputs)
        assert fv.features["spread_cost_impact"] == 1.0  # capped

    def test_spread_negative_volume_treated_as_zero(self):
        """Negative volume is clamped to 0."""
        feat = SpreadBehaviorFeature()
        inputs = {"spread": 2.0, "volume": -100.0}
        fv = feat.compute(inputs)
        assert fv.features["spread_efficiency"] == 0.0


# =============================================================================
# DOMPressureFeature Tests
# =============================================================================


class TestDOMPressureFeature:
    def test_dom_single_level_buy_heavy(self):
        """Single-level TOB with buy-side dominance."""
        feat = DOMPressureFeature()
        inputs = {"bid_sizes": [100.0], "ask_sizes": [30.0]}
        fv = feat.compute(inputs)
        # pressure = (100-30)/(100+30) = 70/130 = ~0.538
        assert fv.features["dom_pressure_1"] > 0.5
        assert fv.features["dom_pressure_1"] < 0.6

    def test_dom_multi_level(self):
        """3 levels of depth are all computed."""
        feat = DOMPressureFeature()
        inputs = {
            "bid_sizes": [100.0, 80.0, 60.0],
            "ask_sizes": [30.0, 40.0, 50.0],
        }
        fv = feat.compute(inputs)
        # Level 1: (100-30)/130 = 0.538
        assert 0.5 < fv.features["dom_pressure_1"] < 0.6
        # Level 2: (80-40)/120 = 0.333
        assert 0.3 < fv.features["dom_pressure_2"] < 0.4
        # Level 3: (60-50)/110 = 0.091
        assert 0.0 < fv.features["dom_pressure_3"] < 0.1

    def test_dom_depth_imbalance_long_thick(self):
        """total_bid > total_ask * 1.5 -> LONG_THICK."""
        feat = DOMPressureFeature()
        inputs = {
            "bid_sizes": [100.0, 80.0, 60.0],
            "ask_sizes": [20.0, 10.0, 10.0],  # total=40, 100*1.5=150 < 240
        }
        fv = feat.compute(inputs)
        assert fv.features["depth_imbalance"] == 1.0   # LONG_THICK

    def test_dom_depth_imbalance_short_thick(self):
        """total_ask > total_bid * 1.5 -> SHORT_THICK."""
        feat = DOMPressureFeature()
        inputs = {
            "bid_sizes": [10.0, 10.0, 10.0],   # total=30
            "ask_sizes": [100.0, 80.0, 60.0],  # total=240, 30*1.5=45 < 240
        }
        fv = feat.compute(inputs)
        assert fv.features["depth_imbalance"] == -1.0  # SHORT_THICK

    def test_dom_depth_imbalance_balanced(self):
        """total_bid and total_ask within 1.5x -> BALANCED."""
        feat = DOMPressureFeature()
        inputs = {
            "bid_sizes": [100.0],
            "ask_sizes": [100.0],
        }
        fv = feat.compute(inputs)
        assert fv.features["depth_imbalance"] == 0.0   # BALANCED

    def test_dom_total_depth(self):
        """total_bid_depth and total_ask_depth are correct sums."""
        feat = DOMPressureFeature()
        inputs = {
            "bid_sizes": [100.0, 50.0, 25.0],
            "ask_sizes": [80.0, 40.0, 20.0],
        }
        fv = feat.compute(inputs)
        assert fv.features["total_bid_depth"] == 175.0
        assert fv.features["total_ask_depth"] == 140.0

    def test_dom_insufficient_data(self):
        """Empty lists return INSUFFICIENT_DATA."""
        feat = DOMPressureFeature()
        inputs = {"bid_sizes": [], "ask_sizes": []}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["dom_pressure_1"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"


# =============================================================================
# DepthThinningFeature Tests
# =============================================================================


class TestDepthThinningFeature:
    def test_depth_thinning_thin(self):
        """Depth significantly below baseline -> THINNING signal."""
        feat = DepthThinningFeature()
        # current_depth = 5000, baseline = 20000 -> ratio = 0.25
        inputs = {"bid_sizes": [2000.0], "ask_sizes": [3000.0]}
        fv = feat.compute(inputs)
        assert fv.features["depth_thinning_ratio"] == 0.25
        assert fv.features["depth_thinning_signal"] == 1.0   # THINNING
        assert fv.features["book_vacuity"] == 0.75

    def test_depth_thinning_normal(self):
        """Depth at or above 40% of baseline -> NORMAL signal."""
        feat = DepthThinningFeature()
        # current_depth = 10000, baseline = 20000 -> ratio = 0.5
        inputs = {"bid_sizes": [5000.0], "ask_sizes": [5000.0]}
        fv = feat.compute(inputs)
        assert fv.features["depth_thinning_ratio"] == 0.5
        assert fv.features["depth_thinning_signal"] == 0.0   # NORMAL
        assert fv.features["book_vacuity"] == 0.5

    def test_depth_thinning_full_book(self):
        """Depth at baseline -> ratio=1.0, vacuity=0."""
        feat = DepthThinningFeature()
        # baseline per side = 10000
        inputs = {"bid_sizes": [10000.0], "ask_sizes": [10000.0]}
        fv = feat.compute(inputs)
        assert fv.features["depth_thinning_ratio"] == 1.0
        assert fv.features["depth_thinning_signal"] == 0.0
        assert fv.features["book_vacuity"] == 0.0

    def test_depth_thinning_extreme(self):
        """Near-zero depth -> ratio near 0, extreme vacuity."""
        feat = DepthThinningFeature()
        # current_depth = 0.002, baseline = 20000 -> ratio ~ 1e-7
        inputs = {"bid_sizes": [0.001], "ask_sizes": [0.001]}
        fv = feat.compute(inputs)
        assert fv.features["depth_thinning_ratio"] < 0.00001
        assert fv.features["book_vacuity"] == pytest.approx(0.9999999, abs=1e-7)

    def test_depth_thinning_empty_inputs(self):
        """Empty lists return INSUFFICIENT_DATA."""
        feat = DepthThinningFeature()
        inputs = {"bid_sizes": [], "ask_sizes": []}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["depth_thinning_ratio"]
        assert conf.feed_quality_tag == "INSUFFICIENT_DATA"

    def test_depth_thinning_zero_elements(self):
        """Lists with zero values are not sufficient data."""
        feat = DepthThinningFeature()
        inputs = {"bid_sizes": [0.0], "ask_sizes": [0.0]}
        fv = feat.compute(inputs)
        conf = fv.feature_confidence["depth_thinning_ratio"]
        # No positive values, quality should be low
        assert conf.quality < 0.75

    def test_depth_thinning_baseline_clamp(self):
        """Ratio is clamped to [0, 1] even above baseline."""
        feat = DepthThinningFeature()
        # current_depth = 50000, baseline = 20000 -> ratio = 2.5, clamped to 1.0
        inputs = {"bid_sizes": [25000.0], "ask_sizes": [25000.0]}
        fv = feat.compute(inputs)
        assert fv.features["depth_thinning_ratio"] == 1.0
        assert fv.features["book_vacuity"] == 0.0


# =============================================================================
# Integration: All features produce FeatureVector
# =============================================================================


class TestFeatureVectorContract:
    def test_all_features_return_featurevector(self):
        """Every feature returns a FeatureVector instance."""
        feat_classes = [
            (SpreadStateFeature(), {"spread": 1.0, "spread_history": [1.0] * 10}),
            (TopOfBookPressureFeature(), {"bid_size": 100.0, "ask_size": 50.0}),
            (AggressionProxyFeature(), {"close_prices": [1.0850, 1.0855], "volumes": [1000.0, 1500.0]}),
            (SpreadBehaviorFeature(), {"spread": 2.0, "volume": 1000.0}),
            (DOMPressureFeature(), {"bid_sizes": [100.0], "ask_sizes": [50.0]}),
            (DepthThinningFeature(), {"bid_sizes": [5000.0], "ask_sizes": [5000.0]}),
        ]
        for feat, inputs in feat_classes:
            fv = feat.compute(inputs)
            assert isinstance(fv, FeatureVector)

    def test_all_features_have_confidence_metadata(self):
        """Every feature populates feature_confidence for all output keys."""
        feat_classes = [
            (SpreadStateFeature(), {"spread": 1.0, "spread_history": [1.0] * 10}),
            (TopOfBookPressureFeature(), {"bid_size": 100.0, "ask_size": 50.0}),
            (AggressionProxyFeature(), {"close_prices": [1.0850, 1.0855], "volumes": [1000.0, 1500.0]}),
            (SpreadBehaviorFeature(), {"spread": 2.0, "volume": 1000.0}),
            (DOMPressureFeature(), {"bid_sizes": [100.0], "ask_sizes": [50.0]}),
            (DepthThinningFeature(), {"bid_sizes": [5000.0], "ask_sizes": [5000.0]}),
        ]
        for feat, inputs in feat_classes:
            fv = feat.compute(inputs)
            for key in feat.output_keys:
                assert key in fv.features, f"Missing feature key: {key}"
                assert key in fv.feature_confidence, f"Missing confidence for: {key}"
                conf = fv.feature_confidence[key]
                assert isinstance(conf, FeatureConfidence)
                assert 0.0 <= conf.quality <= 1.0

    def test_all_features_output_keys_match(self):
        """output_keys property matches keys returned by compute()."""
        feat_classes = [
            (SpreadStateFeature(), {"spread": 1.0, "spread_history": [1.0] * 10}),
            (TopOfBookPressureFeature(), {"bid_size": 100.0, "ask_size": 50.0}),
            (AggressionProxyFeature(), {"close_prices": [1.0850, 1.0855], "volumes": [1000.0, 1500.0]}),
            (SpreadBehaviorFeature(), {"spread": 2.0, "volume": 1000.0}),
            (DOMPressureFeature(), {"bid_sizes": [100.0], "ask_sizes": [50.0]}),
            (DepthThinningFeature(), {"bid_sizes": [5000.0], "ask_sizes": [5000.0]}),
        ]
        for feat, inputs in feat_classes:
            fv = feat.compute(inputs)
            assert fv.features.keys() == feat.output_keys, \
                f"{feat.feature_id}: expected {feat.output_keys}, got {fv.features.keys()}"

    def test_all_features_required_inputs_defined(self):
        """Every feature declares non-empty required_inputs."""
        feats = [
            SpreadStateFeature(),
            TopOfBookPressureFeature(),
            AggressionProxyFeature(),
            SpreadBehaviorFeature(),
            DOMPressureFeature(),
            DepthThinningFeature(),
        ]
        for feat in feats:
            assert len(feat.required_inputs) > 0
            assert len(feat.output_keys) > 0

    def test_all_features_config_defined(self):
        """Every feature has a valid config with feature_id."""
        feats = [
            SpreadStateFeature(),
            TopOfBookPressureFeature(),
            AggressionProxyFeature(),
            SpreadBehaviorFeature(),
            DOMPressureFeature(),
            DepthThinningFeature(),
        ]
        for feat in feats:
            config = feat.config
            assert config.feature_id != ""
            assert config.feature_id.count("/") >= 1  # has family/name format
