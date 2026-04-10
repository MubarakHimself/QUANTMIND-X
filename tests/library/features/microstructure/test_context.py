"""
QuantMindLib V1 — Packet 11C Self-Verification Tests

Tests for MicrostructureContext aggregation (FEATURE-029).

15 tests covering: init, compute_all, compute_batch, get_summary,
is_market_stressed, get_feature_count, partial features.
"""

from __future__ import annotations

import pytest

from src.library.features.microstructure.context import MicrostructureContext
from src.library.features.microstructure.volume_imbalance import (
    VolumeImbalanceFeature,
)
from src.library.features.microstructure.tick_activity import TickActivityFeature
from src.library.features.microstructure.depth import MultiLevelDepthFeature
from src.library.features.microstructure.absorption import AbsorptionProxyFeature
from src.library.features.microstructure.breakout_pressure import (
    BreakoutPressureProxyFeature,
)
from src.library.features.microstructure.liquidity_stress import (
    LiquidityStressProxyFeature,
)


@pytest.fixture
def bar_normal():
    """Normal OHLCV bar for testing."""
    return {
        "open": 1.1010,
        "high": 1.1015,
        "low": 1.1005,
        "close": 1.1012,
        "volume": 5000,
        "spread": 0.5,
    }


@pytest.fixture
def ctx_full():
    """Full MicrostructureContext with all 6 features."""
    return MicrostructureContext(
        volume_imbalance=VolumeImbalanceFeature(),
        tick_activity=TickActivityFeature(),
        depth=MultiLevelDepthFeature(),
        absorption=AbsorptionProxyFeature(),
        breakout_pressure=BreakoutPressureProxyFeature(),
        liquidity_stress=LiquidityStressProxyFeature(),
    )


class TestMicrostructureContextInit:
    """Init tests (MC1-MC4)."""

    def test_full_init_sets_all_features(self, ctx_full):
        """MC1: Full init sets all 6 feature attributes."""
        assert ctx_full.volume_imbalance is not None
        assert ctx_full.tick_activity is not None
        assert ctx_full.depth is not None
        assert ctx_full.absorption is not None
        assert ctx_full.breakout_pressure is not None
        assert ctx_full.liquidity_stress is not None

    def test_partial_init(self):
        """MC2: Partial init sets volume_imbalance, leaves rest as None."""
        ctx = MicrostructureContext(volume_imbalance=VolumeImbalanceFeature())
        assert ctx.volume_imbalance is not None
        assert ctx.tick_activity is None
        assert ctx.get_feature_count() == 1

    def test_empty_init(self):
        """MC3: Empty init get_feature_count() returns 0."""
        ctx = MicrostructureContext()
        assert ctx.get_feature_count() == 0

    def test_mixed_partial_init(self):
        """MC4: Mixed 3-feature init get_feature_count() returns 3."""
        ctx = MicrostructureContext(
            volume_imbalance=VolumeImbalanceFeature(),
            breakout_pressure=BreakoutPressureProxyFeature(),
            liquidity_stress=LiquidityStressProxyFeature(),
        )
        assert ctx.get_feature_count() == 3


class TestComputeAll:
    """compute_all tests (MC5-MC9)."""

    def test_returns_dict(self, ctx_full, bar_normal):
        """MC5: compute_all() returns a dict."""
        result = ctx_full.compute_all(bar_normal)
        assert isinstance(result, dict)

    def test_returns_non_empty_for_valid_bar(self, ctx_full, bar_normal):
        """MC6: compute_all() returns non-empty dict for normal bar."""
        result = ctx_full.compute_all(bar_normal)
        assert len(result) > 0

    def test_includes_expected_feature_keys(self, ctx_full, bar_normal):
        """MC7: compute_all() includes volume_imbalance and tick_activity keys."""
        result = ctx_full.compute_all(bar_normal)
        assert "volume_imbalance" in result
        assert "tick_activity" in result

    def test_handles_zero_volume_bar(self, ctx_full):
        """MC8: compute_all() handles zero-volume bar."""
        bar_zero = {
            "open": 1.1010,
            "high": 1.1010,
            "low": 1.1010,
            "close": 1.1010,
            "volume": 0,
        }
        result = ctx_full.compute_all(bar_zero)
        # Should not raise, returns a dict
        assert isinstance(result, dict)

    def test_empty_context_returns_empty_dict(self, bar_normal):
        """MC9: compute_all() on empty context returns empty dict."""
        ctx = MicrostructureContext()
        result = ctx.compute_all(bar_normal)
        assert isinstance(result, dict)
        assert len(result) == 0


class TestComputeBatch:
    """compute_batch tests (MC10-MC12)."""

    def test_returns_dict(self, ctx_full, bar_normal):
        """MC10: compute_batch() returns a dict."""
        result = ctx_full.compute_batch([bar_normal, bar_normal, bar_normal])
        assert isinstance(result, dict)

    def test_returns_feature_keyed_lists(self, ctx_full, bar_normal):
        """MC11: compute_batch() returns feature-keyed lists with correct length."""
        bars = [bar_normal, bar_normal, bar_normal]
        result = ctx_full.compute_batch(bars)
        assert "volume_imbalance" in result
        assert len(result["volume_imbalance"]) == len(bars)

    def test_empty_bars_returns_empty_dict(self, ctx_full):
        """MC12: compute_batch([]) returns empty dict."""
        result = ctx_full.compute_batch([])
        assert isinstance(result, dict)
        assert len(result) == 0


class TestGetSummary:
    """get_summary tests (MC13-MC14)."""

    def test_returns_dict_with_active_features(self, ctx_full, bar_normal):
        """MC13: get_summary() returns dict with active_features count."""
        ctx_full.compute_all(bar_normal)  # Populate history
        summary = ctx_full.get_summary()
        assert isinstance(summary, dict)
        assert "active_features" in summary
        assert summary["active_features"] == 6

    def test_includes_market_stressed_boolean(self, ctx_full, bar_normal):
        """MC14: get_summary() includes market_stressed boolean."""
        ctx_full.compute_all(bar_normal)
        summary = ctx_full.get_summary()
        assert "market_stressed" in summary
        assert isinstance(summary["market_stressed"], bool)


class TestIsMarketStressed:
    """is_market_stressed tests (MC15-MC17)."""

    def test_returns_bool_with_no_history(self):
        """MC15: is_market_stressed() returns bool (False with no history)."""
        ctx = MicrostructureContext(liquidity_stress=LiquidityStressProxyFeature())
        result = ctx.is_market_stressed()
        assert isinstance(result, bool)

    def test_returns_bool_with_high_stress_bar(self):
        """MC16: is_market_stressed() returns bool with high-stress bar."""
        ctx = MicrostructureContext(liquidity_stress=LiquidityStressProxyFeature())
        bar = {
            "high": 1.15,
            "low": 1.05,
            "close": 1.12,
            "volume": 40000,
            "spread": 3.0,
        }
        ctx.compute_all(bar)
        result = ctx.is_market_stressed()
        assert isinstance(result, bool)

    def test_returns_bool_with_high_breakout_pressure(self):
        """MC17: is_market_stressed() returns bool with high breakout pressure."""
        ctx = MicrostructureContext(breakout_pressure=BreakoutPressureProxyFeature())
        bar = {
            "high": 1.15,
            "low": 1.05,
            "close": 1.12,
            "volume": 30000,
            "spread": 2.0,
        }
        for _ in range(5):
            ctx.compute_all(bar)
        result = ctx.is_market_stressed()
        assert isinstance(result, bool)
