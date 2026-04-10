"""Tests for QuantMindLib V1 type enums."""

import pytest

from src.library.core.types.enums import (
    ActivationState,
    BotHealth,
    BotTier,
    EvaluationMode,
    FeatureConfidenceLevel,
    NewsState,
    OrderFlowSource,
    RegimeType,
    RegistryStatus,
    RiskMode,
    SignalDirection,
    TradeDirection,
)


class TestRegimeType:
    """Tests for RegimeType enum."""

    def test_has_correct_values(self):
        assert RegimeType.TREND_STABLE == "TREND_STABLE"
        assert RegimeType.RANGE_STABLE == "RANGE_STABLE"
        assert RegimeType.HIGH_CHAOS == "HIGH_CHAOS"
        assert RegimeType.BREAKOUT_PRIME == "BREAKOUT_PRIME"
        assert RegimeType.NEWS_EVENT == "NEWS_EVENT"
        assert RegimeType.UNCERTAIN == "UNCERTAIN"

    def test_str_value(self):
        assert str(RegimeType.TREND_STABLE) == "TREND_STABLE"

    def test_membership(self):
        assert RegimeType.TREND_STABLE in RegimeType


class TestNewsState:
    """Tests for NewsState enum."""

    def test_has_correct_values(self):
        assert NewsState.ACTIVE == "ACTIVE"
        assert NewsState.KILL_ZONE == "KILL_ZONE"
        assert NewsState.CLEAR == "CLEAR"


class TestTradeDirection:
    """Tests for TradeDirection enum."""

    def test_has_correct_values(self):
        assert TradeDirection.LONG == "LONG"
        assert TradeDirection.SHORT == "SHORT"


class TestSignalDirection:
    """Tests for SignalDirection enum."""

    def test_has_correct_values(self):
        assert SignalDirection.BULLISH == "BULLISH"
        assert SignalDirection.BEARISH == "BEARISH"
        assert SignalDirection.NEUTRAL == "NEUTRAL"


class TestOrderFlowSource:
    """Tests for OrderFlowSource enum."""

    def test_has_correct_values(self):
        assert OrderFlowSource.CTRADER_NATIVE == "CTRADER_NATIVE"
        assert OrderFlowSource.PROXY_INFERRED == "PROXY_INFERRED"
        assert OrderFlowSource.EXTERNAL == "EXTERNAL"
        assert OrderFlowSource.APPROXIMATED == "APPROXIMATED"
        assert OrderFlowSource.DISABLED == "DISABLED"

    def test_proxy_inferred_exists(self):
        """V1 Proxy Market Microstructure requires PROXY_INFERRED."""
        assert OrderFlowSource.PROXY_INFERRED == "PROXY_INFERRED"


class TestRiskMode:
    """Tests for RiskMode enum."""

    def test_has_correct_values(self):
        assert RiskMode.STANDARD == "STANDARD"
        assert RiskMode.CLAMPED == "CLAMPED"
        assert RiskMode.HALTED == "HALTED"


class TestActivationState:
    """Tests for ActivationState enum."""

    def test_has_correct_values(self):
        assert ActivationState.ACTIVE == "ACTIVE"
        assert ActivationState.PAUSED == "PAUSED"
        assert ActivationState.STOPPED == "STOPPED"


class TestBotHealth:
    """Tests for BotHealth enum."""

    def test_has_correct_values(self):
        assert BotHealth.HEALTHY == "HEALTHY"
        assert BotHealth.DEGRADED == "DEGRADED"
        assert BotHealth.FAILING == "FAILING"
        assert BotHealth.OFFLINE == "OFFLINE"


class TestEvaluationMode:
    """Tests for EvaluationMode enum."""

    def test_has_correct_values(self):
        assert EvaluationMode.VANILLA == "VANILLA"
        assert EvaluationMode.SPICED == "SPICED"
        assert EvaluationMode.VANILLA_FULL == "VANILLA_FULL"
        assert EvaluationMode.SPICED_FULL == "SPICED_FULL"
        assert EvaluationMode.MODE_B == "MODE_B"
        assert EvaluationMode.MODE_C == "MODE_C"


class TestRegistryStatus:
    """Tests for RegistryStatus enum."""

    def test_has_correct_values(self):
        assert RegistryStatus.ACTIVE == "ACTIVE"
        assert RegistryStatus.SUSPENDED == "SUSPENDED"
        assert RegistryStatus.ARCHIVED == "ARCHIVED"


class TestFeatureConfidenceLevel:
    """Tests for FeatureConfidenceLevel enum."""

    def test_has_correct_values(self):
        assert FeatureConfidenceLevel.HIGH == "HIGH"
        assert FeatureConfidenceLevel.MEDIUM == "MEDIUM"
        assert FeatureConfidenceLevel.LOW == "LOW"
        assert FeatureConfidenceLevel.DISABLED == "DISABLED"


class TestBotTier:
    """Tests for BotTier enum — labels only, no threshold logic."""

    def test_has_tiers(self):
        assert BotTier.TIER_1 == "TIER_1"
        assert BotTier.TIER_2 == "TIER_2"
        assert BotTier.TIER_3 == "TIER_3"

    def test_tiers_are_labels_only(self):
        """BotTier is a label — no threshold values embedded in the enum."""
        # Each member has a string value, nothing else
        for tier in BotTier:
            assert isinstance(tier.value, str)
            assert tier.value.startswith("TIER_")


class TestStrEnumBehavior:
    """Tests for StrEnum string value behavior."""

    def test_regime_str_value(self):
        assert str(RegimeType.TREND_STABLE) == "TREND_STABLE"

    def test_regime_membership(self):
        assert RegimeType.TREND_STABLE in RegimeType


class TestImportAll:
    """Test all 12 enums are importable from the expected path."""

    def test_import_all_enums(self):
        """Verify all 12 enums can be imported from the target module."""
        # This is the import test — if it fails, the module is broken
        from src.library.core.types.enums import (
            ActivationState,
            BotHealth,
            BotTier,
            EvaluationMode,
            FeatureConfidenceLevel,
            NewsState,
            OrderFlowSource,
            RegimeType,
            RegistryStatus,
            RiskMode,
            SignalDirection,
            TradeDirection,
        )

        # Count = 12
        assert len(
            [
                ActivationState,
                BotHealth,
                BotTier,
                EvaluationMode,
                FeatureConfidenceLevel,
                NewsState,
                OrderFlowSource,
                RegimeType,
                RegistryStatus,
                RiskMode,
                SignalDirection,
                TradeDirection,
            ]
        ) == 12
