"""Tests for QuantMindLib V1 -- IntentEmitter."""

import time
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.types.enums import (
    RegimeType,
    NewsState,
    TradeDirection,
)
from src.library.runtime.intent_emitter import IntentEmitter, _map_regime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORB_BOT_SPEC = BotSpec(
    id="test_orb_v1",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi", "indicators/atr"],
    confirmations=["spread_ok", "sentinel_allow"],
    execution_profile="orb_v1",
)

SCALPER_BOT_SPEC = BotSpec(
    id="test_scalper_v1",
    archetype="breakout_scalper",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/rsi"],
    confirmations=["spread_ok"],
    execution_profile="scalper_v1",
)

MEAN_REV_BOT_SPEC = BotSpec(
    id="test_mean_rev",
    archetype="mean_reversion",
    symbol_scope=["EURUSD"],
    sessions=["london"],
    features=["indicators/vwap"],
    confirmations=["spread_ok"],
    execution_profile="mean_rev_v1",
)


def _fresh_market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    news_state: NewsState = NewsState.CLEAR,
    regime_confidence: float = 0.75,
    spread_state: str = "NORMAL",
    stale: bool = False,
) -> MarketContext:
    """Create a fresh MarketContext with configurable fields."""
    return MarketContext(
        regime=regime,
        news_state=news_state,
        regime_confidence=regime_confidence,
        spread_state=spread_state,
        is_stale=stale,
        last_update_ms=int(time.time() * 1000),
    )


def _fv(**features: float) -> FeatureVector:
    """Create a FeatureVector with given features."""
    return FeatureVector(
        bot_id="test_orb_v1",
        timestamp=datetime.now(),
        features=dict(features),
        feature_confidence={},
    )


# ---------------------------------------------------------------------------
# Tests: _map_regime helper
# ---------------------------------------------------------------------------

class TestMapRegime:
    def test_breakout_prime_maps_to_breakout_trending(self):
        assert _map_regime(RegimeType.BREAKOUT_PRIME) == "BREAKOUT_TRENDING"

    def test_range_stable_maps_to_range_bound(self):
        assert _map_regime(RegimeType.RANGE_STABLE) == "RANGE_BOUND"

    def test_high_chaos_maps_to_breakout_trending_reverse(self):
        assert _map_regime(RegimeType.HIGH_CHAOS) == "BREAKOUT_TRENDING_REVERSE"

    def test_other_regimes_return_name(self):
        assert _map_regime(RegimeType.NEWS_EVENT) == "NEWS_EVENT"
        assert _map_regime(RegimeType.UNCERTAIN) == "UNCERTAIN"


# ---------------------------------------------------------------------------
# Tests: IntentEmitter initialization
# ---------------------------------------------------------------------------

class TestIntentEmitterInit:
    def test_initializes_with_defaults(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        assert emitter.bot_spec == ORB_BOT_SPEC
        assert emitter.regime_confidence_threshold == 0.5
        assert emitter.min_feature_confidence == 0.3

    def test_initializes_with_custom_thresholds(self):
        emitter = IntentEmitter(
            bot_spec=ORB_BOT_SPEC,
            regime_confidence_threshold=0.7,
            min_feature_confidence=0.5,
        )
        assert emitter.regime_confidence_threshold == 0.7
        assert emitter.min_feature_confidence == 0.5


# ---------------------------------------------------------------------------
# Tests: emit guards
# ---------------------------------------------------------------------------

class TestEmitGuards:
    def test_emit_returns_none_when_regime_confidence_below_threshold(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC, regime_confidence_threshold=0.8)
        market_ctx = _fresh_market_ctx(regime_confidence=0.6)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None

    def test_emit_returns_none_when_news_state_not_clear(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        for news_state in (NewsState.ACTIVE, NewsState.KILL_ZONE):
            market_ctx = _fresh_market_ctx(news_state=news_state)
            fv = _fv(rsi_14=45.0)
            result = emitter.emit(fv, market_ctx, "EURUSD")
            assert result is None, f"Should block on news_state={news_state}"

    def test_emit_returns_none_when_market_ctx_is_stale(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(stale=True)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None

    def test_emit_returns_none_when_market_ctx_is_not_fresh(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        # last_update_ms far in the past
        market_ctx = MarketContext(
            regime=RegimeType.BREAKOUT_PRIME,
            news_state=NewsState.CLEAR,
            regime_confidence=0.75,
            last_update_ms=int(time.time() * 1000) - 10_000,  # 10s ago > 5s threshold
        )
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: archetype signal logic
# ---------------------------------------------------------------------------

class TestORBSignal:
    def test_orb_signal_long_when_regime_breakout_prime(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.8)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 80

    def test_orb_signal_short_when_regime_high_chaos(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.HIGH_CHAOS, regime_confidence=0.7)
        fv = _fv(rsi_14=55.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.SHORT
        assert result.confidence == 70

    def test_orb_signal_from_rsi_extremes_in_range_bound(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.RANGE_STABLE)

        # RSI oversold -> LONG
        fv_buy = _fv(rsi_14=25.0)
        result_buy = emitter.emit(fv_buy, market_ctx, "EURUSD")
        assert result_buy is not None
        assert result_buy.direction == TradeDirection.LONG

        # RSI overbought -> SHORT
        fv_sell = _fv(rsi_14=75.0)
        result_sell = emitter.emit(fv_sell, market_ctx, "EURUSD")
        assert result_sell is not None
        assert result_sell.direction == TradeDirection.SHORT

    def test_orb_signal_returns_none_when_rsi_neutral_in_range(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.RANGE_STABLE)
        fv = _fv(rsi_14=50.0)  # Neutral
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None

    def test_orb_signal_returns_none_for_unknown_regime(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.NEWS_EVENT)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None


class TestScalperSignal:
    def test_scalper_signal_long_from_high_aggression(self):
        emitter = IntentEmitter(bot_spec=SCALPER_BOT_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = FeatureVector(bot_id="test_scalper", features={"aggression_proxy": 0.8})
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.LONG

    def test_scalper_signal_short_from_low_aggression(self):
        emitter = IntentEmitter(bot_spec=SCALPER_BOT_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = FeatureVector(bot_id="test_scalper", features={"aggression_proxy": 0.2})
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.SHORT

    def test_scalper_signal_returns_none_on_wide_spread(self):
        emitter = IntentEmitter(bot_spec=SCALPER_BOT_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="WIDE")
        fv = FeatureVector(bot_id="test_scalper", features={"aggression_proxy": 0.8})
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None

    def test_scalper_signal_returns_none_when_aggression_neutral(self):
        emitter = IntentEmitter(bot_spec=SCALPER_BOT_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = FeatureVector(bot_id="test_scalper", features={"aggression_proxy": 0.5})
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None


class TestMeanReversionSignal:
    def test_mean_reversion_signal_long_from_negative_vwap_distance(self):
        emitter = IntentEmitter(bot_spec=MEAN_REV_BOT_SPEC)
        market_ctx = _fresh_market_ctx()
        fv = _fv(vwap_distance=-0.010)  # 1% below VWAP
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.LONG

    def test_mean_reversion_signal_short_from_positive_vwap_distance(self):
        emitter = IntentEmitter(bot_spec=MEAN_REV_BOT_SPEC)
        market_ctx = _fresh_market_ctx()
        fv = _fv(vwap_distance=0.010)  # 1% above VWAP
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.direction == TradeDirection.SHORT

    def test_mean_reversion_returns_none_when_deviation_too_small(self):
        emitter = IntentEmitter(bot_spec=MEAN_REV_BOT_SPEC)
        market_ctx = _fresh_market_ctx()
        fv = _fv(vwap_distance=0.003)  # 0.3% -- below 0.5% threshold
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: urgency determination
# ---------------------------------------------------------------------------

class TestUrgencyDetermination:
    def test_urgency_immediate_for_breakout_with_confidence_above_0_8(self):
        # ORB signal in BREAKOUT_PRIME uses regime_confidence as signal confidence.
        # Use regime_confidence > 0.8 so signal confidence > 0.8 -> IMMEDIATE.
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.85)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.urgency == "IMMEDIATE"

    def test_urgency_high_for_signal_confidence_above_0_7(self):
        # Scalper signal uses aggression as signal confidence.
        # aggression > 0.7 -> HIGH urgency.
        emitter = IntentEmitter(bot_spec=SCALPER_BOT_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = FeatureVector(bot_id="test_scalper", features={"aggression_proxy": 0.75})
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert result.urgency == "HIGH"

    def test_urgency_normal_for_signal_confidence_above_0_5(self):
        # No archetype naturally produces a signal in the 0.5-0.7 range,
        # but we test NORMAL via direct method call with controlled confidence.
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        assert emitter._determine_urgency(RegimeType.RANGE_STABLE, 0.6) == "NORMAL"

    def test_urgency_low_for_signal_confidence_below_0_5(self):
        # Mean reversion with small deviation produces low confidence.
        # Test via direct method call since signal confidence < 0.5 rarely
        # produces a valid emit (it is filtered by min_feature_confidence).
        emitter = IntentEmitter(bot_spec=MEAN_REV_BOT_SPEC)
        # Mean reversion with 0.3% deviation: confidence = 0.3/2.0 = 0.15
        market_ctx = _fresh_market_ctx(regime_confidence=0.75)
        fv = _fv(vwap_distance=0.003)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        # 0.3% < 0.5% threshold -> no signal emitted
        assert result is None
        # Test urgency determination directly with controlled confidence
        urgency = emitter._determine_urgency(RegimeType.BREAKOUT_PRIME, 0.4)
        assert urgency == "LOW"


# ---------------------------------------------------------------------------
# Tests: reason string building
# ---------------------------------------------------------------------------

class TestReasonString:
    def test_reason_contains_archetype_and_regime(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.75)
        fv = _fv(rsi_14=45.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert "archetype=opening_range_breakout" in result.reason
        assert "BREAKOUT_PRIME" in result.reason
        assert "direction=LONG" in result.reason
        assert "confidence=0.75" in result.reason

    def test_reason_includes_rsi_when_present(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        market_ctx = _fresh_market_ctx(regime=RegimeType.RANGE_STABLE)
        fv = _fv(rsi_14=25.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is not None
        assert "rsi=25.0" in result.reason


# ---------------------------------------------------------------------------
# Tests: emit with valid signal produces correct TradeIntent
# ---------------------------------------------------------------------------

class TestEmitTradeIntent:
    def test_emit_produces_trade_intent_with_correct_fields(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        # regime_confidence > 0.8 so signal confidence > 0.8 -> IMMEDIATE urgency
        market_ctx = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.85)
        fv = _fv(rsi_14=40.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert result.bot_id == "test_orb_v1"
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 85  # regime_confidence=0.85 -> 85
        assert result.urgency == "IMMEDIATE"
        assert result.symbol == "EURUSD"
        assert isinstance(result.timestamp_ms, int)
        assert result.timestamp_ms > 0

    def test_emit_with_invalid_signal_returns_none(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC)
        # UNCERTAIN regime has no signal logic -> returns None
        market_ctx = _fresh_market_ctx(regime=RegimeType.UNCERTAIN)
        fv = _fv(rsi_14=50.0)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None

    def test_emit_with_low_confidence_below_min_threshold(self):
        emitter = IntentEmitter(bot_spec=ORB_BOT_SPEC, min_feature_confidence=0.6)
        market_ctx = _fresh_market_ctx(regime=RegimeType.RANGE_STABLE, regime_confidence=0.5)
        fv = _fv(rsi_14=25.0)  # Would emit but regime_confidence (0.5) is not > min (0.6)
        result = emitter.emit(fv, market_ctx, "EURUSD")
        assert result is None
