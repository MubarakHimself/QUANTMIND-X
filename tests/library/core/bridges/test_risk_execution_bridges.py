"""Tests for QuantMindLib V1 -- RiskBridge and ExecutionBridge."""

import time

import pytest

from src.library.core.bridges.risk_execution_bridges import (
    ExecutionBridge,
    RiskBridge,
    _get_stop_ticks,
    _get_max_slippage_ticks,
)
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.risk_envelope import RiskEnvelope
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.types.enums import NewsState, RegimeType, RiskMode, TradeDirection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _intent(
    bot_id: str = "scalper_bot1",
    direction: TradeDirection = TradeDirection.LONG,
    confidence: int = 75,
    symbol: str = "EURUSD",
    timestamp_ms: int = None,
) -> TradeIntent:
    return TradeIntent(
        bot_id=bot_id,
        direction=direction,
        confidence=confidence,
        urgency="HIGH",
        reason="test reason",
        timestamp_ms=timestamp_ms or int(time.time() * 1000),
        symbol=symbol,
    )


def _fv(bot_id: str = "scalper_bot1") -> FeatureVector:
    return FeatureVector(
        bot_id=bot_id,
        features={"rsi_14": 65.0, "aggression_proxy": 0.7},
        feature_confidence={},
    )


def _market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    news_state: NewsState = NewsState.CLEAR,
    regime_confidence: float = 0.75,
    spread_state: str = "NORMAL",
    trend_strength: float = 0.6,
    volatility_regime: str = "NORMAL",
) -> MarketContext:
    return MarketContext(
        regime=regime,
        news_state=news_state,
        regime_confidence=regime_confidence,
        spread_state=spread_state,
        trend_strength=trend_strength,
        volatility_regime=volatility_regime,
        last_update_ms=int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# TestRiskBridgeAuthorize
# ---------------------------------------------------------------------------

class TestRiskBridgeAuthorize:
    def test_standard_authorization(self):
        """authorize returns STANDARD mode and positive position_size for clear conditions."""
        bridge = RiskBridge(max_position_size=100_000.0)
        intent = _intent(confidence=75)
        fv = _fv()
        ctx = _market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.75)

        result = bridge.authorize(intent, fv, ctx)

        assert result.risk_mode == RiskMode.STANDARD
        assert result.position_size > 0
        assert result.max_position_size == 100_000.0
        assert result.stop_ticks > 0
        assert result.max_slippage_ticks > 0
        assert result.last_check_ms > 0
        assert result.bot_id == intent.bot_id

    def test_halted_on_high_drawdown(self):
        """authorize returns HALTED when spread state is WIDE."""
        bridge = RiskBridge()
        intent = _intent()
        fv = _fv()
        ctx = _market_ctx(spread_state="WIDE")

        result = bridge.authorize(intent, fv, ctx)

        assert result.risk_mode == RiskMode.HALTED

    def test_halted_on_news_event_regime(self):
        """authorize returns HALTED when market regime is NEWS_EVENT."""
        bridge = RiskBridge()
        intent = _intent()
        fv = _fv()
        ctx = _market_ctx(regime=RegimeType.NEWS_EVENT)

        result = bridge.authorize(intent, fv, ctx)

        assert result.risk_mode == RiskMode.HALTED

    def test_archetype_based_sizing(self):
        """authorize uses archetype to determine stop_ticks."""
        bridge = RiskBridge()
        # Scalper archetype
        intent = _intent(bot_id="scalper_m1_bot", confidence=75)
        fv = _fv()
        ctx = _market_ctx()

        result = bridge.authorize(intent, fv, ctx)

        # Scalper stop_ticks should be 8
        assert result.stop_ticks == 8
        assert result.max_slippage_ticks == 4


# ---------------------------------------------------------------------------
# TestExecutionBridgeExecute
# ---------------------------------------------------------------------------

class TestExecutionBridgeExecute:
    def test_long_direction(self):
        """execute returns ExecutionDirective with LONG direction mapped from TradeIntent."""
        bridge = ExecutionBridge()
        intent = TradeIntent(
            bot_id="orb_v1",
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="breakout signal",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        envelope = RiskEnvelope(
            bot_id="orb_v1",
            max_position_size=100_000.0,
            max_daily_loss=500.0,
            current_drawdown=0.05,
            risk_mode=RiskMode.STANDARD,
            daily_loss_used=0.0,
            max_slippage_ticks=6,
            stop_ticks=12,
            position_size=50_000.0,
            open_risk=0.0,
            last_check_ms=int(time.time() * 1000),
        )

        result = bridge.execute(intent, envelope)

        assert result.bot_id == "orb_v1"
        assert result.direction == TradeDirection.LONG
        assert result.symbol == "EURUSD"
        assert result.quantity == 50_000.0
        assert result.risk_mode == RiskMode.STANDARD
        assert result.max_slippage_ticks == 6
        assert result.stop_ticks == 12

    def test_short_direction(self):
        """execute returns ExecutionDirective with SHORT direction mapped from TradeIntent."""
        bridge = ExecutionBridge()
        intent = TradeIntent(
            bot_id="orb_v1",
            direction=TradeDirection.SHORT,
            confidence=70,
            urgency="NORMAL",
            reason="mean reversion",
            timestamp_ms=int(time.time() * 1000),
            symbol="GBPUSD",
        )
        envelope = RiskEnvelope(
            bot_id="orb_v1",
            max_position_size=100_000.0,
            max_daily_loss=500.0,
            current_drawdown=0.10,
            risk_mode=RiskMode.CLAMPED,
            daily_loss_used=25.0,
            max_slippage_ticks=5,
            stop_ticks=10,
            position_size=25_000.0,
            open_risk=0.0,
            last_check_ms=int(time.time() * 1000),
        )

        result = bridge.execute(intent, envelope)

        assert result.direction == TradeDirection.SHORT
        assert result.symbol == "GBPUSD"
        assert result.risk_mode == RiskMode.CLAMPED

    def test_directive_fields(self):
        """execute populates all ExecutionDirective fields correctly."""
        bridge = ExecutionBridge()
        intent = _intent(bot_id="test_bot", confidence=80)
        envelope = RiskEnvelope(
            bot_id="test_bot",
            max_position_size=100_000.0,
            max_daily_loss=500.0,
            current_drawdown=0.03,
            risk_mode=RiskMode.STANDARD,
            daily_loss_used=10.0,
            max_slippage_ticks=6,
            stop_ticks=12,
            position_size=75_000.0,
            open_risk=0.0,
            last_check_ms=int(time.time() * 1000),
        )

        result = bridge.execute(intent, envelope)

        assert result.authorization == "RUNTIME_ORCHESTRATOR"
        assert result.timestamp_ms > 0
        assert result.quantity > 0

    def test_authorization_string(self):
        """execute sets authorization to RUNTIME_ORCHESTRATOR."""
        bridge = ExecutionBridge()
        intent = _intent()
        envelope = RiskEnvelope(
            bot_id=intent.bot_id,
            max_position_size=100_000.0,
            max_daily_loss=500.0,
            current_drawdown=0.0,
            risk_mode=RiskMode.STANDARD,
            daily_loss_used=0.0,
            max_slippage_ticks=6,
            stop_ticks=12,
            position_size=30_000.0,
            open_risk=0.0,
            last_check_ms=int(time.time() * 1000),
        )

        result = bridge.execute(intent, envelope)

        assert result.authorization == "RUNTIME_ORCHESTRATOR"


# ---------------------------------------------------------------------------
# TestArchetypeHelpers
# ---------------------------------------------------------------------------

class TestArchetypeHelpers:
    def test_stop_ticks_orb(self):
        assert _get_stop_ticks("opening_range_breakout") == 15
        assert _get_stop_ticks("london_orb") == 12
        assert _get_stop_ticks("ny_orb") == 12

    def test_stop_ticks_scalper(self):
        assert _get_stop_ticks("breakout_scalper") == 8
        assert _get_stop_ticks("scalper_m1") == 8
        assert _get_stop_ticks("pullback_scalper") == 10

    def test_stop_ticks_mean_reversion(self):
        assert _get_stop_ticks("mean_reversion") == 20

    def test_stop_ticks_default(self):
        assert _get_stop_ticks("unknown_archetype") == 10

    def test_max_slippage_ticks(self):
        assert _get_max_slippage_ticks("scalper_m1") == 4
        assert _get_max_slippage_ticks("london_orb") == 6
        assert _get_max_slippage_ticks("mean_reversion") == 10