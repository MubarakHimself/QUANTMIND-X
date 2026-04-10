"""
Phase 7 Packet 7D: Integration Tests for QuantMindLib V1
tests/library/runtime/test_archetype_signals.py

Archetype-specific signal behavior integration tests.
Tests verify that the IntentEmitter produces correct TradeIntents
for ORB, Scalper, and Pullback archetypes under various market conditions.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.library.core.domain.bot_spec import BotSpec, BotRuntimeProfile
from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.domain.feature_vector import FeatureVector, FeatureConfidence
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.types.enums import (
    ActivationState,
    BotHealth,
    NewsState,
    RegimeType,
    TradeDirection,
)
from src.library.runtime.orchestrator import RuntimeOrchestrator
from src.library.runtime.intent_emitter import IntentEmitter
from src.library.runtime.feature_evaluator import FeatureEvaluator
from src.library.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ORB_SPEC = BotSpec(
    id="orb-signal-test",
    archetype="opening_range_breakout",
    symbol_scope=["EURUSD"],
    sessions=["LONDON"],
    features=["indicators/macd"],
    confirmations=[],
    execution_profile="standard",
)

SCALPER_SPEC = BotSpec(
    id="scalper-signal-test",
    archetype="breakout_scalper",
    symbol_scope=["EURUSD"],
    sessions=["LONDON"],
    features=["indicators/macd"],
    confirmations=[],
    execution_profile="scalper_v1",
)

PULLBACK_SPEC = BotSpec(
    id="pullback-signal-test",
    archetype="pullback_scalper",
    symbol_scope=["EURUSD"],
    sessions=["LONDON"],
    features=["indicators/macd"],
    confirmations=[],
    execution_profile="pullback_v1",
)

ORB_RUNTIME = BotRuntimeProfile(
    bot_id="orb-signal-test",
    activation_state=ActivationState.ACTIVE,
    deployment_target="production",
    health=BotHealth.HEALTHY,
    session_eligibility={"LONDON": True},
    dpr_ranking=1,
    dpr_score=0.85,
    report_ids=[],
)

SCALPER_RUNTIME = BotRuntimeProfile(
    bot_id="scalper_m1_pipeline",
    activation_state=ActivationState.ACTIVE,
    deployment_target="production",
    health=BotHealth.HEALTHY,
    session_eligibility={"LONDON": True},
    dpr_ranking=1,
    dpr_score=0.85,
    report_ids=[],
)

PULLBACK_RUNTIME = BotRuntimeProfile(
    bot_id="pullback-signal-test",
    activation_state=ActivationState.ACTIVE,
    deployment_target="production",
    health=BotHealth.HEALTHY,
    session_eligibility={"LONDON": True},
    dpr_ranking=1,
    dpr_score=0.85,
    report_ids=[],
)


def _fresh_market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    regime_confidence: float = 0.75,
    news_state: NewsState = NewsState.CLEAR,
    spread_state: str = "NORMAL",
) -> MarketContext:
    return MarketContext(
        regime=regime,
        news_state=news_state,
        regime_confidence=regime_confidence,
        spread_state=spread_state,
        trend_strength=0.65,
        volatility_regime="NORMAL",
        last_update_ms=int(time.time() * 1000),
    )


def _fv(**features: float) -> FeatureVector:
    """Create a FeatureVector with given features."""
    return FeatureVector(
        bot_id="signal-test",
        timestamp=datetime.now(),
        features=dict(features),
        feature_confidence={},
    )


def _orchestrator_with_mocked_emitter(spec: BotSpec) -> RuntimeOrchestrator:
    """Create an orchestrator with a mocked intent emitter."""
    orch = RuntimeOrchestrator()
    orch.register_bot(spec)
    return orch


# ---------------------------------------------------------------------------
# TestORBSignalIntegration
# ---------------------------------------------------------------------------

class TestORBSignalIntegration:
    """
    ORB (Opening Range Breakout) archetype signal integration tests.

    Signal logic (from IntentEmitter._orb_signal):
    - BREAKOUT_TRENDING (BREAKOUT_PRIME) -> LONG, confidence = regime_confidence
    - HIGH_CHAOS (BREAKOUT_TRENDING_REVERSE) -> SHORT, confidence = regime_confidence
    - RANGE_STABLE with RSI < 30 -> LONG
    - RANGE_STABLE with RSI > 70 -> SHORT
    - RANGE_STABLE with neutral RSI -> None
    - TREND_STABLE -> None (no signal path)
    - NEWS_EVENT -> None (blocked by news gate)
    """

    def test_orb_long_on_breakout(self):
        """
        BREAKOUT_PRIME regime with RSI below 50 -> LONG signal.
        """
        emitter = IntentEmitter(bot_spec=ORB_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.80,
        )
        fv = _fv(rsi_14=45.0)  # RSI in bullish territory

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert isinstance(result, TradeIntent)
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 80  # regime_confidence * 100
        assert result.symbol == "EURUSD"
        assert "BREAKOUT_PRIME" in result.reason
        assert "direction=LONG" in result.reason
        assert "archetype=opening_range_breakout" in result.reason

    def test_orb_short_on_breakdown(self):
        """
        HIGH_CHAOS regime -> SHORT signal (breakout trending reverse).
        """
        emitter = IntentEmitter(bot_spec=ORB_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.HIGH_CHAOS,
            regime_confidence=0.75,
        )
        fv = _fv(rsi_14=55.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert isinstance(result, TradeIntent)
        assert result.direction == TradeDirection.SHORT
        assert result.confidence == 75
        assert result.symbol == "EURUSD"

    def test_orb_neutral_in_range(self):
        """
        RANGE_STABLE regime with neutral RSI (50) -> no signal.
        """
        emitter = IntentEmitter(bot_spec=ORB_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.RANGE_STABLE,
            regime_confidence=0.70,
        )
        fv = _fv(rsi_14=50.0)  # Neutral

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is None

    def test_orb_weak_on_low_volume(self):
        """
        BREAKOUT_PRIME regime but with low regime_confidence -> reduced confidence.
        Confidence is derived from regime_confidence (0.55 -> confidence=55).
        """
        emitter = IntentEmitter(bot_spec=ORB_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.55,  # Lower confidence
        )
        fv = _fv(rsi_14=45.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 55  # regime_confidence * 100 = 55
        assert result.urgency == "NORMAL"  # confidence 0.55 -> NORMAL


# ---------------------------------------------------------------------------
# TestScalperSignalIntegration
# ---------------------------------------------------------------------------

class TestScalperSignalIntegration:
    """
    Scalper archetype signal integration tests.

    Signal logic (from IntentEmitter._scalper_signal):
    - WIDE spread -> None (blocked)
    - aggression_proxy > 0.65 -> LONG, confidence = aggression
    - aggression_proxy < 0.35 -> SHORT, confidence = 1 - aggression
    - Otherwise -> None
    """

    def test_scalper_long_on_micro_pullback(self):
        """
        High aggression_proxy (> 0.65) with NORMAL spread -> LONG.
        """
        emitter = IntentEmitter(bot_spec=SCALPER_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = _fv(aggression_proxy=0.80)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert isinstance(result, TradeIntent)
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 80  # aggression * 100
        assert result.symbol == "EURUSD"
        assert result.urgency == "HIGH"  # confidence 0.80 -> HIGH

    def test_scalper_short_on_micro_rally(self):
        """
        Low aggression_proxy (< 0.35) with NORMAL spread -> SHORT.
        """
        emitter = IntentEmitter(bot_spec=SCALPER_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="NORMAL")
        fv = _fv(aggression_proxy=0.20)  # 1 - 0.20 = 0.80 confidence

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert isinstance(result, TradeIntent)
        assert result.direction == TradeDirection.SHORT
        assert result.confidence == 80  # (1 - 0.20) * 100 = 80

    def test_scalper_neutral_in_wide_range(self):
        """
        High aggression but WIDE spread -> no signal (spread gate blocks).
        """
        emitter = IntentEmitter(bot_spec=SCALPER_SPEC)
        market_ctx = _fresh_market_ctx(spread_state="WIDE")
        fv = _fv(aggression_proxy=0.80)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is None

    def test_scalper_urgency_immediate(self):
        """
        Very high aggression_proxy (> 0.80) -> IMMEDIATE urgency.
        BREAKOUT_PRIME regime with confidence > 0.8 -> IMMEDIATE.
        """
        emitter = IntentEmitter(bot_spec=SCALPER_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.85,
            spread_state="NORMAL",
        )
        fv = _fv(aggression_proxy=0.85)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        # confidence = 0.85 > 0.8, regime = BREAKOUT_PRIME -> IMMEDIATE
        assert result.urgency == "IMMEDIATE"
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 85


# ---------------------------------------------------------------------------
# TestPullbackSignalIntegration
# ---------------------------------------------------------------------------

class TestPullbackSignalIntegration:
    """
    Pullback scalper archetype signal integration tests.

    Signal logic (from IntentEmitter._pullback_signal):
    - Requires BREAKOUT_TRENDING regime (BREAKOUT_PRIME)
    - RSI in [45, 55] -> LONG
    - Otherwise -> None
    - HIGH_CHAOS -> None
    """

    def test_pullback_long_on_trend_retrace(self):
        """
        BREAKOUT_PRIME regime with RSI in [45, 55] -> LONG.
        """
        emitter = IntentEmitter(bot_spec=PULLBACK_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.75,
        )
        # RSI at 50 -- pullback to mean within bullish trend
        fv = _fv(rsi_14=50.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        assert isinstance(result, TradeIntent)
        assert result.direction == TradeDirection.LONG
        assert result.confidence == 60  # pullback confidence = 0.6
        assert result.symbol == "EURUSD"

    def test_pullback_short_on_trend_retrace(self):
        """
        Pullback archetype only has LONG signal (trade with trend on RSI pullback).
        RSI=50 in BREAKOUT_PRIME -> LONG only, not SHORT.
        This test verifies that pullback does NOT produce SHORT.
        """
        emitter = IntentEmitter(bot_spec=PULLBACK_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.75,
        )
        fv = _fv(rsi_14=50.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is not None
        # Pullback only trades LONG (with the trend)
        assert result.direction == TradeDirection.LONG
        # No SHORT signal possible from pullback archetype

    def test_pullback_neutral_in_chaos(self):
        """
        HIGH_CHAOS regime with RSI in range -> no signal.
        """
        emitter = IntentEmitter(bot_spec=PULLBACK_SPEC)
        market_ctx = _fresh_market_ctx(
            regime=RegimeType.HIGH_CHAOS,
            regime_confidence=0.75,
        )
        fv = _fv(rsi_14=50.0)

        result = emitter.emit(fv, market_ctx, "EURUSD")

        assert result is None


# ---------------------------------------------------------------------------
# TestArchetypePipelineIntegration
# ---------------------------------------------------------------------------

class TestArchetypePipelineIntegration:
    """
    Full pipeline integration tests across archetype signal -> execution.
    Verifies that the complete orchestrator pipeline handles archetype signals correctly.
    """

    def test_orb_pipeline_end_to_end(self):
        """
        Full orchestrator pipeline with ORB archetype: tick -> directive.
        """
        orch = RuntimeOrchestrator()
        spec = BotSpec(
            id="orb-pipeline-1",
            archetype="opening_range_breakout",
            symbol_scope=["EURUSD"],
            sessions=["LONDON"],
            features=["indicators/macd"],
            confirmations=[],
            execution_profile="standard",
            runtime=ORB_RUNTIME,
        )
        orch.register_bot(spec)

        # Override emitter to return a valid LONG intent
        orch._intent_emitters["orb-pipeline-1"].emit = MagicMock(return_value=TradeIntent(
            bot_id="orb-pipeline-1",
            direction=TradeDirection.LONG,
            confidence=80,
            urgency="HIGH",
            reason="archetype=opening_range_breakout; regime=BREAKOUT_PRIME; confidence=0.80; direction=LONG",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        ))

        ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.80,
        )
        bar = {"close": [1.0860], "high": [1.0865], "low": [1.0855], "volume": [1000]}

        result = orch.on_tick("orb-pipeline-1", bar, ctx)

        assert result is not None
        assert isinstance(result, ExecutionDirective)
        assert result.direction == TradeDirection.LONG
        assert result.symbol == "EURUSD"
        assert result.quantity > 0

    def test_scalper_pipeline_end_to_end(self):
        """
        Full orchestrator pipeline with scalper archetype: tick -> directive.
        """
        orch = RuntimeOrchestrator()
        spec = BotSpec(
            id="scalper_m1_pipeline",
            archetype="breakout_scalper",
            symbol_scope=["EURUSD"],
            sessions=["LONDON"],
            features=["indicators/macd"],
            confirmations=[],
            execution_profile="scalper_v1",
            runtime=SCALPER_RUNTIME,
        )
        orch.register_bot(spec)

        orch._intent_emitters["scalper_m1_pipeline"].emit = MagicMock(return_value=TradeIntent(
            bot_id="scalper_m1_pipeline",
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="archetype=breakout_scalper",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        ))

        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
            spread_state="NORMAL",
        )
        bar = {"close": [1.0850], "high": [1.0853], "low": [1.0847], "volume": [1000]}

        result = orch.on_tick("scalper_m1_pipeline", bar, ctx)

        assert result is not None
        assert isinstance(result, ExecutionDirective)
        assert result.symbol == "EURUSD"
        # Scalper archetype -> 8 stop_ticks
        assert result.stop_ticks == 8
        assert result.max_slippage_ticks == 4

    def test_pullback_pipeline_blocks_on_trend_stable(self):
        """
        Pullback archetype in TREND_STABLE (not BREAKOUT_PRIME) -> no signal.
        """
        orch = RuntimeOrchestrator()
        spec = BotSpec(
            id="pullback-pipeline-1",
            archetype="pullback_scalper",
            symbol_scope=["EURUSD"],
            sessions=["LONDON"],
            features=["indicators/macd"],
            confirmations=[],
            execution_profile="pullback_v1",
            runtime=PULLBACK_RUNTIME,
        )
        orch.register_bot(spec)

        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
        )
        bar = {"close": [1.0850], "high": [1.0855], "low": [1.0845], "volume": [1000]}

        result = orch.on_tick("pullback-pipeline-1", bar, ctx)

        # Pullback only works in BREAKOUT_PRIME; TREND_STABLE -> no signal
        assert result is None

    def test_spread_gate_blocks_scalper(self):
        """
        Scalper with WIDE spread should be blocked by SafetyHooks.
        """
        orch = RuntimeOrchestrator()
        spec = BotSpec(
            id="scalper-spread-1",
            archetype="breakout_scalper",
            symbol_scope=["EURUSD"],
            sessions=["LONDON"],
            features=["indicators/macd"],
            confirmations=[],
            execution_profile="scalper_v1",
            runtime=SCALPER_RUNTIME,
        )
        orch.register_bot(spec)

        orch._intent_emitters["scalper-spread-1"].emit = MagicMock(return_value=TradeIntent(
            bot_id="scalper-spread-1",
            direction=TradeDirection.LONG,
            confidence=80,
            urgency="HIGH",
            reason="test",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        ))

        # WIDE spread
        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
            spread_state="WIDE",
        )
        bar = {"close": [1.0850], "high": [1.0853], "low": [1.0847], "volume": [1000]}

        result = orch.on_tick("scalper-spread-1", bar, ctx)

        # Blocked by kill switch: spread not ok
        assert result is None
