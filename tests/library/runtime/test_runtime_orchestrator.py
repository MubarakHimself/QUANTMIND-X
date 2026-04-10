"""Tests for QuantMindLib V1 -- RuntimeOrchestrator."""

import time
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.library.core.domain.bot_spec import BotSpec, BotRuntimeProfile
from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.domain.market_context import MarketContext, RegimeReport
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.types.enums import (
    ActivationState,
    BotHealth,
    NewsState,
    RegimeType,
    RiskMode,
    TradeDirection,
)
from src.library.runtime.orchestrator import RuntimeOrchestrator
from src.library.runtime.state_manager import BotStateManager
from src.library.runtime.feature_evaluator import FeatureEvaluator
from src.library.runtime.intent_emitter import IntentEmitter
from src.library.runtime.safety_hooks import SafetyHooks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _orb_bot_spec(
    bot_id: str = "test_orb_v1",
    runtime_profile: BotRuntimeProfile = None,
) -> BotSpec:
    spec = BotSpec(
        id=bot_id,
        archetype="opening_range_breakout",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/rsi", "indicators/atr"],
        confirmations=["spread_ok", "sentinel_allow"],
        execution_profile="orb_v1",
        runtime=runtime_profile,
    )
    return spec


def _scalper_bot_spec(
    bot_id: str = "test_scalper_v1",
    runtime_profile: BotRuntimeProfile = None,
) -> BotSpec:
    return BotSpec(
        id=bot_id,
        archetype="breakout_scalper",
        symbol_scope=["EURUSD"],
        sessions=["london"],
        features=["indicators/rsi"],
        confirmations=["spread_ok"],
        execution_profile="scalper_v1",
        runtime=runtime_profile,
    )


def _active_runtime_profile(bot_id: str) -> BotRuntimeProfile:
    return BotRuntimeProfile(
        bot_id=bot_id,
        activation_state=ActivationState.ACTIVE,
        deployment_target="production",
        health=BotHealth.HEALTHY,
        session_eligibility={"london": True},
        dpr_ranking=1,
        dpr_score=0.85,
        report_ids=[],
    )


def _fresh_market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    news_state: NewsState = NewsState.CLEAR,
    regime_confidence: float = 0.75,
    spread_state: str = "NORMAL",
    trend_strength: float = 0.6,
) -> MarketContext:
    return MarketContext(
        regime=regime,
        news_state=news_state,
        regime_confidence=regime_confidence,
        spread_state=spread_state,
        trend_strength=trend_strength,
        last_update_ms=int(time.time() * 1000),
    )


def _bar_data(
    close: float = 1.0850,
    high: float = 1.0855,
    low: float = 1.0845,
    volume: int = 1000,
) -> Dict[str, Any]:
    return {
        "close": [close],
        "high": [high],
        "low": [low],
        "volume": [volume],
    }


# ---------------------------------------------------------------------------
# TestOrchestratorInit
# ---------------------------------------------------------------------------

class TestOrchestratorInit:
    def test_default_init(self):
        """Orchestrator creates all components internally with defaults."""
        orch = RuntimeOrchestrator()
        assert orch._state_manager is not None
        assert orch._feature_evaluator is not None
        assert orch._safety_hooks is not None
        assert orch._risk_bridge is not None
        assert orch._execution_bridge is not None
        assert orch._bots == {}
        assert orch._intent_emitters == {}

    def test_custom_components(self):
        """Orchestrator accepts custom components via constructor."""
        sm = BotStateManager()
        fe = FeatureEvaluator()
        sh = SafetyHooks()
        orch = RuntimeOrchestrator(
            state_manager=sm,
            feature_evaluator=fe,
            safety_hooks=sh,
        )
        assert orch._state_manager is sm
        assert orch._feature_evaluator is fe
        assert orch._safety_hooks is sh


# ---------------------------------------------------------------------------
# TestBotRegistration
# ---------------------------------------------------------------------------

class TestBotRegistration:
    def test_register_success(self):
        """register_bot returns True and bot is stored."""
        orch = RuntimeOrchestrator()
        spec = _orb_bot_spec()
        result = orch.register_bot(spec)
        assert result is True
        assert "test_orb_v1" in orch._bots
        assert "test_orb_v1" in orch._intent_emitters

    def test_register_duplicate(self):
        """register_bot returns False for already-registered bot."""
        orch = RuntimeOrchestrator()
        spec = _orb_bot_spec()
        assert orch.register_bot(spec) is True
        assert orch.register_bot(spec) is False  # duplicate

    def test_unregister(self):
        """unregister_bot removes bot and returns True; unknown bot returns False."""
        orch = RuntimeOrchestrator()
        spec = _orb_bot_spec()
        orch.register_bot(spec)
        assert orch.unregister_bot("test_orb_v1") is True
        assert "test_orb_v1" not in orch._bots
        # Unknown bot
        assert orch.unregister_bot("unknown_bot") is False


# ---------------------------------------------------------------------------
# TestOnTick
# ---------------------------------------------------------------------------

class TestOnTick:
    def test_tick_with_full_pipeline(self):
        """on_tick returns ExecutionDirective when full pipeline succeeds."""
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("test_orb_v1")
        spec = _orb_bot_spec(runtime_profile=runtime)
        orch.register_bot(spec)

        # Mock intent emitter to return a real TradeIntent
        mock_intent = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="archetype=opening_range_breakout; regime=BREAKOUT_PRIME; confidence=0.75; direction=LONG",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=mock_intent)

        ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.75,
        )
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is not None
        assert isinstance(result, ExecutionDirective)
        assert result.bot_id == spec.id
        assert result.direction == TradeDirection.LONG
        assert result.symbol == "EURUSD"
        assert result.quantity > 0

    def test_tick_with_no_intent(self):
        """on_tick returns None when IntentEmitter emits no intent."""
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("test_orb_v1")
        spec = _orb_bot_spec(runtime_profile=runtime)
        orch.register_bot(spec)

        # Mock emit to return None (no signal)
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=None)

        ctx = _fresh_market_ctx(regime_confidence=0.75)
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)
        assert result is None

    def test_tick_with_safety_blocked(self):
        """on_tick returns None when SafetyHooks blocks the trade."""
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("test_orb_v1")
        spec = _orb_bot_spec(runtime_profile=runtime)
        orch.register_bot(spec)

        # Mock safety hooks to always block
        blocking_result = MagicMock()
        blocking_result.allowed = False
        blocking_result.reason = "Kill switch -- spread state not ok"
        orch._safety_hooks.check = MagicMock(return_value=blocking_result)

        mock_intent = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="test",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=mock_intent)

        ctx = _fresh_market_ctx(regime_confidence=0.75)
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)
        assert result is None

    def test_tick_with_unknown_bot(self):
        """on_tick returns None for unregistered bot."""
        orch = RuntimeOrchestrator()
        ctx = _fresh_market_ctx()
        bar = _bar_data()
        result = orch.on_tick("unknown_bot", bar, ctx)
        assert result is None

    def test_tick_with_zero_risk_position(self):
        """on_tick returns None when RiskBridge returns zero position_size."""
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("test_orb_v1")
        spec = _orb_bot_spec(runtime_profile=runtime)
        orch.register_bot(spec)

        # Mock RiskBridge to return HALTED (zero position)
        halted_envelope = MagicMock()
        halted_envelope.position_size = 0.0
        halted_envelope.risk_mode = RiskMode.HALTED
        orch._risk_bridge.authorize = MagicMock(return_value=halted_envelope)

        mock_intent = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="test",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=mock_intent)

        ctx = _fresh_market_ctx(regime_confidence=0.75)
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# TestGetState
# ---------------------------------------------------------------------------

class TestGetState:
    def test_returns_combined_state(self):
        """get_state returns FeatureVector and MarketContext for registered bot."""
        orch = RuntimeOrchestrator()
        spec = _orb_bot_spec()
        orch.register_bot(spec)

        ctx = _fresh_market_ctx()
        bar = _bar_data()
        orch.on_tick(spec.id, bar, ctx)

        fv, mc = orch.get_state(spec.id)
        assert fv is not None
        assert mc is not None
        assert mc.regime == ctx.regime

    def test_returns_none_for_unknown(self):
        """get_state returns (None, None) for unknown bot."""
        orch = RuntimeOrchestrator()
        fv, mc = orch.get_state("nonexistent_bot")
        assert fv is None
        assert mc is None


# ---------------------------------------------------------------------------
# TestHealthSummary
# ---------------------------------------------------------------------------

class TestHealthSummary:
    def test_empty_summary(self):
        """get_health_summary returns empty dict when no bots registered."""
        orch = RuntimeOrchestrator()
        summary = orch.get_health_summary()
        assert summary == {}

    def test_summary_with_registered_bots(self):
        """get_health_summary returns health string per bot."""
        orch = RuntimeOrchestrator()
        spec1 = _orb_bot_spec(
            "bot_a",
            runtime_profile=BotRuntimeProfile(
                bot_id="bot_a",
                activation_state=ActivationState.ACTIVE,
                deployment_target="production",
                health=BotHealth.HEALTHY,
                session_eligibility={"london": True},
                dpr_ranking=1,
                dpr_score=0.85,
                report_ids=[],
            ),
        )
        spec2 = _scalper_bot_spec(
            "bot_b",
            runtime_profile=BotRuntimeProfile(
                bot_id="bot_b",
                activation_state=ActivationState.CAUTIOUS,
                deployment_target="production",
                health=BotHealth.CAUTIOUS,
                session_eligibility={"london": True},
                dpr_ranking=2,
                dpr_score=0.70,
                report_ids=[],
            ),
        )
        orch.register_bot(spec1)
        orch.register_bot(spec2)

        summary = orch.get_health_summary()
        assert summary["bot_a"] == "HEALTHY"
        assert summary["bot_b"] == "CAUTIOUS"


# ---------------------------------------------------------------------------
# TestListRegisteredBots
# ---------------------------------------------------------------------------

class TestListRegisteredBots:
    def test_empty(self):
        """list_registered_bots returns empty list when no bots."""
        orch = RuntimeOrchestrator()
        assert orch.list_registered_bots() == []

    def test_with_bots(self):
        """list_registered_bots returns registered bot IDs."""
        orch = RuntimeOrchestrator()
        orch.register_bot(_orb_bot_spec("bot1"))
        orch.register_bot(_scalper_bot_spec("bot2"))
        orch.register_bot(_orb_bot_spec("bot3"))

        bots = orch.list_registered_bots()
        assert set(bots) == {"bot1", "bot2", "bot3"}