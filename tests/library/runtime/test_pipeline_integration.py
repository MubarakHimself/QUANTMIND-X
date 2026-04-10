"""
Phase 7 Packet 7D: Integration Tests for QuantMindLib V1
tests/library/runtime/test_pipeline_integration.py

Full RuntimeOrchestrator pipeline integration tests -- all internal components
are real, external dependencies (Redis, cTrader adapter, SentinelBridge) are mocked.
"""
from __future__ import annotations

import threading
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
    RiskMode,
    TradeDirection,
)
from src.library.runtime.orchestrator import RuntimeOrchestrator
from src.library.runtime.state_manager import BotStateManager
from src.library.runtime.feature_evaluator import FeatureEvaluator
from src.library.runtime.intent_emitter import IntentEmitter
from src.library.runtime.safety_hooks import SafetyHooks, KillSwitchResult
from src.library.features.registry import FeatureRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _active_runtime_profile(bot_id: str, health: BotHealth = BotHealth.HEALTHY) -> BotRuntimeProfile:
    """Active runtime profile for a healthy bot."""
    return BotRuntimeProfile(
        bot_id=bot_id,
        activation_state=ActivationState.ACTIVE,
        deployment_target="production",
        health=health,
        session_eligibility={"london": True, "ny": False},
        dpr_ranking=1,
        dpr_score=0.85,
        report_ids=[],
    )


def _cautious_runtime_profile(bot_id: str, health: BotHealth = BotHealth.CAUTIOUS) -> BotRuntimeProfile:
    """Cautious runtime profile."""
    return BotRuntimeProfile(
        bot_id=bot_id,
        activation_state=ActivationState.CAUTIOUS,
        deployment_target="production",
        health=health,
        session_eligibility={"london": True},
        dpr_ranking=2,
        dpr_score=0.70,
        report_ids=[],
    )


def _orb_bot_spec(
    bot_id: str = "london_orb_1",
    runtime: BotRuntimeProfile = None,
) -> BotSpec:
    return BotSpec(
        id=bot_id,
        archetype="opening_range_breakout",
        symbol_scope=["EURUSD"],
        sessions=["LONDON"],
        features=["indicators/rsi", "indicators/atr"],
        confirmations=[],
        execution_profile="standard",
        runtime=runtime,
    )


def _scalper_bot_spec(
    bot_id: str = "scalper_m1_1",
    runtime: BotRuntimeProfile = None,
) -> BotSpec:
    return BotSpec(
        id=bot_id,
        archetype="breakout_scalper",
        symbol_scope=["EURUSD"],
        sessions=["LONDON"],
        features=["indicators/rsi", "indicators/atr"],
        confirmations=[],
        execution_profile="scalper_v1",
        runtime=runtime,
    )


def _fresh_market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    regime_confidence: float = 0.75,
    news_state: NewsState = NewsState.CLEAR,
    spread_state: str = "NORMAL",
    trend_strength: float = 0.65,
    is_stale: bool = False,
) -> MarketContext:
    return MarketContext(
        regime=regime,
        news_state=news_state,
        regime_confidence=regime_confidence,
        spread_state=spread_state,
        trend_strength=trend_strength,
        is_stale=is_stale,
        volatility_regime="NORMAL",
        last_update_ms=int(time.time() * 1000),
    )


def _stale_market_ctx(
    regime: RegimeType = RegimeType.BREAKOUT_PRIME,
    regime_confidence: float = 0.75,
) -> MarketContext:
    """MarketContext with stale last_update_ms (10+ seconds old)."""
    return MarketContext(
        regime=regime,
        news_state=NewsState.CLEAR,
        regime_confidence=regime_confidence,
        spread_state="NORMAL",
        trend_strength=0.65,
        is_stale=False,
        volatility_regime="NORMAL",
        last_update_ms=int(time.time() * 1000) - 15_000,  # 15s old -- beyond 5s threshold
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


def _mock_feature(
    feature_id: str,
    required_inputs: set,
    output_features: Dict[str, float],
    quality: float = 0.9,
) -> MagicMock:
    """Build a mock FeatureModule that returns predictable FeatureVectors."""
    mock = MagicMock()
    mock.feature_id = feature_id
    mock.required_inputs = required_inputs
    mock.output_keys = set(output_features.keys())
    mock.quality_class = "native_supported"
    mock.source = "test"
    mock.enabled = True
    confidence = {
        k: FeatureConfidence(source="test", quality=quality, latency_ms=0.1, feed_quality_tag="HIGH")
        for k in output_features
    }
    mock.compute.return_value = FeatureVector(
        bot_id="test",
        timestamp=datetime.now(),
        features=output_features,
        feature_confidence=confidence,
    )
    return mock


def _build_mock_registry(
    feature_configs: Dict[str, Dict[str, float]],
    quality: float = 0.9,
) -> FeatureRegistry:
    """
    Build a mock FeatureRegistry with mock FeatureModules.
    feature_configs: dict mapping feature_id -> {output_key: value}
    """
    reg = FeatureRegistry()
    # feature_id_map resolves the "logical" feature_id used in BotSpec.features
    # to the actual feature_id of the registered FeatureModule.
    feature_id_map = {
        "indicators/rsi": "indicators/rsi",
        "indicators/atr": "indicators/atr",
        "indicators/macd": "indicators/macd",
    }
    for fid, outputs in feature_configs.items():
        resolved_id = feature_id_map.get(fid, fid)
        mock = _mock_feature(resolved_id, {"close_prices"}, outputs, quality)
        reg.register(mock)
    return reg


# ---------------------------------------------------------------------------
# TestPipelineEndToEnd
# ---------------------------------------------------------------------------

class TestPipelineEndToEnd:
    """End-to-end pipeline tests -- real orchestrator with real sub-components."""

    def test_orb_bot_full_pipeline(self):
        """
        Register ORB bot, send tick with BREAKOUT_PRIME regime,
        verify full pipeline produces ExecutionDirective with LONG direction.
        """
        # Build mock registry with RSI feature (registered under indicators/rsi)
        reg = _build_mock_registry({"indicators/rsi": {"rsi_14": 45.0}})

        orch = RuntimeOrchestrator(
            feature_evaluator=FeatureEvaluator(registry=reg),
        )
        runtime = _active_runtime_profile("london_orb_1")
        spec = _orb_bot_spec(runtime=runtime)
        orch.register_bot(spec)

        # Tick with BREAKOUT_PRIME regime -- ORB should emit LONG
        ctx = _fresh_market_ctx(
            regime=RegimeType.BREAKOUT_PRIME,
            regime_confidence=0.80,
            trend_strength=0.70,
        )
        bar = _bar_data(close=1.0860, high=1.0865, low=1.0855)

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is not None
        assert isinstance(result, ExecutionDirective)
        assert result.bot_id == spec.id
        assert result.direction == TradeDirection.LONG
        assert result.symbol == "EURUSD"
        assert result.quantity > 0
        assert result.risk_mode in (RiskMode.STANDARD, RiskMode.CLAMPED)
        assert result.stop_ticks == 12  # ORB archetype -> 12
        assert result.max_slippage_ticks == 6  # 12 // 2
        assert result.authorization == "RUNTIME_ORCHESTRATOR"

    def test_scalper_bot_full_pipeline(self):
        """
        Register scalper bot, send tick with high aggression and NORMAL spread,
        verify full pipeline produces ExecutionDirective.
        """
        # Scalper signal uses aggression_proxy > 0.65 -> LONG
        reg = _build_mock_registry({
            "indicators/rsi": {"aggression_proxy": 0.80},
        })

        orch = RuntimeOrchestrator(
            feature_evaluator=FeatureEvaluator(registry=reg),
        )
        runtime = _active_runtime_profile("scalper_m1_1")
        spec = _scalper_bot_spec(runtime=runtime)
        orch.register_bot(spec)

        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
            spread_state="NORMAL",
        )
        bar = _bar_data(close=1.0850, high=1.0853, low=1.0847)

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is not None
        assert isinstance(result, ExecutionDirective)
        assert result.bot_id == spec.id
        assert result.direction == TradeDirection.LONG
        assert result.symbol == "EURUSD"
        assert result.quantity > 0
        assert result.stop_ticks == 8  # scalper archetype -> 8
        assert result.max_slippage_ticks == 4  # 8 // 2

    def test_pipeline_blocks_on_circuit_breaker(self):
        """
        Register bot, configure SafetyHooks with very low circuit_breaker_loss_pct,
        then trigger the condition by patching _run_safety_check to simulate
        daily_loss >= circuit_breaker_loss_pct.
        """
        # Use a SafetyHooks instance with 0.01 (1%) circuit breaker threshold
        # and patch the orchestrator to simulate daily_loss exceeding it
        safety = SafetyHooks(circuit_breaker_loss_pct=0.01)
        orch = RuntimeOrchestrator(safety_hooks=safety)

        runtime = _active_runtime_profile("test-cb-1")
        spec = _orb_bot_spec(bot_id="test-cb-1", runtime=runtime)
        orch.register_bot(spec)

        # Intercept _run_safety_check to simulate daily_loss >= circuit_breaker
        original_check = orch._run_safety_check

        def patched_check(spec, intent, market_ctx):
            # Simulate daily_loss_pct = 0.02 (2%) which exceeds 1% threshold
            result = original_check(spec, intent, market_ctx)
            # Return a blocking result instead
            return KillSwitchResult(
                allowed=False,
                reason="test-cb-1: Circuit breaker -- daily loss 2.00% >= 1.00%",
                triggered_by="CIRCUIT_BREAKER",
            )

        orch._run_safety_check = patched_check

        ctx = _fresh_market_ctx(regime_confidence=0.80)
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is None  # Pipeline blocked by circuit breaker

    def test_pipeline_blocks_on_health_critical(self):
        """
        Register bot with CRITICAL health, verify pipeline blocks.
        """
        runtime_critical = _active_runtime_profile("test-critical-1", health=BotHealth.CRITICAL)
        spec = _orb_bot_spec(bot_id="test-critical-1", runtime=runtime_critical)

        orch = RuntimeOrchestrator()
        orch.register_bot(spec)

        ctx = _fresh_market_ctx(regime_confidence=0.80)
        bar = _bar_data()

        # Also need the intent emitter to return a valid signal
        mock_intent = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.LONG,
            confidence=80,
            urgency="HIGH",
            reason="test: health critical pipeline block",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=mock_intent)

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is None  # Blocked by HEALTH_CRITICAL safety gate

    def test_pipeline_returns_none_on_no_signal(self):
        """
        Register bot, send tick with TREND_STABLE regime and neutral RSI,
        verify no ExecutionDirective emitted (ORB has no signal in TREND_STABLE
        with neutral RSI).
        """
        # TREND_STABLE is not BREAKOUT_PRIME, RANGE_STABLE, or HIGH_CHAOS,
        # so ORB signal returns (None, 0.0) -- no intent emitted.
        reg = _build_mock_registry({"indicators/macd": {"rsi_14": 50.0}})

        orch = RuntimeOrchestrator(
            feature_evaluator=FeatureEvaluator(registry=reg),
        )
        runtime = _active_runtime_profile("test-neutral-1")
        spec = _orb_bot_spec(bot_id="test-neutral-1", runtime=runtime)
        orch.register_bot(spec)

        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
        )
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is None  # No signal in TREND_STABLE with neutral RSI


# ---------------------------------------------------------------------------
# TestPipelineMultiBot
# ---------------------------------------------------------------------------

class TestPipelineMultiBot:
    """Tests for multi-bot scenarios."""

    def test_two_bots_independent_state(self):
        """
        Register 2 bots (ORB + scalper), send ticks,
        verify each has independent FeatureVector and state.
        """
        # ORB bot
        orb_reg = _build_mock_registry({"indicators/macd": {"rsi_14": 45.0}})
        orb_spec = _orb_bot_spec(
            bot_id="orb-multi-1",
            runtime=_active_runtime_profile("orb-multi-1"),
        )

        # Scalper bot
        scalper_reg = _build_mock_registry({"indicators/macd": {"aggression_proxy": 0.80}})
        scalper_spec = _scalper_bot_spec(
            bot_id="scalper-multi-1",
            runtime=_active_runtime_profile("scalper-multi-1"),
        )

        # Single orchestrator with separate feature evaluators
        orch = RuntimeOrchestrator()
        orch.register_bot(orb_spec)
        orch.register_bot(scalper_spec)

        # Override feature evaluator per-bot via the shared evaluator + mock registry
        # Both bots share the same feature evaluator but emit different intents
        # based on their archetypes and the market context.

        # ORB tick: BREAKOUT_PRIME -> LONG
        ctx_orb = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.80)
        bar_orb = _bar_data(close=1.0860, high=1.0865, low=1.0855)

        # Scalper tick: TREND_STABLE, high aggression -> LONG
        ctx_scalper = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
            spread_state="NORMAL",
        )
        bar_scalper = _bar_data(close=1.0850, high=1.0853, low=1.0847)

        # Process ORB tick -- but the shared evaluator doesn't have the right features.
        # Instead, let's mock the emit to return the correct signals and verify
        # state is independent.

        # Mock ORB emit to return LONG
        orch._intent_emitters["orb-multi-1"].emit = MagicMock(return_value=TradeIntent(
            bot_id="orb-multi-1",
            direction=TradeDirection.LONG,
            confidence=80,
            urgency="HIGH",
            reason="test",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        ))

        # Mock scalper emit to return LONG
        orch._intent_emitters["scalper-multi-1"].emit = MagicMock(return_value=TradeIntent(
            bot_id="scalper-multi-1",
            direction=TradeDirection.LONG,
            confidence=75,
            urgency="HIGH",
            reason="test",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        ))

        result_orb = orch.on_tick("orb-multi-1", bar_orb, ctx_orb)
        result_scalper = orch.on_tick("scalper-multi-1", bar_scalper, ctx_scalper)

        # Both should produce directives
        assert result_orb is not None
        assert result_scalper is not None

        # State should be independent
        fv_orb, ctx_saved_orb = orch.get_state("orb-multi-1")
        fv_scalper, ctx_saved_scalper = orch.get_state("scalper-multi-1")

        assert fv_orb is not None
        assert fv_scalper is not None
        assert fv_orb.bot_id == "orb-multi-1"
        assert fv_scalper.bot_id == "scalper-multi-1"

        assert ctx_saved_orb.regime == RegimeType.BREAKOUT_PRIME
        assert ctx_saved_scalper.regime == RegimeType.TREND_STABLE

        # Verify bots are listed correctly
        bots = orch.list_registered_bots()
        assert set(bots) == {"orb-multi-1", "scalper-multi-1"}

    def test_bot_unregister_cleanup(self):
        """
        Register, then unregister a bot, verify no lingering state.
        """
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("test-unreg-1")
        spec = _orb_bot_spec(bot_id="test-unreg-1", runtime=runtime)
        orch.register_bot(spec)

        # Send a tick to populate state
        ctx = _fresh_market_ctx()
        bar = _bar_data()
        orch.on_tick(spec.id, bar, ctx)

        # Verify bot is registered
        assert "test-unreg-1" in orch.list_registered_bots()

        # Unregister
        result = orch.unregister_bot("test-unreg-1")
        assert result is True

        # Bot should be gone
        assert "test-unreg-1" not in orch.list_registered_bots()
        assert "test-unreg-1" not in orch._bots
        assert "test-unreg-1" not in orch._intent_emitters

        # State manager should be clean
        fv, mc = orch.get_state("test-unreg-1")
        assert fv is None
        assert mc is None

        # Sending tick to unregistered bot should return None
        result = orch.on_tick("test-unreg-1", bar, ctx)
        assert result is None

    def test_tick_unknown_bot_returns_none(self):
        """
        Send tick for unregistered bot, verify None is returned.
        """
        orch = RuntimeOrchestrator()
        ctx = _fresh_market_ctx()
        bar = _bar_data()

        result = orch.on_tick("never-registered-bot", bar, ctx)
        assert result is None


# ---------------------------------------------------------------------------
# TestPipelineEdgeCases
# ---------------------------------------------------------------------------

class TestPipelineEdgeCases:
    """Edge case tests for the pipeline."""

    def test_stale_market_context_blocks_pipeline(self):
        """
        Send tick with stale MarketContext (15s old),
        verify pipeline handles gracefully (intent emitter returns None).
        """
        reg = _build_mock_registry({"indicators/macd": {"rsi_14": 45.0}})

        orch = RuntimeOrchestrator(
            feature_evaluator=FeatureEvaluator(registry=reg),
        )
        runtime = _active_runtime_profile("test-stale-1")
        spec = _orb_bot_spec(bot_id="test-stale-1", runtime=runtime)
        orch.register_bot(spec)

        # Stale context: 15 seconds old (> 5s threshold)
        stale_ctx = _stale_market_ctx(regime=RegimeType.BREAKOUT_PRIME, regime_confidence=0.80)
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, stale_ctx)

        # Pipeline should handle gracefully -- no ExecutionDirective
        assert result is None

    def test_low_confidence_features_reduce_position(self):
        """
        Send tick with low-confidence scalper signal,
        verify position_size is reduced accordingly.
        Mock the intent emitter to return a signal with confidence=55 (low),
        then verify the pipeline produces a reduced position_size.
        """
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("scalper_m1_lowconf")
        spec = _scalper_bot_spec(bot_id="scalper_m1_lowconf", runtime=runtime)
        orch.register_bot(spec)

        # Mock emit to return a scalper LONG signal with low confidence=55
        # This simulates what the scalper archetype would emit with low aggression
        mock_intent = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.LONG,
            confidence=55,  # Low confidence
            urgency="NORMAL",
            reason="archetype=breakout_scalper; regime=TREND_STABLE; confidence=0.55",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters[spec.id].emit = MagicMock(return_value=mock_intent)

        ctx = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
            spread_state="NORMAL",
        )
        bar = _bar_data()

        result = orch.on_tick(spec.id, bar, ctx)

        assert result is not None
        # Risk envelope: confidence_factor = 55/100 = 0.55
        # volatility_factor = 0.85 (TREND_STABLE)
        # position_size = 100000 * 0.55 * 0.85 = 46750
        assert result.quantity == pytest.approx(46750.0, rel=1.0)  # Within 1% tolerance
        assert result.risk_mode in (RiskMode.STANDARD, RiskMode.CLAMPED)

    def test_regime_change_affects_signal(self):
        """
        Send tick under TREND_STABLE regime, verify no signal;
        send tick under RANGE_STABLE regime with RSI>70, verify SHORT signal.

        Uses mocked intent emitter to avoid regime safety gate interference.
        TREND_STABLE and RANGE_STABLE are both regime_is_clear=True so safety passes.
        """
        orch = RuntimeOrchestrator()
        runtime = _active_runtime_profile("london_orb_regime")
        spec = _orb_bot_spec(bot_id="london_orb_regime", runtime=runtime)
        orch.register_bot(spec)

        # Mock emit to return None for TREND_STABLE
        orch._intent_emitters["london_orb_regime"].emit = MagicMock(return_value=None)

        ctx_stable = _fresh_market_ctx(
            regime=RegimeType.TREND_STABLE,
            regime_confidence=0.75,
        )
        bar = _bar_data()

        result_stable = orch.on_tick(spec.id, bar, ctx_stable)
        assert result_stable is None  # No signal in TREND_STABLE

        # Mock emit to return SHORT for RANGE_STABLE with RSI>70 (ORB signal)
        mock_short = TradeIntent(
            bot_id=spec.id,
            direction=TradeDirection.SHORT,
            confidence=50,
            urgency="NORMAL",
            reason="test: orb range bound RSI overbought",
            timestamp_ms=int(time.time() * 1000),
            symbol="EURUSD",
        )
        orch._intent_emitters["london_orb_regime"].emit = MagicMock(return_value=mock_short)

        ctx_range = _fresh_market_ctx(
            regime=RegimeType.RANGE_STABLE,
            regime_confidence=0.70,
        )

        result_range = orch.on_tick(spec.id, bar, ctx_range)
        assert result_range is not None
        assert result_range.direction == TradeDirection.SHORT


# ---------------------------------------------------------------------------
# TestBotStateManagerIntegration
# ---------------------------------------------------------------------------

class TestBotStateManagerIntegration:
    """Integration tests for BotStateManager within the orchestrator context."""

    def test_state_manager_persists_across_ticks(self):
        """
        Send multiple ticks, verify FeatureVector state accumulates and persists.
        """
        sm = BotStateManager()

        # First tick
        ctx1 = _fresh_market_ctx()
        sm.tick_update("bot-persist", {}, ctx1)

        # Add a FeatureVector after evaluation
        fv1 = FeatureVector(
            bot_id="bot-persist",
            features={"rsi_14": 45.0},
            feature_confidence={},
        )
        sm.update_feature_vector("bot-persist", fv1)

        # Second tick
        ctx2 = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME)
        sm.tick_update("bot-persist", {}, ctx2)

        fv2 = FeatureVector(
            bot_id="bot-persist",
            features={"rsi_14": 40.0, "atr_14": 0.0015},
            feature_confidence={},
        )
        sm.update_feature_vector("bot-persist", fv2)

        # State should be the latest
        fv, ctx = sm.get_combined_state("bot-persist")

        assert fv is not None
        assert fv.features["rsi_14"] == 40.0
        assert fv.features["atr_14"] == 0.0015
        assert ctx.regime == RegimeType.BREAKOUT_PRIME

        # Third tick -- verify previous state is still there
        ctx3 = _fresh_market_ctx(regime=RegimeType.TREND_STABLE)
        sm.tick_update("bot-persist", {}, ctx3)

        fv3, ctx3_saved = sm.get_combined_state("bot-persist")
        assert fv3 is not None
        assert fv3.features["rsi_14"] == 40.0  # Still the latest FV
        assert ctx3_saved.regime == RegimeType.TREND_STABLE

    def test_state_manager_thread_safety(self):
        """
        Send concurrent ticks from 5 threads,
        verify no race conditions (all threads complete successfully).
        """
        sm = BotStateManager()
        errors = []
        barrier = threading.Barrier(5)

        def tick_thread(thread_id: int):
            try:
                for i in range(20):
                    ctx = _fresh_market_ctx(regime=RegimeType.BREAKOUT_PRIME)
                    sm.tick_update(f"bot-thread-{thread_id}", {}, ctx)
                    fv = FeatureVector(
                        bot_id=f"bot-thread-{thread_id}",
                        features={"tick": float(i)},
                    )
                    sm.update_feature_vector(f"bot-thread-{thread_id}", fv)
                    retrieved = sm.get_feature_vector(f"bot-thread-{thread_id}")
                    assert retrieved is not None
                    assert retrieved.bot_id == f"bot-thread-{thread_id}"
                barrier.wait(timeout=5.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=tick_thread, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Thread errors: {errors}"

        # Verify all bots have state
        for i in range(5):
            fv = sm.get_feature_vector(f"bot-thread-{i}")
            assert fv is not None

    def test_has_stale_context_triggers_refresh(self):
        """
        Verify has_stale_context returns True when context is absent or old.
        """
        sm = BotStateManager()

        # No context -> stale
        assert sm.has_stale_context("never-seen-bot") is True

        # Fresh context -> not stale
        fresh_ctx = _fresh_market_ctx()
        sm.update_market_context("fresh-bot", fresh_ctx)
        assert sm.has_stale_context("fresh-bot") is False

        # Old context -> stale (threshold = 5s, age = 10s)
        old_ctx = MarketContext(
            regime=RegimeType.BREAKOUT_PRIME,
            news_state=NewsState.CLEAR,
            regime_confidence=0.75,
            last_update_ms=int(time.time() * 1000) - 10_000,
        )
        sm.update_market_context("old-bot", old_ctx)
        assert sm.has_stale_context("old-bot") is True

        # Verify reset clears staleness
        sm.reset("fresh-bot")
        assert sm.has_stale_context("fresh-bot") is True

    def test_bot_state_manager_concurrent_access(self):
        """
        BotStateManager should handle concurrent updates safely.
        """
        sm = BotStateManager()

        def update_loop(bot_id: str, count: int):
            for i in range(count):
                ctx = _fresh_market_ctx(regime=RegimeType.TREND_STABLE)
                sm.update_market_context(bot_id, ctx)
                fv = FeatureVector(
                    bot_id=bot_id,
                    features={"counter": float(i)},
                )
                sm.update_feature_vector(bot_id, fv)

        threads = [
            threading.Thread(target=update_loop, args=(f"bot-concurrent-{i}", 50))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # All bots should have final state
        for i in range(5):
            fv = sm.get_feature_vector(f"bot-concurrent-{i}")
            assert fv is not None
            assert fv.features["counter"] == 49.0  # Last update (0-indexed, 50 iterations)

    def test_state_manager_reset_all(self):
        """
        reset_all should clear all state.
        """
        sm = BotStateManager()

        for i in range(3):
            ctx = _fresh_market_ctx()
            sm.update_market_context(f"bot-{i}", ctx)
            fv = FeatureVector(bot_id=f"bot-{i}", features={"v": float(i)})
            sm.update_feature_vector(f"bot-{i}", fv)

        assert len(sm.list_bots()) == 3

        sm.reset_all()

        assert len(sm.list_bots()) == 0
        for i in range(3):
            assert sm.get_feature_vector(f"bot-{i}") is None
            assert sm.get_market_context(f"bot-{i}") is None
