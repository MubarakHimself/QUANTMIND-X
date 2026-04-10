"""
QuantMindLib V1 -- RuntimeOrchestrator
Central coordinator for the tick-to-execution pipeline per bot session.

Pipeline:
  tick/bar data -> FeatureEvaluator -> FeatureVector
                                     |
                              IntentEmitter -> TradeIntent
                                     |
                               SafetyHooks -> KillSwitchResult
                                     |
                              RiskBridge -> RiskEnvelope
                                     |
                         ExecutionBridge -> ExecutionDirective
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.library.core.domain.bot_spec import BotSpec
from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.domain.feature_vector import FeatureVector
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.risk_envelope import RiskEnvelope
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.bridges.risk_execution_bridges import RiskBridge, ExecutionBridge
from src.library.core.types.enums import ActivationState, BotHealth
from src.library.runtime.feature_evaluator import FeatureEvaluator
from src.library.runtime.intent_emitter import IntentEmitter
from src.library.runtime.safety_hooks import SafetyHooks, KillSwitchResult
from src.library.runtime.state_manager import BotStateManager
from src.library.features._registry import get_default_registry

logger = logging.getLogger(__name__)


class RuntimeOrchestrator:
    """
    Central coordinator for the complete tick-to-execution pipeline.

    Orchestrates FeatureEvaluator, IntentEmitter, SafetyHooks, RiskBridge,
    and ExecutionBridge per registered bot.
    """

    def __init__(
        self,
        state_manager: Optional[BotStateManager] = None,
        feature_evaluator: Optional[FeatureEvaluator] = None,
        intent_emitter: Optional[IntentEmitter] = None,
        safety_hooks: Optional[SafetyHooks] = None,
    ) -> None:
        self._state_manager = (
            BotStateManager() if state_manager is None else state_manager
        )
        self._feature_evaluator = (
            FeatureEvaluator(registry=get_default_registry()) if feature_evaluator is None else feature_evaluator
        )
        self._safety_hooks = (
            SafetyHooks() if safety_hooks is None else safety_hooks
        )

        # Risk and Execution bridges -- always created (can be swapped via injection)
        self._risk_bridge = RiskBridge()
        self._execution_bridge = ExecutionBridge()

        # Bot registry: bot_id -> spec
        self._bots: Dict[str, BotSpec] = {}

        # Per-bot IntentEmitter instances (one per bot, needs bot spec)
        self._intent_emitters: Dict[str, IntentEmitter] = {}

    # -------------------------------------------------------------------------
    # Bot Registration
    # -------------------------------------------------------------------------

    def register_bot(self, spec: BotSpec) -> bool:
        """
        Register a bot from its BotSpec.

        Stores the spec and creates a per-bot IntentEmitter.
        Returns True if registered, False if already registered.
        """
        if spec.id in self._bots:
            return False
        self._bots[spec.id] = spec
        self._intent_emitters[spec.id] = IntentEmitter(bot_spec=spec)
        logger.info("RuntimeOrchestrator: registered bot %s (archetype=%s)", spec.id, spec.archetype)
        return True

    def unregister_bot(self, bot_id: str) -> bool:
        """Remove a bot from the registry. Returns True if removed, False if not found."""
        if bot_id not in self._bots:
            return False
        del self._bots[bot_id]
        self._intent_emitters.pop(bot_id, None)
        self._state_manager.reset(bot_id)
        logger.info("RuntimeOrchestrator: unregistered bot %s", bot_id)
        return True

    def list_registered_bots(self) -> List[str]:
        """List all registered bot IDs."""
        return list(self._bots.keys())

    # -------------------------------------------------------------------------
    # Tick Processing (Hot Path)
    # -------------------------------------------------------------------------

    def on_tick(
        self,
        bot_id: str,
        bar_data: Dict[str, Any],
        market_ctx: MarketContext,
    ) -> Optional[ExecutionDirective]:
        """
        Process a market tick for a registered bot.

        Steps:
          1. Update state manager with bar data and market context
          2. Evaluate features -> FeatureVector
          3. Emit trade intent -> Optional[TradeIntent]
          4. Run safety checks -> KillSwitchResult
          5. Authorize risk -> RiskEnvelope
          6. Generate execution directive -> ExecutionDirective

        Returns:
            ExecutionDirective if the full pipeline succeeded and trade is authorized.
            None if no intent, blocked by safety, halted by risk, or unknown bot.
        """
        # Step 0: Check bot is registered
        spec = self._bots.get(bot_id)
        if spec is None:
            logger.debug("RuntimeOrchestrator: unknown bot %s, skipping tick", bot_id)
            return None

        # Step 1: Update state manager
        self._state_manager.tick_update(bot_id, bar_data, market_ctx)

        # Step 2: Build inputs dict for feature evaluation
        inputs: Dict[str, Any] = {
            "close_prices": bar_data.get("close", []),
            "high": bar_data.get("high", []),
            "low": bar_data.get("low", []),
            "volume": bar_data.get("volume", []),
            "market_context": market_ctx,
        }

        # Step 3: Evaluate features
        feature_vector = self._feature_evaluator.evaluate(
            bot_id, spec.features, inputs
        )
        # Update state manager with computed FeatureVector
        self._state_manager.update_feature_vector(bot_id, feature_vector)

        # Step 4: Emit trade intent
        emitter = self._intent_emitters[bot_id]
        symbol = spec.symbol_scope[0] if spec.symbol_scope else "UNKNOWN"
        intent = emitter.emit(feature_vector, market_ctx, symbol)

        if intent is None:
            logger.debug(
                "RuntimeOrchestrator: no intent emitted for bot %s (tick=%s)",
                bot_id,
                market_ctx.last_update_ms,
            )
            return None

        # Step 5: Run safety checks
        safety_result = self._run_safety_check(spec, intent, market_ctx)
        if not safety_result.allowed:
            logger.info(
                "RuntimeOrchestrator: safety blocked bot %s -- %s",
                bot_id,
                safety_result.reason,
            )
            return None

        # Step 6: Authorize risk
        risk_envelope = self._risk_bridge.authorize(intent, feature_vector, market_ctx)

        if risk_envelope.position_size <= 0:
            logger.debug(
                "RuntimeOrchestrator: zero position_size for bot %s", bot_id
            )
            return None

        if risk_envelope.risk_mode.value == "HALTED":
            logger.info(
                "RuntimeOrchestrator: risk mode HALTED for bot %s", bot_id
            )
            return None

        # Step 7: Generate execution directive
        directive = self._execution_bridge.execute(intent, risk_envelope)
        return directive

    def _run_safety_check(
        self,
        spec: BotSpec,
        intent: TradeIntent,
        market_ctx: MarketContext,
    ) -> KillSwitchResult:
        """
        Run SafetyHooks.check() with appropriate defaults.
        Derives health/activation from BotSpec.runtime if present.
        """
        # Extract health and activation state from bot spec runtime profile
        health = BotHealth.UNKNOWN
        activation_state = ActivationState.UNKNOWN
        if spec.runtime is not None:
            health = spec.runtime.health
            activation_state = spec.runtime.activation_state

        # Derive safety gates from market context
        regime_is_clear = market_ctx.regime.value not in (
            "NEWS_EVENT", "HIGH_CHAOS", "UNCERTAIN"
        )
        spread_state_ok = market_ctx.spread_state != "WIDE"
        news_clear = market_ctx.news_state.value == "CLEAR"

        # Default daily loss (simplified -- real impl would track actual P&L)
        daily_loss_pct = 0.0

        return self._safety_hooks.check(
            bot_id=intent.bot_id,
            health=health,
            activation_state=activation_state,
            daily_loss_pct=daily_loss_pct,
            regime_is_clear=regime_is_clear,
            spread_state_ok=spread_state_ok,
            news_clear=news_clear,
        )

    # -------------------------------------------------------------------------
    # State Queries
    # -------------------------------------------------------------------------

    def get_state(
        self, bot_id: str
    ) -> Tuple[Optional[FeatureVector], Optional[MarketContext]]:
        """Proxy to state manager's get_combined_state."""
        return self._state_manager.get_combined_state(bot_id)

    def get_health_summary(self) -> Dict[str, str]:
        """
        Return health status of all registered bots.
        Maps bot_id -> health string.
        """
        summary: Dict[str, str] = {}
        for bot_id, spec in self._bots.items():
            if spec.runtime is not None:
                summary[bot_id] = spec.runtime.health.value
            else:
                summary[bot_id] = "UNKNOWN"
        return summary


__all__ = ["RuntimeOrchestrator"]