"""Tests for QuantMindLib V1 -- SafetyHooks."""

import time

import pytest

from src.library.core.types.enums import (
    BotHealth,
    ActivationState,
)
from src.library.runtime.safety_hooks import SafetyHooks, KillSwitchResult


# ---------------------------------------------------------------------------
# Tests: SafetyHooks initialization
# ---------------------------------------------------------------------------

class TestSafetyHooksInit:
    def test_initializes_with_defaults(self):
        hooks = SafetyHooks()
        assert hooks.kill_switch_enabled is True
        assert hooks.max_daily_loss_pct == 0.05
        assert hooks.circuit_breaker_loss_pct == 0.10

    def test_initializes_with_custom_values(self):
        hooks = SafetyHooks(
            kill_switch_enabled=False,
            max_daily_loss_pct=0.03,
            circuit_breaker_loss_pct=0.08,
        )
        assert hooks.kill_switch_enabled is False
        assert hooks.max_daily_loss_pct == 0.03
        assert hooks.circuit_breaker_loss_pct == 0.08


# ---------------------------------------------------------------------------
# Tests: KillSwitchResult properties
# ---------------------------------------------------------------------------

class TestKillSwitchResult:
    def test_allowed_result_has_reason_and_timestamp(self):
        result = KillSwitchResult(allowed=True, reason="all good")
        assert result.allowed is True
        assert result.reason == "all good"
        assert result.triggered_by is None
        assert isinstance(result.checked_at_ms, int)

    def test_blocked_result_has_triggered_by(self):
        result = KillSwitchResult(
            allowed=False,
            reason="health critical",
            triggered_by="HEALTH_GATE",
        )
        assert result.allowed is False
        assert result.triggered_by == "HEALTH_GATE"
        assert result.reason == "health critical"


# ---------------------------------------------------------------------------
# Tests: check -- passes when all conditions met
# ---------------------------------------------------------------------------

class TestCheckPasses:
    def test_check_passes_when_all_conditions_met(self):
        hooks = SafetyHooks()
        result = hooks.check(
            bot_id="bot-1",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.01,
            regime_is_clear=True,
            spread_state_ok=True,
            news_clear=True,
        )
        assert result.allowed is True
        assert "All checks passed" in result.reason

    def test_check_passes_with_cautious_activation(self):
        hooks = SafetyHooks()
        result = hooks.check(
            bot_id="bot-2",
            health=BotHealth.CAUTIOUS,
            activation_state=ActivationState.CAUTIOUS,
            daily_loss_pct=0.0,
            regime_is_clear=True,
        )
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Tests: check -- blocks on kill_switch regime
# ---------------------------------------------------------------------------

class TestKillSwitchRegime:
    def test_check_blocks_on_kill_switch_regime(self):
        hooks = SafetyHooks(kill_switch_enabled=True)
        result = hooks.check(
            bot_id="bot-regime",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=False,
        )
        assert result.allowed is False
        assert result.triggered_by == "KILL_SWITCH_REGIME"
        assert "regime not clear" in result.reason

    def test_kill_switch_respects_disabled_flag(self):
        hooks = SafetyHooks(kill_switch_enabled=False)
        result = hooks.check(
            bot_id="bot-regime",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=False,
        )
        # With kill switch disabled, regime block is bypassed
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Tests: check -- blocks on kill_switch spread
# ---------------------------------------------------------------------------

class TestKillSwitchSpread:
    def test_check_blocks_on_kill_switch_spread(self):
        hooks = SafetyHooks(kill_switch_enabled=True)
        result = hooks.check(
            bot_id="bot-spread",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=True,
            spread_state_ok=False,
        )
        assert result.allowed is False
        assert result.triggered_by == "KILL_SWITCH_SPREAD"
        assert "spread state not ok" in result.reason


# ---------------------------------------------------------------------------
# Tests: check -- blocks on circuit_breaker daily loss
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_check_blocks_on_circuit_breaker_daily_loss(self):
        hooks = SafetyHooks(circuit_breaker_loss_pct=0.10)
        result = hooks.check(
            bot_id="bot-cb",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.12,
            regime_is_clear=True,
        )
        assert result.allowed is False
        assert result.triggered_by == "CIRCUIT_BREAKER"
        assert "12.00%" in result.reason

    def test_check_passes_at_exactly_circuit_breaker_threshold(self):
        hooks = SafetyHooks(circuit_breaker_loss_pct=0.10)
        result = hooks.check(
            bot_id="bot-cb-edge",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.10,  # Exactly at threshold
            regime_is_clear=True,
        )
        # >= comparison means exactly at threshold is blocked
        assert result.allowed is False

    def test_check_passes_just_below_circuit_breaker(self):
        hooks = SafetyHooks(circuit_breaker_loss_pct=0.10)
        result = hooks.check(
            bot_id="bot-cb-ok",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.099,
            regime_is_clear=True,
        )
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Tests: check -- blocks on health CRITICAL
# ---------------------------------------------------------------------------

class TestHealthGate:
    def test_check_blocks_on_health_critical(self):
        hooks = SafetyHooks()
        result = hooks.check(
            bot_id="bot-critical",
            health=BotHealth.CRITICAL,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=True,
        )
        assert result.allowed is False
        assert result.triggered_by == "HEALTH_GATE"
        assert "CRITICAL" in result.reason

    def test_check_passes_on_health_healthy(self):
        hooks = SafetyHooks()
        result = hooks.check(
            bot_id="bot-healthy",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=True,
        )
        assert result.allowed is True

    def test_check_passes_on_health_cautious(self):
        hooks = SafetyHooks()
        result = hooks.check(
            bot_id="bot-cautious",
            health=BotHealth.CAUTIOUS,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.0,
            regime_is_clear=True,
        )
        assert result.allowed is True


# ---------------------------------------------------------------------------
# Tests: check -- blocks on activation state
# ---------------------------------------------------------------------------

class TestActivationGate:
    def test_check_blocks_on_inactive_activation(self):
        hooks = SafetyHooks()
        for state in (
            ActivationState.INACTIVE,
            ActivationState.DEGRADED,
            ActivationState.UNKNOWN,
            ActivationState.PAUSED,
            ActivationState.STOPPED,
        ):
            result = hooks.check(
                bot_id="bot-activation",
                health=BotHealth.HEALTHY,
                activation_state=state,
                daily_loss_pct=0.0,
                regime_is_clear=True,
            )
            assert result.allowed is False, f"Should block on activation_state={state}"
            assert result.triggered_by == "ACTIVATION_GATE"


# ---------------------------------------------------------------------------
# Tests: check -- warning on max_daily_loss approaching
# ---------------------------------------------------------------------------

class TestDailyLossWarning:
    def test_check_passes_with_warning_on_max_daily_loss(self):
        hooks = SafetyHooks(max_daily_loss_pct=0.05, circuit_breaker_loss_pct=0.10)
        result = hooks.check(
            bot_id="bot-warn",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.06,  # Above warning, below circuit breaker
            regime_is_clear=True,
        )
        assert result.allowed is True
        assert "Warning" in result.reason
        assert "approaching circuit breaker" in result.reason

    def test_check_passes_without_warning_when_loss_low(self):
        hooks = SafetyHooks(max_daily_loss_pct=0.05)
        result = hooks.check(
            bot_id="bot-clean",
            health=BotHealth.HEALTHY,
            activation_state=ActivationState.ACTIVE,
            daily_loss_pct=0.01,
            regime_is_clear=True,
        )
        assert result.allowed is True
        assert "Warning" not in result.reason


# ---------------------------------------------------------------------------
# Tests: quick_health_check
# ---------------------------------------------------------------------------

class TestQuickHealthCheck:
    def test_quick_health_check_false_for_critical(self):
        hooks = SafetyHooks()
        assert hooks.quick_health_check(BotHealth.CRITICAL) is False

    def test_quick_health_check_false_for_degraded(self):
        hooks = SafetyHooks()
        assert hooks.quick_health_check(BotHealth.DEGRADED) is False

    def test_quick_health_check_true_for_healthy(self):
        hooks = SafetyHooks()
        assert hooks.quick_health_check(BotHealth.HEALTHY) is True

    def test_quick_health_check_true_for_cautious(self):
        hooks = SafetyHooks()
        assert hooks.quick_health_check(BotHealth.CAUTIOUS) is True

    def test_quick_health_check_true_for_unknown(self):
        hooks = SafetyHooks()
        assert hooks.quick_health_check(BotHealth.UNKNOWN) is True


# ---------------------------------------------------------------------------
# Tests: session_blackout_check
# ---------------------------------------------------------------------------

class TestSessionBlackoutCheck:
    def test_session_blackout_returns_true_when_session_active(self):
        hooks = SafetyHooks()
        result = hooks.session_blackout_check("london", ["london", "ny"])
        assert result is True

    def test_session_blackout_returns_false_when_session_inactive(self):
        hooks = SafetyHooks()
        result = hooks.session_blackout_check("tokyo", ["london", "ny"])
        assert result is False

    def test_session_blackout_returns_false_for_empty_active_list(self):
        hooks = SafetyHooks()
        result = hooks.session_blackout_check("london", [])
        assert result is False

    def test_session_blackout_returns_true_with_single_matching_session(self):
        hooks = SafetyHooks()
        result = hooks.session_blackout_check("london", ["london"])
        assert result is True
