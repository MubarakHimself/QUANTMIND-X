"""
QuantMindLib V1 -- SafetyHooks
Kill switch and circuit breaker integration for runtime safety.
"""
from __future__ import annotations

import time
from typing import List, Optional

from src.library.core.types.enums import BotHealth, ActivationState


class KillSwitchResult:
    """Result of a kill switch evaluation."""

    allowed: bool
    reason: str
    triggered_by: Optional[str]
    checked_at_ms: int

    def __init__(
        self,
        allowed: bool,
        reason: str = "",
        triggered_by: Optional[str] = None,
    ) -> None:
        self.allowed = allowed
        self.reason = reason
        self.triggered_by = triggered_by
        self.checked_at_ms = int(time.time() * 1000)


class SafetyHooks:
    """
    Pre-trade safety checks.
    Evaluates kill switches, circuit breakers, and health gates.
    """

    def __init__(
        self,
        kill_switch_enabled: bool = True,
        max_daily_loss_pct: float = 0.05,
        circuit_breaker_loss_pct: float = 0.10,
    ) -> None:
        self.kill_switch_enabled = kill_switch_enabled
        self.max_daily_loss_pct = max_daily_loss_pct
        self.circuit_breaker_loss_pct = circuit_breaker_loss_pct

    def check(
        self,
        bot_id: str,
        health: BotHealth,
        activation_state: ActivationState,
        daily_loss_pct: float,
        regime_is_clear: bool,
        spread_state_ok: bool = True,
        news_clear: bool = True,
    ) -> KillSwitchResult:
        """
        Run all safety checks. Returns KillSwitchResult.
        If allowed=False, trade MUST NOT proceed.
        """
        # 1. Kill switch master override
        if self.kill_switch_enabled:
            if not regime_is_clear:
                return KillSwitchResult(
                    allowed=False,
                    reason=f"{bot_id}: Kill switch -- regime not clear",
                    triggered_by="KILL_SWITCH_REGIME",
                )

            if not spread_state_ok:
                return KillSwitchResult(
                    allowed=False,
                    reason=f"{bot_id}: Kill switch -- spread state not ok",
                    triggered_by="KILL_SWITCH_SPREAD",
                )

            if not news_clear:
                return KillSwitchResult(
                    allowed=False,
                    reason=f"{bot_id}: Kill switch -- news event active",
                    triggered_by="KILL_SWITCH_NEWS",
                )

        # 2. Circuit breaker: daily loss exceeded
        if daily_loss_pct >= self.circuit_breaker_loss_pct:
            return KillSwitchResult(
                allowed=False,
                reason=(
                    f"{bot_id}: Circuit breaker -- daily loss "
                    f"{daily_loss_pct:.2%} >= {self.circuit_breaker_loss_pct:.2%}"
                ),
                triggered_by="CIRCUIT_BREAKER",
            )

        # 3. Health check
        if health == BotHealth.CRITICAL:
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Health CRITICAL",
                triggered_by="HEALTH_GATE",
            )

        # 4. Activation state check
        if activation_state not in (
            ActivationState.ACTIVE,
            ActivationState.CAUTIOUS,
        ):
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Activation state {activation_state.value} not tradable",
                triggered_by="ACTIVATION_GATE",
            )

        # 5. Daily loss warning gate
        if daily_loss_pct >= self.max_daily_loss_pct:
            return KillSwitchResult(
                allowed=True,  # Still allowed but with warning
                reason=(
                    f"{bot_id}: Warning -- daily loss {daily_loss_pct:.2%} "
                    "approaching circuit breaker"
                ),
            )

        return KillSwitchResult(
            allowed=True,
            reason=f"{bot_id}: All checks passed",
        )

    def quick_health_check(self, health: BotHealth) -> bool:
        """
        Fast boolean check: is the bot healthy enough to consider trading?
        Used for pre-filtering before full check().
        """
        return health not in (
            BotHealth.CRITICAL,
            BotHealth.DEGRADED,
        )

    def session_blackout_check(
        self,
        session_id: str,
        active_sessions: List[str],
    ) -> bool:
        """
        True if the current session is active (not in blackout).
        """
        return session_id in active_sessions
