"""
QuantMindLib V1 -- SafetyHooks
Kill switch and circuit breaker integration for runtime safety.
Phase 11F: DPRCircuitBreakerMonitor wiring for combined DPR+SSL safety.
"""
from __future__ import annotations

import time
from typing import List, Optional, TYPE_CHECKING

from src.library.core.types.enums import BotHealth, ActivationState

if TYPE_CHECKING:
    from src.library.core.bridges.sentinel_dpr_bridges import DPRScore
    from src.library.core.bridges.safety_integration import DPRCircuitBreakerMonitor
    from src.risk.dpr.dpr_emitter import DPRSSLEmitter


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

    Phase 11F: Optionally integrates DPRCircuitBreakerMonitor for combined
    DPR + SSL safety gating. DPR is checked before health gates.
    """

    def __init__(
        self,
        kill_switch_enabled: bool = True,
        max_daily_loss_pct: float = 0.05,
        circuit_breaker_loss_pct: float = 0.10,
        dpr_monitor: Optional["DPRCircuitBreakerMonitor"] = None,
        dpr_emitter: Optional["DPRSSLEmitter"] = None,
    ) -> None:
        self.kill_switch_enabled = kill_switch_enabled
        self.max_daily_loss_pct = max_daily_loss_pct
        self.circuit_breaker_loss_pct = circuit_breaker_loss_pct
        self._dpr_monitor = dpr_monitor
        self._dpr_emitter = dpr_emitter

    def check(
        self,
        bot_id: str,
        health: BotHealth,
        activation_state: ActivationState,
        daily_loss_pct: float,
        regime_is_clear: bool,
        spread_state_ok: bool = True,
        news_clear: bool = True,
        dpr_score: Optional["DPRScore"] = None,
        ssl_state: Optional[str] = None,
        ssl_tier: Optional[str] = None,
    ) -> KillSwitchResult:
        """
        Run all safety checks. Returns KillSwitchResult.
        If allowed=False, trade MUST NOT proceed.

        Phase 11F: DPR circuit check runs before health gate if dpr_monitor
        is configured. DPR→SSL emission runs after blocked DPR decisions.

        Args:
            bot_id: Bot identifier
            health: Bot health state
            activation_state: Activation state
            daily_loss_pct: Daily loss percentage
            regime_is_clear: Whether market regime is clear
            spread_state_ok: Whether spread is acceptable
            news_clear: Whether news blackout is clear
            dpr_score: Optional DPRScore for DPR circuit check (Phase 11F)
            ssl_state: Optional SSL state string for combined DPR+SSL check
            ssl_tier: Optional SSL tier for combined check
        """
        # 0. DPR circuit breaker check (Phase 11F)
        if self._dpr_monitor is not None and dpr_score is not None:
            if ssl_state is not None:
                # Combined DPR + SSL check via SSLCircuitBreakerDPRMonitor
                dpr_result = self._dpr_monitor.check_ssl_dpr_combined(
                    bot_id=bot_id,
                    dpr_score=dpr_score,
                    ssl_state=ssl_state,
                    ssl_tier=ssl_tier,
                )
            else:
                # DPR-only circuit check
                dpr_result = self._dpr_monitor.check_bot_circuit_state(
                    bot_id=bot_id,
                    dpr_score=dpr_score,
                )

            # DPR blocked — emit DPR→SSL transition and return blocked result
            if not dpr_result.allowed:
                if self._dpr_emitter is not None:
                    self._dpr_emitter.emit_dpr_to_ssl(
                        bot_id=bot_id,
                        dpr_score=dpr_score,
                        reason=dpr_result.triggered_by,
                    )
                return dpr_result

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

        # 6. DPR recovery check (Phase 11F): if DPR improves, emit SSL recovery
        if (
            self._dpr_monitor is not None
            and self._dpr_emitter is not None
            and dpr_score is not None
        ):
            # Only emit recovery if DPR is recovering and bot was transitioned via DPR
            tier = dpr_score.tier
            if tier in ("STANDARD", "PERFORMING", "ELITE"):
                if self._dpr_emitter.is_bot_transitioned(bot_id):
                    self._dpr_emitter.emit_recovery_to_ssl(bot_id, dpr_score)

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
