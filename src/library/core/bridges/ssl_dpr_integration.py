"""
QuantMindLib V1 — SSL + DPR Safety Integration
Phase 10 Packet 10C: SSLCircuitBreaker DPR monitor for combined safety gating.

Bridges SSLCircuitBreaker (per-bot consecutive loss tracking) with the DPR
safety layer (DPR tier circuit breaker), enabling unified kill-switch
decisions that respect both systems simultaneously.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.risk.ssl.circuit_breaker import SSLCircuitBreaker, BotType, FAIL_S02
from src.risk.ssl.state import SSLState
from src.library.runtime.safety_hooks import KillSwitchResult


# ── SSL state constants for DPR monitoring ────────────────────────────────────

# HALTED: FAIL_S02 triggered — 3 losses in a day, halt for rest of day.
# Blocks ALL trades regardless of DPR tier.
SSL_HALTED = "HALTED"

# ACTIVE: Consecutive losses threshold reached — circuit is open.
# Blocks new entries but existing positions may be held.
SSL_ACTIVE = "ACTIVE"

# SSL DPR tier ordering for combined risk evaluation.
_SSL_DPR_TIER_ORDER: Dict[str, int] = {
    "ELITE": 0,
    "PERFORMING": 1,
    "STANDARD": 2,
    "AT_RISK": 3,
    "CIRCUIT_BROKEN": 4,
}

# Combined risk level derived from SSL state + DPR tier.
_SSL_DPR_COMBINED_RISK: Dict[str, str] = {
    # SSL HALTED — highest risk regardless of DPR.
    f"{SSL_HALTED}|CIRCUIT_BROKEN": "CRITICAL",
    f"{SSL_HALTED}|AT_RISK": "CRITICAL",
    f"{SSL_HALTED}|STANDARD": "CRITICAL",
    f"{SSL_HALTED}|PERFORMING": "HIGH",
    f"{SSL_HALTED}|ELITE": "HIGH",
    # SSL ACTIVE with AT_RISK DPR — immediate circuit break.
    f"{SSL_ACTIVE}|AT_RISK": "CRITICAL",
    f"{SSL_ACTIVE}|CIRCUIT_BROKEN": "CRITICAL",
    f"{SSL_ACTIVE}|STANDARD": "HIGH",
    f"{SSL_ACTIVE}|PERFORMING": "HIGH",
    f"{SSL_ACTIVE}|ELITE": "MEDIUM",
    # SSL RECOVERY.
    f"{SSLState.RECOVERY.value}|CIRCUIT_BROKEN": "CRITICAL",
    f"{SSLState.RECOVERY.value}|AT_RISK": "HIGH",
    f"{SSLState.RECOVERY.value}|STANDARD": "MEDIUM",
    f"{SSLState.RECOVERY.value}|PERFORMING": "LOW",
    f"{SSLState.RECOVERY.value}|ELITE": "LOW",
    # SSL PAPER (paper rotation).
    f"{SSLState.PAPER.value}|CIRCUIT_BROKEN": "CRITICAL",
    f"{SSLState.PAPER.value}|AT_RISK": "HIGH",
    f"{SSLState.PAPER.value}|STANDARD": "MEDIUM",
    f"{SSLState.PAPER.value}|PERFORMING": "LOW",
    f"{SSLState.PAPER.value}|ELITE": "LOW",
    # SSL LIVE / NORMAL.
    f"{SSLState.LIVE.value}|CIRCUIT_BROKEN": "HIGH",
    f"{SSLState.LIVE.value}|AT_RISK": "MEDIUM",
    f"{SSLState.LIVE.value}|STANDARD": "LOW",
    f"{SSLState.LIVE.value}|PERFORMING": "LOW",
    f"{SSLState.LIVE.value}|ELITE": "LOW",
    # RETIRED — no active trading.
    f"{SSLState.RETIRED.value}|CIRCUIT_BROKEN": "CRITICAL",
    f"{SSLState.RETIRED.value}|AT_RISK": "HIGH",
    f"{SSLState.RETIRED.value}|STANDARD": "HIGH",
    f"{SSLState.RETIRED.value}|PERFORMING": "HIGH",
    f"{SSLState.RETIRED.value}|ELITE": "MEDIUM",
}


class SSLCircuitBreakerDPRMonitor:
    """
    Tracks SSL state alongside DPR tier for combined kill-switch evaluation.

    Integrates SSLCircuitBreaker (consecutive loss tracking) with DPR tier
    concerns to produce unified KillSwitchResult decisions.

    Key behaviours:
    - SSL HALTED: block ALL trades regardless of DPR tier.
    - SSL ACTIVE: block new entries; existing positions may continue.
    - SSL RECOVERY: allow with caution flag.
    - DPR CIRCUIT_BROKEN: block regardless of SSL state.
    - AT_RISK DPR + SSL ACTIVE: immediate circuit break (combined escalation).
    """

    def __init__(
        self,
        ssl_breaker: Optional[SSLCircuitBreaker] = None,
    ) -> None:
        self._ssl_breaker = ssl_breaker or SSLCircuitBreaker()
        # Tracks SSL HALTED state per bot (FAIL_S02 flag — persists for the day).
        self._halted_bots: Dict[str, str] = {}
        # Tracks SSL ACTIVE state per bot (threshold breached, awaiting paper move).
        self._active_bots: Dict[str, str] = {}
        self._last_check_ms: int = int(time.time() * 1000)

    # ── Public API ─────────────────────────────────────────────────────────────

    def check_ssl_dpr_state(
        self,
        bot_id: str,
        ssl_state: str,
        ssl_tier: str,
        dpr_tier: str,
    ) -> KillSwitchResult:
        """
        Evaluate combined SSL + DPR state for a bot.

        Decision order (highest priority first):
        1. SSL HALTED (FAIL_S02) — block all trades.
        2. DPR tier CIRCUIT_BROKEN — block regardless of SSL.
        3. SSL ACTIVE — block new entries.
        4. AT_RISK DPR + SSL ACTIVE — immediate circuit break.
        5. SSL RECOVERY — allow with caution.
        6. Otherwise — allow (combined risk level in reason).

        Args:
            bot_id: Bot identifier.
            ssl_state: SSL state string (SSLState value or SSL_HALTED/SSL_ACTIVE).
            ssl_tier: Paper trading tier ("TIER_1" | "TIER_2" | None).
            dpr_tier: DPR tier string ("ELITE" | "PERFORMING" | "STANDARD" | "AT_RISK" | "CIRCUIT_BROKEN").

        Returns:
            KillSwitchResult with allowed flag, human-readable reason, and
            triggered_by source.
        """
        self._last_check_ms = int(time.time() * 1000)

        # Guard: treat unknown DPR tier as STANDARD.
        if dpr_tier not in _SSL_DPR_TIER_ORDER:
            dpr_tier = "STANDARD"

        # 1. SSL HALTED — block everything (FAIL_S02).
        if ssl_state == SSL_HALTED or self._is_halted(bot_id):
            self._halted_bots[bot_id] = f"SSL HALTED (FAIL_S02) with DPR tier={dpr_tier}"
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: SSL HALTED (FAIL_S02) — block all trades, DPR tier={dpr_tier}",
                triggered_by="SSL_HALTED",
            )

        # 2. DPR CIRCUIT_BROKEN — block regardless of SSL.
        if dpr_tier == "CIRCUIT_BROKEN":
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: DPR tier CIRCUIT_BROKEN — block all trades, SSL state={ssl_state}",
                triggered_by="DPR_CIRCUIT_BROKEN",
            )

        # 3. SSL ACTIVE — block new entries.
        if ssl_state == SSL_ACTIVE or self._is_active(bot_id):
            self._active_bots[bot_id] = f"SSL ACTIVE — threshold breached, DPR tier={dpr_tier}"
            # 4. AT_RISK DPR + SSL ACTIVE — immediate circuit break.
            if dpr_tier == "AT_RISK":
                self._active_bots.pop(bot_id, None)
                self._ssl_breaker.increment_consecutive_losses(bot_id, "")
                return KillSwitchResult(
                    allowed=False,
                    reason=(
                        f"{bot_id}: AT_RISK DPR + SSL ACTIVE — immediate circuit break, "
                        f"SSL state={ssl_state}, DPR tier={dpr_tier}"
                    ),
                    triggered_by="SSL_DPR_COMBINED_CIRCUIT_BREAK",
                )
            return KillSwitchResult(
                allowed=False,
                reason=(
                    f"{bot_id}: SSL ACTIVE — new entries blocked, "
                    f"existing positions allowed, DPR tier={dpr_tier}"
                ),
                triggered_by="SSL_ACTIVE",
            )

        # 5. SSL RECOVERY — allow with caution.
        if ssl_state == SSLState.RECOVERY.value:
            return KillSwitchResult(
                allowed=True,
                reason=(
                    f"{bot_id}: SSL RECOVERY — allowed with caution, "
                    f"SSL state={ssl_state}, ssl_tier={ssl_tier}, DPR tier={dpr_tier}"
                ),
            )

        # 6. SSL PAPER — paper trading mode, allow entries.
        if ssl_state == SSLState.PAPER.value:
            if dpr_tier == "AT_RISK":
                return KillSwitchResult(
                    allowed=False,
                    reason=(
                        f"{bot_id}: SSL PAPER with AT_RISK DPR — block new entries, "
                        f"ssl_tier={ssl_tier}, DPR tier={dpr_tier}"
                    ),
                    triggered_by="SSL_DPR_COMBINED",
                )
            return KillSwitchResult(
                allowed=True,
                reason=(
                    f"{bot_id}: SSL PAPER — allowed, ssl_tier={ssl_tier}, DPR tier={dpr_tier}"
                ),
            )

        # 7. SSL LIVE — normal trading, DPR-gated.
        if dpr_tier == "AT_RISK":
            return KillSwitchResult(
                allowed=True,
                reason=(
                    f"{bot_id}: SSL LIVE — allowed with DPR AT_RISK caution, "
                    f"DPR tier={dpr_tier}"
                ),
            )

        # Default: allow.
        return KillSwitchResult(
            allowed=True,
            reason=f"{bot_id}: SSL LIVE — all checks passed, DPR tier={dpr_tier}",
        )

    def get_ssl_dpr_summary(self, bot_id: str) -> Dict:
        """
        Return SSL + DPR summary for a bot including combined risk level.

        Args:
            bot_id: Bot identifier.

        Returns:
            Dict with keys:
            - ssl_state: Current SSL state or SSL_HALTED/SSL_ACTIVE.
            - ssl_tier: Paper trading tier or None.
            - dpr_tier: Last observed DPR tier (from tracked bots).
            - halted: Whether FAIL_S02 is active.
            - active: Whether SSL ACTIVE state is active.
            - combined_risk: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL".
            - blocked: Whether currently blocked.
        """
        is_halted = self._is_halted(bot_id)
        is_active = self._is_active(bot_id)

        # Determine effective SSL state string.
        # Prefer internal registries over SSLCircuitBreaker (which may need DB).
        if is_halted:
            effective_ssl = SSL_HALTED
        elif is_active:
            effective_ssl = SSL_ACTIVE
        else:
            # Fallback to LIVE if we have no record.
            effective_ssl = SSLState.LIVE.value

        # DPR tier from tracked state (default to STANDARD for unknown bots).
        dpr_tier = self._get_tracked_dpr_tier(bot_id)

        combined_risk = _SSL_DPR_COMBINED_RISK.get(
            f"{effective_ssl}|{dpr_tier}", "HIGH"
        )

        # A bot is blocked if HALTED, or if ACTIVE and DPR is AT_RISK.
        blocked = is_halted or (
            is_active and dpr_tier == "AT_RISK"
        )

        return {
            "bot_id": bot_id,
            "ssl_state": effective_ssl,
            "ssl_tier": self._ssl_breaker.state_manager.get_tier(bot_id).value
            if self._ssl_breaker.state_manager.get_tier(bot_id)
            else None,
            "dpr_tier": dpr_tier,
            "halted": is_halted,
            "active": is_active,
            "combined_risk": combined_risk,
            "blocked": blocked,
        }

    def reset_ssl_state(self, bot_id: str) -> bool:
        """
        Clear SSL state for a bot (used for new paper rotation).

        Removes HALTED and ACTIVE tracking flags. Does NOT reset the
        underlying SSLCircuitBreaker consecutive loss counter.

        Args:
            bot_id: Bot identifier.

        Returns:
            True if any state was cleared.
        """
        halted_cleared = self._halted_bots.pop(bot_id, None) is not None
        active_cleared = self._active_bots.pop(bot_id, None) is not None
        return halted_cleared or active_cleared

    def set_ssl_halted(self, bot_id: str, reason: str = "") -> None:
        """
        Mark a bot as HALTED (FAIL_S02 — 3 losses in a day).

        Args:
            bot_id: Bot identifier.
            reason: Optional reason string.
        """
        self._halted_bots[bot_id] = reason or f"SSL HALTED (FAIL_S02)"

    def set_ssl_active(self, bot_id: str, reason: str = "") -> None:
        """
        Mark a bot as SSL ACTIVE (consecutive losses threshold breached).

        Args:
            bot_id: Bot identifier.
            reason: Optional reason string.
        """
        self._active_bots[bot_id] = reason or f"SSL ACTIVE — threshold breached"

    def get_halted_bots(self) -> Dict[str, str]:
        """Return dict of bot_id -> halt reason for all halted bots."""
        return dict(self._halted_bots)

    def get_active_bots(self) -> Dict[str, str]:
        """Return dict of bot_id -> active reason for all active bots."""
        return dict(self._active_bots)

    def is_halted(self, bot_id: str) -> bool:
        """Return True if bot is in HALTED state."""
        return self._is_halted(bot_id)

    def is_active(self, bot_id: str) -> bool:
        """Return True if bot is in SSL ACTIVE state."""
        return self._is_active(bot_id)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_halted(self, bot_id: str) -> bool:
        return bot_id in self._halted_bots

    def _is_active(self, bot_id: str) -> bool:
        return bot_id in self._active_bots

    def _get_tracked_dpr_tier(self, bot_id: str) -> str:
        """Extract DPR tier from tracked state, defaulting to STANDARD."""
        # Check haltedbots reason string for tier.
        halted_reason = self._halted_bots.get(bot_id, "")
        if "DPR tier=" in halted_reason:
            tier = halted_reason.split("DPR tier=")[-1].strip()
            if tier in _SSL_DPR_TIER_ORDER:
                return tier
        # Check active bots reason string.
        active_reason = self._active_bots.get(bot_id, "")
        if "DPR tier=" in active_reason:
            tier = active_reason.split("DPR tier=")[-1].strip()
            if tier in _SSL_DPR_TIER_ORDER:
                return tier
        return "STANDARD"
