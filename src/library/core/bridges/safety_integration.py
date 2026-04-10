"""
QuantMindLib V1 — Safety Integration
Phase 10 Packet 10B: SafetyHooks integration with DPR circuit breaker concerns.
Phase 10 Packet 10C: SSLCircuitBreaker DPR monitor integration.
"""
from __future__ import annotations

import time
from typing import Dict, Optional

from pydantic import BaseModel, Field

from src.library.core.bridges.sentinel_dpr_bridges import DPRBridge, DPRScore
from src.library.core.bridges.ssl_dpr_integration import (
    SSLCircuitBreakerDPRMonitor,
    SSL_HALTED,
    SSL_ACTIVE,
)
from src.library.runtime.safety_hooks import SafetyHooks, KillSwitchResult


# DPR tier ordering for comparisons
_DPR_TIER_ORDER: Dict[str, int] = {
    "ELITE": 0,
    "PERFORMING": 1,
    "STANDARD": 2,
    "AT_RISK": 3,
    "CIRCUIT_BROKEN": 4,
}

# Concern level gating: what each level means for trading
_CONCERN_GATE: Dict[str, Dict[str, bool]] = {
    "LOW": {"allow_new": True, "allow_existing": True, "block_all": False},
    "MEDIUM": {"allow_new": True, "allow_existing": True, "block_all": False},
    "HIGH": {"allow_new": False, "allow_existing": True, "block_all": False},
    "CRITICAL": {"allow_new": False, "allow_existing": False, "block_all": True},
}


class DPRCircuitBreakerMonitor:
    """
    Monitors DPR tier and circuit breaker state to inform SafetyHooks.

    DPR concern events → @session_concern tag flow:
    - When DPR tier drops below STANDARD, SafetyHooks blocks new trades
    - When DPR circuit breaker triggers, all trades are halted
    - DPR concern events are translated into session tags for Governor
    """

    def __init__(
        self,
        dpr_bridge: Optional[DPRBridge] = None,
        safety_hooks: Optional[SafetyHooks] = None,
        ssl_monitor: Optional[SSLCircuitBreakerDPRMonitor] = None,
    ) -> None:
        self._dpr_bridge = dpr_bridge or DPRBridge()
        self._safety_hooks = safety_hooks or SafetyHooks()
        self._ssl_monitor = ssl_monitor or SSLCircuitBreakerDPRMonitor()
        self._blocked_bots: Dict[str, str] = {}  # bot_id -> reason
        self._last_check_ms: int = Field(
            default_factory=lambda: int(time.time() * 1000)
        )
        # Track consecutive AT_RISK checks per bot
        self._at_risk_counts: Dict[str, int] = {}

    def check_bot_circuit_state(
        self,
        bot_id: str,
        dpr_score: Optional[DPRScore] = None,
    ) -> KillSwitchResult:
        """
        Check if a bot should be circuit-broken based on DPR score.

        Circuit breaking logic:
        - DPR tier CIRCUIT_BROKEN → blocked
        - DPR tier AT_RISK for > 3 consecutive checks → blocked
        - DPR score < 0.3 → blocked
        - DPR rank > 10 with AT_RISK tier → blocked

        Returns KillSwitchResult indicating if blocked.
        """
        self._last_check_ms = int(time.time() * 1000)

        # Unknown bot: no DPR score available, pass with caution
        if dpr_score is None:
            return KillSwitchResult(
                allowed=True,
                reason=f"{bot_id}: No DPR score available — pass with caution",
            )

        tier = dpr_score.tier
        score = dpr_score.dpr_score
        rank = dpr_score.rank

        # Circuit 1: DPR tier CIRCUIT_BROKEN
        if tier == "CIRCUIT_BROKEN":
            self._block_bot(bot_id, "DPR tier CIRCUIT_BROKEN")
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Circuit broken — DPR tier CIRCUIT_BROKEN (score={score:.3f})",
                triggered_by="DPR_CIRCUIT_BREAKER",
            )

        # Circuit 2: DPR score < 0.3
        if score < 0.3:
            self._block_bot(bot_id, f"DPR score {score:.3f} below threshold 0.3")
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Circuit broken — DPR score {score:.3f} < 0.3",
                triggered_by="DPR_SCORE_THRESHOLD",
            )

        # Circuit 3: AT_RISK for > 3 consecutive checks
        if tier == "AT_RISK":
            self._at_risk_counts[bot_id] = self._at_risk_counts.get(bot_id, 0) + 1
            if self._at_risk_counts[bot_id] > 3:
                self._block_bot(bot_id, "AT_RISK for > 3 consecutive checks")
                return KillSwitchResult(
                    allowed=False,
                    reason=(
                        f"{bot_id}: Circuit broken — AT_RISK "
                        f"for {self._at_risk_counts[bot_id]} consecutive checks"
                    ),
                    triggered_by="DPR_AT_RISK_CONSECUTIVE",
                )
        else:
            # Reset count on non-AT_RISK tier
            self._at_risk_counts[bot_id] = 0

        # Circuit 4: DPR rank > 10 with AT_RISK tier
        if rank > 10 and tier == "AT_RISK":
            self._block_bot(bot_id, f"DPR rank {rank} > 10 with AT_RISK tier")
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Circuit broken — rank {rank} > 10 with AT_RISK tier",
                triggered_by="DPR_RANK_AT_RISK",
            )

        # If previously blocked, check if now eligible to unblock
        self._unblock_bot_if_present(bot_id)

        # All circuits passed
        return KillSwitchResult(
            allowed=True,
            reason=f"{bot_id}: DPR circuit state OK — tier={tier}, score={score:.3f}, rank={rank}",
        )

    def check_concern_events(
        self,
        bot_id: str,
        concern_level: str,  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    ) -> KillSwitchResult:
        """
        Translate DPR concern events into safety gates.

        concern levels:
        - LOW: pass with warning
        - MEDIUM: pass with caution flag
        - HIGH: block new entries, allow existing
        - CRITICAL: block all trades
        """
        valid_levels = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if concern_level not in valid_levels:
            return KillSwitchResult(
                allowed=True,
                reason=f"{bot_id}: Unknown concern level '{concern_level}' — passing",
            )

        gate = _CONCERN_GATE[concern_level]

        if gate["block_all"]:
            self._block_bot(bot_id, f"CRITICAL DPR concern event")
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: Blocked — CRITICAL DPR concern event",
                triggered_by="DPR_CONCERN_CRITICAL",
            )

        if not gate["allow_new"]:
            # HIGH concern: block new entries, allow existing
            self._block_bot(bot_id, f"HIGH DPR concern — new entries blocked")
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: DPR concern HIGH — new entries blocked, existing positions allowed",
                triggered_by="DPR_CONCERN_HIGH",
            )

        if not gate["allow_existing"]:
            # Should not happen with current gate definitions, but handle gracefully
            return KillSwitchResult(
                allowed=False,
                reason=f"{bot_id}: DPR concern {concern_level} — all trades blocked",
                triggered_by=f"DPR_CONCERN_{concern_level}",
            )

        # LOW or MEDIUM: allowed with warning
        warning_suffix = " — caution flagged" if concern_level == "MEDIUM" else ""
        return KillSwitchResult(
            allowed=True,
            reason=f"{bot_id}: DPR concern {concern_level} — passed{warning_suffix}",
        )

    def get_blocked_bots(self) -> Dict[str, str]:
        """Return dict of bot_id -> block reason for all blocked bots."""
        return dict(self._blocked_bots)

    def unblock_bot(self, bot_id: str) -> bool:
        """Remove a bot from the blocked list. Returns True if was blocked."""
        if bot_id in self._blocked_bots:
            del self._blocked_bots[bot_id]
            self._at_risk_counts.pop(bot_id, None)
            return True
        return False

    def get_bot_concern_state(self, bot_id: str) -> Optional[str]:
        """Return concern state for a bot if tracked, else None."""
        if bot_id in self._blocked_bots:
            reason = self._blocked_bots[bot_id]
            if "CIRCUIT_BROKEN" in reason:
                return "CIRCUIT_BROKEN"
            elif "AT_RISK" in reason or "AT_RISK" in self._blocked_bots.get(bot_id, ""):
                return "AT_RISK"
            elif "CRITICAL" in reason:
                return "CRITICAL"
            elif "HIGH" in reason:
                return "HIGH"
        return None

    def check_ssl_dpr_combined(
        self,
        bot_id: str,
        dpr_score: Optional[DPRScore],
        ssl_state: str,
        ssl_tier: Optional[str] = None,
    ) -> KillSwitchResult:
        """
        Check both DPR and SSL circuits simultaneously for a bot.

        Runs the DPR tier circuit check first, then the SSL DPR combined
        check, and returns the most restrictive result. Also updates
        the SSL monitor's blocked-bot registry when SSL causes a block.

        Decision order:
        1. DPR CIRCUIT_BROKEN or score < 0.3 → block (DPR wins).
        2. SSL HALTED → block regardless of DPR.
        3. AT_RISK DPR + SSL ACTIVE → immediate circuit break.
        4. SSL ACTIVE → block new entries.
        5. DPR AT_RISK consecutive → block.
        6. SSL RECOVERY → allow with caution.
        7. Otherwise → DPR result (allow with caution if AT_RISK).

        Args:
            bot_id: Bot identifier.
            dpr_score: DPR score data (optional).
            ssl_state: SSL state string (SSLState value or SSL_HALTED/SSL_ACTIVE).
            ssl_tier: Paper trading tier ("TIER_1" | "TIER_2" | None).

        Returns:
            KillSwitchResult with the most restrictive combined decision.
        """
        dpr_tier = dpr_score.tier if dpr_score else "STANDARD"

        # Delegate to SSL DPR monitor for combined SSL+DPR logic.
        ssl_result = self._ssl_monitor.check_ssl_dpr_state(
            bot_id=bot_id,
            ssl_state=ssl_state,
            ssl_tier=ssl_tier or "TIER_2",
            dpr_tier=dpr_tier,
        )

        # Sync SSL-blocked bots to DPR blocked registry for observability.
        if not ssl_result.allowed and ssl_result.triggered_by:
            self._block_bot(bot_id, ssl_result.reason)

        # Run DPR circuit check independently.
        dpr_result = self.check_bot_circuit_state(bot_id, dpr_score)

        # If DPR blocked, DPR wins (harder constraint).
        if not dpr_result.allowed:
            # Ensure blocked by DPR is reflected in SSL monitor too.
            if ssl_state == SSL_HALTED or self._ssl_monitor.is_halted(bot_id):
                pass  # Already handled above.
            elif ssl_state == SSL_ACTIVE or self._ssl_monitor.is_active(bot_id):
                pass
            return dpr_result

        # If SSL blocked and DPR allowed, SSL result takes precedence.
        if not ssl_result.allowed:
            return ssl_result

        # Both passed: return the most informative result.
        # Prefer SSL RECOVERY caution flag over DPR result.
        if ssl_state == "recovery":
            return ssl_result

        return dpr_result

    @property
    def ssl_monitor(self) -> SSLCircuitBreakerDPRMonitor:
        """Return the SSL DPR monitor instance."""
        return self._ssl_monitor

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _block_bot(self, bot_id: str, reason: str) -> None:
        """Add or update a bot in the blocked list."""
        self._blocked_bots[bot_id] = reason

    def _unblock_bot_if_present(self, bot_id: str) -> None:
        """Remove a bot from the blocked list if it is present."""
        self._blocked_bots.pop(bot_id, None)