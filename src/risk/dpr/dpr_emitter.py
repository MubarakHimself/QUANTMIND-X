"""
DPR → SSL Event Emitter — Translates DPR circuit breaks to SSL state transitions.

Phase 11F: DPR→SSL bridge for unified safety gating.

When DPR detects that a bot should circuit-break, this emitter translates that
decision into an SSL state transition event that SSLCircuitBreaker can emit
via Redis, completing the DPR↔SSL bidirectional integration loop.

DPR conditions that trigger SSL transitions:
- DPR tier CIRCUIT_BROKEN → SSL move_to_paper
- DPR score < 0.3 for LIVE bot → SSL move_to_paper
- DPR AT_RISK for > 3 consecutive checks + LIVE bot → SSL move_to_paper
- DPR rank > 10 with AT_RISK + LIVE bot → SSL move_to_paper
- Recovery from any of the above → SSL recovery_step_1 / recovery_confirmed

Per NFR-M2: DPR is synchronous — NO LLM calls in circuit-breaking path.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Set

from src.risk.ssl.circuit_breaker import SSLCircuitBreaker
from src.events.ssl import SSLCircuitBreakerEvent, SSLState, SSLEventType, TradeOutcome
from src.events.dpr import SSLEvent, SSLEventType as DPRSSLEventType
from src.risk.dpr.scoring_engine import DPRScore


logger = logging.getLogger(__name__)


class DPRSSLEmitter:
    """
    DPR → SSL bridge: translates DPR circuit breaks into SSL state transitions.

    This is the reverse direction of DPRSSLConsumer (SSL→DPR). Where DPRSSLConsumer
    subscribes to Redis ssl:events and routes SSL state changes into DPR system,
    DPRSSLEmitter observes DPR circuit-breaker decisions and triggers equivalent
    SSL state transitions.

    Key behaviors:
    - Only transitions LIVE bots to paper (does not re-paper already-paper bots)
    - Skips transition if bot already in SSL state other than LIVE
    - Tracks which bots have been transitioned to avoid duplicate events
    - Logs all transitions to DPR audit trail via SSLCircuitBreaker

    Usage:
        emitter = DPRSSLEmitter(ssl_breaker=my_ssl_breaker)
        # Call after DPR scoring to catch DPR-triggered circuit breaks
        result = monitor.check_bot_circuit_state(bot_id, dpr_score)
        if not result.allowed:
            emitter.emit_dpr_to_ssl(bot_id, dpr_score, reason=result.triggered_by)
    """

    # DPR circuit-breaker trigger types that warrant SSL state transition
    _TRIGGERS_SSL_TRANSITION: frozenset[str] = frozenset([
        "DPR_CIRCUIT_BREAKER",        # DPR tier CIRCUIT_BROKEN
        "DPR_SCORE_THRESHOLD",         # DPR score < 0.3
        "DPR_AT_RISK_CONSECUTIVE",      # AT_RISK for > 3 checks
        "DPR_RANK_AT_RISK",            # Rank > 10 with AT_RISK
    ])

    def __init__(
        self,
        ssl_breaker: Optional[SSLCircuitBreaker] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ) -> None:
        """
        Initialize DPR SSL Emitter.

        Args:
            ssl_breaker: SSLCircuitBreaker instance for state transitions
            redis_host: Redis host for SSL event publishing
            redis_port: Redis port for SSL event publishing
        """
        self._ssl_breaker = ssl_breaker or SSLCircuitBreaker(
            redis_host=redis_host,
            redis_port=redis_port,
        )
        # Track bots we've already transitioned from LIVE → PAPER via DPR
        # key: bot_id, value: reason for transition
        self._transitioned_to_paper: dict[str, str] = {}
        # Track bots we've already transitioned from PAPER → RECOVERY via DPR
        self._transitioned_to_recovery: dict[str, str] = {}

    def emit_dpr_to_ssl(
        self,
        bot_id: str,
        dpr_score: "DPRScore",
        reason: Optional[str] = None,
    ) -> Optional[SSLCircuitBreakerEvent]:
        """
        Translate a DPR circuit-break decision into an SSL state transition.

        Only transitions bots that are currently LIVE in SSL state.
        Bots already in PAPER, RECOVERY, or RETIRED are skipped (SSL is
        authoritative for those states).

        Args:
            bot_id: Bot identifier
            dpr_score: DPRScore object with tier, score, rank
            reason: DPR circuit-breaker trigger type (e.g. "DPR_CIRCUIT_BROKEN")

        Returns:
            SSLCircuitBreakerEvent if transition was emitted, None if skipped
        """
        if not reason or reason not in self._TRIGGERS_SSL_TRANSITION:
            return None

        # Check if already transitioned via DPR this session
        if bot_id in self._transitioned_to_paper:
            logger.debug(
                f"DPR→SSL: bot {bot_id} already transitioned to paper "
                f"(reason: {self._transitioned_to_paper[bot_id]}), skipping"
            )
            return None

        # Check current SSL state — only transition LIVE bots
        current_ssl_state = self._ssl_breaker.state_manager.get_state(bot_id)
        if current_ssl_state is None:
            # Bot has no SSL record — treat as LIVE and create state
            current_ssl_state = SSLState.LIVE.value
        elif current_ssl_state != SSLState.LIVE.value:
            logger.debug(
                f"DPR→SSL: bot {bot_id} in SSL state {current_ssl_state} "
                f"(not LIVE), DPR transition deferred to SSL authority"
            )
            return None

        # Verify DPR conditions are severe enough
        tier = dpr_score.tier
        score = dpr_score.dpr_score

        should_transition = False
        transition_reason = reason

        if tier == "CIRCUIT_BROKEN":
            should_transition = True
        elif score < 0.3:
            should_transition = True
        elif reason in ("DPR_AT_RISK_CONSECUTIVE", "DPR_RANK_AT_RISK"):
            # Only transition AT_RISK bots that are LIVE
            should_transition = True
        else:
            return None

        if not should_transition:
            return None

        # Mark as transitioned
        self._transitioned_to_paper[bot_id] = reason or "unknown"

        try:
            # Get threshold for the bot (2 for scalping, 3 for ORB)
            threshold = self._ssl_breaker._get_threshold(bot_id)
            magic_number = (
                self._ssl_breaker.state_manager.get_magic_number(bot_id) or ""
            )

            # Build trade outcome with DPR context
            trade_outcome = TradeOutcome(
                pnl=float(dpr_score.daily_pnl or 0),
                win_rate=float(dpr_score.win_rate_today or 0),
                ev_per_trade=0.0,  # EV not available from DPRScore
                session_id="DPR_CIRCUIT_BREAK",
            )

            # Create SSL event: LIVE → PAPER
            event = SSLCircuitBreakerEvent(
                bot_id=bot_id,
                magic_number=magic_number,
                event_type=SSLEventType.MOVE_TO_PAPER,
                consecutive_losses=threshold,
                tier=self._ssl_breaker._determine_tier(bot_id).value,
                previous_state=SSLState.LIVE,
                new_state=SSLState.PAPER,
                trade_outcome=trade_outcome,
                metadata={
                    "dpr_tier": tier,
                    "dpr_score": float(score),
                    "dpr_rank": dpr_score.rank,
                    "dpr_trigger": reason,
                    "transition_source": "dpr_emitter",
                },
            )

            # Emit via SSLCircuitBreaker (persists to DB + publishes to Redis)
            self._ssl_breaker._emit_event(event)

            logger.warning(
                f"DPR→SSL: bot {bot_id} moved LIVE→PAPER via DPR "
                f"(tier={tier}, score={score:.3f}, trigger={reason})"
            )

            return event

        except Exception as e:
            logger.error(
                f"DPR→SSL: failed to emit transition for bot {bot_id}: {e}"
            )
            self._transitioned_to_paper.pop(bot_id, None)
            return None

    def emit_recovery_to_ssl(
        self,
        bot_id: str,
        dpr_score: "DPRScore",
    ) -> Optional[SSLCircuitBreakerEvent]:
        """
        Translate DPR recovery (bot improving back to STANDARD or PERFORMING)
        into an SSL recovery_step_1 event.

        Called when a previously DPR-blocked bot now passes circuit checks.
        Only transitions bots that were moved to paper via DPR.

        Args:
            bot_id: Bot identifier
            dpr_score: DPRScore showing improvement

        Returns:
            SSLCircuitBreakerEvent if recovery was emitted, None if skipped
        """
        current_ssl_state = self._ssl_breaker.state_manager.get_state(bot_id)

        if current_ssl_state is None or current_ssl_state == SSLState.LIVE.value:
            # Bot was never in paper via DPR, or is already live
            return None

        if current_ssl_state != SSLState.PAPER.value:
            # Only handle PAPER → RECOVERY
            return None

        # Check if this paper was caused by DPR
        if bot_id not in self._transitioned_to_paper:
            # Was not a DPR-triggered paper transition — let SSL handle recovery
            return None

        tier = dpr_score.tier
        if tier not in ("STANDARD", "PERFORMING", "ELITE"):
            # Not yet recovered in DPR
            return None

        # Emit recovery_step_1
        try:
            magic_number = (
                self._ssl_breaker.state_manager.get_magic_number(bot_id) or ""
            )
            current_losses = (
                self._ssl_breaker.state_manager.get_consecutive_losses(bot_id)
            )

            event = SSLCircuitBreakerEvent(
                bot_id=bot_id,
                magic_number=magic_number,
                event_type=SSLEventType.RECOVERY_STEP_1,
                consecutive_losses=current_losses,
                tier=self._ssl_breaker._determine_tier(bot_id).value,
                previous_state=SSLState.PAPER,
                new_state=SSLState.RECOVERY,
                recovery_win_count=1,
                metadata={
                    "dpr_tier": tier,
                    "dpr_score": float(dpr_score.dpr_score),
                    "dpr_rank": dpr_score.rank,
                    "recovery_source": "dpr_emitter",
                },
            )

            self._ssl_breaker._emit_event(event)

            # Mark as transitioned to recovery
            self._transitioned_to_paper.pop(bot_id, None)
            self._transitioned_to_recovery[bot_id] = "recovery_confirmed"

            logger.info(
                f"DPR→SSL: bot {bot_id} moved PAPER→RECOVERY via DPR "
                f"(tier={tier}, score={dpr_score.dpr_score:.3f})"
            )

            return event

        except Exception as e:
            logger.error(
                f"DPR→SSL: failed to emit recovery for bot {bot_id}: {e}"
            )
            return None

    def get_transitioned_bots(self) -> dict[str, str]:
        """Return bots currently marked as transitioned to paper via DPR."""
        return dict(self._transitioned_to_paper)

    def is_bot_transitioned(self, bot_id: str) -> bool:
        """Return True if bot was transitioned to paper via DPR this session."""
        return bot_id in self._transitioned_to_paper

    def reset(self) -> None:
        """Clear all transition tracking (call at session start)."""
        self._transitioned_to_paper.clear()
        self._transitioned_to_recovery.clear()
