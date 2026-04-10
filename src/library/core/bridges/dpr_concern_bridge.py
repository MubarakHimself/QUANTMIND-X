"""
QuantMindLib V1 — DPR Concern Bridge

Phase 10 Packet 10C: DPRConcernTag + DPRConcernEmitter
Emits concern tags when DPR tier changes or concern thresholds are crossed.
These tags flow to Governor for session eligibility decisions.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.library.core.bridges.sentinel_dpr_bridges import DPRBridge
from src.library.core.bridges.registry_journal_bridges import RegistryBridge


# DPR tier ordering for computing downgrade depth
_DPR_TIER_ORDER: Dict[str, int] = {
    "ELITE": 0,
    "PERFORMING": 1,
    "STANDARD": 2,
    "AT_RISK": 3,
    "CIRCUIT_BROKEN": 4,
}


class DPRConcernTag(BaseModel):
    """
    A concern tag emitted when DPR tier changes or concern threshold is crossed.
    """

    bot_id: str
    session_id: str
    concern_level: str  # "INFO" | "WARN" | "ALERT" | "CRITICAL"
    tag: str  # e.g., "@session_concern", "@dpr_downgrade", "@circuit_breaker"
    reason: str
    previous_tier: Optional[str]
    current_tier: str
    emitted_at_ms: int
    dpr_score: float


class DPRConcernEmitter:
    """
    Emits concern tags when DPR events occur.

    DPR concern -> @session_concern tag flow:
    - Tier downgrade (ELITE->AT_RISK, etc.) -> @session_concern tag
    - Score below threshold -> @dpr_downgrade tag
    - Circuit breaker trigger -> @circuit_breaker tag
    - Concern event -> @session_concern tag

    These tags flow to Governor for session eligibility decisions.
    """

    def __init__(
        self,
        dpr_bridge: Optional[DPRBridge] = None,
        registry_bridge: Optional[RegistryBridge] = None,
    ) -> None:
        self._dpr_bridge = dpr_bridge or DPRBridge()
        self._registry_bridge = registry_bridge or RegistryBridge()
        self._concern_log: List[DPRConcernTag] = []

    def emit_tier_change(
        self,
        bot_id: str,
        previous_tier: str,
        current_tier: str,
        dpr_score: float,
    ) -> DPRConcernTag:
        """
        Emit a concern tag on tier change.

        Tags:
        - ELITE->AT_RISK: "@session_concern" + "@dpr_downgrade"
        - PERFORMING/STANDARD->AT_RISK: "@session_concern" + "@dpr_downgrade"
        - Any->CIRCUIT_BROKEN: "@session_concern" + "@circuit_breaker"
        - AT_RISK->CIRCUIT_BROKEN: "@circuit_breaker"

        concern_level:
        - downgrade 1 tier -> "WARN"
        - downgrade 2+ tiers -> "ALERT"
        - ->CIRCUIT_BROKEN -> "CRITICAL"
        """
        # Same tier: no emission
        if previous_tier == current_tier:
            # Find and return the most recent concern for this bot, or a no-op tag
            recent = [
                c for c in self._concern_log
                if c.bot_id == bot_id
            ]
            if recent:
                return recent[-1]
            # Return a no-op INFO tag if nothing exists
            noop = DPRConcernTag(
                bot_id=bot_id,
                session_id="",
                concern_level="INFO",
                tag="@no_op",
                reason=f"No tier change: {previous_tier} -> {current_tier}",
                previous_tier=previous_tier,
                current_tier=current_tier,
                emitted_at_ms=int(time.time() * 1000),
                dpr_score=dpr_score,
            )
            self._concern_log.append(noop)
            return noop

        # Determine concern level and tags
        prev_order = _DPR_TIER_ORDER.get(previous_tier, 99)
        curr_order = _DPR_TIER_ORDER.get(current_tier, 99)
        tier_drop = abs(prev_order - curr_order)

        if current_tier == "CIRCUIT_BROKEN":
            concern_level = "CRITICAL"
        elif tier_drop >= 2:
            concern_level = "ALERT"
        elif tier_drop >= 1:
            concern_level = "WARN"
        else:
            concern_level = "INFO"

        # Determine primary tag
        if current_tier == "CIRCUIT_BROKEN":
            if previous_tier == "AT_RISK":
                tag = "@circuit_breaker"
            else:
                tag = "@circuit_breaker"
        elif concern_level in ("WARN", "ALERT"):
            tag = "@session_concern"
        else:
            tag = "@session_concern"

        reason = f"tier change: {previous_tier} -> {current_tier} (score={dpr_score:.3f})"

        # Fetch session_id from registry if available
        session_id = ""
        record = self._registry_bridge.get_record(bot_id)
        if record:
            session_id = getattr(record, "session_id", "") or ""

        emitted = DPRConcernTag(
            bot_id=bot_id,
            session_id=session_id,
            concern_level=concern_level,
            tag=tag,
            reason=reason,
            previous_tier=previous_tier,
            current_tier=current_tier,
            emitted_at_ms=int(time.time() * 1000),
            dpr_score=dpr_score,
        )
        self._concern_log.append(emitted)

        # Emit second tag for ELITE->AT_RISK / PERFORMING->AT_RISK
        if (
            previous_tier in ("ELITE", "PERFORMING", "STANDARD")
            and current_tier == "AT_RISK"
        ):
            second_tag = DPRConcernTag(
                bot_id=bot_id,
                session_id=session_id,
                concern_level=concern_level,
                tag="@dpr_downgrade",
                reason=reason,
                previous_tier=previous_tier,
                current_tier=current_tier,
                emitted_at_ms=int(time.time() * 1000),
                dpr_score=dpr_score,
            )
            self._concern_log.append(second_tag)

        # Emit second tag for Any->CIRCUIT_BROKEN (when not AT_RISK->CIRCUIT_BROKEN)
        if current_tier == "CIRCUIT_BROKEN" and previous_tier != "AT_RISK":
            second_tag = DPRConcernTag(
                bot_id=bot_id,
                session_id=session_id,
                concern_level=concern_level,
                tag="@session_concern",
                reason=reason,
                previous_tier=previous_tier,
                current_tier=current_tier,
                emitted_at_ms=int(time.time() * 1000),
                dpr_score=dpr_score,
            )
            self._concern_log.append(second_tag)

        return emitted

    def emit_concern_event(
        self,
        bot_id: str,
        session_id: str,
        concern_level: str,  # "INFO" | "WARN" | "ALERT" | "CRITICAL"
        reason: str,
        dpr_score: Optional[float] = None,
    ) -> DPRConcernTag:
        """
        Emit a concern event tag (general purpose).
        """
        if dpr_score is None:
            # Try to look up the current score from DPRBridge
            score_obj = self._dpr_bridge.get_score(bot_id)
            dpr_score = score_obj.dpr_score if score_obj else 0.0

        # Get current tier from DPRBridge if available
        current_tier = "STANDARD"
        score_obj = self._dpr_bridge.get_score(bot_id)
        if score_obj:
            current_tier = score_obj.tier

        emitted = DPRConcernTag(
            bot_id=bot_id,
            session_id=session_id,
            concern_level=concern_level,
            tag="@session_concern",
            reason=reason,
            previous_tier=None,
            current_tier=current_tier,
            emitted_at_ms=int(time.time() * 1000),
            dpr_score=dpr_score,
        )
        self._concern_log.append(emitted)
        return emitted

    def get_recent_concerns(
        self, bot_id: Optional[str] = None
    ) -> List[DPRConcernTag]:
        """
        Get recent concern tags, optionally filtered by bot.
        """
        if bot_id is None:
            return list(self._concern_log)
        return [c for c in self._concern_log if c.bot_id == bot_id]

    def get_critical_concerns(self) -> List[DPRConcernTag]:
        """Get all CRITICAL level concerns."""
        return [c for c in self._concern_log if c.concern_level == "CRITICAL"]

    def clear_old_concerns(self, older_than_ms: int = 3600000) -> int:
        """
        Remove concerns older than threshold. Returns count removed.
        """
        now_ms = int(time.time() * 1000)
        before = len(self._concern_log)
        self._concern_log = [
            c for c in self._concern_log
            if (now_ms - c.emitted_at_ms) <= older_than_ms
        ]
        return before - len(self._concern_log)


__all__ = ["DPRConcernTag", "DPRConcernEmitter"]
