"""
Decline recovery workflow support.

This is a lightweight in-process implementation for the bot circuit breaker.
It provides the state machine expected by existing callers without creating a
second policy owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, Any


class DeclineState(str, Enum):
    NORMAL = "normal"
    DECLINING = "declining"
    RECOVERY = "recovery"


@dataclass
class DeclineRecord:
    bot_id: str
    state: DeclineState
    reason: Optional[str] = None
    regime_state: Optional[str] = None
    performance_delta: float = 0.0
    flagged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bot_id": self.bot_id,
            "state": self.state.value,
            "reason": self.reason,
            "regime_state": self.regime_state,
            "performance_delta": self.performance_delta,
            "flagged_at": self.flagged_at,
            "updated_at": self.updated_at,
        }


class DeclineRecoveryEngine:
    """
    Minimal decline/recovery tracker for bot-level workflow escalation.

    This intentionally stays local and simple until a fuller review pipeline is
    introduced. The circuit breaker only needs current-state lookup plus a way
    to flag bots into decline handling.
    """

    def __init__(self):
        self._records: Dict[str, DeclineRecord] = {}

    def get_decline_state(self, bot_id: str) -> DeclineState:
        record = self._records.get(bot_id)
        return record.state if record else DeclineState.NORMAL

    def flag_bot(
        self,
        bot_id: str,
        reason: str,
        regime_state: str = "UNKNOWN",
        performance_delta: float = 0.0,
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        existing = self._records.get(bot_id)
        state = DeclineState.DECLINING if existing is None else existing.state
        record = DeclineRecord(
            bot_id=bot_id,
            state=state,
            reason=reason,
            regime_state=regime_state,
            performance_delta=performance_delta,
            flagged_at=existing.flagged_at if existing else now,
            updated_at=now,
        )
        self._records[bot_id] = record
        return record.to_dict()

    def mark_recovery(self, bot_id: str, reason: str = "manual review complete") -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        record = self._records.get(bot_id)
        if record is None:
            record = DeclineRecord(bot_id=bot_id, state=DeclineState.RECOVERY, reason=reason, updated_at=now)
        else:
            record.state = DeclineState.RECOVERY
            record.reason = reason
            record.updated_at = now
        self._records[bot_id] = record
        return record.to_dict()

    def clear_bot(self, bot_id: str) -> bool:
        return self._records.pop(bot_id, None) is not None

    def get_record(self, bot_id: str) -> Optional[Dict[str, Any]]:
        record = self._records.get(bot_id)
        return record.to_dict() if record else None
