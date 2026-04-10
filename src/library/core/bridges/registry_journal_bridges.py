"""
QuantMindLib V1 — RegistryBridge + JournalBridge

Phase 3 (Bridge Definitions) of QuantMindLib V1 packet delivery.
Packet 3B: Registry operations + append-only trade journal.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from src.library.core.types.enums import (
    ActivationState,
    BotHealth,
    BotTier,
    RegistryStatus,
)
from src.library.core.domain.registry_record import RegistryRecord
from src.library.core.domain.bot_spec import BotRuntimeProfile


class RegistryBridge(BaseModel):
    """
    Manages bot registry operations.
    Tracks registration state and maps RegistryStatus -> ActivationState.
    """

    records: Dict[str, RegistryRecord] = Field(default_factory=dict)

    def register(
        self,
        bot_id: str,
        bot_spec_id: str,
        owner: str,
        variant_ids: Optional[List[str]] = None,
        tier: BotTier = BotTier.STANDARD,
    ) -> RegistryRecord:
        """Register a new bot. Creates a RegistryRecord with ACTIVE status."""
        now_ms = int(time.time() * 1000)
        record = RegistryRecord(
            bot_id=bot_id,
            bot_spec_id=bot_spec_id,
            status=RegistryStatus.ACTIVE,
            tier=tier,
            registered_at_ms=now_ms,
            last_updated_ms=now_ms,
            owner=owner,
            variant_ids=variant_ids or [],
        )
        self.records[bot_id] = record
        return record

    def deactivate(self, bot_id: str) -> bool:
        """Deactivate a bot — sets status to ARCHIVED."""
        if bot_id not in self.records:
            return False
        record = self.records[bot_id]
        record.status = RegistryStatus.ARCHIVED
        record.last_updated_ms = int(time.time() * 1000)
        return True

    def get_record(self, bot_id: str) -> Optional[RegistryRecord]:
        """Get registry record for a bot."""
        return self.records.get(bot_id)

    def get_active_bots(self) -> List[RegistryRecord]:
        """Return all bots with ACTIVE status."""
        return [r for r in self.records.values() if r.status == RegistryStatus.ACTIVE]

    def to_runtime_profile(
        self, bot_id: str, deployment_target: str = "LIVE"
    ) -> Optional[BotRuntimeProfile]:
        """
        Derive a BotRuntimeProfile from a RegistryRecord.
        Maps RegistryStatus -> ActivationState:
          ACTIVE   -> ACTIVE
          TRIAL    -> CAUTIOUS
          ARCHIVED -> INACTIVE
          SUSPENDED -> DEGRADED
        Maps BotTier -> health:
          ELITE               -> HEALTHY
          PERFORMANCE_TEST    -> HEALTHY
          STANDARD            -> HEALTHY
          EVALUATION_CANDIDATE -> CAUTIOUS
          AT_RISK             -> DEGRADED
          CIRCUIT_BROKEN      -> CRITICAL
        """
        record = self.records.get(bot_id)
        if not record:
            return None

        # RegistryStatus -> ActivationState
        status_map = {
            RegistryStatus.ACTIVE: ActivationState.ACTIVE,
            RegistryStatus.TRIAL: ActivationState.CAUTIOUS,
            RegistryStatus.ARCHIVED: ActivationState.INACTIVE,
            RegistryStatus.SUSPENDED: ActivationState.DEGRADED,
        }

        # BotTier -> BotHealth
        health_map = {
            BotTier.ELITE: BotHealth.HEALTHY,
            BotTier.PERFORMANCE_TEST: BotHealth.HEALTHY,
            BotTier.STANDARD: BotHealth.HEALTHY,
            BotTier.EVALUATION_CANDIDATE: BotHealth.CAUTIOUS,
            BotTier.AT_RISK: BotHealth.DEGRADED,
            BotTier.CIRCUIT_BROKEN: BotHealth.CRITICAL,
        }

        return BotRuntimeProfile(
            bot_id=bot_id,
            activation_state=status_map.get(record.status, ActivationState.UNKNOWN),
            deployment_target=deployment_target,
            health=health_map.get(record.tier, BotHealth.UNKNOWN),
            session_eligibility={},
            dpr_ranking=0,
            dpr_score=0.0,
            report_ids=[],
        )

    def get_tier_bots(self, tier: BotTier) -> List[RegistryRecord]:
        """Return all bots of a given tier."""
        return [r for r in self.records.values() if r.tier == tier]


class JournalEntry(BaseModel):
    """
    Single journal event — append-only trade log entry.
    """

    entry_id: str
    bot_id: str
    session_id: str
    event_type: str  # "FILL" | "PNL" | "SESSION_OPEN" | "SESSION_CLOSE" | "SESSION_BLACKOUT"
    timestamp_ms: int
    description: str = ""
    pnl_delta: Optional[float] = None  # None for non-P&L entries
    filled_qty: Optional[float] = None
    price: Optional[float] = None
    order_id: Optional[str] = None

    def is_profitable(self) -> bool:
        """True if this entry represents a profitable event."""
        return self.pnl_delta is not None and self.pnl_delta > 0


class JournalBridge(BaseModel):
    """
    Append-only trade journal.
    Maintains an ordered log of all trade events per bot per session.
    """

    entries: List[JournalEntry] = Field(default_factory=list)
    _by_bot: Dict[str, List[JournalEntry]] = PrivateAttr(default_factory=dict)

    def log_fill(
        self,
        bot_id: str,
        session_id: str,
        order_id: str,
        filled_qty: float,
        price: float,
        pnl_delta: Optional[float] = None,
    ) -> JournalEntry:
        """Log a fill event."""
        entry = JournalEntry(
            entry_id=f"{bot_id}:{session_id}:{int(time.time() * 1000)}",
            bot_id=bot_id,
            session_id=session_id,
            event_type="FILL",
            timestamp_ms=int(time.time() * 1000),
            description=f"Fill {order_id}",
            pnl_delta=pnl_delta,
            filled_qty=filled_qty,
            price=price,
            order_id=order_id,
        )
        self.entries.append(entry)
        if bot_id not in self._by_bot:
            self._by_bot[bot_id] = []
        self._by_bot[bot_id].append(entry)
        return entry

    def log_pnl(
        self,
        bot_id: str,
        session_id: str,
        pnl_delta: float,
        description: str = "",
    ) -> JournalEntry:
        """Log a P&L event."""
        entry = JournalEntry(
            entry_id=f"{bot_id}:{session_id}:{int(time.time() * 1000)}",
            bot_id=bot_id,
            session_id=session_id,
            event_type="PNL",
            timestamp_ms=int(time.time() * 1000),
            description=description,
            pnl_delta=pnl_delta,
        )
        self.entries.append(entry)
        if bot_id not in self._by_bot:
            self._by_bot[bot_id] = []
        self._by_bot[bot_id].append(entry)
        return entry

    def log_session(
        self,
        bot_id: str,
        session_id: str,
        event_type: str,  # "SESSION_OPEN" | "SESSION_CLOSE" | "SESSION_BLACKOUT"
        description: str = "",
    ) -> JournalEntry:
        """Log a session lifecycle event."""
        entry = JournalEntry(
            entry_id=f"{bot_id}:{session_id}:{int(time.time() * 1000)}",
            bot_id=bot_id,
            session_id=session_id,
            event_type=event_type,
            timestamp_ms=int(time.time() * 1000),
            description=description,
        )
        self.entries.append(entry)
        if bot_id not in self._by_bot:
            self._by_bot[bot_id] = []
        self._by_bot[bot_id].append(entry)
        return entry

    def get_bot_entries(self, bot_id: str) -> List[JournalEntry]:
        """Get all journal entries for a bot."""
        return list(self._by_bot.get(bot_id, []))

    def get_session_entries(self, bot_id: str, session_id: str) -> List[JournalEntry]:
        """Get all journal entries for a bot in a specific session."""
        return [e for e in self._by_bot.get(bot_id, []) if e.session_id == session_id]

    def get_total_pnl(self, bot_id: str) -> float:
        """Sum all P&L deltas for a bot."""
        return sum(e.pnl_delta for e in self._by_bot.get(bot_id, []) if e.pnl_delta is not None)

    def get_profitable_entries(self, bot_id: str) -> List[JournalEntry]:
        """Return entries where pnl_delta > 0."""
        return [e for e in self._by_bot.get(bot_id, []) if e.is_profitable()]


__all__ = ["RegistryBridge", "JournalEntry", "JournalBridge"]
