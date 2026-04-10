"""
QuantMindLib V1 — DPR Concern Bridge Tests

Phase 10 Packet 10C: DPRConcernEmitter tests.
"""
from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.library.core.bridges.dpr_concern_bridge import (
    DPRConcernTag,
    DPRConcernEmitter,
)
from src.library.core.bridges.sentinel_dpr_bridges import DPRScore, DPRBridge
from src.library.core.bridges.registry_journal_bridges import RegistryBridge


def _make_dpr_score(
    bot_id: str = "bot-001",
    score: float = 0.85,
    tier: str = "ELITE",
    rank: int = 1,
) -> DPRScore:
    return DPRScore(
        bot_id=bot_id,
        dpr_score=score,
        sharpe_today=score,
        win_rate_today=score,
        daily_pnl=score,
        rank=rank,
        tier=tier,
        computed_at_ms=int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# TestConcernEmitterInit
# ---------------------------------------------------------------------------

class TestConcernEmitterInit:
    def test_init_default_components(self):
        """Initializes with DPRBridge and RegistryBridge."""
        emitter = DPRConcernEmitter()
        assert emitter._dpr_bridge is not None
        assert isinstance(emitter._dpr_bridge, DPRBridge)
        assert emitter._registry_bridge is not None
        assert isinstance(emitter._registry_bridge, RegistryBridge)
        assert emitter._concern_log == []

    def test_init_custom_components(self):
        """Accepts custom DPRBridge and RegistryBridge."""
        dpr = DPRBridge()
        registry = RegistryBridge()
        emitter = DPRConcernEmitter(dpr_bridge=dpr, registry_bridge=registry)
        assert emitter._dpr_bridge is dpr
        assert emitter._registry_bridge is registry
        assert emitter._concern_log == []


# ---------------------------------------------------------------------------
# TestTierChangeEmission
# ---------------------------------------------------------------------------

class TestTierChangeEmission:
    def test_elite_to_at_risk_emits_alert(self):
        """ELITE->AT_RISK (3-tier drop) emits ALERT concern level."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_tier_change(
            bot_id="bot-1",
            previous_tier="ELITE",
            current_tier="AT_RISK",
            dpr_score=0.35,
        )
        assert result.concern_level == "ALERT"
        assert result.tag == "@session_concern"
        assert result.previous_tier == "ELITE"
        assert result.current_tier == "AT_RISK"
        assert result.dpr_score == 0.35
        # Should also emit a @dpr_downgrade tag
        all_concerns = emitter.get_recent_concerns()
        tags = [c.tag for c in all_concerns if c.bot_id == "bot-1"]
        assert "@dpr_downgrade" in tags

    def test_two_tier_drop_emits_alert(self):
        """Downgrade of 2+ tiers emits ALERT."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_tier_change(
            bot_id="bot-2",
            previous_tier="ELITE",
            current_tier="STANDARD",
            dpr_score=0.55,
        )
        assert result.concern_level == "ALERT"
        assert result.previous_tier == "ELITE"
        assert result.current_tier == "STANDARD"

    def test_circuit_broken_emits_critical(self):
        """Transition to CIRCUIT_BROKEN emits CRITICAL."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_tier_change(
            bot_id="bot-3",
            previous_tier="PERFORMING",
            current_tier="CIRCUIT_BROKEN",
            dpr_score=0.15,
        )
        assert result.concern_level == "CRITICAL"
        assert result.tag == "@circuit_breaker"
        assert result.current_tier == "CIRCUIT_BROKEN"
        # Should emit @session_concern as second tag
        all_concerns = emitter.get_recent_concerns()
        tags = [c.tag for c in all_concerns if c.bot_id == "bot-3"]
        assert "@session_concern" in tags

    def test_same_tier_no_emission(self):
        """Same tier emits no new concern (returns existing or no-op)."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_tier_change(
            bot_id="bot-4",
            previous_tier="STANDARD",
            current_tier="STANDARD",
            dpr_score=0.60,
        )
        # Should return a no-op tag since no change
        assert result.tag == "@no_op"
        assert result.concern_level == "INFO"

    def test_performing_to_standard_emits_nothing(self):
        """PERFORMING->STANDARD (1-tier drop) emits WARN."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_tier_change(
            bot_id="bot-5",
            previous_tier="PERFORMING",
            current_tier="STANDARD",
            dpr_score=0.65,
        )
        # 1-tier drop = WARN
        assert result.concern_level == "WARN"
        # PERFORMING->STANDARD is a downgrade, not to AT_RISK, so no @dpr_downgrade
        all_concerns = emitter.get_recent_concerns()
        tags = [c.tag for c in all_concerns if c.bot_id == "bot-5"]
        # Only @session_concern should be emitted, not @dpr_downgrade
        assert "@dpr_downgrade" not in tags


# ---------------------------------------------------------------------------
# TestConcernEventEmission
# ---------------------------------------------------------------------------

class TestConcernEventEmission:
    def test_general_emission(self):
        """emit_concern_event creates a general concern tag."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_concern_event(
            bot_id="bot-evt-1",
            session_id="sess-abc",
            concern_level="WARN",
            reason="High drawdown detected",
            dpr_score=0.45,
        )
        assert result.bot_id == "bot-evt-1"
        assert result.session_id == "sess-abc"
        assert result.concern_level == "WARN"
        assert result.tag == "@session_concern"
        assert result.reason == "High drawdown detected"
        assert result.dpr_score == 0.45

    def test_info_level_emission(self):
        """emit_concern_event supports INFO level."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_concern_event(
            bot_id="bot-evt-2",
            session_id="sess-def",
            concern_level="INFO",
            reason="Routine check passed",
            dpr_score=0.90,
        )
        assert result.concern_level == "INFO"
        assert result.dpr_score == 0.90

    def test_critical_level_emission(self):
        """emit_concern_event supports CRITICAL level."""
        emitter = DPRConcernEmitter()
        result = emitter.emit_concern_event(
            bot_id="bot-evt-3",
            session_id="sess-ghi",
            concern_level="CRITICAL",
            reason="Emergency stop triggered",
            dpr_score=0.10,
        )
        assert result.concern_level == "CRITICAL"
        assert result.tag == "@session_concern"
        assert result.dpr_score == 0.10


# ---------------------------------------------------------------------------
# TestRecentConcerns
# ---------------------------------------------------------------------------

class TestRecentConcerns:
    def test_returns_all_concerns(self):
        """get_recent_concerns returns all stored concerns."""
        emitter = DPRConcernEmitter()
        emitter.emit_tier_change("bot-a", "ELITE", "AT_RISK", 0.35)
        emitter.emit_concern_event("bot-b", "sess-1", "INFO", "test")
        emitter.emit_tier_change("bot-c", "STANDARD", "CIRCUIT_BROKEN", 0.10)
        concerns = emitter.get_recent_concerns()
        assert len(concerns) >= 3

    def test_filters_by_bot(self):
        """get_recent_concerns filters by bot_id."""
        emitter = DPRConcernEmitter()
        emitter.emit_tier_change("bot-x", "ELITE", "AT_RISK", 0.35)
        emitter.emit_concern_event("bot-x", "sess-x", "WARN", "test-x")
        emitter.emit_tier_change("bot-y", "PERFORMING", "STANDARD", 0.60)
        concerns = emitter.get_recent_concerns(bot_id="bot-x")
        assert all(c.bot_id == "bot-x" for c in concerns)

    def test_empty_when_none(self):
        """get_recent_concerns returns empty list when no concerns."""
        emitter = DPRConcernEmitter()
        concerns = emitter.get_recent_concerns()
        assert concerns == []


# ---------------------------------------------------------------------------
# TestCriticalConcerns
# ---------------------------------------------------------------------------

class TestCriticalConcerns:
    def test_returns_critical_only(self):
        """get_critical_concerns returns only CRITICAL level concerns."""
        emitter = DPRConcernEmitter()
        emitter.emit_concern_event("bot-c1", "sess-1", "INFO", "info event")
        emitter.emit_concern_event("bot-c2", "sess-2", "WARN", "warn event")
        emitter.emit_tier_change("bot-c3", "PERFORMING", "CIRCUIT_BROKEN", 0.15)
        critical = emitter.get_critical_concerns()
        assert all(c.concern_level == "CRITICAL" for c in critical)
        assert len(critical) >= 1

    def test_empty_when_none_critical(self):
        """get_critical_concerns returns empty when no CRITICAL concerns."""
        emitter = DPRConcernEmitter()
        emitter.emit_concern_event("bot-ok", "sess-1", "INFO", "fine")
        emitter.emit_concern_event("bot-ok2", "sess-2", "WARN", "warning")
        critical = emitter.get_critical_concerns()
        assert critical == []


# ---------------------------------------------------------------------------
# TestClearOldConcerns
# ---------------------------------------------------------------------------

class TestClearOldConcerns:
    def test_removes_old_concerns(self):
        """clear_old_concerns removes concerns older than threshold."""
        emitter = DPRConcernEmitter()
        emitter.emit_concern_event("bot-old", "sess-1", "WARN", "old event")
        # Manually age the concern
        emitter._concern_log[0].emitted_at_ms = 0
        count = emitter.clear_old_concerns(older_than_ms=3600000)
        assert count == 1
        assert emitter.get_recent_concerns() == []

    def test_returns_count_removed(self):
        """clear_old_concerns returns count of removed concerns."""
        emitter = DPRConcernEmitter()
        emitter.emit_concern_event("bot-r1", "sess-1", "INFO", "event 1")
        emitter.emit_concern_event("bot-r2", "sess-2", "INFO", "event 2")
        emitter._concern_log[0].emitted_at_ms = 0
        emitter._concern_log[1].emitted_at_ms = int(time.time() * 1000)
        count = emitter.clear_old_concerns(older_than_ms=3600000)
        assert count == 1
        remaining = emitter.get_recent_concerns()
        assert len(remaining) == 1
        assert remaining[0].bot_id == "bot-r2"
