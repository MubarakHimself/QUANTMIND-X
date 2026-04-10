"""
Tests for DPRSSLEmitter — Phase 11F DPR→SSL bridge.

Tests the reverse integration of DPRSSLConsumer:
- DPRSSLEmitter translates DPR circuit-break decisions into SSL state transitions
- Only LIVE bots can be transitioned to paper
- Skips bots already in PAPER/RECOVERY/RETIRED
- Tracks transitioned bots to prevent duplicate events
- Recovery translation for PAPER→RECOVERY
"""
from __future__ import annotations

from unittest.mock import Mock, MagicMock, patch
import time

import pytest

from src.events.ssl import SSLCircuitBreakerEvent, SSLState, SSLEventType, TradeOutcome
from src.risk.dpr.dpr_emitter import DPRSSLEmitter


def make_mock_ssl_breaker():
    """Create a fully mocked SSLCircuitBreaker."""
    breaker = Mock()

    # Mock state manager
    mock_state = Mock()
    mock_state.get_state = Mock(return_value=SSLState.LIVE.value)
    mock_state.get_magic_number = Mock(return_value="12345")
    mock_state.get_consecutive_losses = Mock(return_value=2)
    breaker.state_manager = mock_state

    # Mock threshold and tier helpers
    breaker._get_threshold = Mock(return_value=2)
    breaker._determine_tier = Mock(return_value=Mock(value="TIER_1"))
    breaker._emit_event = Mock()

    return breaker


def make_dpr_score(
    bot_id: str,
    score: float,
    tier: str,
    rank: int = 1,
) -> Mock:
    """Helper to build a mock DPRScore for testing."""
    dpr = Mock()
    dpr.bot_id = bot_id
    dpr.dpr_score = score
    dpr.tier = tier
    dpr.rank = rank
    dpr.win_rate_today = score
    dpr.daily_pnl = score * 1000
    return dpr


class TestDPRSSLEmitterInit:
    """Init / construction tests."""

    def test_init_default_ssl_breaker(self):
        """Creates SSLCircuitBreaker if none provided."""
        emitter = DPRSSLEmitter()
        assert emitter._ssl_breaker is not None

    def test_init_custom_ssl_breaker(self):
        """Accepts custom SSLCircuitBreaker."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)
        assert emitter._ssl_breaker is mock_breaker

    def test_init_empty_transition_trackers(self):
        """Starts with empty transition trackers."""
        emitter = DPRSSLEmitter()
        assert emitter._transitioned_to_paper == {}
        assert emitter._transitioned_to_recovery == {}


class TestEmitDPRToSSL:
    """emit_dpr_to_ssl tests."""

    def test_emits_ssl_event_on_circuit_broken_tier(self):
        """DPR tier CIRCUIT_BROKEN triggers LIVE→PAPER SSL event."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-1", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-1", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.previous_state == SSLState.LIVE
        assert event.new_state == SSLState.PAPER
        mock_breaker._emit_event.assert_called_once_with(event)

    def test_emits_ssl_event_on_low_score(self):
        """DPR score < 0.3 triggers LIVE→PAPER SSL event."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-2", score=0.25, tier="AT_RISK", rank=7)
        event = emitter.emit_dpr_to_ssl("bot-2", dpr_score, reason="DPR_SCORE_THRESHOLD")

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER

    def test_emits_ssl_event_on_at_risk_consecutive(self):
        """DPR AT_RISK consecutive check triggers LIVE→PAPER SSL event."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-3", score=0.35, tier="AT_RISK", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-3", dpr_score, reason="DPR_AT_RISK_CONSECUTIVE")

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER

    def test_emits_ssl_event_on_rank_at_risk(self):
        """DPR rank > 10 with AT_RISK triggers LIVE→PAPER SSL event."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-4", score=0.40, tier="AT_RISK", rank=12)
        event = emitter.emit_dpr_to_ssl("bot-4", dpr_score, reason="DPR_RANK_AT_RISK")

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER

    def test_skips_non_trigger_reason(self):
        """Unknown trigger reason returns None without emitting."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-5", score=0.25, tier="AT_RISK", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-5", dpr_score, reason="SOME_OTHER_TRIGGER")

        assert event is None
        mock_breaker._emit_event.assert_not_called()

    def test_skips_none_reason(self):
        """None reason returns None without emitting."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-6", score=0.25, tier="AT_RISK", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-6", dpr_score, reason=None)

        assert event is None
        mock_breaker._emit_event.assert_not_called()

    def test_skips_already_paper_bot(self):
        """Bot already in PAPER state is skipped."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.PAPER.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-paper", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-paper", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert event is None
        mock_breaker._emit_event.assert_not_called()

    def test_skips_already_recovery_bot(self):
        """Bot already in RECOVERY state is skipped."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.RECOVERY.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-recovery", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-recovery", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert event is None
        mock_breaker._emit_event.assert_not_called()

    def test_skips_already_retired_bot(self):
        """Bot already in RETIRED state is skipped."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.RETIRED.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-retired", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-retired", dpr_score, reason="DPR_CIRCUIT_BROKEN")

        assert event is None
        mock_breaker._emit_event.assert_not_called()

    def test_skips_duplicate_transition(self):
        """Same bot transitioned twice in same session is skipped."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-dup", score=0.25, tier="CIRCUIT_BROKEN", rank=5)

        # First transition
        event1 = emitter.emit_dpr_to_ssl("bot-dup", dpr_score, reason="DPR_CIRCUIT_BREAKER")
        assert event1 is not None

        # Second transition — should be skipped
        event2 = emitter.emit_dpr_to_ssl("bot-dup", dpr_score, reason="DPR_CIRCUIT_BREAKER")
        assert event2 is None
        # Only one emit call
        assert mock_breaker._emit_event.call_count == 1

    def test_unknown_bot_treated_as_live(self):
        """Bot with no SSL state record is treated as LIVE."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=None)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-new", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-new", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert event is not None
        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.previous_state == SSLState.LIVE

    def test_event_metadata_includes_dpr_context(self):
        """Emitted event includes DPR tier, score, rank, and trigger in metadata."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-meta", score=0.25, tier="CIRCUIT_BROKEN", rank=7)
        event = emitter.emit_dpr_to_ssl(
            "bot-meta", dpr_score, reason="DPR_CIRCUIT_BREAKER"
        )

        assert event is not None
        assert event.metadata["dpr_tier"] == "CIRCUIT_BROKEN"
        assert event.metadata["dpr_score"] == 0.25
        assert event.metadata["dpr_rank"] == 7
        assert event.metadata["dpr_trigger"] == "DPR_CIRCUIT_BREAKER"
        assert event.metadata["transition_source"] == "dpr_emitter"

    def test_emit_failure_does_not_leak_state(self):
        """If _emit_event fails, bot is not marked as transitioned."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker._emit_event = Mock(side_effect=Exception("Redis error"))
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-fail", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        event = emitter.emit_dpr_to_ssl("bot-fail", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert event is None
        assert "bot-fail" not in emitter._transitioned_to_paper


class TestEmitRecoveryToSSL:
    """emit_recovery_to_ssl tests."""

    def test_emits_recovery_on_tier_improvement(self):
        """DPR tier improving to STANDARD triggers PAPER→RECOVERY SSL event."""
        mock_breaker = make_mock_ssl_breaker()

        # Step 1: Bot starts in LIVE — transition to paper via DPR
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.LIVE.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score_paper = make_dpr_score(
            "bot-rec", score=0.25, tier="CIRCUIT_BROKEN", rank=5
        )
        paper_event = emitter.emit_dpr_to_ssl(
            "bot-rec", dpr_score_paper, reason="DPR_CIRCUIT_BREAKER"
        )
        assert paper_event is not None

        # Step 2: SSL state changes to PAPER (simulated external transition)
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.PAPER.value)

        # Step 3: DPR tier improves to STANDARD — recovery emission
        dpr_score_recovery = make_dpr_score(
            "bot-rec", score=0.60, tier="STANDARD", rank=5
        )
        event = emitter.emit_recovery_to_ssl("bot-rec", dpr_score_recovery)

        assert event is not None
        assert event.event_type == SSLEventType.RECOVERY_STEP_1
        assert event.previous_state == SSLState.PAPER
        assert event.new_state == SSLState.RECOVERY

    def test_skips_recovery_for_non_dpr_paper(self):
        """PAPER bots not transitioned via DPR are not recovered by DPR."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.PAPER.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        # DPR transitioned to paper
        dpr_score_paper = make_dpr_score(
            "bot-nondpr", score=0.25, tier="CIRCUIT_BROKEN", rank=5
        )
        emitter.emit_dpr_to_ssl(
            "bot-nondpr", dpr_score_paper, reason="DPR_CIRCUIT_BREAKER"
        )

        # Recovery attempt for a DIFFERENT bot that was paper via SSL (not DPR)
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.PAPER.value)
        emitter2 = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score_recovery = make_dpr_score(
            "bot-ssl-paper", score=0.60, tier="STANDARD", rank=5
        )
        event = emitter2.emit_recovery_to_ssl("bot-ssl-paper", dpr_score_recovery)

        # No transition since this bot was never transitioned to paper via DPR
        assert event is None

    def test_skips_recovery_for_live_bot(self):
        """LIVE bots are not transitioned for recovery."""
        mock_breaker = make_mock_ssl_breaker()
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.LIVE.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-live", score=0.60, tier="STANDARD", rank=5)
        event = emitter.emit_recovery_to_ssl("bot-live", dpr_score)

        assert event is None

    def test_skips_recovery_for_at_risk_tier(self):
        """AT_RISK DPR tier does not trigger recovery."""
        mock_breaker = make_mock_ssl_breaker()

        # Step 1: Bot starts in LIVE — transition to paper via DPR
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.LIVE.value)
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score_paper = make_dpr_score(
            "bot-atrisk", score=0.25, tier="CIRCUIT_BROKEN", rank=5
        )
        emitter.emit_dpr_to_ssl(
            "bot-atrisk", dpr_score_paper, reason="DPR_CIRCUIT_BREAKER"
        )

        # Step 2: SSL state changes to PAPER
        mock_breaker.state_manager.get_state = Mock(return_value=SSLState.PAPER.value)

        # Step 3: DPR tier is still AT_RISK — recovery NOT emitted
        dpr_score_atrisk = make_dpr_score(
            "bot-atrisk", score=0.35, tier="AT_RISK", rank=7
        )
        event = emitter.emit_recovery_to_ssl("bot-atrisk", dpr_score_atrisk)

        assert event is None


class TestDPRSSLEmitterStateTracking:
    """get_transitioned_bots / is_bot_transitioned / reset tests."""

    def test_get_transitioned_bots(self):
        """Returns dict of transitioned bots."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-a", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        emitter.emit_dpr_to_ssl("bot-a", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        dpr_score2 = make_dpr_score("bot-b", score=0.20, tier="CIRCUIT_BROKEN", rank=5)
        emitter.emit_dpr_to_ssl("bot-b", dpr_score2, reason="DPR_CIRCUIT_BREAKER")

        transitioned = emitter.get_transitioned_bots()
        assert "bot-a" in transitioned
        assert "bot-b" in transitioned
        assert transitioned["bot-a"] == "DPR_CIRCUIT_BREAKER"
        assert transitioned["bot-b"] == "DPR_CIRCUIT_BREAKER"

    def test_is_bot_transitioned(self):
        """Returns True for transitioned bots, False for others."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        assert emitter.is_bot_transitioned("bot-x") is False

        dpr_score = make_dpr_score("bot-x", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        emitter.emit_dpr_to_ssl("bot-x", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert emitter.is_bot_transitioned("bot-x") is True
        assert emitter.is_bot_transitioned("bot-y") is False

    def test_reset_clears_trackers(self):
        """reset() clears all transition tracking."""
        mock_breaker = make_mock_ssl_breaker()
        emitter = DPRSSLEmitter(ssl_breaker=mock_breaker)

        dpr_score = make_dpr_score("bot-reset", score=0.25, tier="CIRCUIT_BROKEN", rank=5)
        emitter.emit_dpr_to_ssl("bot-reset", dpr_score, reason="DPR_CIRCUIT_BREAKER")

        assert emitter.is_bot_transitioned("bot-reset") is True

        emitter.reset()

        assert emitter._transitioned_to_paper == {}
        assert emitter._transitioned_to_recovery == {}
        assert emitter.is_bot_transitioned("bot-reset") is False
