"""
QuantMindLib V1 — SafetyIntegration Tests
Phase 10 Packet 10B: DPRCircuitBreakerMonitor tests.
"""
from __future__ import annotations

import time
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.library.core.bridges.sentinel_dpr_bridges import DPRScore, DPRBridge
from src.library.core.bridges.safety_integration import DPRCircuitBreakerMonitor
from src.library.runtime.safety_hooks import SafetyHooks, KillSwitchResult


def make_dpr_score(
    bot_id: str,
    score: float,
    tier: str,
    rank: int = 1,
) -> DPRScore:
    """Helper to build a DPRScore for testing."""
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


class TestDPRCircuitBreakerMonitorInit:
    """Init / construction tests."""

    def test_init_default_components(self):
        """Initializes with DPRBridge and SafetyHooks."""
        monitor = DPRCircuitBreakerMonitor()
        assert monitor._dpr_bridge is not None
        assert isinstance(monitor._dpr_bridge, DPRBridge)
        assert monitor._safety_hooks is not None
        assert isinstance(monitor._safety_hooks, SafetyHooks)
        assert monitor._blocked_bots == {}

    def test_init_custom_components(self):
        """Accepts custom DPRBridge and SafetyHooks."""
        dpr = DPRBridge()
        hooks = SafetyHooks(kill_switch_enabled=False)
        monitor = DPRCircuitBreakerMonitor(dpr_bridge=dpr, safety_hooks=hooks)
        assert monitor._dpr_bridge is dpr
        assert monitor._safety_hooks is hooks
        assert monitor._blocked_bots == {}


class TestDPRCircuitBreakerMonitorCircuitState:
    """check_bot_circuit_state tests."""

    def test_check_bot_circuit_state_circuit_broken(self):
        """DPR tier CIRCUIT_BROKEN returns blocked KillSwitchResult."""
        monitor = DPRCircuitBreakerMonitor()
        score = make_dpr_score("bot-1", score=0.15, tier="CIRCUIT_BROKEN", rank=5)

        result = monitor.check_bot_circuit_state("bot-1", dpr_score=score)

        assert result.allowed is False
        assert "CIRCUIT_BROKEN" in result.reason
        assert result.triggered_by == "DPR_CIRCUIT_BREAKER"
        assert "bot-1" in monitor.get_blocked_bots()

    def test_check_bot_circuit_state_low_score(self):
        """DPR score < 0.3 returns blocked KillSwitchResult."""
        monitor = DPRCircuitBreakerMonitor()
        score = make_dpr_score("bot-2", score=0.25, tier="AT_RISK", rank=3)

        result = monitor.check_bot_circuit_state("bot-2", dpr_score=score)

        assert result.allowed is False
        assert "0.25" in result.reason or "0.3" in result.reason
        assert result.triggered_by == "DPR_SCORE_THRESHOLD"
        assert "bot-2" in monitor.get_blocked_bots()

    def test_check_bot_circuit_state_elite_passes(self):
        """DPR tier ELITE with score >= 0.85 returns allowed."""
        monitor = DPRCircuitBreakerMonitor()
        score = make_dpr_score("bot-elite", score=0.92, tier="ELITE", rank=1)

        result = monitor.check_bot_circuit_state("bot-elite", dpr_score=score)

        assert result.allowed is True
        assert "ELITE" in result.reason
        assert result.triggered_by is None

    def test_check_bot_circuit_state_at_risk_passes(self):
        """DPR tier AT_RISK with score >= 0.3 returns allowed."""
        monitor = DPRCircuitBreakerMonitor()
        score = make_dpr_score("bot-risk", score=0.45, tier="AT_RISK", rank=7)

        result = monitor.check_bot_circuit_state("bot-risk", dpr_score=score)

        assert result.allowed is True
        assert "AT_RISK" in result.reason

    def test_check_bot_circuit_state_unknown_bot(self):
        """Unknown bot (no DPRScore) returns allowed."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_bot_circuit_state("unknown-bot", dpr_score=None)

        assert result.allowed is True
        assert "No DPR score" in result.reason


class TestDPRCircuitBreakerMonitorConcernEvents:
    """check_concern_events tests."""

    def test_check_concern_events_low_passes(self):
        """LOW concern passes with warning."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_concern_events("bot-1", "LOW")

        assert result.allowed is True
        assert "LOW" in result.reason
        assert result.triggered_by is None

    def test_check_concern_events_critical_blocks(self):
        """CRITICAL concern blocks all trades."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_concern_events("bot-2", "CRITICAL")

        assert result.allowed is False
        assert "CRITICAL" in result.reason
        assert result.triggered_by == "DPR_CONCERN_CRITICAL"
        assert "bot-2" in monitor.get_blocked_bots()

    def test_check_concern_events_high_blocks_new(self):
        """HIGH concern blocks new entries."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_concern_events("bot-3", "HIGH")

        assert result.allowed is False
        assert "HIGH" in result.reason
        assert result.triggered_by == "DPR_CONCERN_HIGH"
        assert "bot-3" in monitor.get_blocked_bots()

    def test_check_concern_events_medium_passes_with_caution(self):
        """MEDIUM concern passes with caution flag."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_concern_events("bot-4", "MEDIUM")

        assert result.allowed is True
        assert "MEDIUM" in result.reason
        assert "caution" in result.reason

    def test_check_concern_events_invalid_level_passes(self):
        """Invalid concern level passes silently."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.check_concern_events("bot-5", "INVALID")

        assert result.allowed is True
        assert "Unknown concern level" in result.reason


class TestDPRCircuitBreakerMonitorBlockedBots:
    """get_blocked_bots / unblock_bot tests."""

    def test_get_blocked_bots(self):
        """Returns dict of blocked bots with reasons."""
        monitor = DPRCircuitBreakerMonitor()
        # Block via concern event
        monitor.check_concern_events("bot-critical", "CRITICAL")
        # Block via circuit state
        score = make_dpr_score("bot-broken", score=0.1, tier="CIRCUIT_BROKEN", rank=8)
        monitor.check_bot_circuit_state("bot-broken", dpr_score=score)

        blocked = monitor.get_blocked_bots()

        assert "bot-critical" in blocked
        assert "bot-broken" in blocked
        assert "CIRCUIT_BROKEN" in blocked["bot-broken"]

    def test_unblock_bot(self):
        """Removes bot from blocked list."""
        monitor = DPRCircuitBreakerMonitor()
        monitor.check_concern_events("bot-x", "CRITICAL")
        assert "bot-x" in monitor.get_blocked_bots()

        result = monitor.unblock_bot("bot-x")

        assert result is True
        assert "bot-x" not in monitor.get_blocked_bots()

    def test_unblock_unknown_bot_returns_false(self):
        """Unblocking unknown bot returns False."""
        monitor = DPRCircuitBreakerMonitor()

        result = monitor.unblock_bot("never-existed")

        assert result is False

    def test_unblock_bot_resets_at_risk_count(self):
        """Unblocking a bot also resets its AT_RISK consecutive count."""
        monitor = DPRCircuitBreakerMonitor()
        # Accumulate AT_RISK checks up to block threshold
        score_at_risk = make_dpr_score("bot-atrisk", score=0.35, tier="AT_RISK", rank=5)
        for _ in range(5):
            monitor.check_bot_circuit_state("bot-atrisk", dpr_score=score_at_risk)

        assert "bot-atrisk" in monitor.get_blocked_bots()

        # Unblock
        monitor.unblock_bot("bot-atrisk")

        # Should be unblocked; subsequent check with same score should pass
        result = monitor.check_bot_circuit_state("bot-atrisk", dpr_score=score_at_risk)
        assert result.allowed is True

    def test_get_bot_concern_state(self):
        """Returns concern state string for a blocked bot."""
        monitor = DPRCircuitBreakerMonitor()
        monitor.check_concern_events("bot-c", "CRITICAL")

        state = monitor.get_bot_concern_state("bot-c")

        assert state == "CRITICAL"

    def test_get_bot_concern_state_unknown_bot(self):
        """Returns None for bot not in blocked list."""
        monitor = DPRCircuitBreakerMonitor()

        state = monitor.get_bot_concern_state("unknown-bot")

        assert state is None

    def test_at_risk_consecutive_blocks_after_3(self):
        """AT_RISK bot is blocked after > 3 consecutive checks (i.e. on the 4th check)."""
        monitor = DPRCircuitBreakerMonitor()
        score = make_dpr_score("bot-atrisk-5x", score=0.35, tier="AT_RISK", rank=5)

        # First 3 checks should pass (counts 1, 2, 3 — none exceed > 3)
        for i in range(3):
            result = monitor.check_bot_circuit_state(
                "bot-atrisk-5x", dpr_score=score
            )
            assert result.allowed is True, f"Check {i + 1} should pass"

        # 4th check should block (count = 4 > 3)
        result = monitor.check_bot_circuit_state("bot-atrisk-5x", dpr_score=score)
        assert result.allowed is False
        assert result.triggered_by == "DPR_AT_RISK_CONSECUTIVE"