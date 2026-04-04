"""
Tests for Chaos Event Models.

Story 14.3: Layer 3 CHAOS + Kill Switch Forced Exit
Tests for ChaosEvent, RVOLWarningEvent, and KillSwitchResult models.
"""

import pytest
from datetime import datetime, timezone

from src.events.chaos import (
    ChaosEvent,
    ChaosLevel,
    ForcedExitOutcome,
    KillSwitchResult,
    RVOLWarningEvent,
)


class TestChaosLevel:
    """Test ChaosLevel enum values."""

    def test_all_chaos_levels_exist(self):
        """Verify all expected chaos levels are defined."""
        assert ChaosLevel.NORMAL.value == "NORMAL"
        assert ChaosLevel.WARNING.value == "WARNING"
        assert ChaosLevel.CRITICAL.value == "CRITICAL"


class TestForcedExitOutcome:
    """Test ForcedExitOutcome enum values."""

    def test_all_outcomes_exist(self):
        """Verify all expected outcome types are defined."""
        assert ForcedExitOutcome.FILLED.value == "filled"
        assert ForcedExitOutcome.PARTIAL.value == "partial"
        assert ForcedExitOutcome.REJECTED.value == "rejected"
        assert ForcedExitOutcome.LOCK_CONFLICT.value == "lock_conflict"
        assert ForcedExitOutcome.SVSS_UNAVAILABLE.value == "svss_unavailable"


class TestChaosEvent:
    """Test ChaosEvent model."""

    def test_create_chaos_event_with_factory(self):
        """Test ChaosEvent.create() factory method with Lyapunov > 0.95."""
        event = ChaosEvent.create(
            lyapunov_value=0.96,
            tickets=[1001, 1002, 1003],
            threshold=0.95,
            metadata={"source": "lyapunov_sensor"}
        )

        assert event.lyapunov_value == 0.96
        assert event.chaos_level == ChaosLevel.WARNING
        assert event.threshold == 0.95
        assert event.tickets == [1001, 1002, 1003]
        assert event.metadata["source"] == "lyapunov_sensor"

    def test_chaos_event_critical_level(self):
        """Test ChaosEvent with Lyapunov >= 0.99 is CRITICAL."""
        event = ChaosEvent.create(
            lyapunov_value=0.99,
            tickets=[2001],
        )

        assert event.lyapunov_value == 0.99
        assert event.chaos_level == ChaosLevel.CRITICAL

    def test_chaos_event_critical_level_above_099(self):
        """Test ChaosEvent with Lyapunov > 0.99 is CRITICAL."""
        event = ChaosEvent.create(
            lyapunov_value=1.05,
            tickets=[2001],
        )

        assert event.lyapunov_value == 1.05
        assert event.chaos_level == ChaosLevel.CRITICAL

    def test_chaos_event_warning_level(self):
        """Test ChaosEvent with 0.95 <= Lyapunov < 0.99 is WARNING."""
        event = ChaosEvent.create(
            lyapunov_value=0.97,
            tickets=[3001],
        )

        assert event.lyapunov_value == 0.97
        assert event.chaos_level == ChaosLevel.WARNING

    def test_chaos_event_normal_level(self):
        """Test ChaosEvent with Lyapunov < 0.95 is NORMAL."""
        event = ChaosEvent.create(
            lyapunov_value=0.90,
            tickets=[4001],
        )

        assert event.lyapunov_value == 0.90
        assert event.chaos_level == ChaosLevel.NORMAL

    def test_chaos_event_boundary_threshold(self):
        """Test ChaosEvent at exactly threshold (0.95) is WARNING, not NORMAL."""
        event = ChaosEvent.create(
            lyapunov_value=0.95,
            tickets=[5001],
        )

        assert event.lyapunov_value == 0.95
        assert event.chaos_level == ChaosLevel.WARNING

    def test_chaos_event_empty_tickets(self):
        """Test ChaosEvent with no tickets to flag."""
        event = ChaosEvent.create(
            lyapunov_value=0.96,
            tickets=[],
        )

        assert event.tickets == []
        assert event.chaos_level == ChaosLevel.WARNING

    def test_chaos_event_str_representation(self):
        """Test string representation."""
        event = ChaosEvent.create(
            lyapunov_value=0.97,
            tickets=[1001, 1002],
        )

        str_repr = str(event)
        assert "WARNING" in str_repr
        assert "0.9700" in str_repr
        assert "tickets=2" in str_repr  # Shows count, not individual tickets

    def test_chaos_event_timestamp(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        event = ChaosEvent.create(
            lyapunov_value=0.96,
            tickets=[1001],
        )
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp_utc <= after

    def test_chaos_event_explicit_level(self):
        """Test ChaosEvent with explicit level construction."""
        event = ChaosEvent(
            lyapunov_value=0.98,
            chaos_level=ChaosLevel.WARNING,
            threshold=0.95,
            tickets=[1001],
        )

        assert event.lyapunov_value == 0.98
        assert event.chaos_level == ChaosLevel.WARNING


class TestRVOLWarningEvent:
    """Test RVOLWarningEvent model."""

    def test_create_rvol_warning_event(self):
        """Test RVOLWarningEvent.create() factory method."""
        event = RVOLWarningEvent.create(
            symbol="EURUSD",
            rvol=0.42,
            has_open_positions=True,
            threshold=0.5,
        )

        assert event.symbol == "EURUSD"
        assert event.rvol == 0.42
        assert event.threshold == 0.5
        assert event.has_open_positions is True
        assert event.blocked_entries is True  # rvol < threshold

    def test_rvol_warning_event_blocks_entries(self):
        """Test RVOL < 0.5 blocks new entries."""
        event = RVOLWarningEvent.create(
            symbol="EURUSD",
            rvol=0.49,
            has_open_positions=False,
        )

        assert event.rvol < event.threshold
        assert event.blocked_entries is True

    def test_rvol_warning_event_does_not_block_above_threshold(self):
        """Test RVOL >= 0.5 does not block entries."""
        event = RVOLWarningEvent.create(
            symbol="EURUSD",
            rvol=0.55,
            has_open_positions=False,
        )

        assert event.rvol >= event.threshold
        assert event.blocked_entries is False

    def test_rvol_warning_event_str_representation(self):
        """Test string representation."""
        event = RVOLWarningEvent.create(
            symbol="GBPUSD",
            rvol=0.35,
            has_open_positions=True,
        )

        str_repr = str(event)
        assert "GBPUSD" in str_repr
        assert "0.35" in str_repr
        assert "True" in str_repr  # has_positions

    def test_rvol_warning_event_timestamp(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        event = RVOLWarningEvent.create(
            symbol="XAUUSD",
            rvol=0.30,
        )
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp_utc <= after

    def test_rvol_warning_event_metadata(self):
        """Test metadata is stored correctly."""
        event = RVOLWarningEvent.create(
            symbol="EURUSD",
            rvol=0.40,
            metadata={"svss_instance": "primary", "correlation": 0.82},
        )

        assert event.metadata["svss_instance"] == "primary"
        assert event.metadata["correlation"] == 0.82


class TestKillSwitchResult:
    """Test KillSwitchResult model."""

    def test_kill_switch_result_filled(self):
        """Test KillSwitchResult with FILLED outcome."""
        result = KillSwitchResult(
            ticket=12345,
            outcome=ForcedExitOutcome.FILLED,
            lyapunov_triggered=True,
            rvol_triggered=False,
            lock_released=True,
        )

        assert result.ticket == 12345
        assert result.outcome == ForcedExitOutcome.FILLED
        assert result.lyapunov_triggered is True
        assert result.rvol_triggered is False
        assert result.lock_released is True
        assert result.error is None

    def test_kill_switch_result_partial(self):
        """Test KillSwitchResult with PARTIAL outcome."""
        result = KillSwitchResult(
            ticket=12346,
            outcome=ForcedExitOutcome.PARTIAL,
            lyapunov_triggered=True,
            partial_volume=0.50,
        )

        assert result.ticket == 12346
        assert result.outcome == ForcedExitOutcome.PARTIAL
        assert result.partial_volume == 0.50

    def test_kill_switch_result_rejected(self):
        """Test KillSwitchResult with REJECTED outcome."""
        result = KillSwitchResult(
            ticket=12347,
            outcome=ForcedExitOutcome.REJECTED,
            lyapunov_triggered=False,
            rvol_triggered=True,
            lock_released=True,
            error="MT5 broker rejected: insufficient margin",
        )

        assert result.ticket == 12347
        assert result.outcome == ForcedExitOutcome.REJECTED
        assert result.rvol_triggered is True
        assert "rejected" in result.error.lower()

    def test_kill_switch_result_lock_conflict(self):
        """Test KillSwitchResult with LOCK_CONFLICT outcome."""
        result = KillSwitchResult(
            ticket=12348,
            outcome=ForcedExitOutcome.LOCK_CONFLICT,
            lock_released=False,
            error="Could not release Layer 2 lock",
        )

        assert result.ticket == 12348
        assert result.outcome == ForcedExitOutcome.LOCK_CONFLICT
        assert result.lock_released is False

    def test_kill_switch_result_str_representation(self):
        """Test string representation."""
        result = KillSwitchResult(
            ticket=12345,
            outcome=ForcedExitOutcome.FILLED,
            lyapunov_triggered=True,
            lock_released=True,
        )

        str_repr = str(result)
        assert "12345" in str_repr
        assert "filled" in str_repr
        assert "lyapunov" in str_repr

    def test_kill_switch_result_timestamp(self):
        """Test timestamp is set correctly."""
        before = datetime.now(timezone.utc)
        result = KillSwitchResult(
            ticket=12345,
            outcome=ForcedExitOutcome.FILLED,
        )
        after = datetime.now(timezone.utc)

        assert before <= result.timestamp_utc <= after


class TestChaosEventIntegration:
    """Integration tests for chaos events with realistic scenarios."""

    def test_lyapunov_threshold_detection(self):
        """Test that Lyapunov > 0.95 triggers CHAOS detection."""
        # Lyapunov just above threshold
        event_095 = ChaosEvent.create(lyapunov_value=0.951, tickets=[1])
        assert event_095.chaos_level == ChaosLevel.WARNING

        # Lyapunov at 0.99
        event_099 = ChaosEvent.create(lyapunov_value=0.99, tickets=[1])
        assert event_099.chaos_level == ChaosLevel.CRITICAL

        # Lyapunov just below threshold
        event_094 = ChaosEvent.create(lyapunov_value=0.949, tickets=[1])
        assert event_094.chaos_level == ChaosLevel.NORMAL

    def test_rvol_threshold_detection(self):
        """Test that RVOL < 0.5 triggers warning."""
        # RVOL below threshold
        event_low = RVOLWarningEvent.create(symbol="EURUSD", rvol=0.49, has_open_positions=True)
        assert event_low.blocked_entries is True

        # RVOL at threshold
        event_050 = RVOLWarningEvent.create(symbol="EURUSD", rvol=0.50, has_open_positions=True)
        assert event_050.blocked_entries is False

        # RVOL above threshold
        event_high = RVOLWarningEvent.create(symbol="EURUSD", rvol=0.51, has_open_positions=True)
        assert event_high.blocked_entries is False

    def test_multiple_tickets_in_chaos_event(self):
        """Test chaos event with many tickets."""
        tickets = list(range(1000, 1100))
        event = ChaosEvent.create(
            lyapunov_value=1.02,
            tickets=tickets,
        )

        assert len(event.tickets) == 100
        assert event.chaos_level == ChaosLevel.CRITICAL

    def test_force_exit_outcome_workflow(self):
        """Test force exit outcome values match expected workflow."""
        # FILLED - complete close
        filled = KillSwitchResult(
            ticket=1,
            outcome=ForcedExitOutcome.FILLED,
        )
        assert filled.outcome == ForcedExitOutcome.FILLED

        # PARTIAL - partial close due to margin issues
        partial = KillSwitchResult(
            ticket=2,
            outcome=ForcedExitOutcome.PARTIAL,
            partial_volume=0.50,
        )
        assert partial.outcome == ForcedExitOutcome.PARTIAL

        # REJECTED - broker rejected
        rejected = KillSwitchResult(
            ticket=3,
            outcome=ForcedExitOutcome.REJECTED,
            error="insufficient margin",
        )
        assert rejected.outcome == ForcedExitOutcome.REJECTED

        # SVSS_UNAVAILABLE - fallback mode
        svss_fallback = KillSwitchResult(
            ticket=4,
            outcome=ForcedExitOutcome.SVSS_UNAVAILABLE,
        )
        assert svss_fallback.outcome == ForcedExitOutcome.SVSS_UNAVAILABLE
