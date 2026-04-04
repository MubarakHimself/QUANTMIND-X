"""
Tests for SSL Event Models.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Tests cover:
- SSLCircuitBreakerEvent model
- SSLState enum
- SSLEventType enum
- BotTier enum
"""

import pytest
from datetime import datetime, timezone

from src.events.ssl import (
    SSLCircuitBreakerEvent,
    SSLState,
    SSLEventType,
    BotTier,
)


class TestSSLState:
    """Test SSLState enum."""

    def test_ssl_state_values(self):
        """Test SSLState enum has expected values."""
        assert SSLState.LIVE.value == "live"
        assert SSLState.PAPER.value == "paper"
        assert SSLState.RECOVERY.value == "recovery"
        assert SSLState.RETIRED.value == "retired"

    def test_ssl_state_count(self):
        """Test SSLState enum has 4 states."""
        assert len(SSLState) == 4


class TestSSLEventType:
    """Test SSLEventType enum."""

    def test_ssl_event_type_values(self):
        """Test SSLEventType enum has expected values."""
        assert SSLEventType.MOVE_TO_PAPER.value == "move_to_paper"
        assert SSLEventType.RECOVERY_STEP_1.value == "recovery_step_1"
        assert SSLEventType.RECOVERY_CONFIRMED.value == "recovery_confirmed"
        assert SSLEventType.RETIRED.value == "retired"

    def test_ssl_event_type_count(self):
        """Test SSLEventType enum has 4 event types."""
        assert len(SSLEventType) == 4


class TestBotTier:
    """Test BotTier enum."""

    def test_bot_tier_values(self):
        """Test BotTier enum has expected values."""
        assert BotTier.TIER_1.value == "TIER_1"
        assert BotTier.TIER_2.value == "TIER_2"

    def test_bot_tier_count(self):
        """Test BotTier enum has 2 tiers."""
        assert len(BotTier) == 2


class TestSSLCircuitBreakerEvent:
    """Test SSLCircuitBreakerEvent model."""

    def test_create_move_to_paper_event(self):
        """Test creating a MOVE_TO_PAPER event."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        assert event.bot_id == "bot-1"
        assert event.magic_number == "12345"
        assert event.event_type == SSLEventType.MOVE_TO_PAPER
        assert event.consecutive_losses == 2
        assert event.tier == "TIER_1"
        assert event.previous_state == SSLState.LIVE
        assert event.new_state == SSLState.PAPER
        assert event.timestamp_utc is not None

    def test_create_recovery_confirmed_event(self):
        """Test creating a RECOVERY_CONFIRMED event."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            consecutive_losses=0,
            previous_state=SSLState.RECOVERY,
            new_state=SSLState.LIVE,
            recovery_win_count=2,
        )

        assert event.bot_id == "bot-1"
        assert event.event_type == SSLEventType.RECOVERY_CONFIRMED
        assert event.new_state == SSLState.LIVE
        assert event.recovery_win_count == 2

    def test_create_with_metadata(self):
        """Test creating an event with metadata."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=3,
            tier="TIER_2",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            metadata={
                "bot_type": "orb",
                "threshold_breached": 3,
            },
        )

        assert event.metadata["bot_type"] == "orb"
        assert event.metadata["threshold_breached"] == 3

    def test_str_representation(self):
        """Test __str__ representation."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        str_repr = str(event)
        assert "bot-1" in str_repr
        assert "move_to_paper" in str_repr
        assert "paper" in str_repr  # state value is lowercase

    def test_str_with_tier_and_recovery(self):
        """Test __str__ representation with tier and recovery info."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            consecutive_losses=0,
            previous_state=SSLState.RECOVERY,
            new_state=SSLState.LIVE,
            recovery_win_count=2,
        )

        str_repr = str(event)
        assert "recovery_wins=2" in str_repr

    def test_to_redis_message(self):
        """Test serialization to Redis JSON message."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            tier="TIER_1",
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        json_msg = event.to_redis_message()
        # JSON is compact (no extra spaces)
        assert '"bot_id":"bot-1"' in json_msg
        assert '"event_type":"move_to_paper"' in json_msg
        assert '"new_state":"paper"' in json_msg

    def test_timestamp_auto_generated(self):
        """Test timestamp_utc is auto-generated if not provided."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
        )

        assert event.timestamp_utc is not None
        assert isinstance(event.timestamp_utc, datetime)

    def test_timestamp_provided(self):
        """Test timestamp_utc is used if provided."""
        specific_time = datetime(2026, 3, 25, 12, 0, 0, tzinfo=timezone.utc)
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            previous_state=SSLState.LIVE,
            new_state=SSLState.PAPER,
            timestamp_utc=specific_time,
        )

        assert event.timestamp_utc == specific_time

    def test_tier_optional(self):
        """Test tier is optional for events that don't need it."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            consecutive_losses=0,
            previous_state=SSLState.RECOVERY,
            new_state=SSLState.LIVE,
        )

        assert event.tier is None

    def test_previous_state_optional(self):
        """Test previous_state is optional for new records."""
        event = SSLCircuitBreakerEvent(
            bot_id="bot-1",
            magic_number="12345",
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=2,
            new_state=SSLState.PAPER,
        )

        assert event.previous_state is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
