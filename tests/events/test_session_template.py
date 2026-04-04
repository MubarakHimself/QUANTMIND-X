"""
Tests for Session Template Event Models.

Story 16.2: Session Template Class — Configurable 10-Window Canonical Cycle

Tests cover:
- SessionTemplateEvent creation and serialization
- SessionTemplateEventType enum values
- Event timestamp handling
"""

import pytest
from datetime import datetime, timezone

from src.events.session_template import SessionTemplateEvent, SessionTemplateEventType


class TestSessionTemplateEventType:
    """Test SessionTemplateEventType enum."""

    def test_event_types_defined(self):
        """Test all event types are defined."""
        assert hasattr(SessionTemplateEventType, "CONFIG_CHANGED")
        assert hasattr(SessionTemplateEventType, "WINDOW_UPDATED")
        assert hasattr(SessionTemplateEventType, "RELOADED")

    def test_event_type_values(self):
        """Test event type string values."""
        assert SessionTemplateEventType.CONFIG_CHANGED.value == "config_changed"
        assert SessionTemplateEventType.WINDOW_UPDATED.value == "window_updated"
        assert SessionTemplateEventType.RELOADED.value == "reloaded"


class TestSessionTemplateEvent:
    """Test SessionTemplateEvent model."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.CONFIG_CHANGED,
            template_name="default",
        )
        assert event.event_type == SessionTemplateEventType.CONFIG_CHANGED
        assert event.template_name == "default"
        assert event.window_name is None
        assert event.timestamp_utc is not None

    def test_event_with_window_name(self):
        """Test event with window name."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.WINDOW_UPDATED,
            template_name="default",
            window_name="London Open",
        )
        assert event.event_type == SessionTemplateEventType.WINDOW_UPDATED
        assert event.window_name == "London Open"

    def test_event_with_metadata(self):
        """Test event with metadata."""
        metadata = {"previous_value": 1.0, "new_value": 1.4}
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.CONFIG_CHANGED,
            template_name="default",
            metadata=metadata,
        )
        assert event.metadata == metadata

    def test_event_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        before = datetime.now(timezone.utc)
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.RELOADED,
            template_name="default",
        )
        after = datetime.now(timezone.utc)

        assert before <= event.timestamp_utc <= after

    def test_event_timestamp_manual_set(self):
        """Test timestamp can be manually set."""
        custom_time = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.CONFIG_CHANGED,
            template_name="default",
            timestamp_utc=custom_time,
        )
        assert event.timestamp_utc == custom_time

    def test_event_serialization(self):
        """Test event serializes to dict correctly."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.WINDOW_UPDATED,
            template_name="default",
            window_name="Tokyo Open",
        )
        data = event.model_dump()

        assert data["event_type"] == "window_updated"
        assert data["template_name"] == "default"
        assert data["window_name"] == "Tokyo Open"
        assert "timestamp_utc" in data

    def test_event_serialization_with_datetime(self):
        """Test event serialization handles datetime correctly."""
        custom_time = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.CONFIG_CHANGED,
            template_name="default",
            timestamp_utc=custom_time,
        )
        data = event.model_dump()

        # Timestamp should be ISO format string after model_dump
        assert isinstance(data["timestamp_utc"], str)
        assert "2026-03-25" in data["timestamp_utc"]

    def test_event_json_serialization(self):
        """Test event serializes to JSON correctly."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.RELOADED,
            template_name="custom",
        )
        json_str = event.model_dump_json()

        assert "reloaded" in json_str
        assert "custom" in json_str


class TestSessionTemplateEventUsage:
    """Test SessionTemplateEvent usage patterns."""

    def test_config_changed_event(self):
        """Test CONFIG_CHANGED event pattern."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.CONFIG_CHANGED,
            template_name="default",
            metadata={
                "changed_by": "operator",
                "reason": "Kelly multiplier adjustment",
            },
        )
        assert event.event_type == SessionTemplateEventType.CONFIG_CHANGED
        assert event.metadata["changed_by"] == "operator"

    def test_window_updated_event(self):
        """Test WINDOW_UPDATED event pattern."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.WINDOW_UPDATED,
            template_name="high-activity",
            window_name="London Open",
            metadata={
                "previous_intensity": "MODERATE",
                "new_intensity": "PREMIUM",
            },
        )
        assert event.event_type == SessionTemplateEventType.WINDOW_UPDATED
        assert event.window_name == "London Open"
        assert event.metadata["new_intensity"] == "PREMIUM"

    def test_reloaded_event(self):
        """Test RELOADED event pattern."""
        event = SessionTemplateEvent(
            event_type=SessionTemplateEventType.RELOADED,
            template_name="default",
            metadata={
                "previous_version": "v1.0",
                "new_version": "v2.0",
            },
        )
        assert event.event_type == SessionTemplateEventType.RELOADED
        assert event.template_name == "default"
