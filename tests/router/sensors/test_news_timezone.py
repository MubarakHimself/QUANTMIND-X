"""
Tests for NewsSensor Timezone Conversion

Tests timezone-aware news event handling and kill zone detection.
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch
from src.router.sensors.news import NewsSensor, NewsEvent


class TestNewsEvent:
    """Test NewsEvent dataclass."""

    def test_news_event_creation(self):
        """Test creating NewsEvent with timezone."""
        event = NewsEvent(
            title="NFP Employment Change",
            impact="HIGH",
            time=datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc),
            timezone_source="America/New_York",
            currency="USD"
        )
        assert event.title == "NFP Employment Change"
        assert event.impact == "HIGH"
        assert event.timezone_source == "America/New_York"
        assert event.currency == "USD"

    def test_news_event_defaults(self):
        """Test NewsEvent with default timezone."""
        event = NewsEvent(
            title="Test Event",
            impact="MEDIUM",
            time=datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        )
        assert event.timezone_source == "UTC"  # Default
        assert event.currency is None  # Default


class TestNewsSensorInitialization:
    """Test NewsSensor initialization with configurable kill zones."""

    def test_default_kill_zones(self):
        """Test default kill zone initialization."""
        sensor = NewsSensor()
        assert sensor.kill_zone_minutes_pre == 15
        assert sensor.kill_zone_minutes_post == 15

    def test_custom_kill_zones(self):
        """Test custom kill zone initialization."""
        sensor = NewsSensor(kill_zone_minutes_pre=30, kill_zone_minutes_post=60)
        assert sensor.kill_zone_minutes_pre == 30
        assert sensor.kill_zone_minutes_post == 60

    def test_initial_events_empty(self):
        """Test sensor starts with empty events list."""
        sensor = NewsSensor()
        assert sensor.events == []


class TestNewsCalendarUpdates:
    """Test update_calendar with timezone conversion."""

    def test_update_calendar_with_est_events(self):
        """Test updating calendar with EST events."""
        sensor = NewsSensor()
        calendar_data = [
            {
                "title": "NFP Employment Change",
                "impact": "HIGH",
                "time": "2026-02-12T08:30:00",
                "timezone": "America/New_York",
                "currency": "USD"
            }
        ]

        sensor.update_calendar(calendar_data)

        assert len(sensor.events) == 1
        event = sensor.events[0]
        assert event.title == "NFP Employment Change"
        # 8:30 AM EST = 13:30 UTC
        assert event.time.hour == 13
        assert event.time.minute == 30
        assert event.timezone_source == "America/New_York"

    def test_update_calendar_with_utc_events(self):
        """Test updating calendar with UTC events."""
        sensor = NewsSensor()
        calendar_data = [
            {
                "title": "BoJ Decision",
                "impact": "HIGH",
                "time": "2026-02-12T03:00:00Z",
                "timezone": "UTC",
                "currency": "JPY"
            }
        ]

        sensor.update_calendar(calendar_data)

        assert len(sensor.events) == 1
        event = sensor.events[0]
        assert event.title == "BoJ Decision"
        assert event.time.hour == 3
        assert event.timezone_source == "UTC"

    def test_update_calendar_with_multiple_events(self):
        """Test updating calendar with multiple events from different timezones."""
        sensor = NewsSensor()
        calendar_data = [
            {
                "title": "NFP",
                "impact": "HIGH",
                "time": "2026-02-12T08:30:00",
                "timezone": "America/New_York",
                "currency": "USD"
            },
            {
                "title": "BoJ Decision",
                "impact": "HIGH",
                "time": "2026-02-12T03:00:00",
                "timezone": "Asia/Tokyo",
                "currency": "JPY"
            },
            {
                "title": "ECB Rate Decision",
                "impact": "HIGH",
                "time": "2026-02-12T12:30:00",
                "timezone": "Europe/London",
                "currency": "EUR"
            }
        ]

        sensor.update_calendar(calendar_data)

        assert len(sensor.events) == 3

        # NFP 8:30 AM EST = 13:30 UTC
        assert sensor.events[0].time.hour == 13
        # BoJ 3:00 AM JST = previous day 18:00 UTC
        assert sensor.events[1].time.hour == 18
        # ECB 12:30 PM GMT = 12:30 UTC
        assert sensor.events[2].time.hour == 12

    def test_update_calendar_with_naive_datetime(self):
        """Test updating calendar with naive datetime (no timezone)."""
        sensor = NewsSensor()
        calendar_data = [
            {
                "title": "Test Event",
                "impact": "HIGH",
                "time": "2026-02-12T10:00:00",  # Naive - treated as UTC
                "timezone": "UTC"
            }
        ]

        sensor.update_calendar(calendar_data)

        assert len(sensor.events) == 1
        event = sensor.events[0]
        # Naive datetime should be made aware as UTC
        assert event.time.tzinfo == timezone.utc

    def test_update_calendar_replaces_events(self):
        """Test that update_calendar replaces existing events (clears stale ones).
        
        Per Comment 2: update_calendar() should clear old events at the start
        to prevent stale events from persisting across calendar updates.
        """
        sensor = NewsSensor()
        sensor.events = [
            NewsEvent(
                title="Existing Event",
                impact="MEDIUM",
                time=datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc),
                timezone_source="UTC"
            )
        ]

        calendar_data = [
            {
                "title": "New Event",
                "impact": "HIGH",
                "time": "2026-02-12T11:00:00Z",
                "timezone": "UTC"
            }
        ]

        sensor.update_calendar(calendar_data)

        # After update_calendar, should only have the new event (old one cleared)
        assert len(sensor.events) == 1
        assert sensor.events[0].title == "New Event"

    def test_repeated_update_calendar_clears_stale_events(self):
        """Test that repeated update_calendar calls don't keep old events.
        
        Per Comment 2: Ensures that NewsSensor clears existing events at the start
        of update_calendar(), preventing stale events from accumulating in persistent
        kill zones.
        """
        sensor = NewsSensor()
        
        # First update with two events
        calendar_data_1 = [
            {
                "title": "NFP Employment Change",
                "impact": "HIGH",
                "time": "2026-02-12T08:30:00",
                "timezone": "America/New_York",
                "currency": "USD"
            },
            {
                "title": "BoJ Interest Rate Decision",
                "impact": "HIGH",
                "time": "2026-02-12T03:00:00",
                "timezone": "Asia/Tokyo",
                "currency": "JPY"
            }
        ]
        sensor.update_calendar(calendar_data_1)
        assert len(sensor.events) == 2
        assert sensor.events[0].title == "NFP Employment Change"
        assert sensor.events[1].title == "BoJ Interest Rate Decision"
        
        # Second update with one new event (different from first)
        calendar_data_2 = [
            {
                "title": "ECB Rate Decision",
                "impact": "HIGH",
                "time": "2026-02-12T12:30:00",
                "timezone": "Europe/London",
                "currency": "EUR"
            }
        ]
        sensor.update_calendar(calendar_data_2)
        
        # After second update, should only have the new event (stale ones cleared)
        assert len(sensor.events) == 1
        assert sensor.events[0].title == "ECB Rate Decision"
        
        # Verify check_state only uses current events
        # At a time between old NFP and new ECB, should be SAFE (not in stale kill zone)
        current_time = datetime(2026, 2, 12, 11, 0, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)
        assert state == "SAFE"  # No old events affecting this


class TestNewsKillZones:
    """Test kill zone detection."""

    def test_inside_pre_news_kill_zone(self):
        """Test detection inside pre-news kill zone."""
        sensor = NewsSensor(kill_zone_minutes_pre=15, kill_zone_minutes_post=15)

        # Event at 13:30 UTC, current time at 13:20 UTC (10 min before)
        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        current_time = datetime(2026, 2, 12, 13, 20, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        assert state == "KILL_ZONE"

    def test_inside_post_news_kill_zone(self):
        """Test detection inside post-news kill zone."""
        sensor = NewsSensor(kill_zone_minutes_pre=15, kill_zone_minutes_post=15)

        # Event at 13:30 UTC, current time at 13:40 UTC (10 min after)
        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        current_time = datetime(2026, 2, 12, 13, 40, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        assert state == "KILL_ZONE"

    def test_pre_news_state(self):
        """Test detection in pre-news window."""
        sensor = NewsSensor(kill_zone_minutes_pre=15, kill_zone_minutes_post=15)

        # Event at 13:30 UTC, current time at 13:05 UTC (25 min before)
        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        current_time = datetime(2026, 2, 12, 13, 5, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        assert state == "PRE_NEWS"

    def test_post_news_state(self):
        """Test detection in post-news recovery."""
        sensor = NewsSensor(kill_zone_minutes_pre=15, kill_zone_minutes_post=15)

        # Event at 13:30 UTC, current time at 14:00 UTC (30 min after)
        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        current_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        assert state == "POST_NEWS"

    def test_safe_state_no_events(self):
        """Test safe state when no high-impact events."""
        sensor = NewsSensor()

        # No events, or only medium/low impact
        current_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        assert state == "SAFE"

    def test_kill_zone_only_for_high_impact(self):
        """Test kill zone only triggers for HIGH impact events."""
        sensor = NewsSensor()

        # Add medium impact event
        sensor.events = [
            NewsEvent(
                title="Medium Event",
                impact="MEDIUM",
                time=datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc),
                timezone_source="UTC"
            )
        ]

        current_time = datetime(2026, 2, 12, 14, 5, tzinfo=timezone.utc)
        state = sensor.check_state(current_time)

        # Should be SAFE (medium impact doesn't trigger kill zone)
        assert state == "SAFE"


class TestNewsSensorTimezoneHandling:
    """Test timezone-aware time handling in check_state."""

    def test_check_state_with_naive_utc(self):
        """Test check_state handles naive UTC datetime."""
        sensor = NewsSensor()

        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        # Pass naive datetime (should be converted to aware UTC)
        naive_time = datetime(2026, 2, 12, 13, 20, tzinfo=None)
        state = sensor.check_state(naive_time)

        assert state == "KILL_ZONE"

    def test_check_state_with_different_timezone(self):
        """Test check_state handles non-UTC timezone."""
        sensor = NewsSensor()

        event_time = datetime(2026, 2, 12, 13, 30, tzinfo=timezone.utc)
        sensor.events = [
            NewsEvent(
                title="NFP",
                impact="HIGH",
                time=event_time,
                timezone_source="UTC"
            )
        ]

        # Pass datetime with different timezone (should be converted to UTC)
        # EST 8:20 AM = UTC 13:20, which is 10 min before event at UTC 13:30
        from zoneinfo import ZoneInfo
        est_time = datetime(2026, 2, 12, 8, 20, tzinfo=ZoneInfo("America/New_York"))
        state = sensor.check_state(est_time)

        # Should still detect kill zone (after conversion)
        assert state == "KILL_ZONE"


class TestNewsUpcomingEvents:
    """Test getting upcoming events."""

    def test_get_upcoming_events(self):
        """Test retrieving upcoming high-impact events."""
        sensor = NewsSensor()
        now = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        # Event in 2 hours and 23 hours (both within 24-hour window)
        sensor.events = [
            NewsEvent(
                title="Event in 2h",
                impact="HIGH",
                time=now + timedelta(hours=2),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Event in 23h",
                impact="HIGH",
                time=now + timedelta(hours=23),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Old Event",
                impact="HIGH",
                time=now - timedelta(hours=1),
                timezone_source="UTC"
            )
        ]

        # Mock datetime.now(timezone.utc) to use fixed time
        with patch('src.router.sensors.news.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            upcoming = sensor.get_upcoming_events(hours_ahead=24)

        assert len(upcoming) == 2
        assert upcoming[0].title == "Event in 2h"
        assert upcoming[1].title == "Event in 23h"

    def test_get_upcoming_events_only_high_impact(self):
        """Test upcoming events only includes HIGH impact."""
        sensor = NewsSensor()
        now = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        sensor.events = [
            NewsEvent(
                title="High Impact",
                impact="HIGH",
                time=now + timedelta(hours=1),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Medium Impact",
                impact="MEDIUM",
                time=now + timedelta(hours=2),
                timezone_source="UTC"
            )
        ]

        # Mock datetime.now(timezone.utc) to use fixed time
        with patch('src.router.sensors.news.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            upcoming = sensor.get_upcoming_events(hours_ahead=24)

        # Should only include HIGH impact
        assert len(upcoming) == 1
        assert upcoming[0].title == "High Impact"


class TestNewsClearOldEvents:
    """Test clearing old events."""

    def test_clear_old_events(self):
        """Test removing old events."""
        sensor = NewsSensor()
        now = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        # Add old and new events
        sensor.events = [
            NewsEvent(
                title="Event 50h ago",
                impact="HIGH",
                time=now - timedelta(hours=50),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Event 30h ago",
                impact="HIGH",
                time=now - timedelta(hours=30),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Event 10h ago",
                impact="HIGH",
                time=now - timedelta(hours=10),
                timezone_source="UTC"
            ),
            NewsEvent(
                title="Recent Event",
                impact="HIGH",
                time=now - timedelta(hours=1),
                timezone_source="UTC"
            )
        ]

        # Mock datetime.now(timezone.utc) to use fixed time
        with patch('src.router.sensors.news.datetime') as mock_datetime:
            mock_datetime.now.return_value = now
            # Clear events older than 24 hours
            removed = sensor.clear_old_events(hours_ago=24)

        assert removed == 2
        assert len(sensor.events) == 2
        assert sensor.events[0].title == "Event 10h ago"
        assert sensor.events[1].title == "Recent Event"
