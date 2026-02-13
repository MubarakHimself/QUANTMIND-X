"""
News Sensor (Time Guardian)
Tracks high-impact news events and enforces Kill Zones.

V3: Timezone-aware event handling with SessionDetector integration.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import logging

from src.router.sessions import SessionDetector

logger = logging.getLogger(__name__)


@dataclass
class NewsEvent:
    """
    Represents a news event with timezone information.

    Args:
        title: Event title/description
        impact: Event impact level (HIGH, MEDIUM, LOW)
        time: Event time in UTC (MUST be UTC for consistency)
        timezone_source: Original timezone of the event (e.g., "America/New_York")
        currency: Associated currency (optional)
    """
    title: str
    impact: str  # HIGH, MEDIUM, LOW
    time: datetime  # MUST be in UTC
    timezone_source: str = "UTC"  # Track original timezone
    currency: Optional[str] = None


class NewsSensor:
    """
    Tracks high-impact news events and enforces Kill Zones.

    V3: Timezone-aware with configurable kill zones.
    """

    def __init__(self, kill_zone_minutes_pre: int = 15, kill_zone_minutes_post: int = 15):
        """
        Initialize NewsSensor with configurable kill zones.

        Args:
            kill_zone_minutes_pre: Minutes before event to block trading (default: 15)
            kill_zone_minutes_post: Minutes after event to block trading (default: 15)
        """
        self.events: List[NewsEvent] = []
        self.kill_zone_minutes_pre = kill_zone_minutes_pre
        self.kill_zone_minutes_post = kill_zone_minutes_post
        logger.info(f"NewsSensor initialized with kill zone: ±{kill_zone_minutes_pre}min")

    def update_calendar(self, calendar_data: List[Dict[str, Any]]) -> None:
        """
        Load events from Crawler/API with timezone conversion.

        V3: Converts event times from various timezones to UTC.

        Args:
            calendar_data: List of event dicts with keys:
                - title: Event description
                - impact: Impact level (HIGH, MEDIUM, LOW)
                - time: Event time in ISO format or datetime
                - timezone: IANA timezone name (default: "UTC")
                - currency: Associated currency (optional)

        Example:
            [
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
        """
        # Clear existing events to prevent stale events from accumulating
        self.events = []
        
        for event_data in calendar_data:
            try:
                # Parse event time
                event_time = event_data.get("time")
                event_tz = event_data.get("timezone", "UTC")

                # Convert to datetime if string
                if isinstance(event_time, str):
                    try:
                        event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                    except ValueError:
                        logger.warning(f"Invalid time format for event: {event_data.get('title')}")
                        continue
                elif not isinstance(event_time, datetime):
                    logger.warning(f"Invalid time type for event: {event_data.get('title')}")
                    continue

                # Convert to UTC if not already UTC
                if event_tz != "UTC":
                    utc_time = SessionDetector.convert_to_utc(event_time, event_tz)
                    timezone_source = event_tz
                else:
                    # Ensure datetime is timezone-aware UTC
                    if event_time.tzinfo is None:
                        utc_time = event_time.replace(tzinfo=timezone.utc)
                    elif event_time.tzinfo != timezone.utc:
                        utc_time = event_time.astimezone(timezone.utc)
                    else:
                        utc_time = event_time
                    timezone_source = "UTC"

                # Create NewsEvent with UTC time
                news_event = NewsEvent(
                    title=event_data.get("title", "Unknown Event"),
                    impact=event_data.get("impact", "MEDIUM"),
                    time=utc_time,
                    timezone_source=timezone_source,
                    currency=event_data.get("currency")
                )

                self.events.append(news_event)
                logger.debug(f"Added news event: {news_event.title} at {utc_time.isoformat()} UTC")

            except Exception as e:
                logger.error(f"Failed to process event {event_data.get('title', 'Unknown')}: {e}")

        logger.info(f"Updated calendar with {len(self.events)} events")

    def check_state(self, current_utc: Optional[datetime] = None) -> str:
        """
        Returns current news state (SAFE, PRE_NEWS, KILL_ZONE, POST_NEWS).

        V3: Uses timezone-aware UTC time comparison.

        Args:
            current_utc: Current time in UTC (default: now)

        Returns:
            State string: SAFE, PRE_NEWS, KILL_ZONE, POST_NEWS

        Note:
            All time comparisons use UTC for consistency.
        """
        # Default to current UTC time
        if current_utc is None:
            current_utc = datetime.now(timezone.utc)
        elif current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)
        elif current_utc.tzinfo != timezone.utc:
            current_utc = current_utc.astimezone(timezone.utc)

        for event in self.events:
            if event.impact != "HIGH":
                continue

            # Calculate time difference (both times are UTC)
            time_diff_minutes = (event.time - current_utc).total_seconds() / 60

            # Inside Kill Zone (-post to +pre)
            if -self.kill_zone_minutes_post <= time_diff_minutes <= self.kill_zone_minutes_pre:
                logger.info(f"Inside KILL ZONE for {event.title}: {time_diff_minutes:.0f} min to event")
                return "KILL_ZONE"

            # Approaching news (within 2x pre window)
            if time_diff_minutes > 0 and time_diff_minutes <= self.kill_zone_minutes_pre * 2:
                return "PRE_NEWS"

            # Post news recovery (within 2x post window)
            if time_diff_minutes < 0 and abs(time_diff_minutes) <= self.kill_zone_minutes_post * 2:
                return "POST_NEWS"

        return "SAFE"

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[NewsEvent]:
        """
        Get upcoming high-impact events within time window.

        Args:
            hours_ahead: Hours to look ahead (default: 24)

        Returns:
            List of upcoming high-impact events sorted by time
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)

        upcoming = [
            event for event in self.events
            if event.impact == "HIGH" and now < event.time <= cutoff
        ]

        # Sort by event time
        upcoming.sort(key=lambda e: e.time)
        return upcoming

    def clear_old_events(self, hours_ago: int = 48) -> int:
        """
        Remove events older than specified hours.

        Args:
            hours_ago: Remove events older than this many hours (default: 48)

        Returns:
            Number of events removed
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        original_count = len(self.events)

        self.events = [
            event for event in self.events
            if event.time > cutoff
        ]

        removed = original_count - len(self.events)
        if removed > 0:
            logger.info(f"Cleared {removed} old events (older than {hours_ago}h)")

        return removed
