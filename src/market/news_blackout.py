"""
NewsBlackoutService — Shared service for economic calendar-driven kill zones.

Replaces the old NewsFeedPoller (which only fed news articles).
This service feeds HIGH-impact economic events into NewsSensor,
enabling per-session kill zone enforcement across Tokyo/London/NY sessions.

Responsibilities:
1. Poll Finnhub /calendar/economic every 30 minutes
2. Map events to sessions via currency exposure
3. Populate NewsSensor via update_calendar()
4. Expose is_blackout_active(session_name) for per-session checks
5. Broadcast news state transitions via WebSocket

Design: docs/superpowers/specs/2026-03-30-news-calendar-kill-switch-design.md
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

try:
    import finnhub
except ImportError:
    finnhub = None

logger = logging.getLogger(__name__)


def is_news_blackout_configured() -> bool:
    """Return True only when the Finnhub dependency and API key are both available."""
    return finnhub is not None and bool(os.environ.get("FINNHUB_API_KEY", "").strip())

# ─────────────────────────────────────────────────────────────────────────────
# Session → Currency exposure mapping
# A session is killed only if an event's currency overlaps with its exposure.
# Source: Trading docs / session definitions in src/router/sessions.py
# ─────────────────────────────────────────────────────────────────────────────

SESSION_CURRENCY_MAP: Dict[str, Set[str]] = {
    "TOKYO":    {"JPY", "AUD", "NZD", "CNY"},
    "SYDNEY":   {"AUD", "NZD"},
    "LONDON":   {"EUR", "GBP", "CHF"},
    "NEW_YORK": {"USD", "CAD"},
    # London-NY overlap — both London and NY currencies exposed
    "OVERLAP":  {"EUR", "GBP", "USD", "CHF", "CAD"},
}

# All tracked currencies
TRACKED_CURRENCIES: Set[str] = set()
for currencies in SESSION_CURRENCY_MAP.values():
    TRACKED_CURRENCIES |= currencies


@dataclass
class SessionBlackoutStatus:
    """Per-session blackout status."""
    session: str
    is_blackout: bool
    reason: Optional[str] = None
    minutes_to_event: Optional[float] = None
    event_name: Optional[str] = None


class NewsBlackoutService:
    """
    Economic calendar service that drives session-level kill zones.

    Uses Finnhub /calendar/economic as the data source.
    Refreshes every 30 minutes. Stores 7 days of upcoming HIGH-impact events.
    """

    def __init__(
        self,
        fetch_interval_minutes: int = 30,
        lookahead_days: int = 7,
        kill_zone_pre_minutes: int = 15,
        kill_zone_post_minutes: int = 15,
    ):
        """
        Args:
            fetch_interval_minutes: How often to poll Finnhub (default 30)
            lookahead_days: How many days ahead to fetch events (default 7)
            kill_zone_pre_minutes: Minutes before event to start kill zone (default 15)
            kill_zone_post_minutes: Minutes after event to end kill zone (default 15)
        """
        self._fetch_interval = fetch_interval_minutes
        self._lookahead_days = lookahead_days
        self._kill_zone_pre = kill_zone_pre_minutes
        self._kill_zone_post = kill_zone_post_minutes

        self._api_key: Optional[str] = None
        self._client = None
        self._events: List[Dict] = []          # Normalized events ready for NewsSensor
        self._last_fetch: Optional[datetime] = None
        self._scheduler = None                # APScheduler job handle

        # Owns the NewsSensor — created here so it's ready before start()
        from src.router.sensors.news import NewsSensor
        self._news_sensor = NewsSensor(
            kill_zone_minutes_pre=kill_zone_pre_minutes,
            kill_zone_minutes_post=kill_zone_post_minutes
        )

        # WebSocket broadcast — set by set_ws_manager()
        self._ws_manager = None

    # ─────────────────────────────────────────────────────────────────────────
    # Initialization / lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def _get_client(self):
        """Lazily create Finnhub client."""
        if finnhub is None:
            raise RuntimeError("finnhub-python package required. Install: pip install finnhub-python>=2.4.0")
        if self._client is None:
            self._api_key = os.environ.get("FINNHUB_API_KEY")
            if not self._api_key:
                raise RuntimeError("FINNHUB_API_KEY not configured")
            self._client = finnhub.Client(api_key=self._api_key)
        return self._client

    def set_news_sensor(self, news_sensor):
        """Wire the NewsSensor from ProgressiveKillSwitch."""
        self._news_sensor = news_sensor
        logger.info("NewsBlackoutService wired to NewsSensor")

    def set_ws_manager(self, ws_manager):
        """Set WebSocket manager for broadcasting news state transitions."""
        self._ws_manager = ws_manager

    def start(self):
        """Start the background scheduler. Call once at app startup."""
        self._fetch_and_update()          # Fetch immediately
        self._start_scheduler()
        logger.info(
            f"NewsBlackoutService started — fetch interval={self._fetch_interval}min, "
            f"lookahead={self._lookahead_days}d, kill zone ±{self._kill_zone_pre}min"
        )

    def stop(self):
        """Stop the background scheduler."""
        if self._scheduler:
            self._scheduler.remove_all_jobs()
            self._scheduler.shutdown(wait=False)
            self._scheduler = None
            logger.info("NewsBlackoutService scheduler stopped")

    def _start_scheduler(self):
        """Set up APScheduler for periodic refreshes."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            self._scheduler = BackgroundScheduler()
            self._scheduler.add_job(
                self._fetch_and_update,
                "interval",
                minutes=self._fetch_interval,
                id="news_blackout_fetch",
                replace_existing=True,
            )
            self._scheduler.start()
        except Exception as e:
            logger.warning(f"Could not start NewsBlackoutService scheduler: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Finnhub fetch + normalization
    # ─────────────────────────────────────────────────────────────────────────

    def _fetch_and_update(self):
        """Fetch from Finnhub, normalize, update NewsSensor, broadcast."""
        try:
            events = self._fetch_finnhub_calendar()
            self._events = events
            self._last_fetch = datetime.now(timezone.utc)

            if self._news_sensor:
                # Convert to NewsSensor format and feed it
                calendar_data = self._events_to_calendar_format(events)
                self._news_sensor.update_calendar(calendar_data)
                logger.info(f"NewsBlackoutService fed {len(events)} events to NewsSensor")
            else:
                logger.warning("NewsBlackoutService: NewsSensor not wired yet — events not fed")

            # Broadcast current state
            self._broadcast_state()

        except Exception as e:
            logger.error(f"NewsBlackoutService fetch failed: {e}")

    def _fetch_finnhub_calendar(self) -> List[Dict]:
        """
        Fetch economic calendar from Finnhub for the lookahead window.
        Returns normalized list of HIGH-impact events.
        """
        client = self._get_client()

        today = datetime.now(timezone.utc)
        date_from = today.strftime("%Y-%m-%d")
        date_to = (today + timedelta(days=self._lookahead_days)).strftime("%Y-%m-%d")

        try:
            resp = client.economic_calendar(from_date=date_from, to_date=date_to)
        except Exception as e:
            logger.error(f"Finnhub economic_calendar call failed: {e}")
            return []

        raw_events = resp.get("economicCalendar", []) if isinstance(resp, dict) else []
        normalized = []

        for ev in raw_events:
            impact = (ev.get("impact") or "").lower()
            if impact != "high":
                continue

            currency = (ev.get("currency") or "").upper()
            if currency not in TRACKED_CURRENCIES:
                continue

            # Parse time — Finnhub returns "YYYY-MM-DD HH:MM:SS" or UNIX timestamp
            event_time_raw = ev.get("time", "")
            event_time_utc = self._parse_event_time(event_time_raw)
            if not event_time_utc:
                continue

            normalized.append({
                "title": ev.get("event", "Unknown Event"),
                "impact": "HIGH",
                "time": event_time_utc,
                "timezone": "UTC",
                "currency": currency,
                "forecast": ev.get("estimate"),
                "actual": ev.get("actual"),
                "previous": ev.get("prev"),
                "unit": ev.get("unit"),
            })

        logger.debug(f"Finnhub fetched {len(normalized)} high-impact events")
        return normalized

    def _parse_event_time(self, time_str: str) -> Optional[datetime]:
        """Parse Finnhub event time string to UTC datetime."""
        if not time_str:
            return None
        try:
            # Finnhub format: "2026-04-03 14:30:00"
            dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            try:
                # ISO format with timezone
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
                return dt.astimezone(timezone.utc)
            except ValueError:
                logger.warning(f"Could not parse event time: {time_str}")
                return None

    def _events_to_calendar_format(self, events: List[Dict]) -> List[Dict]:
        """Convert Finnhub event format to NewsSensor.update_calendar() format."""
        return [
            {
                "title": ev["title"],
                "impact": ev["impact"],
                "time": ev["time"].isoformat() if isinstance(ev["time"], datetime) else ev["time"],
                "timezone": ev.get("timezone", "UTC"),
                "currency": ev.get("currency"),
            }
            for ev in events
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Session-level blackout queries
    # ─────────────────────────────────────────────────────────────────────────

    def is_blackout_active(self, session_name: str) -> bool:
        """
        Check if a given session is currently in a kill zone.
        Called by SessionMonitor or any code that needs per-session kill zone status.

        Args:
            session_name: One of TOKYO, SYDNEY, LONDON, NEW_YORK, OVERLAP

        Returns:
            True if the session is within kill zone of a relevant HIGH-impact event
        """
        session_currencies = SESSION_CURRENCY_MAP.get(session_name.upper(), set())
        if not session_currencies:
            return False

        now = datetime.now(timezone.utc)
        for ev in self._events:
            if ev.get("currency") not in session_currencies:
                continue

            event_time = ev.get("time")
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            if event_time.tzinfo and event_time.tzinfo != timezone.utc:
                event_time = event_time.astimezone(timezone.utc)

            diff_minutes = (event_time - now).total_seconds() / 60

            # Within kill zone: -post <= diff <= pre
            if -self._kill_zone_post <= diff_minutes <= self._kill_zone_pre:
                return True

        return False

    def get_session_status(self, session_name: str) -> SessionBlackoutStatus:
        """
        Get detailed blackout status for a session.
        Used by UI to show countdown timers and event names.
        """
        session_currencies = SESSION_CURRENCY_MAP.get(session_name.upper(), set())
        now = datetime.now(timezone.utc)
        nearest_event: Optional[Dict] = None
        nearest_diff = float("inf")

        for ev in self._events:
            if ev.get("currency") not in session_currencies:
                continue

            event_time = ev.get("time")
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            if event_time.tzinfo and event_time.tzinfo != timezone.utc:
                event_time = event_time.astimezone(timezone.utc)

            diff_minutes = (event_time - now).total_seconds() / 60

            if abs(diff_minutes) < abs(nearest_diff):
                nearest_diff = diff_minutes
                nearest_event = ev

        if nearest_event:
            event_time = nearest_event.get("time")
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))
            is_kill_zone = -self._kill_zone_post <= nearest_diff <= self._kill_zone_pre

            return SessionBlackoutStatus(
                session=session_name,
                is_blackout=is_kill_zone,
                reason=f"HIGH impact: {nearest_event.get('title')}",
                minutes_to_event=round(nearest_diff, 1),
                event_name=nearest_event.get("title"),
            )

        return SessionBlackoutStatus(
            session=session_name,
            is_blackout=False,
            reason=None,
            minutes_to_event=None,
            event_name=None,
        )

    def get_all_session_statuses(self) -> Dict[str, SessionBlackoutStatus]:
        """Get blackout status for all tracked sessions."""
        return {session: self.get_session_status(session) for session in SESSION_CURRENCY_MAP}

    def get_upcoming_high_impact_events(
        self, hours_ahead: int = 48, currencies: Optional[Set[str]] = None
    ) -> List[Dict]:
        """
        Get upcoming HIGH-impact events.
        Used by EconomicCalendarPanel and news kill zone UI.
        """
        if currencies is None:
            currencies = TRACKED_CURRENCIES

        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        events = []

        for ev in self._events:
            currency = ev.get("currency", "")
            if currency not in currencies:
                continue

            event_time = ev.get("time")
            if isinstance(event_time, str):
                event_time = datetime.fromisoformat(event_time.replace("Z", "+00:00"))

            if now <= event_time <= cutoff:
                events.append({**ev, "minutes_to_event": (event_time - now).total_seconds() / 60})

        events.sort(key=lambda e: e["time"])
        return events

    # ─────────────────────────────────────────────────────────────────────────
    # WebSocket broadcast
    # ─────────────────────────────────────────────────────────────────────────

    def _broadcast_state(self):
        """Broadcast current news kill zone state to all WebSocket clients."""
        if not self._ws_manager:
            return

        try:
            statuses = self.get_all_session_statuses()
            payload = {
                "type": "news_blackout_update",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sessions": {
                    s: {
                        "is_blackout": st.is_blackout,
                        "reason": st.reason,
                        "minutes_to_event": st.minutes_to_event,
                        "event_name": st.event_name,
                    }
                    for s, st in statuses.items()
                },
                "upcoming_events": [
                    {
                        "title": e.get("title"),
                        "currency": e.get("currency"),
                        "time": e.get("time").isoformat() if isinstance(e.get("time"), datetime) else e.get("time"),
                        "minutes_to_event": e.get("minutes_to_event"),
                    }
                    for e in self.get_upcoming_high_impact_events(hours_ahead=24)
                ],
            }
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._ws_manager.broadcast(payload))
            else:
                asyncio.run(self._ws_manager.broadcast(payload))
        except Exception as e:
            logger.warning(f"Failed to broadcast news blackout state: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Properties for debugging / monitoring
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def last_fetch(self) -> Optional[datetime]:
        return self._last_fetch

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def is_news_sensor_wired(self) -> bool:
        return self._news_sensor is not None
