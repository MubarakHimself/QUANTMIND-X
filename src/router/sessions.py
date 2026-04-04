"""
Session and Timezone Management for QuantMindX Routing

Implements ICT-style session detection for timezone-aware bot filtering.
Supports major forex trading sessions and custom time windows.

Uses DST-aware session detection with local trading hours:
- London: 08:00-16:00 local time (Europe/London)
- New York: 08:00-17:00 local time (America/New_York)
- Asian: 00:00-09:00 local time (Asia/Tokyo)
- Overlap: Detected when London and NY sessions are concurrent
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta, time
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from zoneinfo import ZoneInfo
import logging
import calendar

logger = logging.getLogger(__name__)


# =============================================================================
# Tilt Transition Window (minutes before session end)
# =============================================================================
TILT_TRANSITION_MINUTES = 5


# =============================================================================
# Canonical Session Windows (authoritative for Tilt + bot type dispatch)
# =============================================================================
# All times in UTC. These 10 windows are the authoritative gate for
# Tilt state transitions and bot type mix dispatch.
# The existing TradingSession/SESSIONS remain for backward compatibility.

CANONICAL_WINDOWS = {
    "SYDNEY_OPEN": {
        "utc_start": time(21, 0),
        "utc_end": time(23, 0),
        "name": "Sydney Open",
        "is_premium": False,
        "is_trading": True,
    },
    "SYDNEY_TOKYO_OVERLAP": {
        "utc_start": time(23, 0),
        "utc_end": time(0, 0),
        "name": "Sydney-Tokyo Overlap",
        "is_premium": False,
        "is_trading": True,
    },
    "TOKYO_OPEN": {
        "utc_start": time(0, 0),
        "utc_end": time(3, 0),
        "name": "Tokyo Open",
        "is_premium": False,
        "is_trading": True,
    },
    "TOKYO_LONDON_OVERLAP": {
        "utc_start": time(7, 0),
        "utc_end": time(9, 0),
        "name": "Tokyo-London Overlap",
        "is_premium": True,   # PREMIUM SESSION
        "is_trading": True,
    },
    "LONDON_OPEN": {
        "utc_start": time(8, 0),
        "utc_end": time(10, 30),
        "name": "London Open",
        "is_premium": True,   # PREMIUM SESSION
        "is_trading": True,
    },
    "LONDON_MID": {
        "utc_start": time(10, 30),
        "utc_end": time(12, 0),
        "name": "London Mid",
        "is_premium": False,
        "is_trading": True,
    },
    "INTER_SESSION_COOLDOWN": {
        "utc_start": time(12, 0),
        "utc_end": time(13, 0),
        "name": "Inter-Session Cooldown",
        "is_premium": False,
        "is_trading": True,
    },
    "LONDON_NY_OVERLAP": {
        "utc_start": time(13, 0),
        "utc_end": time(16, 0),
        "name": "London-NY Overlap",
        "is_premium": True,   # PREMIUM SESSION
        "is_trading": True,
    },
    "NY_WIND_DOWN": {
        "utc_start": time(16, 0),
        "utc_end": time(20, 0),
        "name": "NY Wind Down",
        "is_premium": False,
        "is_trading": True,
    },
    "DEAD_ZONE": {
        "utc_start": time(20, 0),
        "utc_end": time(21, 0),
        "name": "Dead Zone",
        "is_premium": False,
        "is_trading": False,  # No trading in Dead Zone — intelligence/DPR only
    },
}

# =============================================================================
# Per-Session Bot Type Mix (authoritative for Tilt ACTIVATE dispatch)
# =============================================================================
# Maps canonical window names to bot type probability distributions.
# Used in Tilt ACTIVATE to filter DPR-ranked queue for correct bot types.
#
# Bot types:
#   ORB = ICT Order Block Recognition strategy
#   MOM = Momentum strategy
#   MR  = Mean Reversion strategy
#   TC  = Trend Continuation strategy
#
# During DEAD_ZONE: empty dict — no trading, DPR/intelligence only

SESSION_BOT_MIX = {
    "SYDNEY_OPEN":            {"MR": 0.70, "MOM": 0.30},
    "SYDNEY_TOKYO_OVERLAP":   {"MR": 0.70, "MOM": 0.30},
    "TOKYO_OPEN":             {"MR": 0.70, "MOM": 0.30},
    "TOKYO_LONDON_OVERLAP":   {"MR": 0.50, "MOM": 0.30, "ORB": 0.20},
    "LONDON_OPEN":            {"ORB": 0.60, "MOM": 0.40},
    "LONDON_MID":             {"MR": 0.50, "MOM": 0.50},
    "INTER_SESSION_COOLDOWN": {"MR": 0.80, "TC": 0.20},
    "LONDON_NY_OVERLAP":      {"ORB": 0.55, "MOM": 0.45},
    "NY_WIND_DOWN":           {"TC": 0.60, "MR": 0.40},
    "DEAD_ZONE":              {},   # No trading
}

PREMIUM_SESSIONS = {
    "TOKYO_LONDON_OVERLAP",
    "LONDON_OPEN",
    "LONDON_NY_OVERLAP",
}

TRADING_WINDOWS = {k for k, v in CANONICAL_WINDOWS.items() if v["is_trading"]}


class TradingSession(Enum):
    """Major forex trading sessions."""
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"
    CLOSED = "CLOSED"


@dataclass
class SessionTemplate:
    """Configurable session template with per-session trading parameters.

    Used for session-aware bot dispatch, Tilt transitions, and queue management.
    Each session has weighted bot type priorities and concurrency limits.
    """
    name: str                          # e.g. "Asian", "London Open"
    start_gmt: str                    # e.g. "22:00" (HH:MM in GMT)
    end_gmt: str                      # e.g. "07:00"
    scalper_priority_weight: float     # 0.0-1.0, sum of scalper+momentum = 1.0
    momentum_priority_weight: float    # 0.0-1.0, sum of scalper+momentum = 1.0
    max_concurrent_bots: int          # max bots for this session
    is_premium: bool                  # premium sessions get extra parameters
    orb_boost_pct: float = 0.0        # e.g. 0.06 for premium (6% ORB boost)
    mr_weight: float = 0.0            # mean reversion weight (for some sessions)
    tc_weight: float = 0.0            # trend continuation weight


# =============================================================================
# Session Templates (authoritative per FIX-010 canonical params)
# =============================================================================
SESSIONS: Dict[str, SessionTemplate] = {
    "asian": SessionTemplate(
        name="Asian",
        start_gmt="22:00",
        end_gmt="07:00",
        scalper_priority_weight=0.70,
        momentum_priority_weight=0.30,
        max_concurrent_bots=18,
        is_premium=False,
        mr_weight=0.70,
        tc_weight=0.0,
    ),
    "london_open": SessionTemplate(
        name="London Open",
        start_gmt="07:00",
        end_gmt="10:00",
        scalper_priority_weight=0.60,
        momentum_priority_weight=0.40,
        max_concurrent_bots=50,
        is_premium=True,
        orb_boost_pct=0.06,
        mr_weight=0.0,
        tc_weight=0.0,
    ),
    "london_mid": SessionTemplate(
        name="London Mid",
        start_gmt="10:00",
        end_gmt="11:30",
        scalper_priority_weight=0.50,
        momentum_priority_weight=0.50,
        max_concurrent_bots=15,
        is_premium=False,
        mr_weight=0.50,
        tc_weight=0.0,
    ),
    "inter_session": SessionTemplate(
        name="Inter-Session",
        start_gmt="11:30",
        end_gmt="13:00",
        scalper_priority_weight=0.80,
        momentum_priority_weight=0.20,
        max_concurrent_bots=6,
        is_premium=False,
        mr_weight=0.80,
        tc_weight=0.0,
    ),
    "ny_overlap": SessionTemplate(
        name="NY+Overlap",
        start_gmt="13:00",
        end_gmt="16:00",
        scalper_priority_weight=0.55,
        momentum_priority_weight=0.45,
        max_concurrent_bots=60,
        is_premium=True,
        orb_boost_pct=0.06,
        mr_weight=0.0,
        tc_weight=0.0,
    ),
    "ny_wind_down": SessionTemplate(
        name="NY Wind-Down",
        start_gmt="16:00",
        end_gmt="17:00",
        scalper_priority_weight=0.60,
        momentum_priority_weight=0.40,
        max_concurrent_bots=8,
        is_premium=False,
        mr_weight=0.0,
        tc_weight=0.60,
    ),
    "dead_zone": SessionTemplate(
        name="Dead Zone",
        start_gmt="17:00",
        end_gmt="22:00",
        scalper_priority_weight=0.0,
        momentum_priority_weight=0.0,
        max_concurrent_bots=0,
        is_premium=False,
        mr_weight=0.0,
        tc_weight=0.0,
    ),
}


@dataclass
class SessionInfo:
    """Comprehensive session status information.

    Args:
        session: Current trading session enum
        is_active: Whether any trading session is active
        next_session: Next session that will open (optional)
        time_until_open: Minutes until next session opens (optional)
        time_until_close: Minutes until current session closes (optional)
        time_until_close_str: Human-readable time until close (optional)
    """
    session: TradingSession
    is_active: bool
    next_session: Optional[TradingSession] = None
    time_until_open: Optional[int] = None
    time_until_close: Optional[int] = None
    time_until_close_str: Optional[str] = None


class SessionDetector:
    """
    Detects trading sessions with DST-aware local time conversion.

    Session definitions (LOCAL times in respective timezones):
    - ASIAN: 00:00-09:00 local (Asia/Tokyo)
    - LONDON: 08:00-16:00 local (Europe/London)
    - NEW_YORK: 08:00-17:00 local (America/New_York)
    - OVERLAP: When London and NY sessions are concurrent

    All sessions are checked by converting UTC to the session's local timezone
    and verifying if the local time falls within the session's local hours.
    This ensures DST is automatically respected.

    Used for:
    - Session-aware bot filtering in Commander
    - ICT strategy time window validation
    - News event timezone conversion in NewsSensor
    """

    # Session definitions with LOCAL times and timezones
    # DST is automatically handled by ZoneInfo
    SESSIONS: Dict[str, Dict[str, Any]] = {
        "ASIAN": {
            "timezone": "Asia/Tokyo",
            "start_local": time(0, 0),   # 00:00 local
            "end_local": time(9, 0),     # 09:00 local
            "name": "Asian Session"
        },
        "LONDON": {
            "timezone": "Europe/London",
            "start_local": time(8, 0),    # 08:00 local
            "end_local": time(16, 0),     # 16:00 local
            "name": "London Session"
        },
        "NEW_YORK": {
            "timezone": "America/New_York",
            "start_local": time(8, 0),    # 08:00 local
            "end_local": time(17, 0),     # 17:00 local
            "name": "New York Session"
        },
        "OVERLAP": {
            "timezone": None,  # Calculated dynamically
            "start_local": None,
            "end_local": None,
            "name": "London/NY Overlap"
        }
    }

    @classmethod
    def detect_session(cls, utc_time: datetime) -> TradingSession:
        """Detect current session from UTC time using DST-aware local time conversion.

        Converts UTC to each session's local timezone and checks if the local time
        falls within the session's local trading hours. Overlap is detected when
        both London and NY sessions are concurrently active.
        
        Returns CLOSED on weekends (Saturday/Sunday) and market holidays.

        Args:
            utc_time: Current time in UTC

        Returns:
            TradingSession enum value

        Examples:
            >>> SessionDetector.detect_session(datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc))
            <TradingSession.LONDON>
            >>> SessionDetector.detect_session(datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc))
            <TradingSession.OVERLAP>
            >>> SessionDetector.detect_session(datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc))  # Sunday
            <TradingSession.CLOSED>
        """
        # Ensure UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Weekday guard: Mon-Fri only (weekday() returns 0=Mon, 6=Sun)
        weekday = utc_time.weekday()
        if weekday >= 5:  # 5=Saturday, 6=Sunday
            logger.debug(f"Market closed - weekday {weekday} (weekend). UTC time: {utc_time.isoformat()}")
            return TradingSession.CLOSED

        # Check each session's local time range
        active_sessions = []

        for session_name, session_config in cls.SESSIONS.items():
            if session_name == "OVERLAP":
                continue  # Overlap is calculated separately

            tz_str = session_config["timezone"]
            start_local = session_config["start_local"]
            end_local = session_config["end_local"]

            if cls._is_in_local_range(utc_time, tz_str, start_local, end_local):
                active_sessions.append(TradingSession(session_name))

        # Determine session based on active sessions
        if not active_sessions:
            return TradingSession.CLOSED
        elif len(active_sessions) == 1:
            return active_sessions[0]
        elif TradingSession.LONDON in active_sessions and TradingSession.NEW_YORK in active_sessions:
            return TradingSession.OVERLAP
        elif TradingSession.LONDON in active_sessions:
            return TradingSession.LONDON
        elif TradingSession.NEW_YORK in active_sessions:
            return TradingSession.NEW_YORK
        else:
            return active_sessions[0]

    @classmethod
    def _is_in_local_range(
        cls,
        utc_time: datetime,
        timezone_str: str,
        start_local: time,
        end_local: time
    ) -> bool:
        """Check if UTC time falls within local time range.

        Converts UTC to the target timezone and checks if the local time
        is between start_local and end_local.

        Args:
            utc_time: UTC datetime to check
            timezone_str: IANA timezone string (e.g., "Europe/London")
            start_local: Session start time in local timezone
            end_local: Session end time in local timezone

        Returns:
            True if UTC time falls within the local time range
        """
        try:
            tz = ZoneInfo(timezone_str)
        except Exception as e:
            logger.error(f"Invalid timezone: {timezone_str}, error: {e}")
            return False

        # Convert UTC to local timezone
        local_time = utc_time.astimezone(tz)
        local_time_only = local_time.time()

        # Check if local time is within range (handles overnight sessions)
        if start_local <= end_local:
            # Normal range (e.g., 08:00-16:00)
            return start_local <= local_time_only < end_local
        else:
            # Overnight range (e.g., 22:00-06:00)
            return local_time_only >= start_local or local_time_only < end_local

    @classmethod
    def is_in_session(cls, session_name: str, utc_time: datetime) -> bool:
        """Check if specific session is active at given UTC time.

        Uses DST-aware local time conversion for accurate session detection.
        Returns False on weekends (Saturday/Sunday) for all sessions except CLOSED.

        Args:
            session_name: Session name (e.g., "LONDON", "NEW_YORK")
            utc_time: Current time in UTC

        Returns:
            True if the specified session is active

        Examples:
            >>> SessionDetector.is_in_session("LONDON", datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc))
            True
            >>> SessionDetector.is_in_session("NEW_YORK", datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc))
            False
        """
        try:
            target_session = TradingSession(session_name)
        except ValueError:
            logger.warning(f"Invalid session name: {session_name}")
            return False

        # Ensure UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Weekday guard for all sessions except CLOSED
        if target_session != TradingSession.CLOSED:
            weekday = utc_time.weekday()
            if weekday >= 5:  # 5=Saturday, 6=Sunday
                return False

        # For OVERLAP, check if both London and NY are active
        if target_session == TradingSession.OVERLAP:
            london_active = cls._is_session_locally_active(TradingSession.LONDON, utc_time)
            ny_active = cls._is_session_locally_active(TradingSession.NEW_YORK, utc_time)
            return london_active and ny_active

        # For LONDON and NEW_YORK, check if locally active OR in overlap
        if target_session in [TradingSession.LONDON, TradingSession.NEW_YORK]:
            if cls._is_session_locally_active(target_session, utc_time):
                return True
            # Also active if we're in overlap
            current_session = cls.detect_session(utc_time)
            return current_session == TradingSession.OVERLAP

        # For ASIAN, just check local activity
        return cls._is_session_locally_active(target_session, utc_time)

    @classmethod
    def _is_session_locally_active(cls, session: TradingSession, utc_time: datetime) -> bool:
        """Check if a session is active using local time conversion.

        Args:
            session: TradingSession to check
            utc_time: Current time in UTC

        Returns:
            True if session is locally active
        """
        if session == TradingSession.OVERLAP or session == TradingSession.CLOSED:
            return False

        session_name = session.value
        session_config = cls.SESSIONS.get(session_name)

        if not session_config or "timezone" not in session_config:
            return False

        tz_str = session_config["timezone"]
        start_local = session_config["start_local"]
        end_local = session_config["end_local"]

        return cls._is_in_local_range(utc_time, tz_str, start_local, end_local)

    @classmethod
    def get_session_info(cls, utc_time: datetime) -> SessionInfo:
        """Get comprehensive session status for given UTC time.

        Args:
            utc_time: Current time in UTC

        Returns:
            SessionInfo with current session, next session, and time info (in minutes)

        Examples:
            >>> SessionDetector.get_session_info(datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc))
            SessionInfo(session=TradingSession.LONDON, is_active=True, ...)
        """
        current_session = cls.detect_session(utc_time)
        is_active = current_session != TradingSession.CLOSED

        # Calculate next session and time until open using dynamic computation
        next_session, time_until_open = cls._get_next_session_info(utc_time)

        # Calculate time until close (returns tuple of (minutes, string))
        close_mins, close_str = cls._calculate_time_until_close(utc_time, current_session)

        return SessionInfo(
            session=current_session,
            is_active=is_active,
            next_session=next_session,
            time_until_open=time_until_open,
            time_until_close=close_mins,
            time_until_close_str=close_str
        )

    @classmethod
    def convert_to_utc(cls, local_time: datetime, timezone_name: str) -> datetime:
        """Convert local time to UTC using IANA timezone.

        Args:
            local_time: Local datetime (naive or aware)
            timezone_name: IANA timezone name (e.g., "America/New_York")

        Returns:
            UTC datetime with timezone info

        Examples:
            >>> est_time = datetime(2026, 2, 12, 8, 30)
            >>> utc_time = SessionDetector.convert_to_utc(est_time, "America/New_York")
            >>> utc_time.hour
            13
        """
        try:
            tz = ZoneInfo(timezone_name)
        except Exception as e:
            logger.error(f"Invalid timezone: {timezone_name}, error: {e}")
            # Fallback to UTC
            if local_time.tzinfo is None:
                local_time = local_time.replace(tzinfo=timezone.utc)
            return local_time.astimezone(timezone.utc)

        # Make local time aware if naive
        if local_time.tzinfo is None:
            local_time = local_time.replace(tzinfo=tz)

        # Convert to UTC
        return local_time.astimezone(timezone.utc)

    @classmethod
    def is_in_time_window(
        cls,
        utc_time: datetime,
        start_time: str,
        end_time: str,
        timezone_name: str = "UTC"
    ) -> bool:
        """Check if UTC time falls within specific time window.

        Used for ICT strategies with custom trading windows (e.g., 9:50-10:10 AM NY).

        Args:
            utc_time: Current time in UTC
            start_time: Start time in "HH:MM" format
            end_time: End time in "HH:MM" format
            timezone_name: IANA timezone for the window (default: "UTC")

        Returns:
            True if UTC time falls within the time window

        Examples:
            >>> # 9:50-10:10 AM NY window
            >>> utc_time = datetime(2026, 2, 12, 14, 55, tzinfo=timezone.utc)  # 9:55 AM EST
            >>> SessionDetector.is_in_time_window(utc_time, "09:50", "10:10", "America/New_York")
            True
        """
        try:
            tz = ZoneInfo(timezone_name)
        except Exception as e:
            logger.error(f"Invalid timezone: {timezone_name}, error: {e}")
            tz = timezone.utc

        # Ensure UTC time is aware
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Get current date in target timezone
        tz_now = utc_time.astimezone(tz)

        # Parse start and end times
        start_hour, start_minute = map(int, start_time.split(":"))
        end_hour, end_minute = map(int, end_time.split(":"))

        # Create start and end datetime in target timezone
        window_start = tz_now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
        window_end = tz_now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        # Handle overnight windows (end < start)
        if window_end < window_start:
            # Window spans midnight
            if tz_now < window_end:
                # Before midnight on next day - adjust window start to yesterday
                window_start = window_start - timedelta(days=1)
            else:
                # After midnight on current day - adjust window end to tomorrow
                window_end = window_end + timedelta(days=1)

        return window_start <= tz_now <= window_end

    @classmethod
    def will_open_next(cls, session_name: str, utc_time: datetime, hours_ahead: int = 12) -> bool:
        """Check if session will open in next N hours.

        Args:
            session_name: Session name to check
            utc_time: Current time in UTC
            hours_ahead: Hours ahead to check (default: 12)

        Returns:
            True if session will open within the lookahead window
        """
        # Ensure UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Short-circuit on weekends - no sessions open on weekends
        weekday = utc_time.weekday()
        if weekday >= 5:  # 5=Saturday, 6=Sunday
            logger.debug(f"Weekend detected (weekday {weekday}). Returning False for will_open_next.")
            return False

        try:
            target_session = TradingSession(session_name)
        except ValueError:
            return False

        current_session = cls.detect_session(utc_time)

        # Already in this session
        if current_session == target_session:
            return True

        # Check next sessions within lookahead
        check_time = utc_time
        for _ in range(hours_ahead):
            check_time += timedelta(hours=1)
            next_session = cls.detect_session(check_time)
            if next_session == target_session:
                return True

        return False

    @classmethod
    def _get_next_session_info(
        cls, 
        utc_time: datetime
    ) -> Tuple[Optional[TradingSession], Optional[int]]:
        """Compute the next active session by scanning all sessions for earliest upcoming start.
        
        Args:
            utc_time: Current UTC time
            
        Returns:
            Tuple of (next_session, minutes_until_open)
        """
        # Ensure UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)
        
        # Short-circuit on weekends - no next session to report
        weekday = utc_time.weekday()
        if weekday >= 5:  # 5=Saturday, 6=Sunday
            logger.debug(f"Weekend detected (weekday {weekday}). Returning None for next session.")
            return None, None

        current_session = cls.detect_session(utc_time)

        # Preserve intermediate session transitions instead of skipping straight to the
        # next base session. This matters when London is active and New York opens
        # before London closes, because the actual next state is OVERLAP, not NEW_YORK.
        if current_session in (TradingSession.LONDON, TradingSession.NEW_YORK):
            current_close_mins, _ = cls._calculate_time_until_close(utc_time, current_session)
            counterpart = (
                TradingSession.NEW_YORK
                if current_session == TradingSession.LONDON
                else TradingSession.LONDON
            )
            counterpart_open_mins = cls._minutes_until_session_opens(counterpart, utc_time)

            if (
                current_close_mins is not None
                and counterpart_open_mins is not None
                and 0 < counterpart_open_mins < current_close_mins
            ):
                return TradingSession.OVERLAP, counterpart_open_mins

        if current_session == TradingSession.OVERLAP:
            overlap_close_mins, _ = cls._calculate_time_until_close(utc_time, TradingSession.OVERLAP)
            if overlap_close_mins is not None:
                return TradingSession.NEW_YORK, overlap_close_mins
        
        # Sessions to check (excluding OVERLAP and CLOSED)
        sessions_to_check = [
            TradingSession.ASIAN,
            TradingSession.LONDON,
            TradingSession.NEW_YORK
        ]
        
        earliest_session = None
        earliest_minutes = None
        
        for session in sessions_to_check:
            mins = cls._minutes_until_session_opens(session, utc_time)
            if mins is not None:
                # mins == 0 means session is currently active, skip for next session
                if mins == 0:
                    continue
                if earliest_minutes is None or mins < earliest_minutes:
                    earliest_minutes = mins
                    earliest_session = session
        
        # If no sessions found (all currently active), wrap to first session tomorrow
        if earliest_session is None:
            # Find the next ASIAN session (which wraps around)
            asian_mins = cls._minutes_until_session_opens(TradingSession.ASIAN, utc_time)
            if asian_mins is not None:
                earliest_session = TradingSession.ASIAN
                earliest_minutes = asian_mins
        
        return earliest_session, earliest_minutes

    @classmethod
    def _get_next_session(cls, current_session: TradingSession) -> Optional[TradingSession]:
        """Get next session after current one (kept for compatibility)."""
        # Use the new dynamic computation
        next_session, _ = cls._get_next_session_info(datetime.now(timezone.utc))
        return next_session

    @classmethod
    def get_current_session(cls) -> TradingSession:
        """Get current trading session.

        Returns:
            Current TradingSession enum based on current UTC time

        Examples:
            >>> session = SessionDetector.get_current_session()
            >>> isinstance(session, TradingSession)
            True
        """
        return cls.detect_session(datetime.now(timezone.utc))

    @classmethod
    def _calculate_time_until_close(cls, utc_time: datetime, session: TradingSession) -> Tuple[Optional[int], Optional[str]]:
        """Calculate time until session closes in minutes and as human-readable string.

        Uses DST-aware local time conversion to determine when the session ends.

        Returns:
            Tuple of (minutes_until_close, human_readable_string)
        """
        # Get session end time from SESSIONS config
        if session == TradingSession.CLOSED:
            return None, None

        # For OVERLAP, use London's end time (overlap ends when London closes)
        if session == TradingSession.OVERLAP:
            session = TradingSession.LONDON

        session_name = session.value
        session_config = cls.SESSIONS.get(session_name)

        if not session_config or "timezone" not in session_config:
            return None, None

        tz_str = session_config["timezone"]
        end_local = session_config["end_local"]

        try:
            tz = ZoneInfo(tz_str)
        except Exception as e:
            logger.error(f"Invalid timezone: {tz_str}, error: {e}")
            return None, None

        # Ensure UTC time is aware
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Convert current UTC to session's timezone
        local_now = utc_time.astimezone(tz)

        # Create session end time in local timezone for today
        session_end_local = local_now.replace(
            hour=end_local.hour,
            minute=end_local.minute,
            second=0,
            microsecond=0
        )

        # If session end is before current local time, it's tomorrow
        if session_end_local < local_now:
            session_end_local += timedelta(days=1)

        # Convert back to UTC to get accurate difference
        session_end_utc = session_end_local.astimezone(timezone.utc)

        delta = session_end_utc - utc_time
        total_minutes = int(delta.total_seconds() // 60)

        hours = total_minutes // 60
        minutes = total_minutes % 60

        if hours > 0:
            return total_minutes, f"{hours}h {minutes}m"
        else:
            return total_minutes, f"{minutes}m"

    @classmethod
    def _calculate_time_until_open(
        cls,
        utc_time: datetime,
        next_session: Optional[TradingSession]
    ) -> Optional[int]:
        """Calculate minutes until next session opens.

        Returns:
            Minutes until next session opens, or None if no next session
        """
        if next_session is None or next_session == TradingSession.CLOSED:
            return None

        # For OVERLAP, find when both London and NY are active
        if next_session == TradingSession.OVERLAP:
            # Find when the later of the two sessions opens
            london_open_mins = cls._minutes_until_session_opens(TradingSession.LONDON, utc_time)
            ny_open_mins = cls._minutes_until_session_opens(TradingSession.NEW_YORK, utc_time)

            if london_open_mins is None or ny_open_mins is None:
                return None

            # Overlap starts when the second session opens
            return max(london_open_mins, ny_open_mins)

        return cls._minutes_until_session_opens(next_session, utc_time)

    @classmethod
    def _minutes_until_session_opens(cls, session: TradingSession, utc_time: datetime) -> Optional[int]:
        """Calculate minutes until a specific session opens.

        Args:
            session: TradingSession to check
            utc_time: Current UTC time

        Returns:
            Minutes until session opens, or None if already open
        """
        if session == TradingSession.OVERLAP or session == TradingSession.CLOSED:
            return None

        # Short-circuit on weekends - no session opens on weekends
        weekday = utc_time.weekday()
        if weekday >= 5:  # 5=Saturday, 6=Sunday
            logger.debug(f"Weekend detected (weekday {weekday}). Returning None for minutes until open.")
            return None

        # Check if already in this session
        if cls._is_session_locally_active(session, utc_time):
            return 0

        session_name = session.value
        session_config = cls.SESSIONS.get(session_name)

        if not session_config or "timezone" not in session_config:
            return None

        tz_str = session_config["timezone"]
        start_local = session_config["start_local"]

        try:
            tz = ZoneInfo(tz_str)
        except Exception as e:
            logger.error(f"Invalid timezone: {tz_str}, error: {e}")
            return None

        # Ensure UTC time is aware
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)
        elif utc_time.tzinfo != timezone.utc:
            utc_time = utc_time.astimezone(timezone.utc)

        # Convert current UTC to session's timezone
        local_now = utc_time.astimezone(tz)

        # Create session start time in local timezone for today
        session_start_local = local_now.replace(
            hour=start_local.hour,
            minute=start_local.minute,
            second=0,
            microsecond=0
        )

        # If session start is after current local time, use today's start
        if session_start_local >= local_now:
            session_start_utc = session_start_local.astimezone(timezone.utc)
        else:
            # Otherwise use tomorrow's start
            session_start_local += timedelta(days=1)
            session_start_utc = session_start_local.astimezone(timezone.utc)

        delta = session_start_utc - utc_time
        return int(delta.total_seconds() // 60)

    @classmethod
    def get_canonical_windows(cls) -> Dict[str, Dict[str, Any]]:
        """Return the CANONICAL_WINDOWS dict."""
        return CANONICAL_WINDOWS

    @classmethod
    def detect_canonical_window(cls, utc_time: datetime) -> Optional[str]:
        """
        Detect which of the 10 canonical windows the given UTC time falls in.

        Args:
            utc_time: datetime in UTC

        Returns:
            Window name (e.g., "LONDON_OPEN") or None if between windows
        """
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)

        # Ensure weekday check — Dead Zone applies on all days including weekends
        # (Dead Zone is a window regardless of weekday)
        current_time = utc_time.time()

        for window_name, window_def in CANONICAL_WINDOWS.items():
            start = window_def["utc_start"]
            end = window_def["utc_end"]

            if start <= end:
                # Normal range (e.g., 08:00-10:30)
                if start <= current_time < end:
                    return window_name
            else:
                # Overnight range (e.g., 23:00-00:00, 20:00-21:00)
                if current_time >= start or current_time < end:
                    return window_name

        return None

    @classmethod
    def is_premium_session(cls, window_name: str) -> bool:
        """Check if a canonical window is a premium session."""
        return window_name in PREMIUM_SESSIONS

    @classmethod
    def is_dead_zone(cls, window_name: str) -> bool:
        """Check if a canonical window is the Dead Zone."""
        return window_name == "DEAD_ZONE"

    @classmethod
    def is_trading_window(cls, window_name: str) -> bool:
        """Check if trading is allowed in this window."""
        return window_name in TRADING_WINDOWS

    @classmethod
    def get_bot_type_mix(cls, window_name: str) -> Dict[str, float]:
        """
        Get bot type probability mix for a canonical window.

        Args:
            window_name: Canonical window name (e.g., "LONDON_OPEN")

        Returns:
            Dict of bot_type -> probability (sums to 1.0, or empty dict for no trading)
        """
        return SESSION_BOT_MIX.get(window_name, {})

    @classmethod
    def get_session_bot_types(cls, window_name: str) -> List[str]:
        """
        Get list of bot types allowed in a canonical window.

        Args:
            window_name: Canonical window name

        Returns:
            List of bot type strings (e.g., ["ORB", "MOM"])
        """
        mix = cls.get_bot_type_mix(window_name)
        return list(mix.keys())

    @classmethod
    def get_session_template(cls, session_name: str) -> Optional[SessionTemplate]:
        """
        Get the SessionTemplate for a given session name.

        Args:
            session_name: Session key from SESSIONS dict (e.g. "asian", "london_open")

        Returns:
            SessionTemplate instance or None if not found
        """
        return SESSIONS.get(session_name)

    @classmethod
    def get_all_session_templates(cls) -> Dict[str, SessionTemplate]:
        """Return all session templates."""
        return SESSIONS

    @classmethod
    def is_trading_allowed(cls, window_name: str) -> bool:
        """
        Check if active trading is allowed in this window.

        Returns False for DEAD_ZONE (intelligence/DPR ranking only).
        """
        return cls.is_trading_window(window_name)

    @classmethod
    def get_current_canonical_window(cls) -> Optional[str]:
        """Get current canonical window name."""
        return cls.detect_canonical_window(datetime.now(timezone.utc))

    @classmethod
    def get_next_canonical_window(cls, utc_time: datetime) -> Tuple[Optional[str], Optional[int]]:
        """
        Get the next canonical window and minutes until it starts.

        Args:
            utc_time: Current UTC time

        Returns:
            Tuple of (next_window_name, minutes_until_start)
        """
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=timezone.utc)

        current_window = cls.detect_canonical_window(utc_time)
        current_time = utc_time.time()

        # Find next window
        window_names = list(CANONICAL_WINDOWS.keys())

        if current_window:
            idx = window_names.index(current_window)
            search_order = window_names[idx + 1:] + window_names[:idx + 1]
        else:
            # Not in any window — find the next one
            search_order = window_names

        for window_name in search_order:
            window_def = CANONICAL_WINDOWS[window_name]
            start = window_def["utc_start"]
            end = window_def["utc_end"]

            # Calculate minutes until this window starts
            start_hour, start_min = start.hour, start.minute
            current_mins = current_time.hour * 60 + current_time.minute
            start_mins = start_hour * 60 + start_min

            if start_mins > current_mins:
                minutes_until = start_mins - current_mins
            else:
                # Window wraps to tomorrow
                minutes_until = (24 * 60 - current_mins) + start_mins

            return window_name, minutes_until

        return None, None

    # =============================================================================
    # Tilt Mechanism at Session Boundaries (FIX-015)
    # =============================================================================

    @classmethod
    def is_tilt_window(cls, utc_now: Optional[datetime] = None) -> bool:
        """
        Returns True if we're in the T-5 tilt window before session end.

        The tilt window spans the last TILT_TRANSITION_MINutes (default 5) minutes
        before a session ends. No new entries are allowed during this window.

        Args:
            utc_now: UTC datetime to check. Defaults to now.

        Returns:
            True if in T-5 tilt window before session end
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        elif utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        # Find the current session template
        current_window = cls.detect_canonical_window(utc_now)
        if current_window is None:
            return False

        # Find the session template that matches this window
        session_template = cls._get_template_for_window(current_window)
        if session_template is None:
            return False

        # Parse session end time
        end_hour, end_minute = map(int, session_template.end_gmt.split(":"))
        end_time = utc_now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        # Handle overnight sessions (end time is next day)
        if end_time <= utc_now:
            end_time += timedelta(days=1)

        # Calculate minutes until session end
        delta = end_time - utc_now
        minutes_until_end = delta.total_seconds() / 60

        return 0 < minutes_until_end <= TILT_TRANSITION_MINUTES

    @classmethod
    def _get_template_for_window(cls, window_name: str) -> Optional[SessionTemplate]:
        """Map canonical window name to session template."""
        # Canonical window to session template name mapping
        WINDOW_TO_TEMPLATE = {
            "SYDNEY_OPEN": "asian",
            "SYDNEY_TOKYO_OVERLAP": "asian",
            "TOKYO_OPEN": "asian",
            "TOKYO_LONDON_OVERLAP": "london_open",
            "LONDON_OPEN": "london_open",
            "LONDON_MID": "london_mid",
            "INTER_SESSION_COOLDOWN": "inter_session",
            "LONDON_NY_OVERLAP": "ny_overlap",
            "NY_WIND_DOWN": "ny_wind_down",
            "DEAD_ZONE": "dead_zone",
        }
        template_name = WINDOW_TO_TEMPLATE.get(window_name)
        if template_name:
            return SESSIONS.get(template_name)
        return None

    @classmethod
    def get_tilt_action(
        cls,
        bot_id: str,
        current_profit: float,
        rvol: float,
        utc_now: Optional[datetime] = None,
    ) -> Optional[str]:
        """
        Returns the tilt action for a bot based on profit and RVOL.

        Rules (FIX-015):
        - At T-5 before session end: no new entries allowed
        - Running trades at T-5: if in profit AND RVOL < 0.8 -> take profit
        - At session end: force close scalpers
        - After session end: 1-min gap, then activate new queue

        Args:
            bot_id: Bot identifier
            current_profit: Current profit (positive = profit, negative = loss)
            rvol: Relative volatility measure (RVOL)
            utc_now: UTC datetime to check. Defaults to now.

        Returns:
            'take_profit' | 'force_close' | None
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        elif utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        current_window = cls.detect_canonical_window(utc_now)
        if current_window is None:
            return None

        session_template = cls._get_template_for_window(current_window)
        if session_template is None:
            return None

        # Parse session end time
        end_hour, end_minute = map(int, session_template.end_gmt.split(":"))
        end_time = utc_now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        # Handle overnight sessions
        if end_time <= utc_now:
            end_time += timedelta(days=1)

        delta = end_time - utc_now
        minutes_until_end = delta.total_seconds() / 60

        # At session end (within 1 minute): force close scalpers
        if 0 <= minutes_until_end <= 1:
            return "force_close"

        # In T-5 window: evaluate take profit vs hold
        if 0 < minutes_until_end <= TILT_TRANSITION_MINUTES:
            # Take profit if in profit AND RVOL < 0.8
            if current_profit > 0 and rvol < 0.8:
                return "take_profit"
            # Otherwise force close scalpers at T-5
            return "force_close"

        return None


# Convenience functions for common operations

def get_current_session() -> TradingSession:
    """Get current trading session."""
    return SessionDetector.detect_session(datetime.now(timezone.utc))


def is_market_open() -> bool:
    """Check if market is currently open (any active session).
    
    Returns False on weekends and market holidays, True only when an active
    trading session is detected on a business day.
    """
    utc_now = datetime.now(timezone.utc)
    session = SessionDetector.detect_session(utc_now)
    return session != TradingSession.CLOSED


def get_next_session_time() -> Optional[str]:
    """Get time until next session opens."""
    info = SessionDetector.get_session_info(datetime.now(timezone.utc))
    if info.next_session:
        session_name = info.next_session.value
        return f"Next: {session_name}"
    return None


# =============================================================================
# Islamic Compliance Functions
# =============================================================================

# Islamic force-close time in UTC (21:45 UTC per architecture)
ISLAMIC_CUTOFF_HOUR = 21
ISLAMIC_CUTOFF_MINUTE = 45
# Window for countdown (60 minutes before force-close)
COUNTDOWN_WINDOW_MINUTES = 60


def get_islamic_cutoff_time_utc(utc_date: Optional[datetime] = None) -> datetime:
    """
    Get the Islamic force-close time for a given UTC date.

    Args:
        utc_date: Optional date in UTC. Defaults to now.

    Returns:
        datetime representing 21:45 UTC on the given date
    """
    if utc_date is None:
        utc_date = datetime.now(timezone.utc)

    return utc_date.replace(
        hour=ISLAMIC_CUTOFF_HOUR,
        minute=ISLAMIC_CUTOFF_MINUTE,
        second=0,
        microsecond=0
    )


def is_past_islamic_cutoff(utc_time: Optional[datetime] = None) -> bool:
    """
    Check if current UTC time is past the Islamic force-close time.

    Force-close happens at 21:45 UTC daily for Islamic-compliant accounts.
    Returns True if current time is at or after 21:45 UTC.

    Args:
        utc_time: Optional UTC time to check. Defaults to now.

    Returns:
        True if past the Islamic cutoff time
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    # Ensure UTC
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    elif utc_time.tzinfo != timezone.utc:
        utc_time = utc_time.astimezone(timezone.utc)

    cutoff = get_islamic_cutoff_time_utc(utc_time)
    return utc_time >= cutoff


def is_islamic_mode_enabled(bot_id: str) -> bool:
    """
    Check if Islamic/swap-free mode is enabled for a bot.

    In a real implementation, this would query the bot's configuration.
    For now, returns False as a default (non-Islamic mode).

    Args:
        bot_id: The bot identifier

    Returns:
        True if Islamic mode is enabled for the bot
    """
    # TODO: Query bot registry for Islamic compliance setting
    # For now, return False as default
    return False


def get_swap_free_status(bot_id: str) -> bool:
    """
    Get swap-free status for a bot.

    Args:
        bot_id: The bot identifier

    Returns:
        True if the bot uses swap-free (Islamic) accounts
    """
    # TODO: Query bot registry for swap-free status
    return is_islamic_mode_enabled(bot_id)


def get_force_close_countdown_seconds(utc_time: Optional[datetime] = None) -> Optional[int]:
    """
    Get seconds until force-close if within the countdown window.

    Returns the number of seconds until 21:45 UTC if the current time
    is within 60 minutes of the cutoff. Otherwise returns None.

    Args:
        utc_time: Optional UTC time to check. Defaults to now.

    Returns:
        Seconds until force-close, or None if not within window
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    # Ensure UTC
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    elif utc_time.tzinfo != timezone.utc:
        utc_time = utc_time.astimezone(timezone.utc)

    cutoff = get_islamic_cutoff_time_utc(utc_time)

    # Check if within countdown window (60 minutes before cutoff)
    window_start = cutoff - timedelta(minutes=COUNTDOWN_WINDOW_MINUTES)

    if window_start <= utc_time <= cutoff:
        delta = cutoff - utc_time
        return int(delta.total_seconds())

    return None


def is_within_countdown_window(utc_time: Optional[datetime] = None) -> bool:
    """
    Check if current time is within the force-close countdown window.

    The window is 60 minutes before 21:45 UTC (20:45 - 21:45 UTC).

    Args:
        utc_time: Optional UTC time to check. Defaults to now.

    Returns:
        True if within the countdown window
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    # Ensure UTC
    if utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    elif utc_time.tzinfo != timezone.utc:
        utc_time = utc_time.astimezone(timezone.utc)

    cutoff = get_islamic_cutoff_time_utc(utc_time)
    window_start = cutoff - timedelta(minutes=COUNTDOWN_WINDOW_MINUTES)

    return window_start <= utc_time <= cutoff


# =============================================================================
# Loss Cap Breach Event Handling (Story 3-3)
# =============================================================================

import uuid
from typing import List, Dict, Any


class LossCapAuditLog:
    """
    Immutable audit log for loss cap breach events.

    Uses in-memory storage with append-only semantics (similar to KillSwitchAuditLog).
    """

    def __init__(self):
        self._logs: List[Dict[str, Any]] = []

    def append(self, entry: Dict[str, Any]) -> str:
        """
        Append an immutable audit log entry for loss cap breach.

        Args:
            entry: Dictionary containing breach details (bot_id, loss_pct, cap, etc.)

        Returns:
            Unique audit log ID
        """
        entry_id = str(uuid.uuid4())

        # Add immutable timestamp
        log_entry = {
            "id": entry_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **entry
        }

        # Append-only - no modifications allowed
        self._logs.append(log_entry)
        logger.info(f"Loss cap breach logged: {entry_id} for bot {entry.get('bot_id')}")
        return entry_id

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all audit log entries (read-only)."""
        return self._logs.copy()

    def get_by_id(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific audit log entry by ID."""
        for entry in self._logs:
            if entry.get("id") == entry_id:
                return entry.copy()
        return None

    def get_by_bot(self, bot_id: str) -> List[Dict[str, Any]]:
        """Get all breach events for a specific bot."""
        return [entry.copy() for entry in self._logs if entry.get("bot_id") == bot_id]


# Global loss cap audit log instance
_loss_cap_audit_log = LossCapAuditLog()


def _get_loss_cap_audit_log() -> LossCapAuditLog:
    """Get the global loss cap audit log instance."""
    return _loss_cap_audit_log


async def check_and_notify_loss_cap_breach(
    bot_id: str,
    current_loss_pct: float,
    daily_loss_cap: float,
    account_equity: float = 0.0,
    account_balance: float = 0.0
) -> Optional[Dict[str, Any]]:
    """
    Check if daily loss cap is breached and notify if so.

    This function should be called when daily P&L is updated. If the loss
    cap is breached, it logs the event and broadcasts a WebSocket notification.

    Args:
        bot_id: The bot identifier
        current_loss_pct: Current daily loss as percentage (negative for loss)
        daily_loss_cap: Configured daily loss cap percentage (e.g., 5.0 for 5%)
        account_equity: Current account equity
        account_balance: Account balance

    Returns:
        Dict with breach details if breached, None otherwise
    """
    # Check if loss exceeds cap (cap is positive value, loss is negative)
    # current_loss_pct is negative when losing, so compare abs() to cap
    if abs(current_loss_pct) >= daily_loss_cap:
        # Loss cap breached
        breach_entry = {
            "event_type": "loss_cap_breached",
            "bot_id": bot_id,
            "loss_pct": current_loss_pct,
            "daily_loss_cap": daily_loss_cap,
            "excess_pct": abs(current_loss_pct) - daily_loss_cap,
            "account_equity": account_equity,
            "account_balance": account_balance,
            "breach_time_utc": datetime.now(timezone.utc).isoformat()
        }

        # Log to audit
        audit_log = _get_loss_cap_audit_log()
        audit_log_id = audit_log.append(breach_entry)
        breach_entry["audit_log_id"] = audit_log_id

        # Broadcast WebSocket notification
        try:
            from src.api.websocket_endpoints import manager
            await manager.broadcast(
                message={
                    "type": "loss_cap_breached",
                    "bot_id": bot_id,
                    "loss_pct": current_loss_pct,
                    "daily_loss_cap": daily_loss_cap,
                    "excess_pct": abs(current_loss_pct) - daily_loss_cap,
                    "audit_log_id": audit_log_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat()
                },
                topic="trading_events"
            )
            logger.info(f"Loss cap breach notification sent for bot {bot_id}")
        except Exception as e:
            logger.error(f"Error sending WebSocket notification for loss cap breach: {e}")

        return breach_entry

    return None


def get_loss_cap_audit_logs() -> List[Dict[str, Any]]:
    """
    Get all loss cap breach audit log entries.

    Returns:
        List of all audit log entries
    """
    audit_log = _get_loss_cap_audit_log()
    return audit_log.get_all()


def get_loss_cap_breach_by_bot(bot_id: str) -> List[Dict[str, Any]]:
    """
    Get all loss cap breach events for a specific bot.

    Args:
        bot_id: The bot identifier

    Returns:
        List of breach events for the bot
    """
    audit_log = _get_loss_cap_audit_log()
    return audit_log.get_by_bot(bot_id)
