"""
Session Manager Module

Detects session boundaries and manages session lifecycle for SVSS.
Session detection via MT5 ZMQ tick stream (first tick after 08:00, 12:00, 13:00, 16:00, 21:00 GMT).
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """
    Information about the current trading session.

    Attributes:
        session_id: Unique session identifier (e.g., 'london_open_20260325')
        session_type: Type of session ('london_open', 'ny_open', 'lunch', 'ny_close', 'late_close')
        start_time: Session start timestamp
        is_new: True if this is a newly opened session
    """

    session_id: str
    session_type: str
    start_time: datetime
    is_new: bool = False


class SessionManager:
    """
    Manages session boundaries and detects session transitions.

    Session detection is based on GMT hour boundaries:
    - 08:00 GMT: London Open
    - 12:00 GMT: NY AM Session
    - 13:00 GMT: Lunch session
    - 16:00 GMT: NY Close
    - 21:00 GMT: Late Close
    """

    SESSION_TYPES = {
        8: "london_open",
        12: "ny_am",
        13: "lunch",
        16: "ny_close",
        21: "late_close",
    }

    def __init__(self, session_boundaries: list[int] = None):
        """
        Initialize SessionManager.

        Args:
            session_boundaries: List of GMT hours that trigger session transitions.
        """
        self._session_boundaries = session_boundaries or [8, 12, 13, 16, 21]
        self._current_session: Optional[SessionInfo] = None
        self._last_known_hour: Optional[int] = None

    def update(self, timestamp: datetime) -> Optional[SessionInfo]:
        """
        Update session state based on timestamp.

        Detects if a new session has started based on the hour crossing
        a session boundary.

        Args:
            timestamp: Current timestamp (should be UTC)

        Returns:
            SessionInfo if a new session was detected, None otherwise.
        """
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        current_hour = timestamp.hour
        current_minute = timestamp.minute

        # Check if we've crossed a session boundary
        new_session = False
        if self._last_known_hour is not None:
            for boundary in self._session_boundaries:
                # New session if we crossed a boundary (hour >= boundary)
                # and were previously below it
                if (
                    current_hour >= boundary
                    and self._last_known_hour < boundary
                    and current_minute < 5  # Within first 5 minutes of session
                ):
                    new_session = True
                    break

        self._last_known_hour = current_hour

        is_new_session = new_session or self._current_session is None

        if is_new_session:
            session_type = self._determine_session_type(current_hour)
            session_id = self._generate_session_id(timestamp, session_type)

            self._current_session = SessionInfo(
                session_id=session_id,
                session_type=session_type,
                start_time=timestamp,
                is_new=is_new_session,
            )

            logger.info(
                f"Session {'changed' if new_session else 'initialized'}: {session_id} "
                f"(type={session_type})"
            )

            return self._current_session

        return None

    def _determine_session_type(self, hour: int) -> str:
        """Determine session type based on hour."""
        # Find the closest boundary at or before the current hour
        applicable_boundary = 0
        for boundary in sorted(self._session_boundaries, reverse=True):
            if hour >= boundary:
                applicable_boundary = boundary
                break

        return self.SESSION_TYPES.get(applicable_boundary, "unknown")

    def _generate_session_id(self, timestamp: datetime, session_type: str) -> str:
        """Generate unique session ID."""
        date_str = timestamp.strftime("%Y%m%d")
        # Use hour as part of ID for uniqueness
        return f"{session_type}_{date_str}_{timestamp.hour:02d}"

    @property
    def current_session(self) -> Optional[SessionInfo]:
        """Get current session info."""
        return self._current_session

    @property
    def is_new_session(self) -> bool:
        """Check if current session is newly opened."""
        return self._current_session.is_new if self._current_session else True

    def reset(self) -> None:
        """Reset session state (e.g., on disconnection)."""
        self._current_session = None
        self._last_known_hour = None
        logger.info("Session manager reset")
