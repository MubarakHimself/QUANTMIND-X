"""
Trading Session REST API Endpoints (Canonical Windows)

Provides DST-aware session detection for the SessionTimeline frontend component.
Uses CANONICAL_WINDOWS (10 windows) instead of the legacy TradingSession (5 sessions).

CANONICAL_WINDOWS:
    SYDNEY_OPEN, SYDNEY_TOKYO_OVERLAP, TOKYO_OPEN, TOKYO_LONDON_OVERLAP,
    LONDON_OPEN, LONDON_MID, INTER_SESSION_COOLDOWN, LONDON_NY_OVERLAP,
    NY_WIND_DOWN, DEAD_ZONE
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

from src.router.sessions import SessionDetector, CANONICAL_WINDOWS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading", tags=["trading"])


# Module-level singleton — SessionDetector is stateless (class-level SESSIONS + classmethods)
_detector: Optional[SessionDetector] = None


def get_session_detector() -> SessionDetector:
    global _detector
    if _detector is None:
        _detector = SessionDetector()
    return _detector


@router.get("/current-session")
async def get_current_session():
    """
    Returns the current trading session with DST-aware window detection.

    Uses SessionDetector.detect_canonical_window which converts UTC to each
    session's local timezone (Asia/Tokyo, Europe/London, America/New_York)
    to automatically handle DST transitions.

    Response:
        current_window: Canonical window name (e.g., "LONDON_OPEN")
        current_window_start: Window start time in UTC (HH:MM format)
        current_window_end: Window end time in UTC (HH:MM format)
        next_window: Next canonical window name
        next_window_start: Next window start time in UTC (HH:MM format)
        is_premium: True if current window is premium (Tokyo-London, London, London-NY)
        is_dead_zone: True if current window is DEAD_ZONE
        tilt_state: Always None (Tilt state is separate from session detection)
    """
    try:
        detector = get_session_detector()
        now_utc = datetime.now(timezone.utc)

        current_window = detector.detect_canonical_window(now_utc)
        next_window_name, minutes_until = detector.get_next_canonical_window(now_utc)

        # Build current window info
        current_window_start = None
        current_window_end = None
        is_premium = False
        is_dead_zone = False

        if current_window and current_window in CANONICAL_WINDOWS:
            # Note: dict supports `in` membership check (tests key existence) — intentional
            win = CANONICAL_WINDOWS[current_window]
            current_window_start = win["utc_start"].strftime("%H:%M")
            current_window_end = win["utc_end"].strftime("%H:%M")
            is_premium = win.get("is_premium", False)
            is_dead_zone = current_window == "DEAD_ZONE"

        # Build next window info
        next_window_start = None
        if next_window_name and next_window_name in CANONICAL_WINDOWS:
            # Note: dict supports `in` membership check (tests key existence) — intentional
            next_window_start = CANONICAL_WINDOWS[next_window_name]["utc_start"].strftime("%H:%M")

        return {
            "current_window": current_window,
            "current_window_start": current_window_start,
            "current_window_end": current_window_end,
            "next_window": next_window_name,
            "next_window_start": next_window_start,
            "minutes_until_next": minutes_until,
            "is_premium": is_premium,
            "is_dead_zone": is_dead_zone,
            "tilt_state": None,  # Tilt state is separate — managed by WebSocket
        }
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current session: {str(e)}")
