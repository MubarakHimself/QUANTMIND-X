"""
Session Detector Module

This module provides session detection functionality for trading strategies.
It re-exports SessionDetector and TradingSession from sessions.py for backward
compatibility and to provide a dedicated module as expected by commander.py
and mode_runner.py.

Usage:
    from src.router.session_detector import SessionDetector, TradingSession
    
    # Detect current session from UTC time
    session = SessionDetector.detect_session(datetime.now(timezone.utc))
    
    # Check if in specific session
    if session == TradingSession.LONDON:
        print("London session active")
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Re-export all public symbols from sessions.py
from src.router.sessions import (
    SessionDetector,
    TradingSession,
    SessionInfo,
    get_current_session,
    is_market_open,
    get_next_session_time,
)


def get_current_session_snapshot(utc_time: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Return a canonical session snapshot for runtime/API callers.

    This keeps the compatibility module as the stable import surface while
    `sessions.py` remains the implementation authority for session math.
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    elif utc_time.tzinfo is None:
        utc_time = utc_time.replace(tzinfo=timezone.utc)
    else:
        utc_time = utc_time.astimezone(timezone.utc)

    current_session = SessionDetector.detect_session(utc_time)
    session_info = SessionDetector.get_session_info(utc_time)

    return {
        "utc_time": utc_time,
        "current_session": current_session,
        "session_info": session_info,
    }

__all__ = [
    'SessionDetector',
    'TradingSession',
    'SessionInfo',
    'get_current_session',
    'is_market_open',
    'get_next_session_time',
    'get_current_session_snapshot',
]
