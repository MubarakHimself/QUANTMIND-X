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

# Re-export all public symbols from sessions.py
from src.router.sessions import (
    SessionDetector,
    TradingSession,
    SessionInfo,
    get_current_session,
    is_market_open,
    get_next_session_time,
)

__all__ = [
    'SessionDetector',
    'TradingSession',
    'SessionInfo',
    'get_current_session',
    'is_market_open',
    'get_next_session_time',
]