"""
Tests for Session Detector Module

Tests the session_detector.py module which re-exports from sessions.py.
This ensures backward compatibility and verifies the module is loadable
before commander.py and mode_runner.py import from it.

Key test cases:
- Module imports correctly
- SessionDetector.detect_session returns expected TradingSession values
- London/NY/Overlap detection works as expected
- Re-exports match original sessions.py exports
"""

import pytest
from datetime import datetime, timezone


class TestSessionDetectorModule:
    """Test the session_detector module imports and re-exports."""

    def test_module_imports_successfully(self):
        """Test that session_detector module can be imported."""
        from src.router import session_detector
        assert session_detector is not None

    def test_session_detector_class_available(self):
        """Test that SessionDetector class is available."""
        from src.router.session_detector import SessionDetector
        assert SessionDetector is not None

    def test_trading_session_enum_available(self):
        """Test that TradingSession enum is available."""
        from src.router.session_detector import TradingSession
        assert TradingSession is not None
        assert hasattr(TradingSession, 'LONDON')
        assert hasattr(TradingSession, 'NEW_YORK')
        assert hasattr(TradingSession, 'OVERLAP')
        assert hasattr(TradingSession, 'ASIAN')
        assert hasattr(TradingSession, 'CLOSED')

    def test_session_info_available(self):
        """Test that SessionInfo is available."""
        from src.router.session_detector import SessionInfo
        assert SessionInfo is not None

    def test_convenience_functions_available(self):
        """Test that convenience functions are available."""
        from src.router.session_detector import (
            get_current_session,
            is_market_open,
            get_next_session_time
        )
        assert get_current_session is not None
        assert is_market_open is not None
        assert get_next_session_time is not None

    def test_all_exports_defined(self):
        """Test that __all__ is properly defined."""
        from src.router import session_detector
        expected_exports = [
            'SessionDetector',
            'TradingSession',
            'SessionInfo',
            'get_current_session',
            'is_market_open',
            'get_next_session_time',
        ]
        for export in expected_exports:
            assert export in session_detector.__all__


class TestSessionDetectorLondonNYOverlap:
    """Test London/NY/Overlap detection - key sessions for ICT strategies."""

    def test_london_session_detection(self):
        """Test London session is detected during London hours."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        # 10:00 UTC = 10:00 London (within 08:00-16:00 London local)
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.LONDON

    def test_new_york_session_detection(self):
        """Test New York session is detected during NY hours."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        # 18:00 UTC = 13:00 EST (within 08:00-17:00 NY local)
        utc_time = datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.NEW_YORK

    def test_overlap_session_detection(self):
        """Test Overlap session is detected when both London and NY are active."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        # 14:00 UTC = 14:00 London AND 09:00 EST - both active
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.OVERLAP

    def test_overlap_at_15_utc(self):
        """Test Overlap at 15:00 UTC (15:00 London, 10:00 EST)."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        utc_time = datetime(2026, 2, 12, 15, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.OVERLAP

    def test_asian_session_detection(self):
        """Test Asian session is detected during Asian hours."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        # 23:00 UTC = 08:00 JST (within 00:00-09:00 Tokyo local)
        utc_time = datetime(2026, 2, 12, 23, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.ASIAN

    def test_closed_session_on_weekend(self):
        """Test CLOSED session on weekend."""
        from src.router.session_detector import SessionDetector, TradingSession
        
        # Saturday
        utc_time = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.CLOSED
        
        # Sunday
        utc_time = datetime(2026, 2, 15, 10, 0, tzinfo=timezone.utc)
        session = SessionDetector.detect_session(utc_time)
        assert session == TradingSession.CLOSED


class TestSessionDetectorIntegrationWithCommander:
    """Test that session_detector module works with Commander imports."""

    def test_commander_can_import_session_detector(self):
        """Test that Commander can import from session_detector."""
        # This test verifies the fix for Comment 1 - that session_detector.py exists
        # and can be imported before commander.py uses it
        try:
            from src.router.commander import Commander
            from src.router.session_detector import SessionDetector, TradingSession
            assert Commander is not None
            assert SessionDetector is not None
            assert TradingSession is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")

    def test_mode_runner_can_import_session_detector(self):
        """Test that mode_runner can import from session_detector."""
        # This test verifies the fix for Comment 1 - that session_detector.py exists
        # and can be imported before mode_runner.py uses it
        try:
            from src.backtesting.mode_runner import SentinelEnhancedTester
            from src.router.session_detector import SessionDetector, TradingSession
            assert SentinelEnhancedTester is not None
            assert SessionDetector is not None
            assert TradingSession is not None
        except ImportError as e:
            pytest.fail(f"Import failed: {e}")


class TestSessionDetectorConsistency:
    """Test consistency between sessions.py and session_detector.py."""

    def test_session_detector_matches_sessions_module(self):
        """Test that session_detector exports match sessions module."""
        from src.router import session_detector
        from src.router import sessions
        
        # SessionDetector should be the same class
        assert session_detector.SessionDetector is sessions.SessionDetector
        
        # TradingSession should be the same enum
        assert session_detector.TradingSession is sessions.TradingSession
        
        # SessionInfo should be the same dataclass
        assert session_detector.SessionInfo is sessions.SessionInfo

    def test_detection_results_identical(self):
        """Test that detection results are identical from both modules."""
        from src.router.session_detector import SessionDetector as SD1
        from src.router.session_detector import TradingSession as TS1
        from src.router.sessions import SessionDetector as SD2
        from src.router.sessions import TradingSession as TS2
        
        test_times = [
            datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc),  # London
            datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc),  # Overlap
            datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc),  # NY
            datetime(2026, 2, 12, 23, 0, tzinfo=timezone.utc),  # Asian
            datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc),  # Closed (Saturday)
        ]
        
        for utc_time in test_times:
            result1 = SD1.detect_session(utc_time)
            result2 = SD2.detect_session(utc_time)
            assert result1 == result2, f"Mismatch at {utc_time}: {result1} vs {result2}"