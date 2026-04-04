"""
Unit Tests for Session Manager
"""

import pytest
from datetime import datetime, timezone

from svss.session_manager import SessionManager, SessionInfo


class TestSessionManager:
    """Tests for SessionManager."""

    def test_initialization(self):
        """Test SessionManager initialization."""
        manager = SessionManager()
        assert manager._current_session is None
        assert manager.is_new_session is True

    def test_initial_update_creates_session(self):
        """Test that first update creates a session."""
        manager = SessionManager()
        timestamp = datetime(2026, 3, 25, 8, 5, 0, tzinfo=timezone.utc)  # 08:05 GMT

        session = manager.update(timestamp)

        assert session is not None
        assert session.session_type == "london_open"
        assert session.is_new is True
        assert session.session_id == "london_open_20260325_08"

    def test_session_boundary_detection(self):
        """Test detection of session boundary crossing."""
        manager = SessionManager()

        # Start before London open
        timestamp1 = datetime(2026, 3, 25, 7, 59, 0, tzinfo=timezone.utc)
        manager.update(timestamp1)

        # Cross into London open
        timestamp2 = datetime(2026, 3, 25, 8, 1, 0, tzinfo=timezone.utc)
        session = manager.update(timestamp2)

        assert session is not None
        assert session.is_new is True
        assert session.session_type == "london_open"

    def test_no_new_session_within_same_period(self):
        """Test that no new session is detected within same period."""
        manager = SessionManager()

        # Within London open
        timestamp1 = datetime(2026, 3, 25, 8, 30, 0, tzinfo=timezone.utc)
        session1 = manager.update(timestamp1)
        assert session1 is not None

        # Still within London open
        timestamp2 = datetime(2026, 3, 25, 9, 0, 0, tzinfo=timezone.utc)
        session2 = manager.update(timestamp2)

        # Should not create new session
        assert session2 is None

    def test_ny_am_session_detection(self):
        """Test NY AM session detection at 12:00 GMT."""
        manager = SessionManager()

        # Before NY AM
        timestamp1 = datetime(2026, 3, 25, 11, 59, 0, tzinfo=timezone.utc)
        manager.update(timestamp1)

        # Cross into NY AM
        timestamp2 = datetime(2026, 3, 25, 12, 1, 0, tzinfo=timezone.utc)
        session = manager.update(timestamp2)

        assert session is not None
        assert session.session_type == "ny_am"
        assert session.session_id == "ny_am_20260325_12"

    def test_reset(self):
        """Test session manager reset."""
        manager = SessionManager()

        timestamp = datetime(2026, 3, 25, 8, 5, 0, tzinfo=timezone.utc)
        manager.update(timestamp)

        manager.reset()

        assert manager._current_session is None
        assert manager._last_known_hour is None
        assert manager.is_new_session is True
