"""
Tests for Weekday Parameter Guard
=================================

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC5
"""

import pytest
from datetime import datetime, timezone


class TestWeekdayParameterGuard:
    """Test cases for WeekdayParameterGuard."""

    @pytest.fixture
    def guard(self):
        """Create a fresh guard instance."""
        from src.router.weekday_parameter_guard import WeekdayParameterGuard
        guard = WeekdayParameterGuard()
        yield guard

    # Test cases from story spec
    @pytest.mark.parametrize("day,hour,expected", [
        # Weekend - allowed
        ("Friday", 22, True),     # Friday 22:00 - allowed
        ("Saturday", 10, True),    # Saturday 10:00 - allowed
        ("Sunday", 15, True),      # Sunday 15:00 - allowed
        ("Friday", 0, True),       # Friday 00:00 - allowed (start of weekend)
        # Weekdays - blocked
        ("Monday", 5, False),      # Monday 05:00 - blocked
        ("Monday", 12, False),     # Monday 12:00 - blocked
        ("Wednesday", 18, False),  # Wednesday 18:00 - blocked
        ("Thursday", 23, False),   # Thursday 23:59 - blocked
    ])
    def test_is_change_allowed(self, guard, day, hour, expected):
        """Test is_change_allowed for various times."""
        # Create a datetime for the specified day and hour
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }
        weekday = day_map[day]

        # Find next occurrence of that day
        now = datetime.now(timezone.utc)
        days_ahead = weekday - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        target_date = target_date.replace(day=now.day + days_ahead)

        result = guard.is_change_allowed(target_date)
        assert result == expected, f"Expected {expected} for {day} {hour}:00 GMT, got {result}"

    def test_block_error_message(self, guard):
        """Test that block error message is informative."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        # Monday 12:00 - should be blocked
        now = datetime.now(timezone.utc)
        monday = now.replace(hour=12, minute=0, second=0, microsecond=0)
        # Find next Monday
        days_ahead = 0 - now.weekday()
        if days_ahead >= 0:
            days_ahead += 7
        monday = monday.replace(day=now.day + days_ahead + 7)  # Next Monday

        try:
            guard.reject_change("test_bot", "param1", "test", monday)
            pytest.fail("Should have raised WeekdayBlockError")
        except WeekdayBlockError as e:
            assert "Weekday parameter updates are not permitted" in e.message
            assert "weekend cycle only" in e.message

    def test_guard_default_enabled(self, guard):
        """Test that guard is enabled by default."""
        assert guard.is_enabled() is True

    def test_guard_can_be_disabled(self, guard):
        """Test that guard can be disabled for testing."""
        guard.disable()
        assert guard.is_enabled() is False
        guard.enable()
        assert guard.is_enabled() is True

    def test_get_block_status(self, guard):
        """Test block status for UI display."""
        status = guard.get_block_status()

        assert "is_blocked" in status
        assert "current_day" in status
        assert "current_time" in status
        assert "next_allowed_window" in status
        assert "message" in status


class TestWeekdayBlockError:
    """Test cases for WeekdayBlockError."""

    def test_default_message(self):
        """Test default error message."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        error = WeekdayBlockError()
        assert "Weekday parameter updates are not permitted" in error.message
        assert "weekend cycle only" in error.message

    def test_custom_message(self):
        """Test custom error message."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        custom_msg = "Custom block message"
        error = WeekdayBlockError(custom_msg)
        assert error.message == custom_msg
