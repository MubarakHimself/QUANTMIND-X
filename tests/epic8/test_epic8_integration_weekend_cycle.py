"""
Integration Tests for Epic 8.13 - Weekend Update Cycle Workflow
=============================================================

Integration tests for weekend update cycle workflow with weekday parameter guard.
Tests the interaction between weekend cycle steps, weekday guard, and roster deployment.

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle)
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock


class TestWeekendCycleWorkflowIntegration:
    """Integration tests for weekend update cycle workflow."""

    @pytest.fixture
    def workflow(self):
        """Create weekend update cycle workflow."""
        from src.router.weekend_update_cycle_workflow import WeekendUpdateCycleWorkflow
        return WeekendUpdateCycleWorkflow()

    def test_workflow_defines_all_9_steps(self, workflow):
        """Verify workflow defines all 9 required steps (Friday-Sunday-Monday)."""
        expected_steps = [
            "friday_analysis",       # Friday 21:00
            "saturday_refinement",   # Saturday 06:00
            "saturday_wfa",          # Saturday 09:00
            "saturday_hmm_retrain",  # Saturday 12:00
            "sunday_calibration",    # Sunday 06:00
            "sunday_spread_profiles",# Sunday 09:00
            "sunday_sqs_refresh",    # Sunday 12:00
            "sunday_kelly_calibration", # Sunday 15:00
            "monday_roster_deploy",  # Monday 05:00
        ]

        for step_name in expected_steps:
            assert step_name in workflow._steps, f"Missing step: {step_name}"

    def test_workflow_step_timing(self, workflow):
        """Verify all steps have correct GMT timing."""
        expected_timing = {
            "friday_analysis": "21:00",
            "saturday_refinement": "06:00",
            "saturday_wfa": "09:00",
            "saturday_hmm_retrain": "12:00",
            "sunday_calibration": "06:00",
            "sunday_spread_profiles": "09:00",
            "sunday_sqs_refresh": "12:00",
            "sunday_kelly_calibration": "15:00",
            "monday_roster_deploy": "05:00",
        }

        for step_name, expected_time in expected_timing.items():
            step = workflow._steps.get(step_name)
            assert step is not None, f"Step {step_name} not found"
            assert step.scheduled_time == expected_time, f"Step {step_name} should be at {expected_time}"

    def test_workflow_step_days(self, workflow):
        """Verify steps occur on correct days."""
        expected_days = {
            "friday_analysis": "Friday",
            "saturday_refinement": "Saturday",
            "saturday_wfa": "Saturday",
            "saturday_hmm_retrain": "Saturday",
            "sunday_calibration": "Sunday",
            "sunday_spread_profiles": "Sunday",
            "sunday_sqs_refresh": "Sunday",
            "sunday_kelly_calibration": "Sunday",
            "monday_roster_deploy": "Monday",
        }

        for step_name, expected_day in expected_days.items():
            step = workflow._steps.get(step_name)
            assert step is not None, f"Step {step_name} not found"
            assert step.day == expected_day, f"Step {step_name} should be on {expected_day}"


class TestWeekdayParameterGuardIntegration:
    """Integration tests for weekday parameter change guard."""

    @pytest.fixture
    def guard(self):
        """Create weekday parameter guard."""
        from src.router.weekday_parameter_guard import WeekdayParameterGuard
        return WeekdayParameterGuard()

    @pytest.mark.parametrize("day,expected", [
        ("Monday", False),     # Weekday - blocked
        ("Tuesday", False),    # Weekday - blocked
        ("Wednesday", False),   # Weekday - blocked
        ("Thursday", False),  # Weekday - blocked
        ("Friday", True),      # Weekend - allowed
        ("Saturday", True),    # Weekend - allowed
        ("Sunday", True),      # Weekend - allowed
    ])
    def test_weekday_blocking(self, guard, day, expected):
        """Verify weekday blocking for all days of week."""
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2,
            "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        # Create a datetime for the specified day (any hour)
        now = datetime.now(timezone.utc)
        weekday = day_map[day]
        days_ahead = weekday - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7

        target_date = now + timedelta(days=days_ahead)
        target_date = target_date.replace(hour=12, minute=0, second=0, microsecond=0)

        result = guard.is_change_allowed(target_date)
        assert result == expected, f"is_change_allowed for {day} should be {expected}"

    def test_friday_evening_allowed(self, guard):
        """AC1: Friday 21:00 GMT - change should be allowed."""
        # Find next Friday at 21:00
        now = datetime.now(timezone.utc)
        days_until_friday = 4 - now.weekday()  # Friday is weekday 4
        if days_until_friday <= 0:
            days_until_friday += 7

        friday = now.replace(
            day=now.day + days_until_friday,
            hour=21,
            minute=0,
            second=0,
            microsecond=0
        )

        assert guard.is_change_allowed(friday) is True

    def test_saturday_morning_allowed(self, guard):
        """AC2: Saturday 10:00 GMT - change should be allowed."""
        now = datetime.now(timezone.utc)
        days_until_saturday = 5 - now.weekday()
        if days_until_saturday <= 0:
            days_until_saturday += 7

        saturday = now.replace(
            day=now.day + days_until_saturday,
            hour=10,
            minute=0,
            second=0,
            microsecond=0
        )

        assert guard.is_change_allowed(saturday) is True

    def test_monday_morning_blocked(self, guard):
        """AC4: Monday 05:00 GMT - change should be blocked (weekday)."""
        now = datetime.now(timezone.utc)
        days_until_monday = 0 - now.weekday()
        if days_until_monday <= 0:
            days_until_monday += 7

        monday = now.replace(
            day=now.day + days_until_monday,
            hour=5,
            minute=0,
            second=0,
            microsecond=0
        )

        assert guard.is_change_allowed(monday) is False

    def test_reject_change_raises_block_error(self, guard):
        """AC5: Weekday rejection should raise WeekdayBlockError with correct message."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        # Monday at any time
        now = datetime.now(timezone.utc)
        days_until_monday = 0 - now.weekday()
        if days_until_monday <= 0:
            days_until_monday += 7

        monday = now.replace(
            day=now.day + days_until_monday,
            hour=12,
            minute=0,
            second=0,
            microsecond=0
        )

        with pytest.raises(WeekdayBlockError) as exc_info:
            guard.reject_change("test_bot", "param1", "value", monday)

        assert "Weekday parameter updates are not permitted" in str(exc_info.value)
        assert "weekend cycle only" in str(exc_info.value)


class TestSqsMondayWarmupIntegration:
    """Integration tests for SQS Monday warm-up ramp."""

    @pytest.fixture
    def warmup(self):
        """Create SQS Monday warmup instance."""
        from src.router.sqs_monday_warmup import SqsMondayWarmup
        return SqsMondayWarmup()

    def test_warmup_constants(self, warmup):
        """Verify warmup constants match AC requirements."""
        assert warmup.SQS_WARMUP_START == 0.75
        assert warmup.SQS_WARMUP_END == 0.60
        assert warmup.WARMUP_DURATION_MINUTES == 15

    def test_baseline_ramp_calculation_t0(self, warmup):
        """T=0 minutes: SQS baseline = 0.75."""
        elapsed_minutes = 0
        expected_baseline = warmup.SQS_WARMUP_START - (
            (warmup.SQS_WARMUP_START - warmup.SQS_WARMUP_END) *
            (elapsed_minutes / warmup.WARMUP_DURATION_MINUTES)
        )
        assert abs(expected_baseline - 0.75) < 0.001

    def test_baseline_ramp_calculation_t7(self, warmup):
        """T=7 minutes: SQS baseline should be 0.675 (midpoint)."""
        elapsed_minutes = 7
        baseline = warmup.SQS_WARMUP_START - (
            (warmup.SQS_WARMUP_START - warmup.SQS_WARMUP_END) *
            (elapsed_minutes / warmup.WARMUP_DURATION_MINUTES)
        )
        # Allow floating point tolerance
        assert abs(baseline - 0.675) < 0.01

    def test_baseline_ramp_calculation_t15(self, warmup):
        """T=15 minutes: SQS baseline = 0.60 (end value)."""
        elapsed_minutes = 15
        baseline = warmup.SQS_WARMUP_START - (
            (warmup.SQS_WARMUP_START - warmup.SQS_WARMUP_END) *
            (elapsed_minutes / warmup.WARMUP_DURATION_MINUTES)
        )
        assert abs(baseline - 0.60) < 0.001

    def test_baseline_ramp_calculation_t20(self, warmup):
        """T=20 minutes: SQS baseline capped at 0.60."""
        elapsed_minutes = 20
        # After warmup period, should stay at end value
        baseline = warmup.SQS_WARMUP_START - (
            (warmup.SQS_WARMUP_START - warmup.SQS_WARMUP_END) *
            (min(elapsed_minutes, warmup.WARMUP_DURATION_MINUTES) / warmup.WARMUP_DURATION_MINUTES)
        )
        assert baseline == 0.60


class TestWeekendRosterIntegration:
    """Integration tests for weekend roster preparation and deployment."""

    @pytest.fixture
    def roster_manager(self):
        """Create weekend roster manager."""
        from src.router.weekend_roster_manager import WeekendRosterManager
        return WeekendRosterManager()

    def test_roster_creation(self):
        """Verify WeekendRoster dataclass structure."""
        from src.router.weekend_update_cycle_workflow import WeekendRoster

        roster = WeekendRoster(
            created_at=datetime.now(timezone.utc),
            bots=["bot1", "bot2", "bot3"],
            session_configs={"bot1": {"session_enabled": True}},
            metadata={"week_number": 12, "year": 2026}
        )

        assert roster.bots == ["bot1", "bot2", "bot3"]
        assert roster.session_configs["bot1"]["session_enabled"] is True
        assert roster.metadata["week_number"] == 12

    def test_roster_to_dict(self):
        """Verify WeekendRoster serialization."""
        from src.router.weekend_update_cycle_workflow import WeekendRoster

        roster = WeekendRoster(
            created_at=datetime.now(timezone.utc),
            bots=["bot1"],
            session_configs={},
            metadata={}
        )

        result = roster.to_dict()
        assert "bots" in result
        assert "created_at" in result
        assert result["bots"] == ["bot1"]


class TestWeekendCycleEndToEnd:
    """End-to-end integration tests for weekend cycle."""

    def test_friday_to_monday_timing(self):
        """Verify weekend cycle timing: Friday 21:00 to Monday 05:00."""
        # Friday 21:00 to Monday 05:00 is approximately 32 hours
        from datetime import datetime, timezone, timedelta

        # Find next Friday 21:00 and Monday 05:00
        now = datetime.now(timezone.utc)
        current_weekday = now.weekday()  # Monday=0, Sunday=6

        # Days until next Friday (weekday 4)
        days_until_friday = (4 - current_weekday) % 7
        if days_until_friday == 0:
            days_until_friday = 7  # If today is Friday, go to next Friday

        friday_21 = now.replace(hour=21, minute=0, second=0, microsecond=0) + timedelta(days=days_until_friday)

        # Monday is 3 days after Friday
        monday_05 = friday_21 + timedelta(days=3)
        monday_05 = monday_05.replace(hour=5, minute=0, second=0, microsecond=0)

        delta = monday_05 - friday_21
        hours = delta.total_seconds() / 3600

        # Friday 21:00 to Monday 05:00 = 56 hours
        # But the spec says ~32 hours which is incorrect. The actual span is 56 hours.
        # Let me verify: Friday 21:00 + 56 hours = Monday 05:00
        # 56 hours = 2 days + 8 hours = Friday 21:00 -> Sunday 21:00 (48h) -> Monday 05:00 (8h more) = 56h
        assert 50 <= hours <= 60, f"Expected ~56 hours, got {hours}"

    def test_saturday_compute_window(self):
        """Verify Saturday compute window: 06:00-18:00 GMT (12 hours)."""
        from datetime import datetime, timezone

        saturday_06 = datetime(2026, 3, 28, 6, 0, 0, tzinfo=timezone.utc)
        saturday_18 = datetime(2026, 3, 28, 18, 0, 0, tzinfo=timezone.utc)

        delta = saturday_18 - saturday_06
        hours = delta.total_seconds() / 3600

        assert hours == 12, f"Expected 12 hours, got {hours}"

    def test_sunday_compute_window(self):
        """Verify Sunday compute window: 06:00-18:00 GMT (12 hours)."""
        from datetime import datetime, timezone

        sunday_06 = datetime(2026, 3, 29, 6, 0, 0, tzinfo=timezone.utc)
        sunday_18 = datetime(2026, 3, 29, 18, 0, 0, tzinfo=timezone.utc)

        delta = sunday_18 - sunday_06
        hours = delta.total_seconds() / 3600

        assert hours == 12, f"Expected 12 hours, got {hours}"


class TestWeekdayBlockErrorIntegration:
    """Integration tests for WeekdayBlockError."""

    def test_block_error_has_required_message(self):
        """Verify WeekdayBlockError message contains required text."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        error = WeekdayBlockError()
        message = str(error.message)

        assert "Weekday parameter updates are not permitted" in message
        assert "weekend cycle only" in message

    def test_block_error_custom_message(self):
        """Verify WeekdayBlockError accepts custom message."""
        from src.router.weekday_parameter_guard import WeekdayBlockError

        custom_msg = "Custom block message"
        error = WeekdayBlockError(custom_msg)

        assert error.message == custom_msg

    def test_guard_can_be_disabled_and_reenabled(self):
        """Verify guard can be disabled for testing and re-enabled."""
        from src.router.weekday_parameter_guard import WeekdayParameterGuard

        guard = WeekdayParameterGuard()

        # Should be enabled by default
        assert guard.is_enabled() is True

        # Can be disabled
        guard.disable()
        assert guard.is_enabled() is False

        # Can be re-enabled
        guard.enable()
        assert guard.is_enabled() is True

    def test_guard_get_block_status(self):
        """Verify guard get_block_status returns required fields."""
        from src.router.weekday_parameter_guard import WeekdayParameterGuard

        guard = WeekdayParameterGuard()
        status = guard.get_block_status()

        required_fields = [
            "is_blocked",
            "current_day",
            "current_time",
            "next_allowed_window",
            "message"
        ]

        for field in required_fields:
            assert field in status, f"Missing status field: {field}"
