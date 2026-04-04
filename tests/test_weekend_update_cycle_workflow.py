"""
Tests for Weekend Update Cycle Workflow
======================================

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle)
"""

import pytest
from datetime import datetime, timezone


class TestWeekendUpdateCycleWorkflow:
    """Test cases for WeekendUpdateCycleWorkflow."""

    @pytest.fixture
    def workflow(self):
        """Create a fresh workflow instance."""
        from src.router.weekend_update_cycle_workflow import WeekendUpdateCycleWorkflow
        workflow = WeekendUpdateCycleWorkflow()
        yield workflow

    def test_workflow_steps_defined(self, workflow):
        """Test that all workflow steps are defined."""
        expected_steps = [
            "friday_analysis",
            "saturday_refinement",
            "saturday_wfa",
            "saturday_hmm_retrain",
            "sunday_calibration",
            "sunday_spread_profiles",
            "sunday_sqs_refresh",
            "sunday_kelly_calibration",
            "monday_roster_deploy",
        ]

        for step_name in expected_steps:
            assert step_name in workflow._steps, f"Missing step: {step_name}"

    def test_workflow_initial_status(self, workflow):
        """Test that workflow starts in correct state."""
        assert workflow._result is None
        status = workflow.get_status()
        assert status["status"] == "not_started"

    def test_weekday_blocked(self, workflow):
        """Test weekday blocking logic."""
        # Monday
        monday = datetime(2026, 3, 23, 12, 0, 0, tzinfo=timezone.utc)  # Monday
        assert workflow.is_weekday_blocked(monday) is True

        # Wednesday
        wednesday = datetime(2026, 3, 25, 18, 0, 0, tzinfo=timezone.utc)  # Wednesday
        assert workflow.is_weekday_blocked(wednesday) is True

        # Thursday
        thursday = datetime(2026, 3, 26, 23, 59, 0, tzinfo=timezone.utc)  # Thursday
        assert workflow.is_weekday_blocked(thursday) is True

    def test_weekend_not_blocked(self, workflow):
        """Test that weekend days are not blocked."""
        # Friday 21:00
        friday = datetime(2026, 3, 27, 21, 0, 0, tzinfo=timezone.utc)  # Friday
        assert workflow.is_weekday_blocked(friday) is False

        # Saturday
        saturday = datetime(2026, 3, 28, 10, 0, 0, tzinfo=timezone.utc)  # Saturday
        assert workflow.is_weekday_blocked(saturday) is False

        # Sunday
        sunday = datetime(2026, 3, 29, 15, 0, 0, tzinfo=timezone.utc)  # Sunday
        assert workflow.is_weekday_blocked(sunday) is False


class TestWeekendCycleStep:
    """Test cases for WeekendCycleStep dataclass."""

    def test_step_creation(self):
        """Test step creation with defaults."""
        from src.router.weekend_update_cycle_workflow import WeekendCycleStep, WorkflowStepStatus

        step = WeekendCycleStep(
            step_name="test_step",
            scheduled_time="21:00",
            day="Friday"
        )

        assert step.step_name == "test_step"
        assert step.scheduled_time == "21:00"
        assert step.day == "Friday"
        assert step.status == WorkflowStepStatus.PENDING
        assert step.started_at is None
        assert step.completed_at is None
        assert step.output is None
        assert step.error is None

    def test_step_to_dict(self):
        """Test step serialization."""
        from src.router.weekend_update_cycle_workflow import WeekendCycleStep, WorkflowStepStatus

        step = WeekendCycleStep(
            step_name="test_step",
            scheduled_time="21:00",
            day="Friday",
            status=WorkflowStepStatus.COMPLETED
        )

        result = step.to_dict()

        assert result["step_name"] == "test_step"
        assert result["scheduled_time"] == "21:00"
        assert result["day"] == "Friday"
        assert result["status"] == "completed"


class TestWeekendRoster:
    """Test cases for WeekendRoster dataclass."""

    def test_roster_creation(self):
        """Test roster creation."""
        from src.router.weekend_update_cycle_workflow import WeekendRoster

        roster = WeekendRoster(
            created_at=datetime.now(timezone.utc),
            bots=["bot1", "bot2"],
            session_configs={"bot1": {"session_enabled": True}},
            metadata={"week_number": 12, "year": 2026}
        )

        assert roster.bots == ["bot1", "bot2"]
        assert roster.session_configs["bot1"]["session_enabled"] is True
        assert roster.metadata["week_number"] == 12

    def test_roster_to_dict(self):
        """Test roster serialization."""
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
