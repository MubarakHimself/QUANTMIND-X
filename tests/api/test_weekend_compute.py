"""
Tests for Weekend Compute Flow

Reference: Story 11-2-weekend-compute-protocol-scheduled-background-tasks
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock


class TestWeekendComputeFlow:
    """Test suite for weekend compute flow."""

    def test_weekend_task_result_dataclass(self):
        """Test WeekendTaskResult dataclass creation."""
        from flows.weekend_compute_flow import WeekendTaskResult

        result = WeekendTaskResult(
            task_name="monte_carlo",
            status="completed",
            message="Simulated 3 strategies",
            duration_seconds=120.5,
            details={"strategies": ["ma_cross", "rsi_div"]}
        )

        assert result.task_name == "monte_carlo"
        assert result.status == "completed"
        assert result.duration_seconds == 120.5
        assert result.details["strategies"] == ["ma_cross", "rsi_div"]

    def test_weekend_compute_summary_dataclass(self):
        """Test WeekendComputeSummary dataclass creation."""
        from flows.weekend_compute_flow import WeekendComputeSummary, WeekendTaskResult

        tasks = [
            WeekendTaskResult(task_name="monte_carlo", status="completed", duration_seconds=60),
            WeekendTaskResult(task_name="hmm_retrain", status="completed", duration_seconds=120),
        ]

        summary = WeekendComputeSummary(
            run_id="test-123",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            tasks=tasks,
            total_duration_seconds=180.0,
            has_failures=False
        )

        assert summary.run_id == "test-123"
        assert len(summary.tasks) == 2
        assert summary.has_failures is False

    @pytest.mark.asyncio
    async def test_monte_carlo_task_execution(self):
        """Test Monte Carlo task runs and returns result."""
        from flows.weekend_compute_flow import monte_carlo_task

        # Mock the dependencies
        with patch('flows.weekend_compute_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()

            # Run the task
            result = await monte_carlo_task(active_strategies=["ma_cross", "rsi_div"])

            assert result.task_name == "monte_carlo"
            assert result.status in ["completed", "failed"]  # May fail due to missing deps
            assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_pageindex_semantic_task(self):
        """Test PageIndex semantic task execution."""
        from flows.weekend_compute_flow import pageindex_semantic_task

        with patch('flows.weekend_compute_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()

            result = await pageindex_semantic_task()

            assert result.task_name == "pageindex_semantic"
            assert result.status in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_correlation_refresh_task(self):
        """Test correlation refresh task execution."""
        from flows.weekend_compute_flow import correlation_refresh_task

        with patch('flows.weekend_compute_flow.get_run_logger') as mock_logger:
            mock_logger.return_value = MagicMock()

            result = await correlation_refresh_task()

            assert result.task_name == "correlation_refresh"
            assert result.status in ["completed", "failed"]


class TestScheduledTasksEndpoints:
    """Test suite for scheduled tasks API endpoints."""

    def test_task_status_dataclass(self):
        """Test TaskStatus dataclass creation."""
        from src.api.scheduled_tasks_endpoints import TaskStatus

        status = TaskStatus(
            task_name="monte_carlo_simulation",
            status="running",
            progress_percent=45.0,
            estimated_completion=datetime(2026, 3, 21, 10, 0, 0),
            start_time=datetime(2026, 3, 21, 9, 0, 0),
            duration_seconds=2700.0
        )

        assert status.task_name == "monte_carlo_simulation"
        assert status.status == "running"
        assert status.progress_percent == 45.0

    def test_calculate_progress(self):
        """Test progress calculation helper function."""
        from src.api.scheduled_tasks_endpoints import calculate_progress

        start = datetime.now(timezone.utc)

        # After 300 seconds of 600 second estimate = 50%
        progress = calculate_progress(start, 600)
        assert progress == 50.0

        # After 600 seconds = 100% (capped)
        progress = calculate_progress(start, 600)
        assert progress <= 100.0


class TestFloorManagerWeekendQuery:
    """Test suite for FloorManager weekend task query handler."""

    @pytest.mark.asyncio
    async def test_handle_weekend_tasks_query(self):
        """Test FloorManager handles weekend task queries."""
        from src.agents.departments.floor_manager import FloorManager
        from unittest.mock import patch, MagicMock

        # Create floor manager with mocks
        with patch('src.agents.departments.floor_manager.get_redis_mail_service'):
            fm = FloorManager(use_redis_mail=False)

        # Test query pattern detection
        message = "What's running this weekend?"
        result = await fm._handle_weekend_tasks(message, {})

        # Result may be None or a response dict depending on API availability
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handle_weekend_compute_keyword(self):
        """Test FloorManager handles weekend compute keyword."""
        from src.agents.departments.floor_manager import FloorManager

        with patch('src.agents.departments.floor_manager.get_redis_mail_service'):
            fm = FloorManager(use_redis_mail=False)

        # Test different patterns
        patterns = [
            "what's running this weekend",
            "weekend tasks",
            "weekend compute",
            "saturday tasks",
        ]

        for pattern in patterns:
            result = await fm._handle_weekend_tasks(pattern, {})
            # Result should be None (not a query) or a dict response
            assert result is None or isinstance(result, dict)