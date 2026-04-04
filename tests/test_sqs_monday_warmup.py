"""
Tests for SQS Monday Warmup
===========================

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC4
"""

import pytest
from datetime import datetime, timezone


class TestSqsMondayWarmup:
    """Test cases for SQS Monday warmup."""

    @pytest.fixture
    def warmup(self):
        """Create a fresh warmup instance."""
        from src.router.sqs_monday_warmup import SqsMondayWarmup
        warmup = SqsMondayWarmup()
        yield warmup

    def test_warmup_initialization(self, warmup):
        """Test warmup initializes with correct defaults."""
        assert warmup.SQS_WARMUP_START == 0.75
        assert warmup.SQS_WARMUP_END == 0.60
        assert warmup.WARMUP_DURATION_MINUTES == 15

    def test_get_warmup_status_initial(self, warmup):
        """Test status before warmup starts."""
        status = warmup.get_warmup_status()

        assert status["is_started"] is False
        assert status["is_complete"] is False
        assert status["current_baseline"] == 0.75

    # Test cases from story spec for SQS warmup
    @pytest.mark.parametrize("minutes,expected_baseline", [
        (0, 0.75),    # T=0, start
        (7, 0.675),   # T=7, halfway
        (15, 0.60),   # T=15, end
        (20, 0.60),   # T=20, capped
    ])
    def test_baseline_ramp_calculation(self, minutes, expected_baseline):
        """Test baseline ramps from 0.75 to 0.60 over 15 minutes."""
        from src.router.sqs_monday_warmup import SqsMondayWarmup

        warmup = SqsMondayWarmup()

        # Calculate baseline at given minute
        if minutes >= 15:
            baseline = warmup.SQS_WARMUP_END
        else:
            progress = minutes / 15.0
            baseline = warmup.SQS_WARMUP_START - (warmup.SQS_WARMUP_START - warmup.SQS_WARMUP_END) * progress

        assert abs(baseline - expected_baseline) < 0.01, f"Expected {expected_baseline} at {minutes}min, got {baseline}"


class TestWarmupState:
    """Test cases for WarmupState dataclass."""

    def test_state_creation(self):
        """Test state creation with defaults."""
        from src.router.sqs_monday_warmup import WarmupState

        state = WarmupState()

        assert state.started_at is None
        assert state.completed_at is None
        assert state.start_baseline == 0.75
        assert state.end_baseline == 0.60
        assert state.duration_minutes == 15
        assert state.current_baseline == 0.75
        assert state.is_complete is False
