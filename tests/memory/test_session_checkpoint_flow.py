"""Tests for SessionCheckpointService checkpoint flow integration."""
import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.agents.memory.session_checkpoint_service import (
    SessionCheckpointService,
    DEFAULT_CHECKPOINT_INTERVAL_MINUTES,
    DEFAULT_STALE_DRAFT_THRESHOLD_HOURS,
)


class TestSessionCheckpointConfig:
    """Test configurable checkpoint settings."""

    def test_default_config_values(self):
        """Test default configuration values are correct."""
        service = SessionCheckpointService()

        assert service.checkpoint_interval_minutes == DEFAULT_CHECKPOINT_INTERVAL_MINUTES
        assert service.stale_draft_threshold_hours == DEFAULT_STALE_DRAFT_THRESHOLD_HOURS
        assert service.checkpoint_on_milestone is True

    def test_custom_config_values(self):
        """Test custom configuration values."""
        service = SessionCheckpointService(
            checkpoint_interval_minutes=10,
            stale_draft_threshold_hours=48,
            checkpoint_on_milestone=False,
        )

        assert service.checkpoint_interval_minutes == 10
        assert service.stale_draft_threshold_hours == 48
        assert service.checkpoint_on_milestone is False

    def test_env_variable_override(self):
        """Test environment variable configuration override."""
        with patch.dict(
            os.environ,
            {
                "SESSION_CHECKPOINT_INTERVAL_MINUTES": "15",
                "SESSION_STALE_DRAFT_THRESHOLD_HOURS": "36",
                "SESSION_CHECKPOINT_ON_MILESTONE": "false",
            },
        ):
            service = SessionCheckpointService()

            assert service.checkpoint_interval_minutes == 15
            assert service.stale_draft_threshold_hours == 36
            assert service.checkpoint_on_milestone is False


class TestMilestoneCheckpoint:
    """Test milestone-based checkpoint triggers."""

    @pytest.mark.asyncio
    async def test_checkpoint_on_milestone_disabled(self):
        """Test milestone checkpoint when disabled."""
        service = SessionCheckpointService(checkpoint_on_milestone=False)

        result = await service.checkpoint_on_agent_milestone(
            session_id="test-session",
            milestone_type="task_completed",
        )

        assert result["checkpoint_created"] is False
        assert result["reason"] == "milestone_checkpoint_disabled"

    @pytest.mark.asyncio
    async def test_checkpoint_on_milestone_enabled(self):
        """Test milestone checkpoint when enabled."""
        service = SessionCheckpointService(checkpoint_on_milestone=True)

        with patch.object(service, "create_checkpoint") as mock_create, patch.object(
            service, "trigger_reflection"
        ) as mock_reflection:
            mock_create.return_value = "checkpoint-123"
            mock_reflection.return_value = {"committed_count": 5}

            result = await service.checkpoint_on_agent_milestone(
                session_id="test-session",
                milestone_type="task_completed",
            )

            assert result["checkpoint_created"] is True
            assert result["checkpoint_id"] == "checkpoint-123"
            assert result["milestone_type"] == "task_completed"
            mock_create.assert_called_once()
            mock_reflection.assert_called_once()


class TestAutoCheckpoint:
    """Test automatic interval-based checkpointing."""

    def test_first_checkpoint_allowed(self):
        """Test first checkpoint is always allowed for a session."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # No previous checkpoint
        assert service.should_auto_checkpoint("new-session") is True

    def test_checkpoint_within_interval_blocked(self):
        """Test checkpoint within interval is blocked."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Set last checkpoint to now
        service._last_checkpoint_time["session-1"] = datetime.now(timezone.utc)

        # Should be blocked (within 5 minutes)
        assert service.should_auto_checkpoint("session-1") is False

    def test_checkpoint_after_interval_allowed(self):
        """Test checkpoint after interval is allowed."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Set last checkpoint to 10 minutes ago
        service._last_checkpoint_time["session-1"] = datetime.now(
            timezone.utc
        ) - timedelta(minutes=10)

        # Should be allowed (after 5 minutes)
        assert service.should_auto_checkpoint("session-1") is True


class TestStaleDraftCleanup:
    """Test stale draft cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_drafts_calls_executor(self):
        """Test cleanup delegates to ReflectionExecutor."""
        service = SessionCheckpointService(stale_draft_threshold_hours=24)

        with patch("src.agents.memory.session_checkpoint_service.Path") as mock_path:
            mock_path.return_value = "test_db_path"

            with patch(
                "src.memory.graph.reflection_executor.create_reflection_executor"
            ) as mock_create_executor:
                mock_executor = MagicMock()
                mock_executor.cleanup_stale_drafts.return_value = {
                    "archived_count": 3,
                    "deleted_count": 1,
                }
                mock_create_executor.return_value = mock_executor

                result = await service.cleanup_stale_drafts()

                assert result["archived_count"] == 3
                assert result["deleted_count"] == 1
                mock_executor.cleanup_stale_drafts.assert_called_once_with(
                    threshold_hours=24
                )


# Import timedelta for the tests
from datetime import timedelta
