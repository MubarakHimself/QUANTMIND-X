"""P1 Tests: SessionCheckpointService milestone and interval triggers."""

import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.memory.session_checkpoint_service import (
    SessionCheckpointService,
    DEFAULT_CHECKPOINT_INTERVAL_MINUTES,
    DEFAULT_STALE_DRAFT_THRESHOLD_HOURS,
)


class TestMilestoneCheckpointEdgeCases:
    """P1: Test milestone checkpoint trigger edge cases."""

    @pytest.mark.asyncio
    async def test_milestone_checkpoint_respects_disabled_flag(self):
        """[P1] Milestone checkpoint MUST NOT fire when checkpoint_on_milestone=False."""
        service = SessionCheckpointService(checkpoint_on_milestone=False)

        with patch.object(service, "create_checkpoint", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None

            result = await service.checkpoint_on_agent_milestone(
                session_id="test-session",
                milestone_type="task_completed",
            )

            assert result["checkpoint_created"] is False
            assert result["reason"] == "milestone_checkpoint_disabled"
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_milestone_checkpoint_creates_on_valid_milestone(self):
        """[P1] Valid milestone type SHOULD trigger checkpoint creation."""
        service = SessionCheckpointService(
            checkpoint_on_milestone=True,
            checkpoint_interval_minutes=5,
        )

        with patch.object(service, "create_checkpoint", new_callable=AsyncMock) as mock_create, \
             patch.object(service, "trigger_reflection", new_callable=AsyncMock) as mock_reflect:

            mock_create.return_value = "cp-123"
            mock_reflect.return_value = {"committed_count": 3}

            result = await service.checkpoint_on_agent_milestone(
                session_id="test-session",
                milestone_type="task_completed",
            )

            assert result["checkpoint_created"] is True
            assert result["checkpoint_id"] == "cp-123"
            mock_create.assert_called_once_with(session_id="test-session")
            mock_reflect.assert_called_once()

    @pytest.mark.asyncio
    async def test_milestone_type_validates_input(self):
        """[P1] Invalid milestone type should be handled gracefully."""
        service = SessionCheckpointService(checkpoint_on_milestone=True)

        with patch.object(service, "create_checkpoint", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = None

            # Unknown milestone type - should still attempt checkpoint
            result = await service.checkpoint_on_agent_milestone(
                session_id="test-session",
                milestone_type="unknown_action",
            )

            # Should attempt checkpoint but may not create one
            assert "checkpoint_created" in result


class TestIntervalCheckpointBoundary:
    """P1: Test interval checkpoint boundary conditions."""

    def test_first_checkpoint_always_allowed(self):
        """[P1] First checkpoint for a session MUST be allowed regardless of interval."""
        service = SessionCheckpointService(checkpoint_interval_minutes=60)

        # No previous checkpoint - should allow
        assert service.should_auto_checkpoint("new-session") is True

    def test_checkpoint_at_exact_interval_boundary(self):
        """[P1] Checkpoint at exact interval boundary should be allowed."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Set checkpoint to exactly 5 minutes ago
        service._last_checkpoint_time["session-boundary"] = datetime.now(timezone.utc) - timedelta(minutes=5)

        # At exact boundary, should be allowed
        assert service.should_auto_checkpoint("session-boundary") is True

    def test_checkpoint_just_before_interval_blocked(self):
        """[P1] Checkpoint just before interval should be blocked."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Set checkpoint to 4 minutes 59 seconds ago
        service._last_checkpoint_time["session-just-before"] = datetime.now(timezone.utc) - timedelta(minutes=4, seconds=59)

        # Should be blocked (not yet at 5 minutes)
        assert service.should_auto_checkpoint("session-just-before") is False

    def test_checkpoint_after_long_interval_allowed(self):
        """[P1] Checkpoint after very long interval (>2x) should be allowed."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Set checkpoint to 15 minutes ago (3x interval)
        service._last_checkpoint_time["session-long"] = datetime.now(timezone.utc) - timedelta(minutes=15)

        # Should be allowed
        assert service.should_auto_checkpoint("session-long") is True

    def test_multiple_sessions_independent_interval_tracking(self):
        """[P1] Each session should have independent interval tracking."""
        service = SessionCheckpointService(checkpoint_interval_minutes=5)

        # Session A: recent checkpoint
        service._last_checkpoint_time["session-A"] = datetime.now(timezone.utc)

        # Session B: no checkpoint
        # Session C: old checkpoint
        service._last_checkpoint_time["session-C"] = datetime.now(timezone.utc) - timedelta(minutes=10)

        assert service.should_auto_checkpoint("session-A") is False
        assert service.should_auto_checkpoint("session-B") is True
        assert service.should_auto_checkpoint("session-C") is True


class TestStaleDraftCleanupThreshold:
    """P1: Test stale draft cleanup with various thresholds."""

    @pytest.mark.asyncio
    async def test_cleanup_respects_custom_threshold(self):
        """[P1] Cleanup should use configurable threshold hours."""
        service = SessionCheckpointService(stale_draft_threshold_hours=12)

        with patch("src.agents.memory.session_checkpoint_service.Path") as mock_path, \
             patch("src.memory.graph.reflection_executor.create_reflection_executor") as mock_create:

            mock_executor = MagicMock()
            mock_executor.cleanup_stale_drafts = AsyncMock(return_value={
                "archived_count": 5,
                "deleted_count": 2,
            })
            mock_create.return_value = mock_executor

            result = await service.cleanup_stale_drafts()

            mock_executor.cleanup_stale_drafts.assert_called_once_with(threshold_hours=12)
            assert result["archived_count"] == 5

    @pytest.mark.asyncio
    async def test_cleanup_default_threshold(self):
        """[P1] Cleanup should use DEFAULT_STALE_DRAFT_THRESHOLD_HOURS when not configured."""
        service = SessionCheckpointService()  # Uses defaults

        with patch("src.agents.memory.session_checkpoint_service.Path") as mock_path, \
             patch("src.memory.graph.reflection_executor.create_reflection_executor") as mock_create:

            mock_executor = MagicMock()
            mock_executor.cleanup_stale_drafts = AsyncMock(return_value={
                "archived_count": 0,
                "deleted_count": 0,
            })
            mock_create.return_value = mock_executor

            await service.cleanup_stale_drafts()

            mock_executor.cleanup_stale_drafts.assert_called_once_with(
                threshold_hours=DEFAULT_STALE_DRAFT_THRESHOLD_HOURS
            )

    @pytest.mark.asyncio
    async def test_cleanup_returns_zero_when_no_stale_drafts(self):
        """[P1] Cleanup should return zero counts when no stale drafts exist."""
        service = SessionCheckpointService(stale_draft_threshold_hours=24)

        with patch("src.agents.memory.session_checkpoint_service.Path") as mock_path, \
             patch("src.memory.graph.reflection_executor.create_reflection_executor") as mock_create:

            mock_executor = MagicMock()
            mock_executor.cleanup_stale_drafts = AsyncMock(return_value={
                "archived_count": 0,
                "deleted_count": 0,
            })
            mock_create.return_value = mock_executor

            result = await service.cleanup_stale_drafts()

            assert result["archived_count"] == 0
            assert result["deleted_count"] == 0
