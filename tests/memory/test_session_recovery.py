"""Tests for session recovery and GraphMemoryFacade load_committed_state."""
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.types import MemoryNode, MemoryNodeType, MemoryCategory, SessionStatus


class TestLoadCommittedState:
    """Test GraphMemoryFacade.load_committed_state method."""

    def test_load_committed_state_empty_session(self):
        """Test loading committed state for session with no memories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_memory.db"
            facade = GraphMemoryFacade(db_path=db_path)

            result = facade.load_committed_state("empty-session")

            assert result["committed_nodes_count"] == 0
            assert result["nodes"] == []

            facade.close()

    def test_load_committed_state_method_exists(self):
        """Test load_committed_state method exists on facade."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/test_memory.db"
            facade = GraphMemoryFacade(db_path=db_path)

            # Verify method exists
            assert hasattr(facade, "load_committed_state")
            assert callable(facade.load_committed_state)

            facade.close()


class TestReflectionExecutorRecovery:
    """Test ReflectionExecutor session recovery."""

    def test_recovery_method_exists(self):
        """Test recover_session method exists on ReflectionExecutor."""
        from src.memory.graph.reflection_executor import ReflectionExecutor

        assert hasattr(ReflectionExecutor, "recover_session")
        assert callable(getattr(ReflectionExecutor, "recover_session"))

    def test_cleanup_stale_drafts_method_exists(self):
        """Test cleanup_stale_drafts method exists on ReflectionExecutor."""
        from src.memory.graph.reflection_executor import ReflectionExecutor

        assert hasattr(ReflectionExecutor, "cleanup_stale_drafts")
        assert callable(getattr(ReflectionExecutor, "cleanup_stale_drafts"))


class TestCheckpointServiceIntegration:
    """Integration tests for checkpoint service."""

    @pytest.mark.asyncio
    async def test_trigger_reflection_integration(self):
        """Test trigger_reflection method exists and is callable."""
        from src.agents.memory.session_checkpoint_service import SessionCheckpointService

        service = SessionCheckpointService()

        # Check methods exist
        assert hasattr(service, "trigger_reflection")
        assert hasattr(service, "checkpoint_on_agent_milestone")
        assert hasattr(service, "auto_checkpoint_if_due")
        assert hasattr(service, "cleanup_stale_drafts")
        assert hasattr(service, "should_checkpoint_on_milestone")
        assert hasattr(service, "should_auto_checkpoint")

    def test_checkpoint_service_configuration(self):
        """Test checkpoint service has correct configuration attributes."""
        from src.agents.memory.session_checkpoint_service import (
            SessionCheckpointService,
            DEFAULT_CHECKPOINT_INTERVAL_MINUTES,
            DEFAULT_STALE_DRAFT_THRESHOLD_HOURS,
        )

        service = SessionCheckpointService()

        # Verify config attributes
        assert hasattr(service, "checkpoint_interval_minutes")
        assert hasattr(service, "stale_draft_threshold_hours")
        assert hasattr(service, "checkpoint_on_milestone")
        assert hasattr(service, "_last_checkpoint_time")

        # Verify defaults
        assert service.checkpoint_interval_minutes == DEFAULT_CHECKPOINT_INTERVAL_MINUTES
        assert service.stale_draft_threshold_hours == DEFAULT_STALE_DRAFT_THRESHOLD_HOURS
