"""
Unit tests for Checkpoint System

Tests the checkpoint, heartbeat, progress tracking, and interrupt handling
functionality for long-running agents.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

from src.agents.checkpoint import (
    CheckpointManager,
    HeartbeatManager,
    ProgressTracker,
    InterruptHandler,
    LongRunningAgent,
    AgentState,
    HeartbeatStatus,
)


class TestCheckpointManager:
    """Test suite for CheckpointManager."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, temp_checkpoint_dir):
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        checkpoint = await manager.save_checkpoint(
            agent_id="test_agent",
            task_id="test_task",
            session_id="test_session",
            conversation_history=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            tool_outputs=[
                {"tool_name": "search", "arguments": {"query": "test"}, "result": "result"}
            ],
            partial_results={"step1": "done"},
            progress={"percent": 50.0, "current_step": "processing"},
            metadata={"extra": "data"},
        )

        assert checkpoint.agent_id == "test_agent"
        assert checkpoint.task_id == "test_task"
        assert checkpoint.checkpoint_number == 1
        assert checkpoint.conversation_history[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, temp_checkpoint_dir):
        """Test retrieving the latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save multiple checkpoints
        await manager.save_checkpoint(
            agent_id="test_agent",
            task_id="test_task",
            session_id="test_session",
            conversation_history=[{"role": "user", "content": "Hello"}],
            tool_outputs=[],
            partial_results={},
            progress={"percent": 10.0},
        )

        await manager.save_checkpoint(
            agent_id="test_agent",
            task_id="test_task",
            session_id="test_session",
            conversation_history=[{"role": "user", "content": "Hello"}],
            tool_outputs=[],
            partial_results={},
            progress={"percent": 50.0},
        )

        latest = manager.get_latest_checkpoint("test_agent", "test_task")
        assert latest is not None
        assert latest.checkpoint_number == 2
        assert latest.progress["percent"] == 50.0

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, temp_checkpoint_dir):
        """Test listing checkpoints."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save checkpoints
        for i in range(3):
            await manager.save_checkpoint(
                agent_id="test_agent",
                task_id="test_task",
                session_id="test_session",
                conversation_history=[],
                tool_outputs=[],
                partial_results={},
                progress={"percent": i * 30.0},
            )

        checkpoints = manager.list_checkpoints("test_agent", "test_task")
        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test automatic cleanup of old checkpoints."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            max_checkpoints=2,
        )

        # Save more checkpoints than the limit
        for i in range(5):
            await manager.save_checkpoint(
                agent_id="test_agent",
                task_id="test_task",
                session_id="test_session",
                conversation_history=[],
                tool_outputs=[],
                partial_results={},
                progress={"percent": i * 20.0},
            )

        # Should only have 2 checkpoints remaining
        checkpoints = manager.list_checkpoints("test_agent", "test_task")
        assert len(checkpoints) == 2
        assert checkpoints[0]["checkpoint_number"] == 5
        assert checkpoints[1]["checkpoint_number"] == 4


class TestHeartbeatManager:
    """Test suite for HeartbeatManager."""

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Test registering an agent for heartbeat monitoring."""
        manager = HeartbeatManager(heartbeat_interval_seconds=30)

        heartbeat = await manager.register_agent("test_agent", "test_task", 0.0, "starting")

        assert heartbeat.agent_id == "test_agent"
        assert heartbeat.status == HeartbeatStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_send_heartbeat(self):
        """Test sending heartbeat updates."""
        manager = HeartbeatManager(heartbeat_interval_seconds=30)

        await manager.register_agent("test_agent", "test_task", 0.0, "starting")

        heartbeat = await manager.send_heartbeat(
            "test_agent", "test_task", 0.5, "processing", eta_seconds=120.0
        )

        assert heartbeat.progress == 0.5
        assert heartbeat.current_step == "processing"
        assert heartbeat.eta_seconds == 120.0

    @pytest.mark.asyncio
    async def test_get_all_heartbeats(self):
        """Test getting all heartbeats."""
        manager = HeartbeatManager(heartbeat_interval_seconds=30)

        await manager.register_agent("agent1", "task1", 0.0, "starting")
        await manager.register_agent("agent2", "task2", 0.5, "processing")

        heartbeats = await manager.get_all_heartbeats()
        assert len(heartbeats) == 2


class TestProgressTracker:
    """Test suite for ProgressTracker."""

    @pytest.mark.asyncio
    async def test_progress_tracking(self):
        """Test progress tracking."""
        tracker = ProgressTracker(total_steps=100)

        assert tracker.progress_percent == 0.0

        await tracker.update(10, "step 1")
        assert tracker.completed_steps == 10
        assert tracker.progress_percent == 10.0

    @pytest.mark.asyncio
    async def test_eta_estimation(self):
        """Test ETA estimation."""
        tracker = ProgressTracker(total_steps=100)

        # Update progress with explicit step counts multiple times
        for i in range(10):
            await tracker.update(completed_steps=i + 1, step_name=f"step {i}")
            # Add some time to make ETA calculable
            tracker.step_times.append(0.1)

        # ETA should be calculated after some updates
        eta = tracker.eta_seconds
        assert eta is not None
        assert eta >= 0

    @pytest.mark.asyncio
    async def test_increment_progress(self):
        """Test incrementing progress without specifying steps."""
        tracker = ProgressTracker(total_steps=10)

        await tracker.update()
        assert tracker.completed_steps == 1

        await tracker.update()
        assert tracker.completed_steps == 2


class TestLongRunningAgent:
    """Test suite for LongRunningAgent."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, temp_checkpoint_dir):
        """Test long-running agent lifecycle."""
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        heartbeat_manager = HeartbeatManager()

        agent = LongRunningAgent(
            agent_id="test_agent",
            task_id="test_task",
            session_id="test_session",
            checkpoint_manager=checkpoint_manager,
            heartbeat_manager=heartbeat_manager,
        )

        # Add test data
        agent.add_message("user", "Hello")
        agent.add_tool_output("search", {"query": "test"}, "result")

        # Start agent
        await agent.start()
        assert agent.state == AgentState.RUNNING

        # Update progress
        progress = await agent.update_progress(50, "processing")
        assert progress.progress_percent == 50.0
        assert progress.current_step == "processing"

        # Stop agent
        await agent.stop("completed")
        assert agent.state == AgentState.COMPLETED

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, temp_checkpoint_dir):
        """Test resuming from checkpoint."""
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        heartbeat_manager = HeartbeatManager()

        # Create and start an agent
        agent1 = LongRunningAgent(
            agent_id="test_agent",
            task_id="test_task",
            session_id="test_session",
            checkpoint_manager=checkpoint_manager,
            heartbeat_manager=heartbeat_manager,
        )

        agent1.add_message("user", "Hello")
        agent1.add_tool_output("search", {"query": "test"}, "result")

        await agent1.start()
        await agent1.update_progress(50, "halfway")
        checkpoint = await agent1.save_checkpoint()
        await agent1.stop("completed")

        # Resume from checkpoint
        agent2 = await LongRunningAgent.resume_from_checkpoint(
            checkpoint,
            checkpoint_manager,
            heartbeat_manager,
        )

        assert agent2.agent_id == "test_agent"
        assert agent2.state == AgentState.RUNNING
        assert len(agent2.conversation_history) == 1


class TestInterruptHandler:
    """Test suite for InterruptHandler."""

    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create a temporary checkpoint directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_shutdown_detection(self, temp_checkpoint_dir):
        """Test shutdown detection."""
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        handler = InterruptHandler(checkpoint_manager)

        assert not handler.is_shutting_down()

        # Note: Full signal testing requires signal module which is limited in tests
        # This just tests the flag mechanism


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
