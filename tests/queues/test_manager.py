"""
Tests for Queue Manager - Task Group 12

Tests queue operations including:
- Enqueue operations (add task to queue)
- Dequeue operations (FIFO ordering)
- Status updates (pending -> processing -> complete)
- Queue persistence (file read/write)
"""

import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

from src.queues.manager import QueueManager, QueueFileFormatError


class TestQueueManagerInitialization:
    """Test QueueManager initialization and setup."""

    def test_init_creates_directory_if_not_exists(self):
        """Test that QueueManager creates queue directory if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir) / "queues"
            manager = QueueManager(queue_dir_path=str(queue_dir))

            assert queue_dir.exists()
            assert queue_dir.is_dir()

    def test_init_with_existing_directory(self):
        """Test QueueManager initialization with existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir) / "queues"
            queue_dir.mkdir(parents=True, exist_ok=True)

            manager = QueueManager(queue_dir_path=str(queue_dir))

            assert manager.queue_dir == queue_dir

    def test_init_creates_queue_files_if_not_exist(self):
        """Test that missing queue files are created with empty arrays."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir) / "queues"
            manager = QueueManager(queue_dir_path=str(queue_dir))

            for queue_name in ["analyst", "quant", "executor"]:
                queue_file = queue_dir / f"{queue_name}_queue.json"
                assert queue_file.exists()

                with open(queue_file, "r") as f:
                    data = json.load(f)
                    assert data == []


class TestEnqueueOperations:
    """Test enqueue operations."""

    def test_enqueue_adds_task_to_queue(self):
        """Test that enqueue adds a task with correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            payload = {"test": "data", "value": 123}
            task_id = manager.enqueue("analyst", payload)

            assert isinstance(task_id, str)
            # Verify UUID format
            uuid_obj = uuid.UUID(task_id)
            assert str(uuid_obj) == task_id

            # Verify file content
            queue_file = Path(temp_dir) / "analyst_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            assert len(tasks) == 1
            assert tasks[0]["task_id"] == task_id
            assert tasks[0]["status"] == "pending"
            assert tasks[0]["payload"] == payload
            assert "created_at" in tasks[0]
            assert isinstance(tasks[0]["created_at"], (int, float))

    def test_enqueue_adds_multiple_tasks_in_order(self):
        """Test that multiple tasks are added in FIFO order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task_ids = []
            for i in range(3):
                task_id = manager.enqueue("quant", {"index": i})
                task_ids.append(task_id)

            queue_file = Path(temp_dir) / "quant_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            assert len(tasks) == 3
            # Verify order maintained
            assert tasks[0]["task_id"] == task_ids[0]
            assert tasks[1]["task_id"] == task_ids[1]
            assert tasks[2]["task_id"] == task_ids[2]

    def test_enqueue_invalid_agent_name_raises_error(self):
        """Test that invalid agent names raise appropriate error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            with pytest.raises(ValueError, match="Invalid agent name"):
                manager.enqueue("invalid_agent", {"data": "test"})


class TestDequeueOperations:
    """Test dequeue operations with FIFO ordering."""

    def test_dequeue_returns_next_pending_task(self):
        """Test that dequeue returns the first pending task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task_id = manager.enqueue("executor", {"action": "deploy"})

            task = manager.dequeue("executor")

            assert task is not None
            assert task["task_id"] == task_id
            assert task["status"] == "processing"
            assert task["payload"] == {"action": "deploy"}

    def test_dequeue_maintains_fifo_order(self):
        """Test that tasks are dequeued in FIFO order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            # Add tasks with identifiable payloads
            task_ids = []
            for i in range(3):
                task_id = manager.enqueue("analyst", {"order": i})
                task_ids.append(task_id)

            # Dequeue in order
            for i in range(3):
                task = manager.dequeue("analyst")
                assert task["task_id"] == task_ids[i]
                assert task["payload"]["order"] == i

    def test_dequeue_skips_non_pending_tasks(self):
        """Test that dequeue skips tasks that are not pending."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task1_id = manager.enqueue("quant", {"data": "first"})
            task2_id = manager.enqueue("quant", {"data": "second"})

            # Mark first task as complete
            manager.update_status("quant", task1_id, "complete")

            # Dequeue should return second task
            task = manager.dequeue("quant")
            assert task["task_id"] == task2_id

    def test_dequeue_returns_none_when_empty(self):
        """Test that dequeue returns None when queue is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task = manager.dequeue("analyst")
            assert task is None


class TestStatusUpdates:
    """Test status update operations."""

    def test_update_status_changes_task_status(self):
        """Test that update_status correctly changes task status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task_id = manager.enqueue("analyst", {"test": "data"})

            # Update to processing
            manager.update_status("analyst", task_id, "processing")

            queue_file = Path(temp_dir) / "analyst_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            assert tasks[0]["status"] == "processing"

    def test_update_status_with_complete(self):
        """Test status transition to complete."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task_id = manager.enqueue("executor", {"deploy": "bot"})

            manager.update_status("executor", task_id, "complete")

            queue_file = Path(temp_dir) / "executor_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            assert tasks[0]["status"] == "complete"

    def test_update_status_invalid_transition(self):
        """Test that invalid status transitions are rejected."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            task_id = manager.enqueue("quant", {"data": "test"})

            with pytest.raises(ValueError, match="Invalid status"):
                manager.update_status("quant", task_id, "invalid_status")

    def test_update_status_nonexistent_task(self):
        """Test updating status for non-existent task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)

            with pytest.raises(KeyError, match="Task not found"):
                manager.update_status("analyst", str(uuid.uuid4()), "processing")


class TestQueuePersistence:
    """Test queue file persistence and recovery."""

    def test_queue_persists_across_manager_instances(self):
        """Test that queue data persists when creating new manager instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First manager instance
            manager1 = QueueManager(queue_dir_path=temp_dir)
            task_id = manager1.enqueue("analyst", {"persistent": "data"})

            # Create new manager instance
            manager2 = QueueManager(queue_dir_path=temp_dir)

            # Verify task is still there
            task = manager2.dequeue("analyst")
            assert task is not None
            assert task["task_id"] == task_id

    def test_corrupted_queue_file_raises_error(self):
        """Test that corrupted queue file raises appropriate error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir) / "queues"
            queue_dir.mkdir(parents=True, exist_ok=True)

            # Create corrupted file
            queue_file = queue_dir / "analyst_queue.json"
            with open(queue_file, "w") as f:
                f.write("invalid json content")

            with pytest.raises(QueueFileFormatError):
                QueueManager(queue_dir_path=str(queue_dir))

    def test_status_changes_persist_to_file(self):
        """Test that status changes are immediately persisted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)
            task_id = manager.enqueue("quant", {"test": "persistence"})

            manager.update_status("quant", task_id, "complete")

            # Read file directly to verify persistence
            queue_file = Path(temp_dir) / "quant_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            assert tasks[0]["status"] == "complete"


class TestQueueFileFormat:
    """Test queue file format and structure."""

    def test_queue_file_has_required_fields(self):
        """Test that queue files contain all required fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)
            manager.enqueue("analyst", {"test": "data"})

            queue_file = Path(temp_dir) / "analyst_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            task = tasks[0]
            required_fields = ["task_id", "status", "payload", "created_at"]
            for field in required_fields:
                assert field in task

    def test_created_at_is_timestamp(self):
        """Test that created_at is a valid timestamp."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = QueueManager(queue_dir_path=temp_dir)
            manager.enqueue("executor", {"test": "timestamp"})

            queue_file = Path(temp_dir) / "executor_queue.json"
            with open(queue_file, "r") as f:
                tasks = json.load(f)

            created_at = tasks[0]["created_at"]
            assert isinstance(created_at, (int, float))
            # Verify it's a recent timestamp (within last minute)
            import time
            current_time = time.time()
            assert current_time - 60 < created_at <= current_time
