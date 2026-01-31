"""
Unit tests for TaskQueue - Task Group 1

Tests TaskQueue operations including:
- Enqueue operations (add task to queue)
- Dequeue operations (FIFO ordering)
- Peek operations (view without removing)
- Size operations (get queue length)
- File locking for concurrent access
"""

import json
import tempfile
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from src.queues.task_queue import TaskQueue


class TestTaskQueueInitialization:
    """Test TaskQueue initialization and setup."""

    def test_init_creates_queue_file_if_not_exists(self):
        """Test that TaskQueue creates queue file if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            queue_file = Path(temp_dir) / "analyst_tasks.json"
            assert queue_file.exists()
            
            with open(queue_file, 'r') as f:
                data = json.load(f)
                assert data == []

    def test_init_with_existing_queue_file(self):
        """Test TaskQueue initialization with existing queue file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_file = Path(temp_dir) / "quant_tasks.json"
            
            # Create existing file with data
            existing_data = [{"task_data": {"test": "data"}, "timestamp": time.time()}]
            with open(queue_file, 'w') as f:
                json.dump(existing_data, f)
            
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            # Verify existing data is preserved
            assert queue.size() == 1

    def test_init_invalid_queue_type_raises_error(self):
        """Test that invalid queue types raise ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="Invalid queue_type"):
                TaskQueue("invalid_type", queue_dir=temp_dir)

    def test_init_creates_directory_if_not_exists(self):
        """Test that TaskQueue creates queue directory if missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_dir = Path(temp_dir) / "queues"
            queue = TaskQueue("executor", queue_dir=str(queue_dir))
            
            assert queue_dir.exists()
            assert queue_dir.is_dir()


class TestEnqueueOperations:
    """Test enqueue operations."""

    def test_enqueue_adds_task_to_queue(self):
        """Test that enqueue adds a task to the queue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            task = {"action": "analyze", "data": "test"}
            queue.enqueue(task)
            
            assert queue.size() == 1
            
            # Verify file content
            queue_file = Path(temp_dir) / "analyst_tasks.json"
            with open(queue_file, 'r') as f:
                tasks = json.load(f)
            
            assert len(tasks) == 1
            assert tasks[0]["task_data"] == task
            assert "timestamp" in tasks[0]

    def test_enqueue_multiple_tasks_maintains_order(self):
        """Test that multiple enqueued tasks maintain FIFO order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            tasks = [
                {"order": 1, "data": "first"},
                {"order": 2, "data": "second"},
                {"order": 3, "data": "third"}
            ]
            
            for task in tasks:
                queue.enqueue(task)
            
            assert queue.size() == 3
            
            # Verify order in file
            queue_file = Path(temp_dir) / "quant_tasks.json"
            with open(queue_file, 'r') as f:
                stored_tasks = json.load(f)
            
            for i, task in enumerate(tasks):
                assert stored_tasks[i]["task_data"] == task

    def test_enqueue_with_complex_task_data(self):
        """Test enqueue with complex nested task data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            complex_task = {
                "action": "deploy",
                "config": {
                    "strategy": "momentum",
                    "parameters": {
                        "period": 14,
                        "threshold": 0.8
                    }
                },
                "metadata": ["tag1", "tag2"]
            }
            
            queue.enqueue(complex_task)
            
            dequeued = queue.dequeue()
            assert dequeued == complex_task


class TestDequeueOperations:
    """Test dequeue operations with FIFO ordering."""

    def test_dequeue_returns_first_task(self):
        """Test that dequeue returns the first task (FIFO)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            task1 = {"order": 1}
            task2 = {"order": 2}
            
            queue.enqueue(task1)
            queue.enqueue(task2)
            
            dequeued = queue.dequeue()
            assert dequeued == task1
            assert queue.size() == 1

    def test_dequeue_maintains_fifo_order(self):
        """Test that dequeue maintains FIFO order across multiple operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            tasks = [{"id": i} for i in range(5)]
            
            for task in tasks:
                queue.enqueue(task)
            
            # Dequeue all tasks and verify order
            for expected_task in tasks:
                dequeued = queue.dequeue()
                assert dequeued == expected_task

    def test_dequeue_returns_none_when_empty(self):
        """Test that dequeue returns None when queue is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            result = queue.dequeue()
            assert result is None

    def test_dequeue_removes_task_from_file(self):
        """Test that dequeue removes task from the queue file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            queue.enqueue({"test": "data"})
            assert queue.size() == 1
            
            queue.dequeue()
            assert queue.size() == 0
            
            # Verify file is empty
            queue_file = Path(temp_dir) / "analyst_tasks.json"
            with open(queue_file, 'r') as f:
                tasks = json.load(f)
            assert tasks == []


class TestPeekOperations:
    """Test peek operations."""

    def test_peek_returns_first_task_without_removing(self):
        """Test that peek returns first task without removing it."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            task = {"action": "backtest"}
            queue.enqueue(task)
            
            peeked = queue.peek()
            assert peeked == task
            assert queue.size() == 1  # Size unchanged

    def test_peek_returns_none_when_empty(self):
        """Test that peek returns None when queue is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            result = queue.peek()
            assert result is None

    def test_peek_multiple_times_returns_same_task(self):
        """Test that multiple peeks return the same task."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            task = {"data": "consistent"}
            queue.enqueue(task)
            
            peek1 = queue.peek()
            peek2 = queue.peek()
            peek3 = queue.peek()
            
            assert peek1 == peek2 == peek3 == task
            assert queue.size() == 1

    def test_peek_shows_first_task_in_fifo_order(self):
        """Test that peek shows the first task in FIFO order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            queue.enqueue({"order": 1})
            queue.enqueue({"order": 2})
            queue.enqueue({"order": 3})
            
            peeked = queue.peek()
            assert peeked == {"order": 1}


class TestSizeOperations:
    """Test size operations."""

    def test_size_returns_zero_for_empty_queue(self):
        """Test that size returns 0 for empty queue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            assert queue.size() == 0

    def test_size_increases_with_enqueue(self):
        """Test that size increases with each enqueue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            assert queue.size() == 0
            
            queue.enqueue({"task": 1})
            assert queue.size() == 1
            
            queue.enqueue({"task": 2})
            assert queue.size() == 2
            
            queue.enqueue({"task": 3})
            assert queue.size() == 3

    def test_size_decreases_with_dequeue(self):
        """Test that size decreases with each dequeue."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            for i in range(3):
                queue.enqueue({"task": i})
            
            assert queue.size() == 3
            
            queue.dequeue()
            assert queue.size() == 2
            
            queue.dequeue()
            assert queue.size() == 1
            
            queue.dequeue()
            assert queue.size() == 0

    def test_size_unchanged_by_peek(self):
        """Test that size is unchanged by peek operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            queue.enqueue({"test": "data"})
            initial_size = queue.size()
            
            queue.peek()
            queue.peek()
            queue.peek()
            
            assert queue.size() == initial_size


class TestFileLocking:
    """Test file locking for concurrent access."""

    def test_concurrent_enqueue_operations(self):
        """Test that concurrent enqueue operations are thread-safe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            num_tasks = 50
            
            def enqueue_task(task_id):
                queue.enqueue({"task_id": task_id})
                return task_id
            
            # Enqueue tasks concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(enqueue_task, i) for i in range(num_tasks)]
                results = [f.result() for f in as_completed(futures)]
            
            # Verify all tasks were enqueued
            assert queue.size() == num_tasks
            assert len(results) == num_tasks

    def test_concurrent_dequeue_operations(self):
        """Test that concurrent dequeue operations are thread-safe."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            # Enqueue tasks
            num_tasks = 50
            for i in range(num_tasks):
                queue.enqueue({"task_id": i})
            
            def dequeue_task():
                return queue.dequeue()
            
            # Dequeue tasks concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(dequeue_task) for _ in range(num_tasks)]
                results = [f.result() for f in as_completed(futures)]
            
            # Verify all tasks were dequeued
            assert queue.size() == 0
            # All results should be non-None
            assert all(r is not None for r in results)
            # Should have unique task_ids
            task_ids = [r["task_id"] for r in results]
            assert len(set(task_ids)) == num_tasks

    def test_concurrent_mixed_operations(self):
        """Test concurrent mix of enqueue, dequeue, peek, and size operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            # Pre-populate queue
            for i in range(20):
                queue.enqueue({"initial": i})
            
            def mixed_operations(op_id):
                if op_id % 4 == 0:
                    queue.enqueue({"new": op_id})
                elif op_id % 4 == 1:
                    queue.dequeue()
                elif op_id % 4 == 2:
                    queue.peek()
                else:
                    queue.size()
                return op_id
            
            # Execute mixed operations concurrently
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(mixed_operations, i) for i in range(100)]
                results = [f.result() for f in as_completed(futures)]
            
            # Verify operations completed
            assert len(results) == 100
            # Queue should still be functional
            final_size = queue.size()
            assert final_size >= 0


class TestQueuePersistence:
    """Test queue persistence across instances."""

    def test_queue_persists_across_instances(self):
        """Test that queue data persists when creating new instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # First instance
            queue1 = TaskQueue("analyst", queue_dir=temp_dir)
            queue1.enqueue({"persistent": "data"})
            queue1.enqueue({"more": "data"})
            
            # Create new instance
            queue2 = TaskQueue("analyst", queue_dir=temp_dir)
            
            # Verify data persisted
            assert queue2.size() == 2
            assert queue2.dequeue() == {"persistent": "data"}
            assert queue2.dequeue() == {"more": "data"}

    def test_operations_persist_to_file_immediately(self):
        """Test that operations are immediately persisted to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            task = {"immediate": "persistence"}
            queue.enqueue(task)
            
            # Read file directly
            queue_file = Path(temp_dir) / "quant_tasks.json"
            with open(queue_file, 'r') as f:
                tasks = json.load(f)
            
            assert len(tasks) == 1
            assert tasks[0]["task_data"] == task


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_enqueue_empty_dict(self):
        """Test enqueue with empty dictionary."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            queue.enqueue({})
            assert queue.size() == 1
            assert queue.dequeue() == {}

    def test_enqueue_none_values(self):
        """Test enqueue with None values in task data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            task = {"key": None, "value": "test"}
            queue.enqueue(task)
            
            dequeued = queue.dequeue()
            assert dequeued == task

    def test_large_task_data(self):
        """Test enqueue/dequeue with large task data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            # Create large task
            large_task = {
                "data": ["item" + str(i) for i in range(1000)],
                "metadata": {f"key{i}": f"value{i}" for i in range(100)}
            }
            
            queue.enqueue(large_task)
            dequeued = queue.dequeue()
            
            assert dequeued == large_task

    def test_corrupted_queue_file_recovery(self):
        """Test that corrupted queue file is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_file = Path(temp_dir) / "executor_tasks.json"
            
            # Create corrupted file
            with open(queue_file, 'w') as f:
                f.write("invalid json content {{{")
            
            # TaskQueue should handle corruption gracefully
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            # Should be able to use queue normally
            assert queue.size() == 0
            queue.enqueue({"recovered": "data"})
            assert queue.size() == 1
