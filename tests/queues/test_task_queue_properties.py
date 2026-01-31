"""
Property-Based Tests for TaskQueue - Task Group 1

Tests universal properties of TaskQueue operations using Hypothesis.

**Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
For any sequence of task submissions to agent queues, tasks SHALL be processed 
in first-in-first-out order without reordering.

**Validates: Requirements 1.5**
"""

import tempfile
from pathlib import Path
from typing import List, Dict, Any

import pytest
from hypothesis import given, strategies as st, settings

from src.queues.task_queue import TaskQueue


# Strategy for generating task data
task_data_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=20, alphabet=st.characters(blacklist_characters=['\x00'])),
    values=st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=50, alphabet=st.characters(blacklist_characters=['\x00'])),
        st.booleans(),
        st.none()
    ),
    min_size=1,
    max_size=10
)


class TestTaskQueueFIFOProperty:
    """
    Property-based tests for FIFO ordering guarantee.
    
    **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
    """
    
    @given(tasks=st.lists(task_data_strategy, min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_property(self, tasks: List[Dict[str, Any]]):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: For any sequence of task submissions to agent queues, 
        tasks SHALL be processed in first-in-first-out order without reordering.
        
        **Validates: Requirements 1.5**
        
        This property test verifies that:
        1. Tasks are dequeued in the exact order they were enqueued
        2. No task reordering occurs regardless of task content
        3. FIFO ordering is maintained across all queue operations
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            # Add unique identifiers to track order
            indexed_tasks = [{"index": i, "data": task} for i, task in enumerate(tasks)]
            
            # Enqueue all tasks
            for task in indexed_tasks:
                queue.enqueue(task)
            
            # Dequeue all tasks and verify FIFO order
            dequeued_tasks = []
            while queue.size() > 0:
                dequeued = queue.dequeue()
                dequeued_tasks.append(dequeued)
            
            # Verify exact FIFO ordering
            assert len(dequeued_tasks) == len(indexed_tasks), \
                "All enqueued tasks should be dequeued"
            
            for i, (original, dequeued) in enumerate(zip(indexed_tasks, dequeued_tasks)):
                assert dequeued == original, \
                    f"Task at position {i} should match enqueue order"
                assert dequeued["index"] == i, \
                    f"Task index should be {i}, got {dequeued['index']}"
    
    @given(
        initial_tasks=st.lists(task_data_strategy, min_size=1, max_size=20),
        additional_tasks=st.lists(task_data_strategy, min_size=1, max_size=20)
    )
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_with_interleaved_operations(
        self, 
        initial_tasks: List[Dict[str, Any]], 
        additional_tasks: List[Dict[str, Any]]
    ):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: FIFO ordering is maintained even when enqueue and dequeue 
        operations are interleaved.
        
        **Validates: Requirements 1.5**
        
        This test verifies that:
        1. Partial dequeuing doesn't affect FIFO order of remaining tasks
        2. New enqueues after partial dequeues maintain FIFO order
        3. Order is preserved across mixed operations
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            # Add indices to track order
            indexed_initial = [{"batch": "initial", "index": i, "data": task} 
                             for i, task in enumerate(initial_tasks)]
            indexed_additional = [{"batch": "additional", "index": i, "data": task} 
                                for i, task in enumerate(additional_tasks)]
            
            # Enqueue initial tasks
            for task in indexed_initial:
                queue.enqueue(task)
            
            # Dequeue half of initial tasks
            dequeue_count = len(indexed_initial) // 2
            first_batch = []
            for _ in range(dequeue_count):
                first_batch.append(queue.dequeue())
            
            # Enqueue additional tasks
            for task in indexed_additional:
                queue.enqueue(task)
            
            # Dequeue remaining tasks
            remaining_tasks = []
            while queue.size() > 0:
                remaining_tasks.append(queue.dequeue())
            
            # Verify first batch maintains FIFO order
            for i, task in enumerate(first_batch):
                assert task["batch"] == "initial"
                assert task["index"] == i
            
            # Verify remaining tasks maintain FIFO order
            # Should be: rest of initial tasks, then all additional tasks
            expected_remaining = indexed_initial[dequeue_count:] + indexed_additional
            
            assert len(remaining_tasks) == len(expected_remaining)
            for dequeued, expected in zip(remaining_tasks, expected_remaining):
                assert dequeued == expected
    
    @given(
        tasks=st.lists(task_data_strategy, min_size=2, max_size=30),
        peek_count=st.integers(min_value=1, max_value=10)
    )
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_unaffected_by_peek(
        self, 
        tasks: List[Dict[str, Any]], 
        peek_count: int
    ):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: Peek operations do not affect FIFO ordering of subsequent dequeues.
        
        **Validates: Requirements 1.5**
        
        This test verifies that:
        1. Multiple peek operations don't change queue order
        2. Peek always returns the first task in FIFO order
        3. Dequeue after peek maintains FIFO order
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            # Add indices to track order
            indexed_tasks = [{"index": i, "data": task} for i, task in enumerate(tasks)]
            
            # Enqueue all tasks
            for task in indexed_tasks:
                queue.enqueue(task)
            
            # Perform multiple peeks
            for _ in range(peek_count):
                peeked = queue.peek()
                # Peek should always return first task
                assert peeked == indexed_tasks[0]
            
            # Dequeue all tasks and verify FIFO order unchanged
            dequeued_tasks = []
            while queue.size() > 0:
                dequeued_tasks.append(queue.dequeue())
            
            # Verify FIFO order maintained
            assert dequeued_tasks == indexed_tasks
    
    @given(
        queue_type=st.sampled_from(["analyst", "quant", "executor"]),
        tasks=st.lists(task_data_strategy, min_size=1, max_size=40)
    )
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_across_queue_types(
        self, 
        queue_type: str, 
        tasks: List[Dict[str, Any]]
    ):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: FIFO ordering is maintained for all queue types 
        (analyst, quant, executor).
        
        **Validates: Requirements 1.5, 1.6**
        
        This test verifies that:
        1. All queue types maintain FIFO ordering
        2. Queue type doesn't affect ordering behavior
        3. Implementation is consistent across queue types
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue(queue_type, queue_dir=temp_dir)
            
            # Add indices to track order
            indexed_tasks = [{"index": i, "data": task} for i, task in enumerate(tasks)]
            
            # Enqueue all tasks
            for task in indexed_tasks:
                queue.enqueue(task)
            
            # Dequeue all tasks
            dequeued_tasks = []
            while queue.size() > 0:
                dequeued_tasks.append(queue.dequeue())
            
            # Verify FIFO order
            assert dequeued_tasks == indexed_tasks
    
    @given(tasks=st.lists(task_data_strategy, min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_persists_across_instances(
        self, 
        tasks: List[Dict[str, Any]]
    ):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: FIFO ordering is preserved when queue is accessed by 
        different TaskQueue instances.
        
        **Validates: Requirements 1.5, 1.7**
        
        This test verifies that:
        1. Queue order persists to file correctly
        2. New instances read queue in correct FIFO order
        3. File-based persistence maintains ordering
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Add indices to track order
            indexed_tasks = [{"index": i, "data": task} for i, task in enumerate(tasks)]
            
            # First instance: enqueue tasks
            queue1 = TaskQueue("analyst", queue_dir=temp_dir)
            for task in indexed_tasks:
                queue1.enqueue(task)
            
            # Second instance: dequeue tasks
            queue2 = TaskQueue("analyst", queue_dir=temp_dir)
            dequeued_tasks = []
            while queue2.size() > 0:
                dequeued_tasks.append(queue2.dequeue())
            
            # Verify FIFO order maintained across instances
            assert dequeued_tasks == indexed_tasks
    
    @given(
        tasks=st.lists(task_data_strategy, min_size=5, max_size=30),
        dequeue_indices=st.lists(st.integers(min_value=0, max_value=4), min_size=1, max_size=5)
    )
    @settings(max_examples=100, deadline=None)
    def test_fifo_ordering_with_partial_dequeues(
        self, 
        tasks: List[Dict[str, Any]], 
        dequeue_indices: List[int]
    ):
        """
        **Feature: quantmindx-unified-backend, Property 2: Task Queue FIFO Ordering**
        
        Property: FIFO ordering is maintained when tasks are dequeued in batches.
        
        **Validates: Requirements 1.5**
        
        This test verifies that:
        1. Partial dequeues maintain FIFO order
        2. Remaining tasks stay in correct order
        3. Multiple partial dequeues preserve overall FIFO ordering
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("quant", queue_dir=temp_dir)
            
            # Add indices to track order
            indexed_tasks = [{"index": i, "data": task} for i, task in enumerate(tasks)]
            
            # Enqueue all tasks
            for task in indexed_tasks:
                queue.enqueue(task)
            
            # Dequeue in batches
            all_dequeued = []
            for batch_size in dequeue_indices:
                batch = []
                for _ in range(min(batch_size, queue.size())):
                    dequeued = queue.dequeue()
                    if dequeued is not None:
                        batch.append(dequeued)
                all_dequeued.extend(batch)
            
            # Dequeue any remaining
            while queue.size() > 0:
                dequeued = queue.dequeue()
                if dequeued is not None:
                    all_dequeued.append(dequeued)
            
            # Verify complete FIFO order
            assert len(all_dequeued) == len(indexed_tasks)
            assert all_dequeued == indexed_tasks


class TestTaskQueueSizeProperty:
    """
    Property-based tests for queue size consistency.
    
    Related to FIFO ordering - size should accurately reflect queue state.
    """
    
    @given(tasks=st.lists(task_data_strategy, min_size=0, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_size_equals_enqueued_count(self, tasks: List[Dict[str, Any]]):
        """
        Property: Queue size equals the number of enqueued tasks.
        
        This test verifies that:
        1. Size accurately reflects number of tasks
        2. Size updates correctly with each enqueue
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("executor", queue_dir=temp_dir)
            
            for i, task in enumerate(tasks):
                queue.enqueue(task)
                assert queue.size() == i + 1
            
            assert queue.size() == len(tasks)
    
    @given(tasks=st.lists(task_data_strategy, min_size=1, max_size=50))
    @settings(max_examples=100, deadline=None)
    def test_size_decreases_with_dequeue(self, tasks: List[Dict[str, Any]]):
        """
        Property: Queue size decreases by 1 with each dequeue.
        
        This test verifies that:
        1. Size decreases correctly with dequeue
        2. Size reaches 0 when all tasks dequeued
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            queue = TaskQueue("analyst", queue_dir=temp_dir)
            
            # Enqueue all tasks
            for task in tasks:
                queue.enqueue(task)
            
            initial_size = len(tasks)
            
            # Dequeue all tasks
            for i in range(len(tasks)):
                expected_size = initial_size - i
                assert queue.size() == expected_size
                queue.dequeue()
            
            assert queue.size() == 0
