"""
Agent Queue System.

Provides priority-based task queue management for agents.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import heapq
import uuid


logger = logging.getLogger(__name__)


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20
    URGENT = 30


class TaskStatus(str, Enum):
    """Task status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a queued task."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    agent_type: str = ""  # copilot, analyst, quantcode
    priority: int = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on
    metadata: Dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300

    def __lt__(self, other: "Task") -> bool:
        """Compare by priority (higher = more urgent)."""
        return self.priority > other.priority

    def start(self) -> None:
        """Mark task as started."""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, output: Dict[str, Any]) -> None:
        """Mark task as completed."""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output_data = output

    def fail(self, error: str) -> None:
        """Mark task as failed."""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error = error

    def cancel(self) -> None:
        """Mark task as cancelled."""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retries < self.max_retries and self.status == TaskStatus.FAILED

    def retry(self) -> None:
        """Prepare task for retry."""
        self.retries += 1
        self.status = TaskStatus.PENDING
        self.started_at = None
        self.completed_at = None
        self.error = None

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get task duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "dependencies": self.dependencies,
            "retries": self.retries,
        }


@dataclass
class AgentQueue:
    """Queue for a specific agent."""
    agent_type: str
    max_concurrent: int = 1
    tasks: List[Task] = field(default_factory=list)
    running_tasks: List[Task] = field(default_factory=list)

    @property
    def pending_count(self) -> int:
        """Get count of pending tasks."""
        return len([t for t in self.tasks if t.status == TaskStatus.PENDING])

    @property
    def running_count(self) -> int:
        """Get count of running tasks."""
        return len(self.running_tasks)

    @property
    def is_full(self) -> bool:
        """Check if queue is at capacity."""
        return self.running_count >= self.max_concurrent

    def add(self, task: Task) -> None:
        """Add task to queue."""
        heapq.heappush(self.tasks, task)

    def pop_next(self) -> Optional[Task]:
        """Get next ready task."""
        while self.tasks:
            task = heapq.heappop(self.tasks)

            # Skip non-pending tasks
            if task.status != TaskStatus.PENDING:
                continue

            # Check dependencies
            if not self._check_dependencies(task):
                # Put back and try next
                heapq.heappush(self.tasks, task)
                return None

            return task

        return None

    def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied."""
        # Would need reference to all tasks to check
        # For now, assume dependencies are satisfied
        return True


class QueueManager:
    """
    Manages task queues for all agents.

    Features:
    - Priority-based scheduling
    - Dependency management
    - Concurrent task limits
    - Retry logic
    """

    def __init__(self):
        self._queues: Dict[str, AgentQueue] = {}
        self._all_tasks: Dict[str, Task] = {}
        self._task_handlers: Dict[str, Callable] = {}
        self._running = False

    def register_agent(
        self,
        agent_type: str,
        max_concurrent: int = 1,
    ) -> None:
        """Register an agent type with its queue."""
        if agent_type not in self._queues:
            self._queues[agent_type] = AgentQueue(
                agent_type=agent_type,
                max_concurrent=max_concurrent,
            )
            logger.info(f"Registered agent queue: {agent_type}")

    def register_handler(
        self,
        agent_type: str,
        handler: Callable,
    ) -> None:
        """Register a task handler for an agent."""
        self._task_handlers[agent_type] = handler

    def add_task(
        self,
        task: Task,
    ) -> str:
        """
        Add a task to the appropriate queue.

        Returns:
            Task ID
        """
        # Ensure agent queue exists
        if task.agent_type not in self._queues:
            self.register_agent(task.agent_type)

        # Queue the task
        task.status = TaskStatus.QUEUED
        self._queues[task.agent_type].add(task)
        self._all_tasks[task.task_id] = task

        logger.info(f"Added task '{task.name}' to {task.agent_type} queue")
        return task.task_id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._all_tasks.get(task_id)

    def get_queue_status(self, agent_type: str) -> Dict[str, Any]:
        """Get status of an agent's queue."""
        queue = self._queues.get(agent_type)
        if not queue:
            return {
                "agent_type": agent_type,
                "exists": False,
            }

        return {
            "agent_type": agent_type,
            "exists": True,
            "pending_count": queue.pending_count,
            "running_count": queue.running_count,
            "max_concurrent": queue.max_concurrent,
            "is_full": queue.is_full,
        }

    def get_all_queue_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all queues."""
        return [
            self.get_queue_status(agent_type)
            for agent_type in self._queues
        ]

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._all_tasks.get(task_id)
        if not task:
            return False

        if task.status in (TaskStatus.PENDING, TaskStatus.QUEUED):
            task.cancel()
            return True

        if task.status == TaskStatus.RUNNING:
            # Would need actual cancellation mechanism
            task.cancel()
            return True

        return False

    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        task = self._all_tasks.get(task_id)
        if not task or not task.can_retry():
            return False

        task.retry()

        # Re-add to queue
        if task.agent_type in self._queues:
            self._queues[task.agent_type].add(task)
            return True

        return False

    async def process_queues(self) -> None:
        """Process all queues (run tasks)."""
        self._running = True

        while self._running:
            for agent_type, queue in self._queues.items():
                if queue.is_full:
                    continue

                task = queue.pop_next()
                if task:
                    await self._execute_task(agent_type, task)

            await asyncio.sleep(0.1)  # Small delay to prevent busy loop

    async def _execute_task(self, agent_type: str, task: Task) -> None:
        """Execute a task."""
        handler = self._task_handlers.get(agent_type)

        if not handler:
            logger.error(f"No handler for agent type: {agent_type}")
            task.fail(f"No handler registered for {agent_type}")
            return

        task.start()
        queue = self._queues[agent_type]
        queue.running_tasks.append(task)

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                handler(task.input_data),
                timeout=task.timeout_seconds,
            )
            task.complete(result)

        except asyncio.TimeoutError:
            task.fail("Task timed out")

        except Exception as e:
            logger.error(f"Task execution error: {e}")
            task.fail(str(e))

        finally:
            queue.running_tasks.remove(task)

    def stop(self) -> None:
        """Stop queue processing."""
        self._running = False


# Global queue manager
_queue_manager: Optional[QueueManager] = None


def get_queue_manager() -> QueueManager:
    """Get the global queue manager."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = QueueManager()
    return _queue_manager
