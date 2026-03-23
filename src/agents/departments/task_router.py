"""
Task Router Module

Provides concurrent task routing and scheduling for the Trading Floor.
Implements Redis Streams-based task queues with priority support.

Key Features:
- Concurrent dispatch to multiple departments (≥5 simultaneous tasks)
- Priority-based scheduling (HIGH/MEDIUM/LOW)
- Session-based task isolation
- Task status tracking (running/queued/completed/failed)
- Result aggregation with parallelism overhead calculation

Architecture:
- Uses Redis Streams for task persistence (extends Story 7.6 patterns)
- Stream keys: task:dept:{dept}:{session_id}:queue
- Consumer groups: task:dept:{dept}:group
"""

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from concurrent.futures import ThreadPoolExecutor

import redis
from redis import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from src.agents.departments.types import Department
from src.agents.departments.department_mail import Priority

logger = logging.getLogger(__name__)

# Redis connection settings (shared with department mail)
DEFAULT_REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
DEFAULT_REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
DEFAULT_REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
DEFAULT_REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", None)


# ============================================================================
# Enums
# ============================================================================

class TaskPriority(str, Enum):
    """Task priority levels for scheduling."""
    HIGH = "high"      # Immediate tasks (kill switch, emergency risk)
    MEDIUM = "medium" # Standard department work
    LOW = "low"       # Background analysis, reflections


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Task:
    """
    Task model for concurrent dispatch.

    Attributes:
        task_id: Unique task identifier
        task_type: Type of task (e.g., "research", "development")
        department: Target department
        priority: Task priority (HIGH/MEDIUM/LOW)
        payload: Task payload (dict)
        depends_on: List of task IDs this task depends on
        session_id: Session namespace for isolation
        status: Current execution status
        created_at: Task creation timestamp
        started_at: When task started execution
        completed_at: When task completed
        result: Task result data
        error: Error message if failed
    """
    task_id: str
    task_type: str
    department: str
    priority: TaskPriority
    payload: Dict[str, Any]
    depends_on: List[str] = field(default_factory=list)
    session_id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "department": self.department,
            "priority": self.priority.value,
            "payload": self.payload,
            "depends_on": self.depends_on,
            "session_id": self.session_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            department=data["department"],
            priority=TaskPriority(data["priority"]),
            payload=data.get("payload", {}),
            depends_on=data.get("depends_on", []),
            session_id=data.get("session_id", ""),
            status=TaskStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else datetime.now(timezone.utc),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            error=data.get("error"),
        )


@dataclass
class TaskResult:
    """
    Aggregated result from multiple concurrent tasks.

    Attributes:
        session_id: Session ID for the batch
        tasks: List of completed tasks
        total_wall_clock_time: Total time from start to all complete
        max_individual_time: Longest individual task time
        parallelism_overhead: Percentage overhead (should be ≤20%)
        all_completed: Whether all tasks completed successfully
    """
    session_id: str
    tasks: List[Task]
    total_wall_clock_time: float
    max_individual_time: float
    parallelism_overhead: float
    all_completed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "tasks": [t.to_dict() for t in self.tasks],
            "total_wall_clock_time": self.total_wall_clock_time,
            "max_individual_time": self.max_individual_time,
            "parallelism_overhead": self.parallelism_overhead,
            "all_completed": self.all_completed,
        }


# ============================================================================
# Task Router
# ============================================================================

class TaskRouterError(Exception):
    """Base exception for TaskRouter errors."""
    pass


class RedisConnectionFailed(TaskRouterError):
    """Raised when Redis connection fails."""
    pass


class TaskDependencyError(TaskRouterError):
    """Raised when task dependencies are not met."""
    pass


class TaskRouter:
    """
    Concurrent task routing engine using Redis Streams.

    Handles concurrent dispatch to multiple departments with:
    - Priority-based scheduling (HIGH preempts MEDIUM)
    - Session-based task isolation
    - Task status tracking and result aggregation

    Key Patterns (extends Story 7.6):
    - Stream keys: task:dept:{dept}:{session_id}:queue
    - Consumer groups: task:dept:{dept}:group
    """

    # Maximum queue depth per department
    MAX_QUEUE_DEPTH = 100

    # Default task timeout (seconds)
    DEFAULT_TIMEOUT = 300

    # Message retention (7 days)
    MESSAGE_RETENTION = 7 * 24 * 60 * 60

    def __init__(
        self,
        host: str = DEFAULT_REDIS_HOST,
        port: int = DEFAULT_REDIS_PORT,
        db: int = DEFAULT_REDIS_DB,
        password: Optional[str] = DEFAULT_REDIS_PASSWORD,
        consumer_name: Optional[str] = None,
    ):
        """
        Initialize the Task Router.

        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            consumer_name: Unique consumer name
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.consumer_name = consumer_name or f"task-router-{uuid.uuid4().hex[:8]}"

        # Connection pool
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[Redis] = None

        # In-memory task tracking (mirrors Redis state)
        self._active_tasks: Dict[str, Task] = {}
        self._task_results: Dict[str, Any] = {}

        # Task execution handlers (set by FloorManager)
        self._task_handlers: Dict[str, Callable[[Task], Awaitable[Dict[str, Any]]]] = {}

    def _create_connection_pool(self) -> redis.ConnectionPool:
        """Create Redis connection pool."""
        return redis.ConnectionPool(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=20,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            decode_responses=True,
        )

    def _get_client(self) -> Redis:
        """Get Redis client from pool."""
        if self._client is None:
            try:
                if self._pool is None:
                    self._pool = self._create_connection_pool()
                self._client = Redis(connection_pool=self._pool)
                self._client.ping()
                logger.debug(f"Task router connected to {self.host}:{self.port}")
            except RedisConnectionError as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise RedisConnectionFailed(f"Cannot connect to Redis at {self.host}:{self.port}") from e
        return self._client

    def close(self):
        """Close Redis connections."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis client: {e}")
            finally:
                self._client = None

        if self._pool:
            try:
                self._pool.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting pool: {e}")
            finally:
                self._pool = None

        logger.info("Task router closed")

    # =========================================================================
    # Stream Key Management
    # =========================================================================

    def _get_task_stream_key(self, dept: str, session_id: str) -> str:
        """
        Get task stream key for department and session.

        Key format: task:dept:{dept}:{session_id}:queue
        """
        return f"task:dept:{dept.lower()}:{session_id}:queue"

    def _get_task_group(self, dept: str) -> str:
        """Get consumer group name for department."""
        return f"task:dept:{dept.lower()}:group"

    def _get_task_status_key(self, task_id: str) -> str:
        """Get task status key for quick lookup."""
        return f"task:status:{task_id}"

    def _get_session_tasks_key(self, session_id: str) -> str:
        """Get session tasks set key."""
        return f"session:{session_id}:tasks"

    # =========================================================================
    # Task Dispatch
    # =========================================================================

    def dispatch_task(
        self,
        task_type: str,
        department: Department,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM,
        depends_on: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Task:
        """
        Dispatch a task to a department queue.

        Args:
            task_type: Type of task (e.g., "research", "development")
            department: Target department
            payload: Task payload data
            priority: Task priority (HIGH/MEDIUM/LOW)
            depends_on: List of task IDs this task depends on
            session_id: Session ID for isolation (auto-generated if not provided)

        Returns:
            Created Task object
        """
        # Generate session ID if not provided
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create task
        task = Task(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            task_type=task_type,
            department=department.value,
            priority=priority,
            payload=payload,
            depends_on=depends_on or [],
            session_id=session_id,
            status=TaskStatus.PENDING,
        )

        # Check dependencies (AC-5)
        if task.depends_on:
            self._wait_for_dependencies(task.depends_on, timeout=self.DEFAULT_TIMEOUT)

        # Update status to queued
        task.status = TaskStatus.QUEUED

        # Add to Redis stream
        try:
            client = self._get_client()
            stream_key = self._get_task_stream_key(department.value, session_id)

            # Build stream payload
            payload_dict = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "department": task.department,
                "priority": task.priority.value,
                "payload": json.dumps(task.payload),
                "depends_on": json.dumps(task.depends_on),
                "session_id": session_id,
                "created_at": task.created_at.isoformat(),
            }

            # Add to stream
            client.xadd(stream_key, payload_dict, maxlen=self.MESSAGE_RETENTION, approximate=True)

            # Update task status in Redis
            status_key = self._get_task_status_key(task.task_id)
            client.setex(status_key, self.MESSAGE_RETENTION, json.dumps(task.to_dict()))

            # Add to session tasks set
            session_key = self._get_session_tasks_key(session_id)
            client.sadd(session_key, task.task_id)

            logger.info(f"Dispatched task {task.task_id} to {department.value} (priority: {priority.value})")

        except RedisError as e:
            logger.error(f"Failed to dispatch task to Redis: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            raise TaskRouterError(f"Failed to dispatch task: {e}") from e

        # Track in memory
        self._active_tasks[task.task_id] = task

        return task

    def dispatch_concurrent(
        self,
        tasks: List[Dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> List[Task]:
        """
        Dispatch multiple tasks concurrently (AC-1).

        Args:
            tasks: List of task definitions with keys:
                - task_type: Type of task
                - department: Target department (Department enum or string)
                - payload: Task payload
                - priority: Optional priority
            session_id: Session ID for isolation

        Returns:
            List of created Task objects
        """
        # Generate session ID
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        dispatched_tasks = []

        for task_def in tasks:
            # Parse department
            if isinstance(task_def["department"], Department):
                dept = task_def["department"]
            else:
                dept = Department(task_def["department"].lower())

            task = self.dispatch_task(
                task_type=task_def["task_type"],
                department=dept,
                payload=task_def.get("payload", {}),
                priority=task_def.get("priority", TaskPriority.MEDIUM),
                depends_on=task_def.get("depends_on"),
                session_id=session_id,
            )
            dispatched_tasks.append(task)

        logger.info(f"Dispatched {len(dispatched_tasks)} concurrent tasks for session {session_id}")
        return dispatched_tasks

    # =========================================================================
    # Priority Handling
    # =========================================================================

    def _wait_for_dependencies(self, depends_on: List[str], timeout: int) -> None:
        """
        Wait for dependent tasks to complete (AC-5).

        Args:
            depends_on: List of task IDs to wait for
            timeout: Timeout in seconds

        Raises:
            TaskDependencyError: If timeout exceeded
        """
        start_time = datetime.now(timezone.utc)

        for dep_id in depends_on:
            while True:
                # Check if task completed
                status_key = self._get_task_status_key(dep_id)
                try:
                    client = self._get_client()
                    status_data = client.get(status_key)

                    if status_data:
                        task_dict = json.loads(status_data)
                        task_status = task_dict.get("status")

                        if task_status == TaskStatus.COMPLETED.value:
                            break
                        elif task_status == TaskStatus.FAILED.value:
                            raise TaskDependencyError(f"Dependency {dep_id} failed")

                    # Check timeout
                    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
                    if elapsed > timeout:
                        raise TaskDependencyError(f"Timeout waiting for dependency {dep_id}")

                    time.sleep(0.1)  # Brief wait (sync method, not coroutine)

                except RedisError as e:
                    logger.error(f"Failed to check dependency status: {e}")
                    raise TaskDependencyError(f"Failed to check dependency: {e}")

    def preempt_medium_task(self, department: Department, session_id: str) -> Optional[Task]:
        """
        Preempt a MEDIUM priority task for HIGH priority task (AC-3).

        Finds a running MEDIUM task and preempts it (moves to queue head).

        Args:
            department: Department to preempt from
            session_id: Session ID

        Returns:
            Preempted Task or None if no MEDIUM tasks running
        """
        # Find running MEDIUM task
        for task_id, task in self._active_tasks.items():
            if (task.department == department.value and
                task.session_id == session_id and
                task.status == TaskStatus.RUNNING and
                task.priority == TaskPriority.MEDIUM):
                # Preempt this task
                task.status = TaskStatus.PREEMPTED
                logger.info(f"Preempted task {task_id} for HIGH priority task")

                # Update in Redis
                try:
                    client = self._get_client()
                    status_key = self._get_task_status_key(task_id)
                    client.setex(status_key, self.MESSAGE_RETENTION, json.dumps(task.to_dict()))
                except RedisError as e:
                    logger.warning(f"Failed to update preempted task status: {e}")

                return task

        return None

    # =========================================================================
    # Task Execution
    # =========================================================================

    async def execute_task(
        self,
        task: Task,
        handler: Callable[[Task], Awaitable[Dict[str, Any]]],
    ) -> Task:
        """
        Execute a task with the provided handler.

        Args:
            task: Task to execute
            handler: Async handler function

        Returns:
            Completed Task
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now(timezone.utc)

        try:
            # Execute handler
            result = await handler(task)

            # Mark completed
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.result = result

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            # Check if retryable
            error_str = str(e).lower()
            non_retryable = any(err in error_str for err in ["timeout", "invalid", "auth", "permission"])

            if non_retryable:
                # Non-retryable failure (AC-4)
                task.status = TaskStatus.FAILED
                task.error = f"Non-retryable error: {str(e)}"
                logger.error(f"Task {task.task_id} failed (non-retryable): {e}")
            else:
                # Retryable - keep as pending for retry
                task.status = TaskStatus.PENDING
                task.error = f"Retryable error: {str(e)}"
                logger.warning(f"Task {task.task_id} failed (retryable): {e}")

            task.completed_at = datetime.now(timezone.utc)

        # Update status in Redis
        try:
            client = self._get_client()
            status_key = self._get_task_status_key(task.task_id)
            client.setex(status_key, self.MESSAGE_RETENTION, json.dumps(task.to_dict()))
        except RedisError as e:
            logger.warning(f"Failed to update task status in Redis: {e}")

        return task

    async def execute_concurrent(
        self,
        tasks: List[Task],
        handlers: Dict[str, Callable[[Task], Awaitable[Dict[str, Any]]]],
        max_concurrent: int = 5,
    ) -> List[Task]:
        """
        Execute multiple tasks concurrently (AC-1, AC-2).

        Args:
            tasks: List of tasks to execute
            handlers: Dict mapping task_type to handler functions
            max_concurrent: Maximum concurrent executions

        Returns:
            List of completed tasks
        """
        start_time = datetime.now(timezone.utc)

        # Execute tasks concurrently
        semaphore = asyncio.Semaphore(max_concurrent)

        async def execute_with_semaphore(task: Task) -> Task:
            async with semaphore:
                handler = handlers.get(task.task_type)
                if not handler:
                    task.status = TaskStatus.FAILED
                    task.error = f"No handler for task type: {task.task_type}"
                    return task

                return await self.execute_task(task, handler)

        # Run all tasks concurrently
        results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=False
        )

        end_time = datetime.now(timezone.utc)
        total_time = (end_time - start_time).total_seconds()

        # Calculate parallelism overhead
        max_individual = 0.0
        for task in results:
            if task.started_at and task.completed_at:
                task_time = (task.completed_at - task.started_at).total_seconds()
                max_individual = max(max_individual, task_time)

        overhead = ((total_time - max_individual) / max_individual * 100) if max_individual > 0 else 0

        logger.info(
            f"Concurrent execution complete: {len(results)} tasks, "
            f"total_time={total_time:.2f}s, max_individual={max_individual:.2f}s, "
            f"overhead={overhead:.1f}%"
        )

        return results

    # =========================================================================
    # Result Aggregation
    # =========================================================================

    async def aggregate_results(
        self,
        tasks: List[Task],
    ) -> TaskResult:
        """
        Aggregate results from multiple concurrent tasks (AC-2).

        Args:
            tasks: List of completed tasks

        Returns:
            Aggregated TaskResult
        """
        if not tasks:
            return TaskResult(
                session_id="",
                tasks=[],
                total_wall_clock_time=0,
                max_individual_time=0,
                parallelism_overhead=0,
                all_completed=True
            )

        # Calculate times
        start_times = [t.started_at for t in tasks if t.started_at]
        end_times = [t.completed_at for t in tasks if t.completed_at]

        if not start_times or not end_times:
            return TaskResult(
                session_id=tasks[0].session_id if tasks else "",
                tasks=tasks,
                total_wall_clock_time=0,
                max_individual_time=0,
                parallelism_overhead=0,
                all_completed=False
            )

        earliest_start = min(start_times)
        latest_end = max(end_times)

        total_time = (latest_end - earliest_start).total_seconds()

        # Find max individual time
        max_individual = 0.0
        for task in tasks:
            if task.started_at and task.completed_at:
                task_time = (task.completed_at - task.started_at).total_seconds()
                max_individual = max(max_individual, task_time)

        # Calculate overhead
        overhead = ((total_time - max_individual) / max_individual * 100) if max_individual > 0 else 0

        # Check if all completed
        all_completed = all(t.status == TaskStatus.COMPLETED for t in tasks)

        return TaskResult(
            session_id=tasks[0].session_id if tasks else "",
            tasks=tasks,
            total_wall_clock_time=total_time,
            max_individual_time=max_individual,
            parallelism_overhead=overhead,
            all_completed=all_completed
        )

    # =========================================================================
    # Status Tracking
    # =========================================================================

    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get task status by ID."""
        if task_id in self._active_tasks:
            return self._active_tasks[task_id].status

        # Try Redis
        try:
            client = self._get_client()
            status_key = self._get_task_status_key(task_id)
            status_data = client.get(status_key)

            if status_data:
                task_dict = json.loads(status_data)
                return TaskStatus(task_dict.get("status", "pending"))
        except RedisError as e:
            logger.warning(f"Failed to get task status from Redis: {e}")

        return None

    def get_department_status(self, department: Department, session_id: str) -> Dict[str, str]:
        """
        Get status of all tasks for a department (AC-1).

        Returns dict mapping task_id to status for Agent Panel display.
        """
        status: Dict[str, str] = {}

        try:
            client = self._get_client()
            stream_key = self._get_task_stream_key(department.value, session_id)

            # Get all tasks in stream
            tasks = client.xrange(stream_key, count=100)

            for task_id, task_data in tasks:
                task_id_val = task_data.get("task_id", task_id)
                priority = task_data.get("priority", "medium")
                status[task_id_val] = f"{priority}"

            # Overlay running tasks
            for task_id, task in self._active_tasks.items():
                if task.department == department.value and task.session_id == session_id:
                    status[task_id] = f"{task.status.value}"

        except RedisError as e:
            logger.warning(f"Failed to get department status: {e}")

        return status

    def get_all_department_status(self, session_id: str) -> Dict[str, Dict[str, str]]:
        """
        Get status of all departments for Agent Panel (AC-1).

        Returns:
            Dict mapping department name to status dict
        """
        result = {}

        for dept in Department:
            result[dept.value] = self.get_department_status(dept, session_id)

        return result


# ============================================================================
# Factory
# ============================================================================

_task_router: Optional[TaskRouter] = None


def get_task_router(
    host: str = DEFAULT_REDIS_HOST,
    port: int = DEFAULT_REDIS_PORT,
    db: int = DEFAULT_REDIS_DB,
    password: Optional[str] = DEFAULT_REDIS_PASSWORD,
) -> TaskRouter:
    """
    Get or create the singleton TaskRouter instance.

    Args:
        host: Redis server host
        port: Redis server port
        db: Redis database number
        password: Optional Redis password

    Returns:
        TaskRouter singleton
    """
    global _task_router
    if _task_router is None:
        _task_router = TaskRouter(
            host=host,
            port=port,
            db=db,
            password=password,
        )
    return _task_router


def reset_task_router() -> None:
    """Reset the TaskRouter singleton. Useful for testing."""
    global _task_router
    if _task_router is not None:
        _task_router.close()
        _task_router = None