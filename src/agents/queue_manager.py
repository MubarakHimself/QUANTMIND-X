"""
Agent Queue Execution Engine

Manages task queues for each agent type with concurrent execution,
persistence, and real-time WebSocket status updates.
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Awaitable
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a task in the queue."""
    task_id: str
    name: str
    description: str
    agent_type: str
    priority: int
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    retries: int = 0
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AgentQueueConfig:
    """Configuration for an agent's queue."""
    agent_type: str
    max_concurrent: int = 1
    retry_delay_seconds: float = 5.0
    task_timeout_seconds: float = 300.0


class AgentQueueManager:
    """
    Manages task queues for all agent types with concurrent execution.

    Features:
    - Per-agent task queues with configurable concurrency
    - Task prioritization (lower priority value = higher priority)
    - Automatic retry on failure
    - Persistence to disk
    - WebSocket status broadcasting
    """

    def __init__(
        self,
        persistence_path: Optional[str] = None,
        websocket_manager: Optional[Any] = None
    ):
        self._queues: Dict[str, List[Task]] = defaultdict(list)
        self._configs: Dict[str, AgentQueueConfig] = {}
        self._running_tasks: Dict[str, List[Task]] = defaultdict(list)
        self._task_handlers: Dict[str, Callable] = {}
        self._persistence_path = persistence_path or "./data/agent_queues"
        self._websocket_manager = websocket_manager
        self._lock = threading.Lock()
        self._running = False
        self._executor_task: Optional[asyncio.Task] = None

        # Initialize default agent configs
        self._init_default_configs()

        # Load persisted state
        self._load_state()

    def _init_default_configs(self):
        """Initialize default configurations for known agent types."""
        default_configs = [
            AgentQueueConfig("copilot", max_concurrent=2, task_timeout_seconds=300.0),
            AgentQueueConfig("analyst", max_concurrent=1, task_timeout_seconds=600.0),
            AgentQueueConfig("quantcode", max_concurrent=1, task_timeout_seconds=900.0),
        ]

        for config in default_configs:
            self._configs[config.agent_type] = config

    def register_agent(self, config: AgentQueueConfig):
        """Register an agent type with its queue configuration."""
        self._configs[config.agent_type] = config
        if config.agent_type not in self._queues:
            self._queues[config.agent_type] = []

    def register_handler(
        self,
        agent_type: str,
        handler: Callable[[Task], Awaitable[Dict[str, Any]]]
    ):
        """Register an async handler function for an agent type."""
        self._task_handlers[agent_type] = handler
        logger.info(f"Registered handler for agent type: {agent_type}")

    async def start(self):
        """Start the queue executor."""
        if self._running:
            return

        self._running = True
        self._executor_task = asyncio.create_task(self._execute_loop())
        logger.info("Agent queue manager started")

    async def stop(self):
        """Stop the queue executor."""
        self._running = False
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("Agent queue manager stopped")

    def submit_task(
        self,
        agent_type: str,
        name: str,
        description: str = "",
        priority: int = 5,
        input_data: Dict[str, Any] = None
    ) -> Task:
        """
        Submit a new task to an agent's queue.

        Args:
            agent_type: Target agent type
            name: Task name
            description: Task description
            priority: Priority (lower = higher priority)
            input_data: Input data for the task

        Returns:
            The created task
        """
        if agent_type not in self._configs:
            raise ValueError(f"Unknown agent type: {agent_type}")

        task = Task(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            agent_type=agent_type,
            priority=priority,
            status=TaskStatus.QUEUED.value,
            created_at=datetime.now(timezone.utc).isoformat(),
            input_data=input_data or {},
        )

        with self._lock:
            self._queues[agent_type].append(task)

        logger.info(f"Submitted task {task.task_id} to {agent_type} queue")

        # Broadcast update
        asyncio.create_task(self._broadcast_queue_update(agent_type))

        # Persist
        self._save_state()

        return task

    def get_queue_status(self, agent_type: str) -> Dict[str, Any]:
        """Get queue status for an agent type."""
        config = self._configs.get(agent_type, AgentQueueConfig(agent_type))

        with self._lock:
            pending = [t for t in self._queues[agent_type] if t.status in [TaskStatus.PENDING.value, TaskStatus.QUEUED.value]]
            running = [t for t in self._queues[agent_type] if t.status == TaskStatus.RUNNING.value]

        return {
            "agent_type": agent_type,
            "exists": agent_type in self._configs,
            "pending_count": len(pending),
            "running_count": len(running),
            "max_concurrent": config.max_concurrent,
            "is_full": len(running) >= config.max_concurrent,
        }

    def get_all_queue_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get queue status for all agent types."""
        return {agent: self.get_queue_status(agent) for agent in self._configs}

    def get_tasks(
        self,
        agent_type: str,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Task]:
        """Get tasks from an agent's queue."""
        with self._lock:
            tasks = list(self._queues[agent_type])

        # Filter by status
        if status:
            tasks = [t for t in tasks if t.status == status]

        # Sort by priority, then by created_at
        tasks = sorted(tasks, key=lambda t: (t.priority, t.created_at))

        return tasks[:limit]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a specific task by ID."""
        for agent_type in self._queues:
            for task in self._queues[agent_type]:
                if task.task_id == task_id:
                    return task
        return None

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or queued task."""
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status not in [TaskStatus.PENDING.value, TaskStatus.QUEUED.value]:
            return False

        task.status = TaskStatus.CANCELLED.value
        task.completed_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Cancelled task {task_id}")

        asyncio.create_task(self._broadcast_task_update(task))
        self._save_state()

        return True

    def retry_task(self, task_id: str) -> bool:
        """Retry a failed task."""
        task = self.get_task(task_id)
        if not task:
            return False

        if task.status != TaskStatus.FAILED.value:
            return False

        if task.retries >= task.max_retries:
            return False

        task.status = TaskStatus.QUEUED.value
        task.retries += 1
        task.error = None
        task.started_at = None
        task.completed_at = None

        logger.info(f"Retrying task {task_id} (attempt {task.retries})")

        asyncio.create_task(self._broadcast_task_update(task))
        self._save_state()

        return True

    async def _execute_loop(self):
        """Main loop for executing tasks."""
        while self._running:
            try:
                # Check each agent's queue
                for agent_type in list(self._configs.keys()):
                    await self._process_queue(agent_type)

                # Wait before next iteration
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in executor loop: {e}")
                await asyncio.sleep(1.0)

    async def _process_queue(self, agent_type: str):
        """Process pending tasks for an agent type."""
        config = self._configs.get(agent_type)
        if not config:
            return

        # Check if we have capacity
        with self._lock:
            running = [t for t in self._queues[agent_type] if t.status == TaskStatus.RUNNING.value]

        if len(running) >= config.max_concurrent:
            return

        # Get next task
        with self._lock:
            pending = [t for t in self._queues[agent_type] if t.status == TaskStatus.QUEUED.value]
            pending = sorted(pending, key=lambda t: (t.priority, t.created_at))

        if not pending:
            return

        task = pending[0]

        # Execute task in background
        asyncio.create_task(self._execute_task(task, config))

    async def _execute_task(self, task: Task, config: AgentQueueConfig):
        """Execute a single task."""
        task.status = TaskStatus.RUNNING.value
        task.started_at = datetime.now(timezone.utc).isoformat()

        logger.info(f"Executing task {task.task_id} for agent {task.agent_type}")

        await self._broadcast_task_update(task)

        try:
            # Get handler
            handler = self._task_handlers.get(task.agent_type)

            if handler:
                # Execute with timeout
                result = await asyncio.wait_for(
                    handler(task),
                    timeout=config.task_timeout_seconds
                )
                task.output_data = result
            else:
                # Default handler - simulate execution
                await asyncio.sleep(2.0)  # Simulate work
                task.output_data = {"result": f"Task {task.name} completed (no handler registered)"}

            task.status = TaskStatus.COMPLETED.value
            logger.info(f"Task {task.task_id} completed successfully")

        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED.value
            task.error = f"Task timed out after {config.task_timeout_seconds}s"
            logger.error(f"Task {task.task_id} timed out")

        except Exception as e:
            task.status = TaskStatus.FAILED.value
            task.error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")

        finally:
            task.completed_at = datetime.now(timezone.utc).isoformat()

            # Calculate duration
            if task.started_at and task.completed_at:
                start = datetime.fromisoformat(task.started_at.replace('Z', '+00:00'))
                end = datetime.fromisoformat(task.completed_at.replace('Z', '+00:00'))
                task.duration_seconds = (end - start).total_seconds()

            await self._broadcast_task_update(task)
            self._save_state()

    async def _broadcast_task_update(self, task: Task):
        """Broadcast task status update via WebSocket."""
        if not self._websocket_manager:
            return

        try:
            await self._websocket_manager.broadcast({
                "type": "task_update",
                "data": task.to_dict()
            }, topic="agent_queue")
        except Exception as e:
            logger.warning(f"Failed to broadcast task update: {e}")

    async def _broadcast_queue_update(self, agent_type: str):
        """Broadcast queue status update via WebSocket."""
        if not self._websocket_manager:
            return

        try:
            status = self.get_queue_status(agent_type)
            await self._websocket_manager.broadcast({
                "type": "queue_update",
                "data": status
            }, topic="agent_queue")
        except Exception as e:
            logger.warning(f"Failed to broadcast queue update: {e}")

    def _save_state(self):
        """Persist queue state to disk."""
        if not self._persistence_path:
            return

        try:
            os.makedirs(self._persistence_path, exist_ok=True)

            state = {
                "queues": {
                    agent: [t.to_dict() for t in tasks]
                    for agent, tasks in self._queues.items()
                },
                "configs": {
                    agent: asdict(config)
                    for agent, config in self._configs.items()
                }
            }

            path = Path(self._persistence_path) / "queue_state.json"
            with open(path, "w") as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save queue state: {e}")

    def _load_state(self):
        """Load persisted queue state from disk."""
        if not self._persistence_path:
            return

        try:
            path = Path(self._persistence_path) / "queue_state.json"
            if not path.exists():
                return

            with open(path, "r") as f:
                state = json.load(f)

            # Restore queues
            for agent, tasks_data in state.get("queues", {}).items():
                self._queues[agent] = [
                    Task(**t) for t in tasks_data
                ]

            # Restore configs
            for agent, config_data in state.get("configs", {}).items():
                self._configs[agent] = AgentQueueConfig(**config_data)

            logger.info(f"Loaded queue state from {path}")

        except Exception as e:
            logger.error(f"Failed to load queue state: {e}")


# Global queue manager instance
_queue_manager: Optional[AgentQueueManager] = None


def get_queue_manager() -> AgentQueueManager:
    """Get the global queue manager instance."""
    global _queue_manager
    if _queue_manager is None:
        _queue_manager = AgentQueueManager()
    return _queue_manager


async def initialize_queue_manager(
    persistence_path: Optional[str] = None,
    websocket_manager: Optional[Any] = None
) -> AgentQueueManager:
    """
    Initialize the global queue manager.

    Args:
        persistence_path: Path for state persistence
        websocket_manager: WebSocket manager for real-time updates

    Returns:
        Initialized queue manager
    """
    global _queue_manager

    _queue_manager = AgentQueueManager(
        persistence_path=persistence_path,
        websocket_manager=websocket_manager
    )

    # Register default task handlers
    await _register_default_handlers(_queue_manager)

    # Start the executor
    await _queue_manager.start()

    return _queue_manager


async def _register_default_handlers(manager: AgentQueueManager):
    """Register default task handlers for each agent type."""

    async def copilot_handler(task: Task) -> Dict[str, Any]:
        """Handle copilot tasks by invoking the agent."""
        try:
            from src.agents.copilot_v2 import compile_copilot_graph
            from langchain_core.messages import HumanMessage

            graph = compile_copilot_graph(use_tool_node=True)
            message = task.input_data.get("message", task.name)

            result = graph.invoke({
                "messages": [HumanMessage(content=message)]
            })

            return {"result": str(result.get("messages", [])[-1])}
        except ImportError:
            return {"result": f"Copilot handler: {task.name}"}

    async def analyst_handler(task: Task) -> Dict[str, Any]:
        """Handle analyst tasks by invoking the agent."""
        try:
            from src.agents.analyst_v2 import compile_analyst_graph
            from langchain_core.messages import HumanMessage

            graph = compile_analyst_graph(use_tool_node=True)
            message = task.input_data.get("message", task.name)

            result = graph.invoke({
                "messages": [HumanMessage(content=message)]
            })

            return {"result": str(result.get("messages", [])[-1])}
        except ImportError:
            return {"result": f"Analyst handler: {task.name}"}

    async def quantcode_handler(task: Task) -> Dict[str, Any]:
        """Handle quantcode tasks by invoking the agent."""
        try:
            from src.agents.quantcode_v2 import compile_quantcode_graph
            from langchain_core.messages import HumanMessage

            graph = compile_quantcode_graph(use_tool_node=True)
            message = task.input_data.get("message", task.name)

            result = graph.invoke({
                "messages": [HumanMessage(content=message)]
            })

            return {"result": str(result.get("messages", [])[-1])}
        except ImportError:
            return {"result": f"QuantCode handler: {task.name}"}

    manager.register_handler("copilot", copilot_handler)
    manager.register_handler("analyst", analyst_handler)
    manager.register_handler("quantcode", quantcode_handler)
