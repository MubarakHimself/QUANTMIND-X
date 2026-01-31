"""
Queue Manager for Agent Task Queues - Task Group 12

Implements FIFO queue operations for analyst, quant, and executor agents.
Provides file-based persistence for async task management.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class QueueFileFormatError(Exception):
    """Raised when queue file format is invalid or corrupted."""
    pass


class QueueManager:
    """
    Manages agent queue operations with file-based persistence.

    Provides FIFO (First-In-First-Out) queue operations for async tasks
    across three agent types: analyst, quant, and executor.

    Queue file format:
    [
        {
            "task_id": "uuid",
            "status": "pending" | "processing" | "complete",
            "payload": {...},
            "created_at": timestamp
        }
    ]
    """

    VALID_AGENTS = ["analyst", "quant", "executor"]
    VALID_STATUSES = ["pending", "processing", "complete"]

    def __init__(self, queue_dir_path: str):
        """
        Initialize QueueManager with queue directory.

        Args:
            queue_dir_path: Path to directory containing queue files

        Raises:
            QueueFileFormatError: If existing queue files are corrupted
        """
        self.queue_dir = Path(queue_dir_path)
        self._ensure_queue_directory()
        self._initialize_queue_files()

    def _ensure_queue_directory(self) -> None:
        """Create queue directory if it doesn't exist."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def _initialize_queue_files(self) -> None:
        """
        Initialize queue files for all valid agents.

        Creates empty array for files that don't exist.
        Validates format for existing files.

        Raises:
            QueueFileFormatError: If existing queue file has invalid format
        """
        for agent in self.VALID_AGENTS:
            queue_file = self.queue_dir / f"{agent}_queue.json"

            if not queue_file.exists():
                # Create new file with empty array
                self._write_queue_file(agent, [])
            else:
                # Validate existing file format
                try:
                    data = self._read_queue_file(agent)
                    if not isinstance(data, list):
                        raise QueueFileFormatError(
                            f"Queue file {queue_file} must contain a JSON array"
                        )
                except json.JSONDecodeError as e:
                    raise QueueFileFormatError(
                        f"Queue file {queue_file} contains invalid JSON: {e}"
                    )

    def _get_queue_file_path(self, agent_name: str) -> Path:
        """Get the file path for an agent's queue."""
        return self.queue_dir / f"{agent_name}_queue.json"

    def _read_queue_file(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Read and parse queue file for an agent.

        Args:
            agent_name: Name of the agent (analyst, quant, executor)

        Returns:
            List of task dictionaries

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
        """
        queue_file = self._get_queue_file_path(agent_name)
        with open(queue_file, "r") as f:
            return json.load(f)

    def _write_queue_file(self, agent_name: str, tasks: List[Dict[str, Any]]) -> None:
        """
        Write tasks to queue file.

        Args:
            agent_name: Name of the agent
            tasks: List of task dictionaries to write
        """
        queue_file = self._get_queue_file_path(agent_name)
        with open(queue_file, "w") as f:
            json.dump(tasks, f, indent=2)

    def _validate_agent_name(self, agent_name: str) -> None:
        """
        Validate that agent name is supported.

        Args:
            agent_name: Name of the agent to validate

        Raises:
            ValueError: If agent name is not valid
        """
        if agent_name not in self.VALID_AGENTS:
            raise ValueError(
                f"Invalid agent name: {agent_name}. "
                f"Must be one of: {', '.join(self.VALID_AGENTS)}"
            )

    def _validate_status(self, status: str) -> None:
        """
        Validate that status is supported.

        Args:
            status: Status to validate

        Raises:
            ValueError: If status is not valid
        """
        if status not in self.VALID_STATUSES:
            raise ValueError(
                f"Invalid status: {status}. "
                f"Must be one of: {', '.join(self.VALID_STATUSES)}"
            )

    def enqueue(self, agent_name: str, payload: Dict[str, Any]) -> str:
        """
        Add a new task to an agent's queue.

        Args:
            agent_name: Name of the agent (analyst, quant, executor)
            payload: Task payload data

        Returns:
            Task ID (UUID string)

        Raises:
            ValueError: If agent_name is invalid
        """
        self._validate_agent_name(agent_name)

        # Create new task
        task_id = str(uuid.uuid4())
        task = {
            "task_id": task_id,
            "status": "pending",
            "payload": payload,
            "created_at": time.time()
        }

        # Read existing tasks, append new task, write back
        tasks = self._read_queue_file(agent_name)
        tasks.append(task)
        self._write_queue_file(agent_name, tasks)

        return task_id

    def dequeue(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Remove and return the next pending task from queue.

        Implements FIFO ordering by returning the first pending task.
        Updates task status to 'processing' before returning.

        Args:
            agent_name: Name of the agent

        Returns:
            Task dictionary or None if queue is empty/no pending tasks

        Raises:
            ValueError: If agent_name is invalid
        """
        self._validate_agent_name(agent_name)

        tasks = self._read_queue_file(agent_name)

        # Find first pending task
        for i, task in enumerate(tasks):
            if task["status"] == "pending":
                # Update status to processing
                task["status"] = "processing"
                tasks[i] = task
                self._write_queue_file(agent_name, tasks)
                return task

        # No pending tasks found
        return None

    def update_status(
        self,
        agent_name: str,
        task_id: str,
        status: str
    ) -> None:
        """
        Update the status of a specific task.

        Args:
            agent_name: Name of the agent
            task_id: UUID of the task to update
            status: New status (pending, processing, complete)

        Raises:
            ValueError: If agent_name or status is invalid
            KeyError: If task_id is not found
        """
        self._validate_agent_name(agent_name)
        self._validate_status(status)

        tasks = self._read_queue_file(agent_name)

        # Find task and update status
        task_found = False
        for i, task in enumerate(tasks):
            if task["task_id"] == task_id:
                task["status"] = status
                tasks[i] = task
                task_found = True
                break

        if not task_found:
            raise KeyError(f"Task not found: {task_id}")

        self._write_queue_file(agent_name, tasks)

    def get_queue_status(self, agent_name: str) -> Dict[str, int]:
        """
        Get status summary for an agent's queue.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary with counts by status: {pending: X, processing: Y, complete: Z}

        Raises:
            ValueError: If agent_name is invalid
        """
        self._validate_agent_name(agent_name)

        tasks = self._read_queue_file(agent_name)

        status_counts = {status: 0 for status in self.VALID_STATUSES}
        for task in tasks:
            status_counts[task["status"]] += 1

        return status_counts

    def clear_complete_tasks(self, agent_name: str) -> int:
        """
        Remove all completed tasks from queue.

        Args:
            agent_name: Name of the agent

        Returns:
            Number of tasks removed

        Raises:
            ValueError: If agent_name is invalid
        """
        self._validate_agent_name(agent_name)

        tasks = self._read_queue_file(agent_name)

        # Filter out completed tasks
        active_tasks = [task for task in tasks if task["status"] != "complete"]
        removed_count = len(tasks) - len(active_tasks)

        if removed_count > 0:
            self._write_queue_file(agent_name, active_tasks)

        return removed_count
