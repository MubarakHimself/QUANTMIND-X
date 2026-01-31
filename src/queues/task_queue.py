"""
TaskQueue implementation for Agent Workspaces - Task Group 1

Implements FIFO queue operations with file locking for concurrent access.
Provides simple interface for task queue management across agents.
"""

import json
import fcntl
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TaskQueue:
    """
    FIFO task queue with file-based persistence and locking.
    
    Provides thread-safe and process-safe queue operations through file locking.
    Each queue instance manages a single JSON file containing task data.
    
    Queue file format:
    [
        {"task_data": {...}, "timestamp": 1234567890.123},
        {"task_data": {...}, "timestamp": 1234567890.456},
        ...
    ]
    
    Attributes:
        queue_file: Path to the queue JSON file
        lock_file: Path to the lock file for concurrent access control
    """
    
    def __init__(self, queue_type: str, queue_dir: str = "data/queues"):
        """
        Initialize TaskQueue for a specific agent type.
        
        Args:
            queue_type: Type of queue (analyst, quant, executor)
            queue_dir: Directory containing queue files (default: data/queues)
        
        Raises:
            ValueError: If queue_type is invalid
        """
        valid_types = ["analyst", "quant", "executor"]
        if queue_type not in valid_types:
            raise ValueError(
                f"Invalid queue_type: {queue_type}. "
                f"Must be one of: {', '.join(valid_types)}"
            )
        
        self.queue_type = queue_type
        self.queue_dir = Path(queue_dir)
        self.queue_file = self.queue_dir / f"{queue_type}_tasks.json"
        self.lock_file = self.queue_dir / f"{queue_type}_tasks.json.lock"
        
        # Ensure directory exists
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize queue file if it doesn't exist
        if not self.queue_file.exists():
            self._write_queue([])
    
    def _acquire_lock(self, lock_fd) -> None:
        """
        Acquire exclusive lock on the lock file.
        
        Args:
            lock_fd: File descriptor for the lock file
        """
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    
    def _release_lock(self, lock_fd) -> None:
        """
        Release lock on the lock file.
        
        Args:
            lock_fd: File descriptor for the lock file
        """
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
    
    def _read_queue(self) -> list:
        """
        Read queue data from file.
        
        Returns:
            List of task dictionaries
        """
        try:
            with open(self.queue_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If file is corrupted or missing, return empty list
            return []
    
    def _write_queue(self, tasks: list) -> None:
        """
        Write queue data to file.
        
        Args:
            tasks: List of task dictionaries to write
        """
        with open(self.queue_file, 'w') as f:
            json.dump(tasks, f, indent=2)
    
    def enqueue(self, task: Dict[str, Any]) -> None:
        """
        Add task to the end of the queue (FIFO).
        
        Thread-safe and process-safe through file locking.
        
        Args:
            task: Task data dictionary to enqueue
        """
        # Open lock file
        with open(self.lock_file, 'w') as lock_fd:
            try:
                # Acquire exclusive lock
                self._acquire_lock(lock_fd)
                
                # Read current queue
                tasks = self._read_queue()
                
                # Add new task with timestamp
                task_entry = {
                    "task_data": task,
                    "timestamp": time.time()
                }
                tasks.append(task_entry)
                
                # Write updated queue
                self._write_queue(tasks)
                
            finally:
                # Always release lock
                self._release_lock(lock_fd)
    
    def dequeue(self) -> Optional[Dict[str, Any]]:
        """
        Remove and return the first task from the queue (FIFO).
        
        Thread-safe and process-safe through file locking.
        
        Returns:
            Task data dictionary or None if queue is empty
        """
        # Open lock file
        with open(self.lock_file, 'w') as lock_fd:
            try:
                # Acquire exclusive lock
                self._acquire_lock(lock_fd)
                
                # Read current queue
                tasks = self._read_queue()
                
                # Return None if empty
                if not tasks:
                    return None
                
                # Remove first task (FIFO)
                task_entry = tasks.pop(0)
                
                # Write updated queue
                self._write_queue(tasks)
                
                # Return task data (without timestamp)
                return task_entry["task_data"]
                
            finally:
                # Always release lock
                self._release_lock(lock_fd)
    
    def peek(self) -> Optional[Dict[str, Any]]:
        """
        View the first task without removing it from the queue.
        
        Thread-safe and process-safe through file locking.
        
        Returns:
            Task data dictionary or None if queue is empty
        """
        # Open lock file
        with open(self.lock_file, 'w') as lock_fd:
            try:
                # Acquire exclusive lock
                self._acquire_lock(lock_fd)
                
                # Read current queue
                tasks = self._read_queue()
                
                # Return None if empty
                if not tasks:
                    return None
                
                # Return first task data (without removing)
                return tasks[0]["task_data"]
                
            finally:
                # Always release lock
                self._release_lock(lock_fd)
    
    def size(self) -> int:
        """
        Get the current number of tasks in the queue.
        
        Thread-safe and process-safe through file locking.
        
        Returns:
            Number of tasks in the queue
        """
        # Open lock file
        with open(self.lock_file, 'w') as lock_fd:
            try:
                # Acquire exclusive lock
                self._acquire_lock(lock_fd)
                
                # Read current queue
                tasks = self._read_queue()
                
                # Return count
                return len(tasks)
                
            finally:
                # Always release lock
                self._release_lock(lock_fd)
