"""
Task List Tool with Hooks Integration

Provides task CRUD operations with pre/post hooks for task state changes.
Supports task statuses: pending, in_progress, completed, blocked
"""

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


# Type alias for hook callbacks
HookCallback = Callable[[Any, Any], Any]


@dataclass
class Task:
    """Task data model."""
    id: str
    title: str
    description: str
    status: TaskStatus
    priority: int
    assignee: Optional[str]
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class TaskListTool:
    """Task management tool with hooks integration."""

    def __init__(self, db_path: str = ".quantmind/tasks.db"):
        self.db_path = db_path
        self._pre_hooks: List[HookCallback] = []
        self._post_hooks: List[HookCallback] = []
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the database schema."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, description TEXT DEFAULT '',
                status TEXT DEFAULT 'pending', priority INTEGER DEFAULT 0, assignee TEXT,
                tags TEXT DEFAULT '[]', metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def add_pre_hook(self, hook: HookCallback) -> None:
        """Add a pre-hook that fires before state changes."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: HookCallback) -> None:
        """Add a post-hook that fires after state changes."""
        self._post_hooks.append(hook)

    def remove_pre_hook(self, hook: HookCallback) -> None:
        if hook in self._pre_hooks:
            self._pre_hooks.remove(hook)

    def remove_post_hook(self, hook: HookCallback) -> None:
        if hook in self._post_hooks:
            self._post_hooks.remove(hook)

    def _execute_hooks(self, hooks: List[HookCallback], before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a list of hooks and return the final state."""
        current = after
        for hook in hooks:
            try:
                current = hook(before, current)
            except Exception as e:
                logger.error(f"Hook {hook.__name__} failed: {e}")
        return current

    def _row_to_task(self, row: tuple) -> Task:
        return Task(
            id=row[0], title=row[1], description=row[2], status=TaskStatus(row[3]),
            priority=row[4], assignee=row[5], tags=json.loads(row[6]) if row[6] else [],
            metadata=json.loads(row[7]) if row[7] else {}, created_at=row[8], updated_at=row[9]
        )

    def _task_to_dict(self, task: Task) -> Dict[str, Any]:
        return {
            "id": task.id, "title": task.title, "description": task.description,
            "status": task.status.value, "priority": task.priority, "assignee": task.assignee,
            "tags": task.tags, "metadata": task.metadata, "created_at": task.created_at, "updated_at": task.updated_at
        }

    # =====================================================================
    # CRUD Operations
    # =====================================================================

    def create_task(self, task_id: str, title: str, description: str = "", priority: int = 0,
                   assignee: Optional[str] = None, tags: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new task."""
        now = datetime.utcnow().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT INTO tasks (id, title, description, status, priority, assignee, tags, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, title, description, TaskStatus.PENDING.value, priority, assignee,
                  json.dumps(tags or []), json.dumps(metadata or {}), now, now))
            conn.commit()

            task_dict = {
                "id": task_id, "title": title, "description": description, "status": TaskStatus.PENDING.value,
                "priority": priority, "assignee": assignee, "tags": tags or [], "metadata": metadata or {},
                "created_at": now, "updated_at": now
            }

            after_state = {"operation": "create", "task": task_dict}
            result_state = self._execute_hooks(self._post_hooks, {"operation": "create", "task_id": task_id}, after_state)

            return {"success": True, "task": result_state.get("task", task_dict)}

        except sqlite3.IntegrityError:
            return {"success": False, "error": f"Task {task_id} already exists"}
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get a task by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if row:
                task = self._row_to_task(row)
                return {"success": True, "task": self._task_to_dict(task)}
            return {"success": False, "error": "Task not found"}
        except Exception as e:
            logger.error(f"Failed to get task: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def update_task(self, task_id: str, title: Optional[str] = None, description: Optional[str] = None,
                   status: Optional[TaskStatus] = None, priority: Optional[int] = None,
                   assignee: Optional[str] = None, tags: Optional[List[str]] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Update a task."""
        before_state = {"operation": "update", "task_id": task_id}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return {"success": False, "error": "Task not found"}

            current_task = self._row_to_task(row)
            before_state["old_status"] = current_task.status.value

            updates, values = [], []
            if title is not None: updates.append("title = ?"); values.append(title)
            if description is not None: updates.append("description = ?"); values.append(description)
            if status is not None: updates.append("status = ?"); values.append(status.value)
            if priority is not None: updates.append("priority = ?"); values.append(priority)
            if assignee is not None: updates.append("assignee = ?"); values.append(assignee)
            if tags is not None: updates.append("tags = ?"); values.append(json.dumps(tags))
            if metadata is not None: updates.append("metadata = ?"); values.append(json.dumps(metadata))

            if not updates:
                return {"success": False, "error": "No fields to update"}

            now = datetime.utcnow().isoformat()
            updates.append("updated_at = ?")
            values.append(now)
            values.append(task_id)

            cursor.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?", values)
            conn.commit()

            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            updated_task = self._row_to_task(cursor.fetchone())

            after_state = {
                "operation": "update", "task": self._task_to_dict(updated_task),
                "old_status": before_state.get("old_status"), "new_status": updated_task.status.value
            }

            if status is not None and status != current_task.status:
                self._execute_hooks(self._pre_hooks, before_state, after_state)

            result_state = self._execute_hooks(self._post_hooks, before_state, after_state)
            return {"success": True, "task": result_state.get("task", self._task_to_dict(updated_task))}

        except Exception as e:
            logger.error(f"Failed to update task: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def delete_task(self, task_id: str) -> Dict[str, Any]:
        """Delete a task."""
        before_state = {"operation": "delete", "task_id": task_id}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
            row = cursor.fetchone()
            if not row:
                return {"success": False, "error": "Task not found"}

            task_dict = self._task_to_dict(self._row_to_task(row))
            cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
            conn.commit()

            after_state = {"operation": "delete", "task": task_dict}
            self._execute_hooks(self._pre_hooks, before_state, after_state)
            self._execute_hooks(self._post_hooks, before_state, after_state)

            return {"success": True, "message": "Task deleted"}

        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
            return {"success": False, "error": str(e)}
        finally:
            conn.close()

    def list_tasks(self, status: Optional[TaskStatus] = None, assignee: Optional[str] = None,
                   tags: Optional[List[str]] = None, limit: int = 100) -> Dict[str, Any]:
        """List tasks with optional filters."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            query, params = "SELECT * FROM tasks WHERE 1=1", []
            if status: query += " AND status = ?"; params.append(status.value)
            if assignee: query += " AND assignee = ?"; params.append(assignee)
            query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            tasks = [self._task_to_dict(self._row_to_task(row)) for row in cursor.fetchall()]

            if tags:
                tasks = [t for t in tasks if any(tag in t["tags"] for tag in tags)]

            return {"success": True, "tasks": tasks, "total": len(tasks)}

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return {"success": False, "error": str(e), "tasks": []}
        finally:
            conn.close()

    def update_task_status(self, task_id: str, status: TaskStatus) -> Dict[str, Any]:
        """Update task status with hooks."""
        return self.update_task(task_id, status=status)


# =====================================================================
# Hook Examples
# =====================================================================

def task_notification_hook(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Example post-hook: Send notification on status change."""
    if "old_status" in before and "new_status" in after:
        old, new = before["old_status"], after["new_status"]
        if old != new:
            logger.info(f"Status change: {before.get('task_id')} {old} -> {new}")
            after["notification_sent"] = True
    return after


def task_audit_hook(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Example post-hook: Audit log for task changes."""
    logger.info(f"Audit: {after.get('operation')} on task {before.get('task_id')}")
    after["audited"] = True
    return after


def task_validation_hook(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Example pre-hook: Validate task before update."""
    if after.get("operation") == "update":
        task = after.get("task", {})
        if task.get("status") == TaskStatus.COMPLETED.value and not task.get("title"):
            raise ValueError("Cannot complete task without title")
    return after


# =====================================================================
# Tool Registry
# =====================================================================

_task_list_tool: Optional[TaskListTool] = None


def get_task_list_tool(db_path: str = ".quantmind/tasks.db") -> TaskListTool:
    """Get or create the global TaskListTool instance."""
    global _task_list_tool
    if _task_list_tool is None:
        _task_list_tool = TaskListTool(db_path)
    return _task_list_tool


TASK_LIST_TOOLS = {
    "create_task": {
        "function": lambda **kwargs: get_task_list_tool().create_task(**kwargs),
        "description": "Create a new task",
        "parameters": {
            "task_id": {"type": "string", "required": True},
            "title": {"type": "string", "required": True},
            "description": {"type": "string", "required": False, "default": ""},
            "priority": {"type": "integer", "required": False, "default": 0},
            "assignee": {"type": "string", "required": False},
            "tags": {"type": "array", "required": False},
            "metadata": {"type": "object", "required": False}
        }
    },
    "get_task": {
        "function": lambda task_id: get_task_list_tool().get_task(task_id),
        "description": "Get a task by ID",
        "parameters": {"task_id": {"type": "string", "required": True}}
    },
    "update_task": {
        "function": lambda **kwargs: get_task_list_tool().update_task(**kwargs),
        "description": "Update a task",
        "parameters": {
            "task_id": {"type": "string", "required": True},
            "title": {"type": "string", "required": False},
            "description": {"type": "string", "required": False},
            "status": {"type": "string", "required": False},
            "priority": {"type": "integer", "required": False},
            "assignee": {"type": "string", "required": False},
            "tags": {"type": "array", "required": False},
            "metadata": {"type": "object", "required": False}
        }
    },
    "delete_task": {
        "function": lambda task_id: get_task_list_tool().delete_task(task_id),
        "description": "Delete a task",
        "parameters": {"task_id": {"type": "string", "required": True}}
    },
    "list_tasks": {
        "function": lambda **kwargs: get_task_list_tool().list_tasks(**kwargs),
        "description": "List tasks with filters",
        "parameters": {
            "status": {"type": "string", "required": False},
            "assignee": {"type": "string", "required": False},
            "tags": {"type": "array", "required": False},
            "limit": {"type": "integer", "required": False, "default": 100}
        }
    },
    "update_task_status": {
        "function": lambda task_id, status: get_task_list_tool().update_task_status(task_id, TaskStatus(status)),
        "description": "Update task status",
        "parameters": {
            "task_id": {"type": "string", "required": True},
            "status": {"type": "string", "required": True}
        }
    }
}


def get_task_list_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Get a task list tool by name."""
    return TASK_LIST_TOOLS.get(name)


def list_task_list_tools() -> List[str]:
    """List all available task list tools."""
    return list(TASK_LIST_TOOLS.keys())
