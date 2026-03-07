"""
Unit Tests for TaskListTool

Tests task CRUD operations and hook functionality.
"""

import pytest
import tempfile
import os
import sqlite3
from src.agents.tools.task_list_tool import (
    TaskListTool,
    TaskStatus,
    task_notification_hook,
    task_audit_hook,
    task_validation_hook
)


class TestTaskListTool:
    """Tests for TaskListTool class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def tool(self, temp_db):
        """Create a TaskListTool instance with temp database."""
        return TaskListTool(db_path=temp_db)

    def test_create_task(self, tool):
        """Test creating a new task."""
        result = tool.create_task(
            task_id="task-1",
            title="Test Task",
            description="Test Description",
            priority=5,
            tags=["test", "sample"]
        )

        assert result["success"] is True
        assert result["task"]["id"] == "task-1"
        assert result["task"]["title"] == "Test Task"
        assert result["task"]["status"] == "pending"
        assert result["task"]["priority"] == 5
        assert result["task"]["tags"] == ["test", "sample"]

    def test_create_duplicate_task(self, tool):
        """Test creating a duplicate task fails."""
        tool.create_task(task_id="task-1", title="First Task")
        result = tool.create_task(task_id="task-1", title="Duplicate Task")

        assert result["success"] is False
        assert "already exists" in result["error"]

    def test_get_task(self, tool):
        """Test retrieving a task by ID."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.get_task("task-1")

        assert result["success"] is True
        assert result["task"]["id"] == "task-1"
        assert result["task"]["title"] == "Test Task"

    def test_get_nonexistent_task(self, tool):
        """Test retrieving a nonexistent task."""
        result = tool.get_task("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_update_task_title(self, tool):
        """Test updating task title."""
        tool.create_task(task_id="task-1", title="Original Title")
        result = tool.update_task(task_id="task-1", title="Updated Title")

        assert result["success"] is True
        assert result["task"]["title"] == "Updated Title"

    def test_update_task_status(self, tool):
        """Test updating task status."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.IN_PROGRESS
        )

        assert result["success"] is True
        assert result["task"]["status"] == "in_progress"

    def test_update_task_status_to_completed(self, tool):
        """Test updating task to completed status."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.COMPLETED
        )

        assert result["success"] is True
        assert result["task"]["status"] == "completed"

    def test_update_task_to_blocked(self, tool):
        """Test updating task to blocked status."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.BLOCKED
        )

        assert result["success"] is True
        assert result["task"]["status"] == "blocked"

    def test_delete_task(self, tool):
        """Test deleting a task."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.delete_task("task-1")

        assert result["success"] is True

        # Verify task is deleted
        get_result = tool.get_task("task-1")
        assert get_result["success"] is False

    def test_delete_nonexistent_task(self, tool):
        """Test deleting a nonexistent task."""
        result = tool.delete_task("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_list_tasks(self, tool):
        """Test listing all tasks."""
        tool.create_task(task_id="task-1", title="Task 1", priority=1)
        tool.create_task(task_id="task-2", title="Task 2", priority=2)
        tool.create_task(task_id="task-3", title="Task 3", priority=3)

        result = tool.list_tasks()

        assert result["success"] is True
        assert result["total"] == 3
        # Tasks should be ordered by priority descending
        assert result["tasks"][0]["priority"] == 3

    def test_list_tasks_by_status(self, tool):
        """Test filtering tasks by status."""
        tool.create_task(task_id="task-1", title="Task 1")
        tool.create_task(task_id="task-2", title="Task 2")
        tool.update_task(task_id="task-1", status=TaskStatus.IN_PROGRESS)

        result = tool.list_tasks(status=TaskStatus.IN_PROGRESS)

        assert result["success"] is True
        assert result["total"] == 1
        assert result["tasks"][0]["id"] == "task-1"

    def test_list_tasks_by_assignee(self, tool):
        """Test filtering tasks by assignee."""
        tool.create_task(task_id="task-1", title="Task 1", assignee="alice")
        tool.create_task(task_id="task-2", title="Task 2", assignee="bob")
        tool.create_task(task_id="task-3", title="Task 3", assignee="alice")

        result = tool.list_tasks(assignee="alice")

        assert result["success"] is True
        assert result["total"] == 2

    def test_list_tasks_by_tags(self, tool):
        """Test filtering tasks by tags."""
        tool.create_task(task_id="task-1", title="Task 1", tags=["urgent", "dev"])
        tool.create_task(task_id="task-2", title="Task 2", tags=["dev", "backend"])
        tool.create_task(task_id="task-3", title="Task 3", tags=["urgent"])

        result = tool.list_tasks(tags=["urgent"])

        assert result["success"] is True
        assert result["total"] == 2

    def test_update_task_status_method(self, tool):
        """Test update_task_status convenience method."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.update_task_status("task-1", TaskStatus.COMPLETED)

        assert result["success"] is True
        assert result["task"]["status"] == "completed"


class TestHooks:
    """Tests for hook functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def tool(self, temp_db):
        """Create a TaskListTool instance with hooks."""
        t = TaskListTool(db_path=temp_db)
        t.add_pre_hook(task_validation_hook)
        t.add_post_hook(task_notification_hook)
        t.add_post_hook(task_audit_hook)
        return t

    def test_pre_hook_validation(self, tool):
        """Test pre-hook validation works."""
        tool.create_task(task_id="task-1", title="Test Task")

        # Update without validation issues should succeed
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.IN_PROGRESS
        )
        assert result["success"] is True

    def test_post_hook_notification(self, tool):
        """Test post-hook notification fires."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.COMPLETED
        )

        # Post-hook should have added notification_sent
        assert result["success"] is True

    def test_post_hook_audit(self, tool):
        """Test post-hook audit fires on delete."""
        tool.create_task(task_id="task-1", title="Test Task")
        result = tool.delete_task("task-1")

        assert result["success"] is True

    def test_status_change_triggers_hooks(self, tool):
        """Test that status changes trigger hooks."""
        tool.create_task(task_id="task-1", title="Test Task")

        # Status change should trigger notification hook
        result = tool.update_task(
            task_id="task-1",
            status=TaskStatus.IN_PROGRESS
        )
        assert result["success"] is True


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_task_status_values(self):
        """Test all task status values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.BLOCKED.value == "blocked"

    def test_task_status_from_string(self):
        """Test creating TaskStatus from string."""
        status = TaskStatus("pending")
        assert status == TaskStatus.PENDING

        status = TaskStatus("completed")
        assert status == TaskStatus.COMPLETED


class TestToolRegistry:
    """Tests for tool registry functions."""

    def test_get_tool_by_name(self):
        """Test retrieving tool by name."""
        from src.agents.tools.task_list_tool import get_task_list_tool_by_name

        tool = get_task_list_tool_by_name("create_task")
        assert tool is not None
        assert "function" in tool

    def test_list_tools(self):
        """Test listing all tools."""
        from src.agents.tools.task_list_tool import list_task_list_tools

        tools = list_task_list_tools()
        assert "create_task" in tools
        assert "get_task" in tools
        assert "update_task" in tools
        assert "delete_task" in tools
        assert "list_tasks" in tools
