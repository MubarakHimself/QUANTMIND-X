"""
Workflow and Queue Tools for QuantMind agents.

These tools provide workflow and queue management:
- trigger_analyst: Trigger Analyst agent with VideoIngest
- trigger_quantcode: Trigger QuantCode agent with TRD
- get_workflow_status: Get current workflow status
- cancel_workflow: Cancel running workflow
- add_task: Add task to agent queue
- get_queue_status: Get current queue status
- get_task_status: Get specific task status
- cancel_task: Cancel pending or running task
- retry_task: Retry failed task
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType as ToolAgentType
from ..queue import (
    QueueManager,
    Task,
    TaskPriority,
    TaskStatus,
    get_queue_manager,
)
from ..workflows.state import get_workflow_store
from ..workflows.video_ingest_to_ea import create_video_ingest_to_ea_workflow


logger = logging.getLogger(__name__)


# Input Schemas
class TriggerAnalystInput(BaseModel):
    """Input for trigger_analyst tool."""
    video_ingest_content: str = Field(description="VideoIngest content to process")
    workspace_path: Optional[str] = Field(default=None, description="Workspace path")
    auto_proceed: bool = Field(default=True, description="Auto-proceed to QuantCode")


class TriggerQuantCodeInput(BaseModel):
    """Input for trigger_quantcode tool."""
    trd_content: str = Field(description="TRD content to process")
    workspace_path: Optional[str] = Field(default=None, description="Workspace path")


class GetWorkflowStatusInput(BaseModel):
    """Input for get_workflow_status tool."""
    workflow_id: str = Field(description="Workflow ID to check")


class CancelWorkflowInput(BaseModel):
    """Input for cancel_workflow tool."""
    workflow_id: str = Field(description="Workflow ID to cancel")


class AddTaskInput(BaseModel):
    """Input for add_task tool."""
    agent_type: str = Field(description="Target agent (copilot, analyst, quantcode)")
    name: str = Field(description="Task name")
    description: str = Field(default="", description="Task description")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input data")
    priority: int = Field(default=5, description="Task priority (1-30)")
    dependencies: List[str] = Field(default_factory=list, description="Task IDs this depends on")


class GetQueueStatusInput(BaseModel):
    """Input for get_queue_status tool."""
    agent_type: Optional[str] = Field(default=None, description="Agent type (all if not specified)")


class GetTaskStatusInput(BaseModel):
    """Input for get_task_status tool."""
    task_id: str = Field(description="Task ID")


class CancelTaskInput(BaseModel):
    """Input for cancel_task tool."""
    task_id: str = Field(description="Task ID to cancel")


class RetryTaskInput(BaseModel):
    """Input for retry_task tool."""
    task_id: str = Field(description="Task ID to retry")


@register_tool(
    agent_types=[ToolAgentType.COPILOT],
    tags=["workflow", "analyst", "trigger"],
)
class TriggerAnalystTool(QuantMindTool):
    """Trigger Analyst agent with VideoIngest."""

    name: str = "trigger_analyst"
    description: str = """Trigger the Analyst agent to process a VideoIngest.
    Parses and validates the VideoIngest, then generates a TRD.
    Can optionally auto-proceed to QuantCode for code generation."""

    args_schema: type[BaseModel] = TriggerAnalystInput
    category: ToolCategory = ToolCategory.WORKFLOW
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        video_ingest_content: str,
        workspace_path: Optional[str] = None,
        auto_proceed: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute analyst trigger."""
        # Create workflow
        workflow = create_video_ingest_to_ea_workflow(workspace_path)

        # Run workflow (async)
        try:
            result = asyncio.run(workflow.run(video_ingest_content))
        except RuntimeError:
            # Already in async context
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    workflow.run(video_ingest_content)
                )
                result = future.result()

        return ToolResult.ok(
            data={
                "workflow_id": result.workflow_id,
                "status": result.status.value,
                "success": result.success,
                "trd": result.intermediate_outputs.get("trd"),
                "mql5_code": result.intermediate_outputs.get("mql5_code") if auto_proceed else None,
            },
            metadata={
                "auto_proceed": auto_proceed,
                "duration_seconds": result.duration_seconds,
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT, ToolAgentType.ANALYST],
    tags=["workflow", "quantcode", "trigger"],
)
class TriggerQuantCodeTool(QuantMindTool):
    """Trigger QuantCode agent with TRD."""

    name: str = "trigger_quantcode"
    description: str = """Trigger the QuantCode agent to generate MQL5 code from a TRD.
    Generates, validates, and compiles the Expert Advisor."""

    args_schema: type[BaseModel] = TriggerQuantCodeInput
    category: ToolCategory = ToolCategory.WORKFLOW
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        trd_content: str,
        workspace_path: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute quantcode trigger."""
        from ..tools.quantcode import GenerateMQL5Tool, CompileMQL5Tool

        # Generate MQL5
        generate_tool = GenerateMQL5Tool()
        if workspace_path:
            generate_tool.set_workspace(workspace_path)

        gen_result = generate_tool.execute(trd_content=trd_content)

        if not gen_result.success:
            return gen_result

        mql5_code = gen_result.data.get("code")

        # Compile
        compile_tool = CompileMQL5Tool()
        if workspace_path:
            compile_tool.set_workspace(workspace_path)

        compile_result = compile_tool.execute(code=mql5_code)

        return ToolResult.ok(
            data={
                "mql5_code": mql5_code,
                "compilation": compile_result.data,
                "ea_path": compile_result.data.get("output_path"),
            },
            metadata={
                "lines_of_code": len(mql5_code.split("\n")),
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT, ToolAgentType.ANALYST, ToolAgentType.QUANTCODE],
    tags=["workflow", "status"],
)
class GetWorkflowStatusTool(QuantMindTool):
    """Get current workflow status."""

    name: str = "get_workflow_status"
    description: str = """Get the current status of a workflow.
    Returns progress, current step, and intermediate results."""

    args_schema: type[BaseModel] = GetWorkflowStatusInput
    category: ToolCategory = ToolCategory.WORKFLOW
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        workflow_id: str,
        **kwargs
    ) -> ToolResult:
        """Execute status check."""
        store = get_workflow_store()
        state = store.get(workflow_id)

        if not state:
            raise ToolError(
                f"Workflow not found: {workflow_id}",
                tool_name=self.name,
                error_code="WORKFLOW_NOT_FOUND"
            )

        return ToolResult.ok(
            data=state.to_dict(),
            metadata={
                "checked_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT],
    tags=["workflow", "cancel"],
)
class CancelWorkflowTool(QuantMindTool):
    """Cancel running workflow."""

    name: str = "cancel_workflow"
    description: str = """Cancel a running workflow.
    Stops execution and marks workflow as cancelled."""

    args_schema: type[BaseModel] = CancelWorkflowInput
    category: ToolCategory = ToolCategory.WORKFLOW
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        workflow_id: str,
        **kwargs
    ) -> ToolResult:
        """Execute workflow cancellation."""
        store = get_workflow_store()
        state = store.get(workflow_id)

        if not state:
            raise ToolError(
                f"Workflow not found: {workflow_id}",
                tool_name=self.name,
                error_code="WORKFLOW_NOT_FOUND"
            )

        state.cancel()
        store.save(state)

        return ToolResult.ok(
            data={
                "cancelled": True,
                "workflow_id": workflow_id,
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT],
    tags=["queue", "task", "add"],
)
class AddTaskTool(QuantMindTool):
    """Add task to agent queue."""

    name: str = "add_task"
    description: str = """Add a new task to an agent's queue.
    Supports priorities and dependencies."""

    args_schema: type[BaseModel] = AddTaskInput
    category: ToolCategory = ToolCategory.QUEUE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        agent_type: str,
        name: str,
        description: str = "",
        input_data: Dict[str, Any] = None,
        priority: int = 5,
        dependencies: List[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute task addition."""
        queue_manager = get_queue_manager()

        task = Task(
            name=name,
            description=description,
            agent_type=agent_type,
            priority=priority,
            input_data=input_data or {},
            dependencies=dependencies or [],
        )

        task_id = queue_manager.add_task(task)

        return ToolResult.ok(
            data={
                "task_id": task_id,
                "status": TaskStatus.QUEUED.value,
                "agent_type": agent_type,
                "priority": priority,
            },
            metadata={
                "added_at": datetime.now().isoformat(),
                "dependencies_count": len(dependencies or []),
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT, ToolAgentType.ANALYST, ToolAgentType.QUANTCODE],
    tags=["queue", "status"],
)
class GetQueueStatusTool(QuantMindTool):
    """Get current queue status."""

    name: str = "get_queue_status"
    description: str = """Get the status of agent task queues.
    Shows pending and running task counts."""

    args_schema: type[BaseModel] = GetQueueStatusInput
    category: ToolCategory = ToolCategory.QUEUE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        agent_type: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute queue status check."""
        queue_manager = get_queue_manager()

        if agent_type:
            status = queue_manager.get_queue_status(agent_type)
            statuses = [status]
        else:
            statuses = queue_manager.get_all_queue_statuses()

        return ToolResult.ok(
            data={
                "queues": statuses,
            },
            metadata={
                "checked_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT, ToolAgentType.ANALYST, ToolAgentType.QUANTCODE],
    tags=["queue", "task", "status"],
)
class GetTaskStatusTool(QuantMindTool):
    """Get specific task status."""

    name: str = "get_task_status"
    description: str = """Get the status of a specific task.
    Returns progress, result, or error information."""

    args_schema: type[BaseModel] = GetTaskStatusInput
    category: ToolCategory = ToolCategory.QUEUE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        task_id: str,
        **kwargs
    ) -> ToolResult:
        """Execute task status check."""
        queue_manager = get_queue_manager()
        task = queue_manager.get_task(task_id)

        if not task:
            raise ToolError(
                f"Task not found: {task_id}",
                tool_name=self.name,
                error_code="TASK_NOT_FOUND"
            )

        return ToolResult.ok(
            data=task.to_dict(),
            metadata={
                "checked_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT],
    tags=["queue", "task", "cancel"],
)
class CancelTaskTool(QuantMindTool):
    """Cancel pending or running task."""

    name: str = "cancel_task"
    description: str = """Cancel a pending or running task.
    Cannot cancel completed tasks."""

    args_schema: type[BaseModel] = CancelTaskInput
    category: ToolCategory = ToolCategory.QUEUE
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        task_id: str,
        **kwargs
    ) -> ToolResult:
        """Execute task cancellation."""
        queue_manager = get_queue_manager()
        success = queue_manager.cancel_task(task_id)

        if not success:
            raise ToolError(
                f"Could not cancel task: {task_id}",
                tool_name=self.name,
                error_code="CANCEL_FAILED"
            )

        return ToolResult.ok(
            data={
                "cancelled": True,
                "task_id": task_id,
            }
        )


@register_tool(
    agent_types=[ToolAgentType.COPILOT],
    tags=["queue", "task", "retry"],
)
class RetryTaskTool(QuantMindTool):
    """Retry failed task."""

    name: str = "retry_task"
    description: str = """Retry a failed task.
    Tasks can be retried up to their max_retries limit."""

    args_schema: type[BaseModel] = RetryTaskInput
    category: ToolCategory = ToolCategory.QUEUE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        task_id: str,
        **kwargs
    ) -> ToolResult:
        """Execute task retry."""
        queue_manager = get_queue_manager()
        success = queue_manager.retry_task(task_id)

        if not success:
            raise ToolError(
                f"Could not retry task: {task_id}",
                tool_name=self.name,
                error_code="RETRY_FAILED"
            )

        return ToolResult.ok(
            data={
                "retried": True,
                "task_id": task_id,
            }
        )


# Export all tools
__all__ = [
    "TriggerAnalystTool",
    "TriggerQuantCodeTool",
    "GetWorkflowStatusTool",
    "CancelWorkflowTool",
    "AddTaskTool",
    "GetQueueStatusTool",
    "GetTaskStatusTool",
    "CancelTaskTool",
    "RetryTaskTool",
]
