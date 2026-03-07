"""
Department Hook Integration Utilities

This module provides utilities for integrating department-aware hooks
into the Trading Floor workflow.
"""

import logging
from typing import Any, Dict, Optional, Callable, Awaitable

from src.agents.departments.types import (
    Department,
    get_department_config,
    DepartmentHeadConfig,
)
from src.agents.hooks.department_hooks import (
    DepartmentHookType,
    DepartmentHookContext,
    DepartmentHookResult,
    execute_department_hooks,
    get_department_registry,
    post_department_task_hook,
    pre_department_task_hook,
    department_error_hook,
    department_memory_hook,
)

logger = logging.getLogger(__name__)


class DepartmentHookExecutor:
    """
    Executes department-aware hooks for the Trading Floor.

    This class provides methods to integrate department hooks into
    the floor manager and department head workflows.

    Usage:
        hook_executor = DepartmentHookExecutor()
        result = await hook_executor.execute_pre_task(department, task_input)
    """

    def __init__(self):
        """Initialize the hook executor."""
        self._default_pre_hook = pre_department_task_hook
        self._default_post_hook = post_department_task_hook
        self._default_error_hook = department_error_hook
        self._default_memory_hook = department_memory_hook

    def _create_context(
        self,
        department: Department,
        agent_id: str,
        task_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> DepartmentHookContext:
        """Create a department hook context."""
        dept_config = get_department_config(department)
        config_dict = {}
        if dept_config:
            config_dict = {
                "agent_type": dept_config.agent_type,
                "memory_namespace": dept_config.memory_namespace,
                "max_workers": dept_config.max_workers,
            }

        return DepartmentHookContext(
            agent_id=agent_id,
            task_id=task_id,
            input=input_data or {},
            session_id=session_id,
            department=department,
            department_config=config_dict,
        )

    async def execute_pre_task(
        self,
        department: Department,
        agent_id: str,
        task_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> DepartmentHookResult:
        """
        Execute pre-task hooks for a department.

        This is called before a task is dispatched to a department.

        Args:
            department: Target department
            agent_id: ID of the agent handling the task
            task_id: ID of the task
            input_data: Input data for the task
            session_id: Optional session ID

        Returns:
            Hook result with possibly modified context
        """
        context = self._create_context(
            department=department,
            agent_id=agent_id,
            task_id=task_id,
            input_data=input_data,
            session_id=session_id,
        )

        logger.debug(f"Executing pre-task hooks for {department.value}")

        return await execute_department_hooks(
            DepartmentHookType.PRE_DEPARTMENT_TASK,
            context,
        )

    async def execute_post_task(
        self,
        department: Department,
        agent_id: str,
        task_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> DepartmentHookResult:
        """
        Execute post-task hooks for a department.

        This is called after a department task completes.

        Args:
            department: Target department
            agent_id: ID of the agent that handled the task
            task_id: ID of the task
            input_data: Input data for the task
            output_data: Output data from the task
            session_id: Optional session ID

        Returns:
            Hook result with metrics and memory namespace
        """
        context = self._create_context(
            department=department,
            agent_id=agent_id,
            task_id=task_id,
            input_data=input_data,
            session_id=session_id,
        )
        context.output = output_data or {}

        logger.debug(f"Executing post-task hooks for {department.value}")

        return await execute_department_hooks(
            DepartmentHookType.POST_DEPARTMENT_TASK,
            context,
        )

    async def execute_error(
        self,
        department: Department,
        agent_id: str,
        task_id: str,
        error: Exception,
        input_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> DepartmentHookResult:
        """
        Execute error hooks for a department.

        This is called when an error occurs in a department.

        Args:
            department: Target department
            agent_id: ID of the agent that encountered the error
            task_id: ID of the task
            error: The exception that occurred
            input_data: Input data for the task
            session_id: Optional session ID

        Returns:
            Hook result with error_handled flag
        """
        context = self._create_context(
            department=department,
            agent_id=agent_id,
            task_id=task_id,
            input_data=input_data,
            session_id=session_id,
        )
        context.error = error

        logger.debug(f"Executing error hooks for {department.value}")

        return await execute_department_hooks(
            DepartmentHookType.DEPARTMENT_ERROR,
            context,
        )

    async def execute_memory(
        self,
        department: Department,
        agent_id: str,
        task_id: str,
        memory_data: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> DepartmentHookResult:
        """
        Execute memory hooks for a department.

        This is called to route memories to department namespaces.

        Args:
            department: Target department
            agent_id: ID of the agent
            task_id: ID of the task
            memory_data: Data to be stored in memory
            session_id: Optional session ID

        Returns:
            Hook result with memory_namespace
        """
        context = self._create_context(
            department=department,
            agent_id=agent_id,
            task_id=task_id,
            input_data=memory_data,
            session_id=session_id,
        )

        logger.debug(f"Executing memory hooks for {department.value}")

        return await execute_department_hooks(
            DepartmentHookType.DEPARTMENT_MEMORY,
            context,
        )


# Global executor instance
_hook_executor: Optional[DepartmentHookExecutor] = None


def get_hook_executor() -> DepartmentHookExecutor:
    """Get the global hook executor instance."""
    global _hook_executor
    if _hook_executor is None:
        _hook_executor = DepartmentHookExecutor()
    return _hook_executor


# Decorator utilities for easy hook registration


def pre_task_for_department(
    department: Department,
    priority: int = 0,
) -> Callable:
    """
    Decorator to register a pre-task hook for a specific department.

    Usage:
        @pre_task_for_department(Department.RESEARCH, priority=100)
        async def my_pre_task_hook(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Args:
        department: Target department
        priority: Hook priority (higher executes first)

    Returns:
        Decorator function
    """
    from src.agents.hooks.department_hooks import register_department_hook

    def decorator(
        func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
    ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
        register_department_hook(
            hook_type=DepartmentHookType.PRE_DEPARTMENT_TASK,
            handler=func,
            departments={department},
            priority=priority,
            hook_id=f"custom_{department.value}_pre_task",
        )
        return func

    return decorator


def post_task_for_department(
    department: Department,
    priority: int = 0,
) -> Callable:
    """
    Decorator to register a post-task hook for a specific department.

    Usage:
        @post_task_for_department(Department.RISK, priority=50)
        async def my_post_task_hook(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Args:
        department: Target department
        priority: Hook priority

    Returns:
        Decorator function
    """
    from src.agents.hooks.department_hooks import register_department_hook

    def decorator(
        func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
    ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
        register_department_hook(
            hook_type=DepartmentHookType.POST_DEPARTMENT_TASK,
            handler=func,
            departments={department},
            priority=priority,
            hook_id=f"custom_{department.value}_post_task",
        )
        return func

    return decorator


def error_handler_for_department(
    department: Department,
    priority: int = 0,
) -> Callable:
    """
    Decorator to register an error handler hook for a specific department.

    Usage:
        @error_handler_for_department(Department.TRADING, priority=100)
        async def my_error_handler(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Args:
        department: Target department
        priority: Hook priority

    Returns:
        Decorator function
    """
    from src.agents.hooks.department_hooks import register_department_hook

    def decorator(
        func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
    ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
        register_department_hook(
            hook_type=DepartmentHookType.DEPARTMENT_ERROR,
            handler=func,
            departments={department},
            priority=priority,
            hook_id=f"custom_{department.value}_error_handler",
        )
        return func

    return decorator


__all__ = [
    "DepartmentHookExecutor",
    "get_hook_executor",
    "pre_task_for_department",
    "post_task_for_department",
    "error_handler_for_department",
]
