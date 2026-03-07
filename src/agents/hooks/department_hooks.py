"""
Department-Aware Hook System for QuantMindX

This module extends the advanced hooks system with department awareness.
It allows hooks to be aware of which department is being used and adapt behavior accordingly.

The 5 Departments:
- development (replaces analysis)
- research
- risk
- trading (replaces execution)
- portfolio

Hook Types Made Department-Aware:
- Pre-task hooks - can inspect department and modify task
- Post-task hooks - can log department-specific metrics
- Error hooks - can handle department-specific error handling
- Memory hooks - can route memories to department namespaces
"""

import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    TypeVar,
    Awaitable,
    Union,
)
from functools import total_ordering
from uuid import uuid4

from src.agents.departments.types import Department
from src.agents.hooks.advanced_hooks import (
    HookContext as BaseHookContext,
    HookCondition as BaseHookCondition,
    HookResult as BaseHookResult,
    Hook as BaseHook,
    HookRegistry as BaseHookRegistry,
    HookType as BaseHookType,
    get_global_registry,
    register_hook as base_register_hook,
    execute_hooks as base_execute_hooks,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class DepartmentHookType(str, Enum):
    """
    Department-specific hook types.

    These extend the base HookType with department-aware hooks:
    - PRE_DEPARTMENT_TASK: Before a task is dispatched to a department
    - POST_DEPARTMENT_TASK: After a department task completes
    - DEPARTMENT_ERROR: When an error occurs in a department
    - DEPARTMENT_MEMORY: For routing memories to department namespaces
    """

    # Task lifecycle hooks (department-aware)
    PRE_DEPARTMENT_TASK = "pre_department_task"
    POST_DEPARTMENT_TASK = "post_department_task"
    DEPARTMENT_ERROR = "department_error"
    DEPARTMENT_MEMORY = "department_memory"

    # Forward base hook types for compatibility
    PRE_TOOL_USE = BaseHookType.PRE_TOOL_USE
    POST_TOOL_USE = BaseHookType.POST_TOOL_USE
    USER_PROMPT_SUBMIT = BaseHookType.USER_PROMPT_SUBMIT
    SESSION_START = BaseHookType.SESSION_START
    SESSION_END = BaseHookType.SESSION_END
    PRE_AGENT_SPAWN = BaseHookType.PRE_AGENT_SPAWN
    POST_AGENT_SPAWN = BaseHookType.POST_AGENT_SPAWN


@dataclass
class DepartmentHookContext(BaseHookContext):
    """
    Extended context for department-aware hooks.

    Adds department-specific fields to the base HookContext:
    - department: The department this hook is running in
    - department_config: Configuration for the department
    - task_metadata: Department-specific task metadata

    Inherits all fields from BaseHookContext:
    - agent_id, task_id, session_id
    - input, output, timestamp
    - metadata, tool_name, agent_type
    - parent_agent_id, error
    """

    department: Optional[Department] = None
    department_config: Optional[Dict[str, Any]] = None
    task_metadata: Dict[str, Any] = field(default_factory=dict)

    def with_department(
        self,
        department: Department,
        config: Optional[Dict[str, Any]] = None,
    ) -> "DepartmentHookContext":
        """Create a new context with department info."""
        return replace(
            self,
            department=department,
            department_config=config or {},
        )

    def get_memory_namespace(self) -> str:
        """Get the memory namespace for this department."""
        if self.department:
            return f"dept_{self.department.value}"
        return "default"

    def get_error_context(self) -> Dict[str, Any]:
        """Get department-specific error context."""
        return {
            "department": self.department.value if self.department else None,
            "agent_id": self.agent_id,
            "task_id": self.task_id,
            "error": str(self.error) if self.error else None,
            "task_metadata": self.task_metadata,
        }


@dataclass
class DepartmentHookCondition(BaseHookCondition):
    """
    Extended condition for department-aware hooks.

    Adds department filtering to the base HookCondition:
    - departments: Only execute for specific departments
    - exclude_departments: Skip execution for specific departments

    Inherits from BaseHookCondition:
    - agent_types: Only execute for specific agent types
    - tools: Only execute for specific tools
    - custom: Custom predicate function
    """

    departments: Optional[Set[Department]] = None
    exclude_departments: Optional[Set[Department]] = None

    def matches(self, context: DepartmentHookContext) -> bool:
        """Check if the condition matches the given context."""
        # Check base conditions first
        if not super().matches(context):
            return False

        # Check department inclusion
        if self.departments and context.department not in self.departments:
            return False

        # Check department exclusion
        if self.exclude_departments and context.department in self.exclude_departments:
            return False

        return True


@dataclass
class DepartmentHookResult(BaseHookResult):
    """
    Result from a department-aware hook execution.

    Extends BaseHookResult with department-specific fields:
    - memory_namespace: Where to route memories
    - metrics: Department-specific metrics
    - error_handled: Whether the hook handled the error

    Inherits from BaseHookResult:
    - context: The (possibly modified) context
    - stop: If True, stop executing further hooks
    - skip: If True, skip the main operation
    - data: Additional data from the hook
    """

    memory_namespace: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_handled: bool = False


@total_ordering
@dataclass
class DepartmentHook(BaseHook):
    """
    Internal representation of a registered department-aware hook.

    Extends BaseHook with department-specific fields:
    - department_filter: Filter hooks by department
    - is_department_specific: Whether this is a department-specific hook

    Inherits from BaseHook:
    - hook_type: The type of hook
    - handler: The async handler function
    - priority: Higher priority hooks execute first
    - condition: Optional condition for execution
    - id: Unique hook identifier
    - enabled: Whether the hook is enabled
    """

    department_filter: Optional[Set[Department]] = None
    is_department_specific: bool = False


class DepartmentHookRegistry(BaseHookRegistry):
    """
    Extended registry for department-aware hooks.

    Features:
    - Register/unregister department-specific hooks
    - Execute hooks with department context
    - Filter hooks by department
    - Department-specific metrics collection
    """

    def __init__(self):
        """Initialize the department-aware hook registry."""
        super().__init__()
        self._department_metrics: Dict[str, Dict[str, Any]] = {}

    def register_department_hook(
        self,
        hook_type: Union[DepartmentHookType, BaseHookType, str],
        handler: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]],
        *,
        departments: Optional[Set[Department]] = None,
        priority: int = 0,
        condition: Optional[DepartmentHookCondition] = None,
        hook_id: Optional[str] = None,
    ) -> str:
        """
        Register a department-aware hook.

        Args:
            hook_type: Type of hook to register
            handler: Async handler function
            departments: Optional set of departments to filter by
            priority: Higher priority hooks execute first
            condition: Optional condition for execution
            hook_id: Optional custom hook ID

        Returns:
            The hook ID
        """
        # Convert hook_type to string for dictionary key
        if hasattr(hook_type, 'value'):
            hook_type_key = hook_type.value
        else:
            hook_type_key = str(hook_type)

        # Initialize the list if it doesn't exist
        if hook_type_key not in self._hooks:
            self._hooks[hook_type_key] = []

        hook = DepartmentHook(
            hook_type=hook_type,
            handler=handler,
            priority=priority,
            condition=condition,
            id=hook_id or f"hook_{uuid4().hex[:12]}",
            department_filter=departments,
            is_department_specific=departments is not None,
        )

        self._hooks[hook_type_key].append(hook)
        self._hooks[hook_type_key].sort()
        logger.info(
            f"Registered department hook {hook.id} for {hook_type_key} "
            f"(departments={departments}, priority={priority})"
        )
        return hook.id

    async def execute_department_hooks(
        self,
        hook_type: Union[DepartmentHookType, str],
        context: DepartmentHookContext,
    ) -> DepartmentHookResult:
        """
        Execute department-aware hooks.

        Args:
            hook_type: Type of hooks to execute
            context: Department-aware context

        Returns:
            Final hook result with department-specific data
        """
        # Convert hook_type to string for dictionary lookup
        hook_type_key = hook_type.value if hasattr(hook_type, 'value') else str(hook_type)

        # Get hooks for this type
        hooks = self._hooks.get(hook_type_key, [])

        # Filter enabled hooks, check department, and check conditions
        active_hooks = []
        for hook in hooks:
            if not hook.enabled:
                continue

            # Check department filter
            if hook.department_filter and context.department not in hook.department_filter:
                continue

            # Check condition
            if hook.condition and not hook.condition.matches(context):
                continue

            active_hooks.append(hook)

        if not active_hooks:
            return DepartmentHookResult(context=context)

        result = DepartmentHookResult(context=context)

        for hook in active_hooks:
            try:
                hook_result = await hook.handler(result.context)

                # Update context for next hook
                result.context = hook_result.context

                # Check for stop signal
                if hook_result.stop:
                    logger.debug(f"Department hook {hook.id} requested stop")
                    result.stop = True
                    break

                # Check for skip signal
                if hook_result.skip:
                    result.skip = True

                # Merge metrics
                result.metrics.update(hook_result.metrics)

                # Track memory namespace
                if hook_result.memory_namespace:
                    result.memory_namespace = hook_result.memory_namespace

                # Track error handling
                if hook_result.error_handled:
                    result.error_handled = True

                # Merge data
                result.data.update(hook_result.data)

            except Exception as e:
                logger.error(f"Department hook {hook.id} failed: {e}", exc_info=True)
                # Continue executing other hooks despite errors

        return result

    def record_department_metric(
        self,
        department: Department,
        metric_name: str,
        value: Any,
    ) -> None:
        """Record a metric for a department."""
        if department.value not in self._department_metrics:
            self._department_metrics[department.value] = {}
        self._department_metrics[department.value][metric_name] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_department_metrics(
        self,
        department: Department,
    ) -> Dict[str, Any]:
        """Get all metrics for a department."""
        return self._department_metrics.get(department.value, {})

    def get_all_department_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all departments."""
        return self._department_metrics.copy()


# Global department-aware registry
_department_registry: Optional[DepartmentHookRegistry] = None
_registry_lock: Any = None


def get_department_registry() -> DepartmentHookRegistry:
    """Get the global department-aware hook registry instance."""
    global _department_registry
    if _department_registry is None:
        _department_registry = DepartmentHookRegistry()
    return _department_registry


def register_department_hook(
    hook_type: Union[DepartmentHookType, BaseHookType],
    handler: Optional[Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]] = None,
    *,
    departments: Optional[Set[Department]] = None,
    priority: int = 0,
    condition: Optional[DepartmentHookCondition] = None,
    hook_id: Optional[str] = None,
) -> Union[str, Callable]:
    """
    Register a department-aware hook with the global registry.

    Can be used as a decorator or direct function call.

    Usage as decorator:
        @register_department_hook(DepartmentHookType.PRE_DEPARTMENT_TASK, departments={Department.RESEARCH})
        async def my_hook(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Usage as function:
        register_department_hook(DepartmentHookType.PRE_DEPARTMENT_TASK, my_handler, departments={Department.RESEARCH})

    Args:
        hook_type: Type of hook to register
        handler: Async handler function (optional if used as decorator)
        departments: Optional set of departments to filter by
        priority: Higher priority hooks execute first
        condition: Optional condition for execution
        hook_id: Optional custom hook ID

    Returns:
        The hook ID if used as function, or the decorator function if used as decorator
    """
    registry = get_department_registry()

    # If handler is None, we're being used as a decorator factory
    if handler is None:
        def decorator(
            func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
        ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
            registry.register_department_hook(
                hook_type=hook_type,
                handler=func,
                departments=departments,
                priority=priority,
                condition=condition,
                hook_id=hook_id,
            )
            return func
        return decorator

    # Direct function call
    return registry.register_department_hook(
        hook_type=hook_type,
        handler=handler,
        departments=departments,
        priority=priority,
        condition=condition,
        hook_id=hook_id,
    )


async def execute_department_hooks(
    hook_type: DepartmentHookType,
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """
    Execute department-aware hooks using the global registry.

    Args:
        hook_type: Type of hooks to execute
        context: Department-aware context

    Returns:
        Final hook result
    """
    registry = get_department_registry()
    return await registry.execute_department_hooks(hook_type, context)


# Convenience functions for common department-aware hook patterns


async def pre_department_task_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """
    Default pre-department task hook.

    Can be customized per department to:
    - Validate task input
    - Enrich context with department-specific data
    - Modify task before execution

    Args:
        context: The department hook context

    Returns:
        Hook result with possibly modified context
    """
    # Default implementation: just pass through
    return DepartmentHookResult(context=context)


async def post_department_task_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """
    Default post-department task hook.

    Can be customized per department to:
    - Log department-specific metrics
    - Store results to department memory namespace
    - Trigger follow-up actions

    Args:
        context: The department hook context

    Returns:
        Hook result with metrics
    """
    result = DepartmentHookResult(context=context)

    # Record default metrics
    if context.department:
        result.metrics = {
            "task_completed": True,
            "department": context.department.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        result.memory_namespace = f"dept_{context.department.value}"

    return result


async def department_error_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """
    Default department error hook.

    Can be customized per department to:
    - Handle department-specific errors
    - Log error details to appropriate namespace
    - Attempt recovery actions

    Args:
        context: The department hook context with error set

    Returns:
        Hook result with error_handled flag
    """
    result = DepartmentHookResult(context=context)

    if context.error:
        logger.error(
            f"Department error in {context.department}: {context.error}",
            extra=context.get_error_context(),
        )
        result.error_handled = False  # Let caller decide if handled

    return result


async def department_memory_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """
    Default department memory hook.

    Routes memories to appropriate department namespace.

    Args:
        context: The department hook context

    Returns:
        Hook result with memory_namespace
    """
    result = DepartmentHookResult(context=context)
    result.memory_namespace = context.get_memory_namespace()
    return result


# Pre-built department-specific hook decorators


def for_department(*departments: Department):
    """
    Decorator factory to create department-specific hooks.

    Usage:
        @for_department(Department.RESEARCH, Department.DEVELOPMENT)
        async def my_hook(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Args:
        *departments: Departments this hook applies to

    Returns:
        Decorator function
    """
    dept_set = set(departments)

    def decorator(
        func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
    ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
        # Store department info on function for later registration
        func._department_filter = dept_set  # type: ignore
        return func

    return decorator


def for_all_departments():
    """
    Decorator to mark a hook as applicable to all departments.

    Usage:
        @for_all_departments()
        async def my_hook(context: DepartmentHookContext) -> DepartmentHookResult:
            ...

    Returns:
        Decorator function
    """
    def decorator(
        func: Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]
    ) -> Callable[[DepartmentHookContext], Awaitable[DepartmentHookResult]]:
        func._all_departments = True
        return func

    return decorator


# Export all department-aware components
__all__ = [
    # Department hook types
    "DepartmentHookType",
    # Context and conditions
    "DepartmentHookContext",
    "DepartmentHookCondition",
    "DepartmentHookResult",
    "DepartmentHook",
    # Registry
    "DepartmentHookRegistry",
    "get_department_registry",
    "register_department_hook",
    "execute_department_hooks",
    # Default hooks
    "pre_department_task_hook",
    "post_department_task_hook",
    "department_error_hook",
    "department_memory_hook",
    # Decorators
    "for_department",
    "for_all_departments",
    # Re-export base types for compatibility
    "Department",
    "BaseHookType",
]
