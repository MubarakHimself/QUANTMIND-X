"""
Advanced Hook System for QuantMindX Agents

A comprehensive hook system inspired by OpenClaw architecture that allows
interception and modification of agent behavior at various lifecycle points.

This module provides:
- HookType enum defining all available hook points
- HookContext dataclass for rich context passing
- HookRegistry for centralized hook management
- Async hook execution with error handling
- Hook priority and ordering
- Conditional hook execution based on agent type or tool
"""

import asyncio
import logging
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from functools import total_ordering
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
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")


class HookType(str, Enum):
    """
    Enum defining all available hook types in the agent lifecycle.

    Hooks are executed at specific points during agent execution:
    - PRE_TOOL_USE: Before a tool is executed
    - POST_TOOL_USE: After a tool has executed
    - USER_PROMPT_SUBMIT: When user submits a prompt
    - SESSION_START: When a session begins
    - SESSION_END: When a session ends
    - PRE_AGENT_SPAWN: Before a sub-agent is spawned
    - POST_AGENT_SPAWN: After a sub-agent has spawned
    """

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_AGENT_SPAWN = "pre_agent_spawn"
    POST_AGENT_SPAWN = "post_agent_spawn"


@dataclass
class HookContext:
    """
    Rich context object passed to and returned from hooks.

    Attributes:
        agent_id: ID of the current agent
        task_id: ID of the current task
        session_id: ID of the current session (optional)
        input: Input data for the operation
        output: Output data from the operation (for post hooks)
        timestamp: When the hook was triggered
        metadata: Additional metadata for hooks
        tool_name: Name of the tool being used (for tool hooks)
        agent_type: Type of agent being spawned (for spawn hooks)
        parent_agent_id: ID of parent agent (for spawn hooks)
        error: Any error that occurred (for post hooks)
    """

    agent_id: str
    task_id: str
    input: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    agent_type: Optional[str] = None
    parent_agent_id: Optional[str] = None
    error: Optional[Exception] = None

    def with_input(self, **kwargs: Any) -> "HookContext":
        """Create a new context with additional input data."""
        new_input = {**self.input, **kwargs}
        return replace(self, input=new_input)

    def with_output(self, **kwargs: Any) -> "HookContext":
        """Create a new context with additional output data."""
        new_output = {**self.output, **kwargs}
        return replace(self, output=new_output)

    def with_metadata(self, **kwargs: Any) -> "HookContext":
        """Create a new context with additional metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return replace(self, metadata=new_metadata)


@dataclass
class HookCondition:
    """
    Condition for hook execution.

    Allows hooks to only execute under specific circumstances:
    - agent_types: Only execute for specific agent types
    - tools: Only execute for specific tools
    - custom: Custom predicate function
    """

    agent_types: Optional[Set[str]] = None
    tools: Optional[Set[str]] = None
    custom: Optional[Callable[[HookContext], bool]] = None

    def matches(self, context: HookContext) -> bool:
        """Check if the condition matches the given context."""
        # Check agent type
        if self.agent_types and context.agent_type not in self.agent_types:
            return False

        # Check tool name
        if self.tools and context.tool_name not in self.tools:
            return False

        # Check custom predicate
        if self.custom and not self.custom(context):
            return False

        return True


@dataclass
class HookResult:
    """
    Result returned from a hook execution.

    Attributes:
        context: The (possibly modified) context
        stop: If True, stop executing further hooks
        skip: If True, skip the main operation
        data: Additional data from the hook
    """

    context: HookContext
    stop: bool = False
    skip: bool = False
    data: Dict[str, Any] = field(default_factory=dict)


@total_ordering
@dataclass
class Hook:
    """
    Internal representation of a registered hook.

    Attributes:
        hook_type: The type of hook
        handler: The async handler function
        priority: Higher priority hooks execute first (default: 0)
        condition: Optional condition for execution
        id: Unique hook identifier
        enabled: Whether the hook is enabled
    """

    hook_type: HookType
    handler: Callable[[HookContext], Awaitable[HookResult]]
    priority: int = 0
    condition: Optional[HookCondition] = None
    id: str = field(default_factory=lambda: f"hook_{uuid4().hex[:12]}")
    enabled: bool = True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Hook):
            return False
        return self.priority == other.priority

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Hook):
            return NotImplemented
        return self.priority > other.priority  # Higher priority first


class HookRegistry:
    """
    Central registry for managing hooks.

    Features:
    - Register/unregister hooks
    - Execute hooks with error handling
    - Hook priority ordering
    - Conditional execution
    - Enable/disable hooks by ID
    """

    def __init__(self):
        """Initialize an empty hook registry."""
        self._hooks: Dict[HookType, List[Hook]] = {hook_type: [] for hook_type in HookType}
        self._lock = asyncio.Lock()

    def register(
        self,
        hook_type: HookType,
        *,
        priority: int = 0,
        condition: Optional[HookCondition] = None,
        hook_id: Optional[str] = None,
    ) -> Callable:
        """
        Decorator to register a hook.

        Args:
            hook_type: Type of hook to register
            priority: Higher priority hooks execute first
            condition: Optional condition for execution
            hook_id: Optional custom hook ID

        Returns:
            Decorator function

        Example:
            ```python
            registry = HookRegistry()

            @registry.register(HookType.PRE_AGENT_SPAWN, priority=100)
            async def my_hook(context: HookContext) -> HookResult:
                return HookResult(context)
            ```
        """

        def decorator(
            func: Callable[[HookContext], Awaitable[HookResult]]
        ) -> Callable[[HookContext], Awaitable[HookResult]]:
            hook = Hook(
                hook_type=hook_type,
                handler=func,
                priority=priority,
                condition=condition,
                id=hook_id or f"hook_{uuid4().hex[:12]}",
            )
            self._hooks[hook_type].append(hook)
            self._hooks[hook_type].sort()
            logger.debug(f"Registered hook {hook.id} for {hook_type.value} (priority={priority})")
            return func

        return decorator

    def register_hook(
        self,
        hook_type: HookType,
        handler: Callable[[HookContext], Awaitable[HookResult]],
        *,
        priority: int = 0,
        condition: Optional[HookCondition] = None,
        hook_id: Optional[str] = None,
    ) -> str:
        """
        Register a hook function directly.

        Args:
            hook_type: Type of hook to register
            handler: Async handler function
            priority: Higher priority hooks execute first
            condition: Optional condition for execution
            hook_id: Optional custom hook ID

        Returns:
            The hook ID
        """
        hook = Hook(
            hook_type=hook_type,
            handler=handler,
            priority=priority,
            condition=condition,
            id=hook_id or f"hook_{uuid4().hex[:12]}",
        )
        self._hooks[hook_type].append(hook)
        self._hooks[hook_type].sort()
        logger.info(f"Registered hook {hook.id} for {hook_type.value} (priority={priority})")
        return hook.id

    def unregister(self, hook_type: HookType, hook_id: str) -> bool:
        """
        Unregister a hook by ID.

        Args:
            hook_type: Type of hook
            hook_id: ID of hook to unregister

        Returns:
            True if hook was found and removed
        """
        hooks = self._hooks[hook_type]
        original_length = len(hooks)
        self._hooks[hook_type] = [h for h in hooks if h.id != hook_id]
        removed = len(self._hooks[hook_type]) < original_length
        if removed:
            logger.info(f"Unregistered hook {hook_id} from {hook_type.value}")
        return removed

    def enable_hook(self, hook_id: str) -> bool:
        """Enable a hook by ID."""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.id == hook_id:
                    hook.enabled = True
                    logger.debug(f"Enabled hook {hook_id}")
                    return True
        return False

    def disable_hook(self, hook_id: str) -> bool:
        """Disable a hook by ID."""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.id == hook_id:
                    hook.enabled = False
                    logger.debug(f"Disabled hook {hook_id}")
                    return True
        return False

    def get_hooks(self, hook_type: HookType) -> List[Hook]:
        """Get all hooks for a given type."""
        return self._hooks.get(hook_type, []).copy()

    async def execute_hooks(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> HookResult:
        """
        Execute all hooks for a given type.

        Hooks are executed in priority order (highest first).
        If a hook returns stop=True, no further hooks are executed.

        Args:
            hook_type: Type of hooks to execute
            context: Context to pass to hooks

        Returns:
            Final hook result with potentially modified context
        """
        hooks = self.get_hooks(hook_type)

        # Filter enabled hooks and check conditions
        active_hooks = [
            h for h in hooks
            if h.enabled and (h.condition is None or h.condition.matches(context))
        ]

        if not active_hooks:
            return HookResult(context=context)

        result = HookResult(context=context)

        for hook in active_hooks:
            try:
                hook_result = await hook.handler(result.context)

                # Update context for next hook
                result.context = hook_result.context

                # Check for stop signal
                if hook_result.stop:
                    logger.debug(f"Hook {hook.id} requested stop")
                    result.stop = True
                    break

                # Check for skip signal
                if hook_result.skip:
                    result.skip = True

                # Merge data
                result.data.update(hook_result.data)

            except Exception as e:
                logger.error(f"Hook {hook.id} failed: {e}", exc_info=True)
                # Continue executing other hooks despite errors

        return result

    async def execute_hooks_parallel(
        self,
        hook_type: HookType,
        context: HookContext,
    ) -> List[HookResult]:
        """
        Execute all hooks for a given type in parallel.

        Note: Parallel hooks cannot modify each other's contexts.
        Use this for fire-and-forget style hooks.

        Args:
            hook_type: Type of hooks to execute
            context: Context to pass to hooks

        Returns:
            List of hook results
        """
        hooks = self.get_hooks(hook_type)

        # Filter enabled hooks and check conditions
        active_hooks = [
            h for h in hooks
            if h.enabled and (h.condition is None or h.condition.matches(context))
        ]

        if not active_hooks:
            return []

        # Execute all hooks in parallel
        tasks = [
            self._safe_execute_hook(hook, context)
            for hook in active_hooks
        ]

        return await asyncio.gather(*tasks, return_exceptions=False)

    async def _safe_execute_hook(
        self,
        hook: Hook,
        context: HookContext,
    ) -> HookResult:
        """Execute a hook with error handling."""
        try:
            return await hook.handler(context)
        except Exception as e:
            logger.error(f"Hook {hook.id} failed: {e}", exc_info=True)
            return HookResult(context=context, data={"error": str(e)})


# Global registry instance
_global_registry: Optional[HookRegistry] = None
_registry_lock = asyncio.Lock()


def get_global_registry() -> HookRegistry:
    """Get the global hook registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = HookRegistry()
    return _global_registry


def register_hook(
    hook_type: HookType,
    handler: Callable[[HookContext], Awaitable[HookResult]],
    *,
    priority: int = 0,
    condition: Optional[HookCondition] = None,
    hook_id: Optional[str] = None,
) -> str:
    """
    Register a hook with the global registry.

    Args:
        hook_type: Type of hook to register
        handler: Async handler function
        priority: Higher priority hooks execute first
        condition: Optional condition for execution
        hook_id: Optional custom hook ID

    Returns:
        The hook ID
    """
    registry = get_global_registry()
    return registry.register_hook(
        hook_type=hook_type,
        handler=handler,
        priority=priority,
        condition=condition,
        hook_id=hook_id,
    )


async def execute_hooks(
    hook_type: HookType,
    context: HookContext,
) -> HookResult:
    """
    Execute hooks using the global registry.

    Args:
        hook_type: Type of hooks to execute
        context: Context to pass to hooks

    Returns:
        Final hook result
    """
    registry = get_global_registry()
    return await registry.execute_hooks(hook_type, context)
