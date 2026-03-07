"""
Example: Using Department-Aware Hooks

This module demonstrates how to use the department-aware hook system.
"""

import asyncio
import logging
from src.agents.hooks.department_hooks import (
    Department,
    DepartmentHookType,
    DepartmentHookContext,
    DepartmentHookResult,
    register_department_hook,
    execute_department_hooks,
    get_department_registry,
    for_department,
    pre_department_task_hook,
    post_department_task_hook,
    department_error_hook,
    department_memory_hook,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Example 1: Register a hook for a specific department
@register_department_hook(
    hook_type=DepartmentHookType.PRE_DEPARTMENT_TASK,
    departments={Department.RESEARCH},
    priority=100,
    hook_id="research_pre_task_logger",
)
async def research_pre_task_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """Pre-task hook that adds research-specific context."""
    logger.info(f"Research department task: {context.task_id}")

    # Add research-specific metadata
    enhanced_context = context.with_metadata(
        research_focus="alpha_discovery",
        required_tools=["market_data", "backtester"],
    )

    return DepartmentHookResult(
        context=enhanced_context,
        metrics={"hook": "research_pre_task"},
    )


# Example 2: Register a hook for multiple departments
@register_department_hook(
    hook_type=DepartmentHookType.POST_DEPARTMENT_TASK,
    departments={Department.RISK, Department.TRADING},
    priority=50,
    hook_id="risk_trading_post_task",
)
async def risk_trading_post_task_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """Post-task hook for risk and trading departments."""
    logger.info(f"Risk/Trading department task completed: {context.task_id}")

    return DepartmentHookResult(
        context=context,
        metrics={
            "task_completed": True,
            "department": context.department.value if context.department else "unknown",
        },
    )


# Example 3: Using the for_department decorator
@for_department(Department.DEVELOPMENT)
async def development_pre_task_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """Pre-task hook for Development department."""
    logger.info(f"Development department pre-task: {context.task_id}")

    return DepartmentHookResult(
        context=context.with_metadata(requires_ea_build=True),
    )


# Example 4: Register a department-specific memory hook
@register_department_hook(
    hook_type=DepartmentHookType.DEPARTMENT_MEMORY,
    departments={Department.PORTFOLIO},
    priority=200,
    hook_id="portfolio_memory_hook",
)
async def portfolio_memory_hook(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """Route portfolio memories to a dedicated namespace."""
    logger.info(f"Routing portfolio memory for task: {context.task_id}")

    return DepartmentHookResult(
        context=context,
        memory_namespace="dept_portfolio_strategic",
    )


# Example 5: Error handling hook for trading department
@register_department_hook(
    hook_type=DepartmentHookType.DEPARTMENT_ERROR,
    departments={Department.TRADING},
    priority=100,
    hook_id="trading_error_handler",
)
async def trading_error_handler(
    context: DepartmentHookContext,
) -> DepartmentHookResult:
    """Handle errors specifically for the Trading department."""
    logger.error(f"Trading department error: {context.error}")

    return DepartmentHookResult(
        context=context,
        error_handled=True,
        data={"recovery_action": "retry_order"},
    )


async def demonstrate_department_hooks():
    """Demonstrate the department-aware hook system."""

    # Create a context for a RESEARCH task
    research_context = DepartmentHookContext(
        agent_id="research_head_001",
        task_id="task_alpha_discovery",
        input={"strategy_type": "momentum"},
        department=Department.RESEARCH,
    )

    # Execute pre-task hooks
    result = await execute_department_hooks(
        DepartmentHookType.PRE_DEPARTMENT_TASK,
        research_context,
    )

    logger.info(f"Pre-task result: {result.metrics}")
    logger.info(f"Context metadata: {result.context.metadata}")

    # Create a context for a TRADING task
    trading_context = DepartmentHookContext(
        agent_id="trading_head_001",
        task_id="task_order_execution",
        input={"order_type": "market", "symbol": "EURUSD"},
        department=Department.TRADING,
    )

    # Execute post-task hooks
    result = await execute_department_hooks(
        DepartmentHookType.POST_DEPARTMENT_TASK,
        trading_context,
    )

    logger.info(f"Post-task result: {result.metrics}")

    # Execute memory hooks
    result = await execute_department_hooks(
        DepartmentHookType.DEPARTMENT_MEMORY,
        research_context,
    )

    logger.info(f"Memory namespace: {result.memory_namespace}")

    # Get department metrics
    registry = get_department_registry()
    all_metrics = registry.get_all_department_metrics()
    logger.info(f"Department metrics: {all_metrics}")


if __name__ == "__main__":
    asyncio.run(demonstrate_department_hooks())
