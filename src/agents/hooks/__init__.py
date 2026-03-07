"""
Advanced Hook System for QuantMindX Agents

This module provides both the legacy agent hooks, the advanced hook system,
and the department-aware hooks.

Legacy Hooks (from hooks.py):
- pre_analyst_hook, post_analyst_hook
- pre_quantcode_hook, post_quantcode_hook
- pre_copilot_hook, post_copilot_hook
- pre_router_hook, post_router_hook
- pre_executor_hook, post_executor_hook
- pre_pinescript_hook, post_pinescript_hook

Advanced Hooks (from advanced_hooks.py):
- HookType enum for all lifecycle events
- HookRegistry for centralized hook management
- HookContext for rich context passing

Department-Aware Hooks (from department_hooks.py):
- DepartmentHookType enum with department-specific hooks
- DepartmentHookContext with department field
- DepartmentHookCondition with department filters
- DepartmentHookRegistry for department-aware hooks
- register_department_hook for registering department hooks
- for_department decorator for department-specific hooks
"""

# Import legacy hooks from the original hooks.py file
# Note: We import from the sibling module using a different name to avoid circular imports
import sys
from pathlib import Path

# Load the original hooks.py module
_legacy_hooks_path = Path(__file__).parent.parent / "hooks.py"
if _legacy_hooks_path.exists():
    import importlib.util
    _spec = importlib.util.spec_from_file_location("_legacy_hooks", _legacy_hooks_path)
    _legacy_hooks = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_legacy_hooks)

    # Export legacy hooks
    pre_analyst_hook = _legacy_hooks.pre_analyst_hook
    post_analyst_hook = _legacy_hooks.post_analyst_hook
    pre_quantcode_hook = _legacy_hooks.pre_quantcode_hook
    post_quantcode_hook = _legacy_hooks.post_quantcode_hook
    pre_copilot_hook = _legacy_hooks.pre_copilot_hook
    post_copilot_hook = _legacy_hooks.post_copilot_hook
    pre_router_hook = _legacy_hooks.pre_router_hook
    post_router_hook = _legacy_hooks.post_router_hook
    pre_executor_hook = _legacy_hooks.pre_executor_hook
    post_executor_hook = _legacy_hooks.post_executor_hook
    pre_pinescript_hook = _legacy_hooks.pre_pinescript_hook
    post_pinescript_hook = _legacy_hooks.post_pinescript_hook

# Import advanced hooks system
from .advanced_hooks import (
    HookType,
    HookContext,
    HookCondition,
    HookResult,
    HookRegistry,
    get_global_registry,
    register_hook,
    execute_hooks,
)

# Import department-aware hooks
from .department_hooks import (
    DepartmentHookType,
    DepartmentHookContext,
    DepartmentHookCondition,
    DepartmentHookResult,
    DepartmentHook,
    DepartmentHookRegistry,
    get_department_registry,
    register_department_hook,
    execute_department_hooks,
    pre_department_task_hook,
    post_department_task_hook,
    department_error_hook,
    department_memory_hook,
    for_department,
    for_all_departments,
    Department,
)

# Import workflow integration utilities
from .workflow_integration import (
    DepartmentHookExecutor,
    get_hook_executor,
    pre_task_for_department,
    post_task_for_department,
    error_handler_for_department,
)

__all__ = [
    # Legacy hooks
    "pre_analyst_hook",
    "post_analyst_hook",
    "pre_quantcode_hook",
    "post_quantcode_hook",
    "pre_copilot_hook",
    "post_copilot_hook",
    "pre_router_hook",
    "post_router_hook",
    "pre_executor_hook",
    "post_executor_hook",
    "pre_pinescript_hook",
    "post_pinescript_hook",
    # Advanced hooks
    "HookType",
    "HookContext",
    "HookCondition",
    "HookResult",
    "HookRegistry",
    "get_global_registry",
    "register_hook",
    "execute_hooks",
    # Department-aware hooks
    "DepartmentHookType",
    "DepartmentHookContext",
    "DepartmentHookCondition",
    "DepartmentHookResult",
    "DepartmentHook",
    "DepartmentHookRegistry",
    "get_department_registry",
    "register_department_hook",
    "execute_department_hooks",
    "pre_department_task_hook",
    "post_department_task_hook",
    "department_error_hook",
    "department_memory_hook",
    "for_department",
    "for_all_departments",
    "Department",
    # Workflow integration
    "DepartmentHookExecutor",
    "get_hook_executor",
    "pre_task_for_department",
    "post_task_for_department",
    "error_handler_for_department",
]
