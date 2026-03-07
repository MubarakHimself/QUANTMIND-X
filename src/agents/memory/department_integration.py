"""
Department Memory Integration Module

Provides integration between department-specific memories and the global memory system.
- Department-specific memory namespaces
- Global memory accessible to departments
- Hierarchical memory structure with routing
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from src.agents.departments.types import Department

logger = logging.getLogger(__name__)


# Global namespace constants
GLOBAL_NAMESPACE = "global"
DEPARTMENT_PREFIX = "dept_"
AGENT_NAMESPACE = "agent"
SESSION_NAMESPACE = "session"
PATTERNS_NAMESPACE = "patterns"


class MemorySharingRule(str, Enum):
    """Defines how memories can be shared between departments."""
    PRIVATE = "private"           # Department-only, not shared
    DEPARTMENT = "department"      # Shared within department
    GLOBAL = "global"              # Available globally
    RESTRICTED = "restricted"     # Only specific departments can access


@dataclass
class DepartmentMemoryConfig:
    """Configuration for department memory behavior."""
    department: Department
    namespace: str
    sharing_rule: MemorySharingRule = MemorySharingRule.DEPARTMENT
    can_access_global: bool = True
    accessible_from: Set[str] = field(default_factory=set)  # Departments that can access
    tags: List[str] = field(default_factory=list)


# Default department memory configurations
DEPARTMENT_CONFIGS: Dict[str, DepartmentMemoryConfig] = {
    "research": DepartmentMemoryConfig(
        department=Department.RESEARCH,
        namespace=f"{DEPARTMENT_PREFIX}research",
        sharing_rule=MemorySharingRule.GLOBAL,  # Research findings are global
        can_access_global=True,
        tags=["strategy", "analysis", "alpha"],
    ),
    "development": DepartmentMemoryConfig(
        department=Department.DEVELOPMENT,
        namespace=f"{DEPARTMENT_PREFIX}development",
        sharing_rule=MemorySharingRule.DEPARTMENT,
        can_access_global=True,
        tags=["code", "implementation", "bots"],
    ),
    "trading": DepartmentMemoryConfig(
        department=Department.TRADING,
        namespace=f"{DEPARTMENT_PREFIX}trading",
        sharing_rule=MemorySharingRule.RESTRICTED,
        can_access_global=True,
        accessible_from={"risk", "portfolio"},
        tags=["execution", "orders", "fills"],
    ),
    "risk": DepartmentMemoryConfig(
        department=Department.RISK,
        namespace=f"{DEPARTMENT_PREFIX}risk",
        sharing_rule=MemorySharingRule.GLOBAL,  # Risk metrics are global
        can_access_global=True,
        tags=["risk", "metrics", "limits", "drawdown"],
    ),
    "portfolio": DepartmentMemoryConfig(
        department=Department.PORTFOLIO,
        namespace=f"{DEPARTMENT_PREFIX}portfolio",
        sharing_rule=MemorySharingRule.GLOBAL,  # Portfolio allocations are global
        can_access_global=True,
        tags=["allocation", "performance", "positions"],
    ),
}


def get_department_config(department: Department) -> DepartmentMemoryConfig:
    """Get the memory configuration for a department."""
    dept_value = department.value
    return DEPARTMENT_CONFIGS.get(dept_value, DepartmentMemoryConfig(
        department=department,
        namespace=f"{DEPARTMENT_PREFIX}{dept_value}",
    ))


def get_department_namespace(department: Department) -> str:
    """Get the memory namespace for a department."""
    return get_department_config(department).namespace


def is_memory_accessible(
    source_department: Department,
    target_namespace: str,
) -> bool:
    """
    Check if a department can access a specific namespace.

    Args:
        source_department: The department trying to access memory
        target_namespace: The namespace being accessed

    Returns:
        True if access is allowed
    """
    # Global namespace is always accessible
    if target_namespace == GLOBAL_NAMESPACE:
        return True

    # Agent and session namespaces are private
    if target_namespace in (AGENT_NAMESPACE, SESSION_NAMESPACE):
        return True

    # Check department-specific namespace
    if target_namespace.startswith(DEPARTMENT_PREFIX):
        target_dept = target_namespace.replace(DEPARTMENT_PREFIX, "")

        # Get target department config
        config = DEPARTMENT_CONFIGS.get(target_dept)
        if not config:
            return False

        # Check sharing rule
        if config.sharing_rule == MemorySharingRule.GLOBAL:
            return True

        if config.sharing_rule == MemorySharingRule.DEPARTMENT:
            return source_department.value == target_dept

        if config.sharing_rule == MemorySharingRule.RESTRICTED:
            return source_department.value in config.accessible_from

        if config.sharing_rule == MemorySharingRule.PRIVATE:
            return False

    # Default: allow access
    return True


def route_memory(
    department: Optional[Department],
    key: str,
    value: str,
    namespace: Optional[str] = None,
    sharing_rule: Optional[MemorySharingRule] = None,
) -> Dict[str, Any]:
    """
    Route memory to appropriate namespace based on department and sharing rules.

    Args:
        department: Source department (if any)
        key: Memory key
        value: Memory value
        namespace: Requested namespace
        sharing_rule: Explicit sharing rule

    Returns:
        Dictionary with resolved namespace and metadata
    """
    # Default to global if no department specified
    if department is None:
        return {
            "namespace": namespace or GLOBAL_NAMESPACE,
            "key": key,
            "value": value,
            "sharing_rule": MemorySharingRule.GLOBAL,
            "is_global": True,
        }

    config = get_department_config(department)

    # If explicit namespace provided
    if namespace:
        # Check access permissions
        if not is_memory_accessible(department, namespace):
            logger.warning(
                f"Department {department.value} denied access to namespace {namespace}"
            )
            # Fall back to department namespace
            namespace = config.namespace
        return {
            "namespace": namespace,
            "key": key,
            "value": value,
            "sharing_rule": sharing_rule or config.sharing_rule,
            "is_global": namespace == GLOBAL_NAMESPACE,
            "source_department": department.value,
        }

    # Default routing based on sharing rule
    if sharing_rule == MemorySharingRule.GLOBAL or config.sharing_rule == MemorySharingRule.GLOBAL:
        return {
            "namespace": GLOBAL_NAMESPACE,
            "key": key,
            "value": value,
            "sharing_rule": sharing_rule or config.sharing_rule,
            "is_global": True,
            "source_department": department.value,
        }

    # Department-private memory
    return {
        "namespace": config.namespace,
        "key": key,
        "value": value,
        "sharing_rule": sharing_rule or config.sharing_rule,
        "is_global": False,
        "source_department": department.value,
    }


def search_across_namespaces(
    department: Department,
    query: str,
    include_global: bool = True,
    include_restricted: bool = False,
    limit: int = 10,
) -> List[str]:
    """
    Get list of namespaces to search based on department permissions.

    Args:
        department: The department performing the search
        query: Search query (for logging)
        include_global: Include global namespace in search
        include_restricted: Include restricted namespaces

    Returns:
        List of namespace strings to search
    """
    namespaces = []

    # Always include department's own namespace
    config = get_department_config(department)
    namespaces.append(config.namespace)

    # Include global if permitted
    if config.can_access_global and include_global:
        namespaces.append(GLOBAL_NAMESPACE)

    # Check restricted namespaces if allowed
    if include_restricted:
        for dept_name, dept_config in DEPARTMENT_CONFIGS.items():
            if dept_config.sharing_rule == MemorySharingRule.RESTRICTED:
                if department.value in dept_config.accessible_from:
                    namespaces.append(dept_config.namespace)

    logger.debug(
        f"Department {department.value} searching namespaces: {namespaces}"
    )

    return namespaces


def get_cross_department_memories(
    department: Department,
    category: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get memories accessible to a department from other departments.

    Args:
        department: The department requesting cross-department memories
        category: Optional category filter

    Returns:
        Dictionary mapping department names to their accessible memories
    """
    config = get_department_config(department)
    results = {}

    if not config.can_access_global:
        return results

    # Get global memories
    results[GLOBAL_NAMESPACE] = []  # Placeholder for actual retrieval

    # Get accessible department memories
    for dept_name, dept_config in DEPARTMENT_CONFIGS.items():
        if dept_name == department.value:
            continue

        if is_memory_accessible(department, dept_config.namespace):
            results[dept_name] = []  # Placeholder for actual retrieval

    return results
