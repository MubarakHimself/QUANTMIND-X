"""Agent memory module for cross-session persistence."""

from src.agents.memory.agent_memory import (
    AgentMemory,
    AgentDBMemoryBackend,
    AgentMemoryWithDepartment,
    FileMemoryBackend,
    MemoryEntry,
    get_agent_memory,
    get_agent_memory_with_department,
)

from src.agents.memory.vector_memory import (
    VectorMemory,
    VectorMemoryEntry,
    get_vector_memory,
)

from src.agents.memory.department_integration import (
    DEPARTMENT_CONFIGS,
    DEPARTMENT_PREFIX,
    GLOBAL_NAMESPACE,
    MemorySharingRule,
    DepartmentMemoryConfig,
    get_department_config,
    get_department_namespace,
    is_memory_accessible,
    route_memory,
    search_across_namespaces,
)

from src.agents.memory.unified_memory_facade import (
    UnifiedMemoryFacade,
    UnifiedMemoryEntry,
    UnifiedSearchResult,
    UnifiedMemoryStats,
    get_unified_memory,
)

__all__ = [
    # Core memory
    "AgentMemory",
    "AgentMemoryWithDepartment",
    "AgentDBMemoryBackend",
    "FileMemoryBackend",
    "MemoryEntry",
    "get_agent_memory",
    "get_agent_memory_with_department",
    # Vector memory
    "VectorMemory",
    "VectorMemoryEntry",
    "get_vector_memory",
    # Department integration
    "DEPARTMENT_CONFIGS",
    "DEPARTMENT_PREFIX",
    "GLOBAL_NAMESPACE",
    "MemorySharingRule",
    "DepartmentMemoryConfig",
    "get_department_config",
    "get_department_namespace",
    "is_memory_accessible",
    "route_memory",
    "search_across_namespaces",
    # Unified facade
    "UnifiedMemoryFacade",
    "UnifiedMemoryEntry",
    "UnifiedSearchResult",
    "UnifiedMemoryStats",
    "get_unified_memory",
]
