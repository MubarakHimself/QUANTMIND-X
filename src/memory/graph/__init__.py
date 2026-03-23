"""Graph-Based Memory System.

This package provides graph-based memory management with tiered storage
(Hot/Warm/Cold), context-aware compaction, and Retain/Recall/Reflect
operations based on the Hindsight paper.
"""

from src.memory.graph.compaction import ContextCompactionTrigger
from src.memory.graph.embedding_service import EmbeddingService, get_embedding_service
from src.memory.graph.facade import GraphMemoryFacade, get_graph_memory
from src.memory.graph.integrations import DepartmentMailIntegration
from src.memory.graph.migration import (
    DualWriteWrapper,
    MemoryMigrator,
    add_embedding_column,
    add_opinion_columns,
    add_session_status_column,
    migrate_graph_memory_db,
    rename_timestamps_to_utc,
)
from src.memory.graph.operations import MemoryOperations
from src.memory.graph.reflection_executor import ReflectionExecutor, create_reflection_executor
from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.tier_manager import MemoryTierManager
from src.memory.graph.tools import GraphMemoryTools
from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
    RelationType,
    SessionStatus,
)

__all__ = [
    # Core classes
    "GraphMemoryStore",
    "MemoryOperations",
    "MemoryTierManager",
    "ContextCompactionTrigger",
    "GraphMemoryFacade",
    # Reflection
    "ReflectionExecutor",
    "create_reflection_executor",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # Tools
    "GraphMemoryTools",
    # Integration
    "DepartmentMailIntegration",
    # Migration
    "MemoryMigrator",
    "DualWriteWrapper",
    "add_session_status_column",
    "add_embedding_column",
    "add_opinion_columns",
    "rename_timestamps_to_utc",
    "migrate_graph_memory_db",
    # Types
    "MemoryNode",
    "MemoryEdge",
    "MemoryNodeType",
    "MemoryCategory",
    "MemoryTier",
    "RelationType",
    "SessionStatus",
    # Factory
    "get_graph_memory",
    "get_embedding_service",
]
