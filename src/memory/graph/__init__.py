"""Graph-Based Memory System.

This package provides graph-based memory management with tiered storage
(Hot/Warm/Cold), context-aware compaction, and Retain/Recall/Reflect
operations based on the Hindsight paper.
"""

from src.memory.graph.compaction import ContextCompactionTrigger
from src.memory.graph.facade import GraphMemoryFacade, get_graph_memory
from src.memory.graph.integrations import DepartmentMailIntegration
from src.memory.graph.migration import DualWriteWrapper, MemoryMigrator
from src.memory.graph.operations import MemoryOperations
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
)

__all__ = [
    # Core classes
    "GraphMemoryStore",
    "MemoryOperations",
    "MemoryTierManager",
    "ContextCompactionTrigger",
    "GraphMemoryFacade",
    # Tools
    "GraphMemoryTools",
    # Integration
    "DepartmentMailIntegration",
    # Migration
    "MemoryMigrator",
    "DualWriteWrapper",
    # Types
    "MemoryNode",
    "MemoryEdge",
    "MemoryNodeType",
    "MemoryCategory",
    "MemoryTier",
    "RelationType",
    # Factory
    "get_graph_memory",
]
