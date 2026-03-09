"""Unified Graph Memory Facade for QuantMindX.

This module provides the GraphMemoryFacade class that wraps all graph-based
memory components (store, operations, tier manager, compaction) into a single
unified interface.
"""
import logging
from pathlib import Path
from typing import Any, Optional

from src.memory.graph.compaction import ContextCompactionTrigger
from src.memory.graph.operations import MemoryOperations
from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.tier_manager import MemoryTierManager
from src.memory.graph.types import MemoryEdge, MemoryNode, MemoryTier

logger = logging.getLogger(__name__)

# Global singleton instance
_facade_instance: Optional["GraphMemoryFacade"] = None


class GraphMemoryFacade:
    """Unified facade for graph-based memory system.

    This class wraps all components of the graph-based memory system:
    - GraphMemoryStore: Persistence layer
    - MemoryOperations: Retain/Recall/Reflect operations
    - MemoryTierManager: Hot/Warm/Cold tier management
    - ContextCompactionTrigger: Context-aware compaction

    Provides a simple interface for common memory operations.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        compaction_threshold: float = 50.0,
    ) -> None:
        """Initialize the graph memory facade.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory store.
            compaction_threshold: Percentage threshold for triggering compaction.
        """
        if db_path is None:
            db_path = Path(":memory:")

        self.store = GraphMemoryStore(db_path)
        self.operations = MemoryOperations(self.store)
        self.tier_manager = MemoryTierManager(self.store)
        self.compaction = ContextCompactionTrigger(
            self.store, threshold=compaction_threshold
        )

        logger.info(f"Initialized GraphMemoryFacade with db_path: {db_path}")

    def retain(
        self,
        content: str,
        source: str = "unknown",
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        related_to: Optional[list[str]] = None,
    ) -> str:
        """Store a new memory (RETAIN operation).

        Args:
            content: The memory content to store.
            source: Source of the memory.
            department: Department associated with this memory.
            agent_id: Agent that created this memory.
            session_id: Session where this memory was created.
            importance: Importance score (0-1).
            tags: List of tags for categorization.
            related_to: List of node IDs this memory relates to.

        Returns:
            The ID of the created memory node.
        """
        node = self.operations.retain(
            content=content,
            source=source,
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            importance=importance,
            tags=tags,
            related_to=related_to,
        )
        return str(node.id)

    def recall(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        tags: Optional[list[str]] = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[MemoryNode]:
        """Retrieve memories (RECALL operation).

        Args:
            query: Text search query.
            department: Filter by department.
            agent_id: Filter by agent ID.
            session_id: Filter by session ID.
            role: Filter by role.
            tags: Filter by tags (any match).
            min_importance: Minimum importance score.
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects sorted by relevance.
        """
        return self.operations.recall(
            query=query,
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            tags=tags,
            min_importance=min_importance,
            limit=limit,
        )

    def reflect(
        self,
        query: str,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Synthesize answer from memories (REFLECT operation).

        Args:
            query: The question to answer.
            department: Filter by department.
            agent_id: Filter by agent ID.
            context: Additional context for the query.

        Returns:
            Dictionary with answer, sources, and synthesized flag.
        """
        return self.operations.reflect(
            query=query,
            department=department,
            agent_id=agent_id,
            context=context,
        )

    def link(
        self,
        source_id: str,
        target_ids: list[str],
        relation_type: str = "related_to",
        strength: float = 0.8,
    ) -> list[MemoryEdge]:
        """Create relationship edges between nodes.

        Args:
            source_id: Source node ID.
            target_ids: List of target node IDs.
            relation_type: Type of relationship.
            strength: Relationship strength (0-1).

        Returns:
            List of created MemoryEdge objects.
        """
        return self.operations.link_nodes(
            source_id=source_id,
            target_ids=target_ids,
            relation_type=relation_type,
            strength=strength,
        )

    # Tier management methods

    def get_hot_nodes(self, limit: int = 50) -> list[MemoryNode]:
        """Get nodes in the hot tier.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the hot tier.
        """
        return self.tier_manager.get_hot_nodes(limit)

    def get_warm_nodes(self, limit: int = 100) -> list[MemoryNode]:
        """Get nodes in the warm tier.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the warm tier.
        """
        return self.tier_manager.get_warm_nodes(limit)

    def get_cold_nodes(self, limit: int = 100) -> list[MemoryNode]:
        """Get nodes in the cold tier.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the cold tier.
        """
        return self.tier_manager.get_cold_nodes(limit)

    def move_to_hot(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the hot tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        return self.tier_manager.move_to_hot(node_id)

    def move_to_warm(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the warm tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        return self.tier_manager.move_to_warm(node_id)

    def move_to_cold(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the cold tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        return self.tier_manager.move_to_cold(node_id)

    def promote_to_hot(self, node_id: str) -> Optional[MemoryNode]:
        """Promote a cold node to hot tier.

        Args:
            node_id: ID of the node to promote.

        Returns:
            The updated node, or None if not found.
        """
        return self.tier_manager.promote_hot(node_id)

    # Compaction methods

    def should_compact(self, context_percent: float) -> bool:
        """Determine if compaction should be triggered.

        Args:
            context_percent: Current context usage percentage.

        Returns:
            True if compaction should be triggered.
        """
        return self.compaction.should_compact(context_percent)

    def check_and_compact(
        self,
        context_percent: float,
        max_importance: float = 0.5,
        max_nodes: int = 100,
    ) -> dict[str, Any]:
        """Check if compaction is needed and perform it if triggered.

        Args:
            context_percent: Current context usage percentage.
            max_importance: Maximum importance threshold for candidates.
            max_nodes: Maximum number of nodes to compact.

        Returns:
            Dictionary with compaction results.
        """
        triggered = self.should_compact(context_percent)

        if triggered:
            result = self.compaction.compact_and_anchor(
                max_importance=max_importance,
                max_nodes=max_nodes,
                create_anchors=True,
            )
            result["triggered"] = True
            return result

        return {"triggered": False}

    def compact_old_nodes(
        self,
        days_old: int = 30,
        max_nodes: int = 100,
        min_importance: float = 0.3,
    ) -> dict[str, Any]:
        """Compact old low-importance nodes to cold tier.

        Args:
            days_old: Age threshold in days.
            max_nodes: Maximum number of nodes to compact.
            min_importance: Maximum importance threshold.

        Returns:
            Dictionary with compaction results.
        """
        return self.tier_manager.compact_old_nodes(
            days_old=days_old,
            max_nodes=max_nodes,
            min_importance=min_importance,
        )

    # Statistics

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with node counts and tier distribution.
        """
        tier_stats = self.tier_manager.get_tier_stats()

        # Get total node count
        all_nodes = self.store.query_nodes(limit=10000)

        return {
            "total_nodes": len(all_nodes),
            "hot": tier_stats.get(MemoryTier.HOT, 0),
            "warm": tier_stats.get(MemoryTier.WARM, 0),
            "cold": tier_stats.get(MemoryTier.COLD, 0),
        }

    def close(self) -> None:
        """Close the database connection."""
        self.store.close()


def get_graph_memory(
    db_path: Optional[Path] = None,
    compaction_threshold: float = 50.0,
) -> GraphMemoryFacade:
    """Get or create the global GraphMemoryFacade singleton.

    Args:
        db_path: Path to SQLite database file.
        compaction_threshold: Percentage threshold for compaction.

    Returns:
        The GraphMemoryFacade singleton instance.
    """
    global _facade_instance

    if _facade_instance is None:
        _facade_instance = GraphMemoryFacade(
            db_path=db_path,
            compaction_threshold=compaction_threshold,
        )

    return _facade_instance
