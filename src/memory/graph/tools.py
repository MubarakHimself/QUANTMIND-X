"""GraphMemoryTools - Dict-based API for agents.

This module provides a simplified, dict-based interface to the graph memory
system, making it easy for agents to use without dealing with complex object
structures. All methods return dictionaries for easy serialization.
"""
import logging
from datetime import datetime
from typing import Any, Optional

from src.memory.graph.facade import GraphMemoryFacade
from src.memory.graph.types import MemoryNode

logger = logging.getLogger(__name__)


class GraphMemoryTools:
    """Dict-based API for graph memory operations.

    This class wraps GraphMemoryFacade with a simpler dict-based interface
    that is more convenient for agents. All return values are dictionaries
    that can be easily serialized.

    Usage:
        facade = GraphMemoryFacade(db_path=Path(":memory:"))
        tools = GraphMemoryTools(facade)

        # Store a memory
        result = tools.retain(
            content="Remember that user prefers momentum strategies",
            importance=0.8,
            tags=["preference", "trading"],
            agent_id="agent-001",
            session_id="session-001",
        )

        # Recall memories
        memories = tools.recall(
            query="user preferences",
            agent_id="agent-001",
        )

        # Reflect on memories
        answer = tools.reflect(
            query="What trading strategies does the user prefer?",
            agent_id="agent-001",
        )
    """

    def __init__(self, facade: GraphMemoryFacade) -> None:
        """Initialize tools with a facade.

        Args:
            facade: GraphMemoryFacade instance.
        """
        self.facade = facade

    def retain(
        self,
        content: str,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        related_to: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Store a new memory.

        Args:
            content: The memory content to store.
            importance: Importance score (0-1).
            tags: List of tags for categorization.
            department: Department associated with this memory.
            agent_id: Agent that created this memory.
            session_id: Session where this memory was created.
            role: Role of the agent.
            related_to: List of node IDs this memory relates to.

        Returns:
            Dictionary with 'node_id' key.
        """
        node_id = self.facade.retain(
            content=content,
            source="agent_tool",
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            importance=importance,
            tags=tags,
            related_to=related_to,
        )

        logger.info(f"Retained memory: {node_id}")

        return {
            "node_id": node_id,
            "success": True,
        }

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
        traverse: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve memories.

        Args:
            query: Text search query.
            department: Filter by department.
            agent_id: Filter by agent ID.
            session_id: Filter by session ID.
            role: Filter by role.
            tags: Filter by tags.
            min_importance: Minimum importance score.
            limit: Maximum number of results.
            traverse: Whether to perform graph traversal.

        Returns:
            List of dictionaries representing memory nodes.
        """
        nodes = self.facade.recall(
            query=query,
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            tags=tags,
            min_importance=min_importance,
            limit=limit,
        )

        # Serialize nodes to dicts
        results = [self._serialize_node(node) for node in nodes]

        logger.info(f"Recalled {len(results)} memories for query: {query}")

        return results

    def reflect(
        self,
        query: str,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Synthesize answer from memories.

        Args:
            query: The question to answer.
            department: Filter by department.
            agent_id: Filter by agent ID.
            context: Additional context for the query.

        Returns:
            Dictionary with 'answer', 'sources', and 'synthesized' keys.
        """
        result = self.facade.reflect(
            query=query,
            department=department,
            agent_id=agent_id,
            context=context,
        )

        logger.info(f"Reflected on query: {query}")

        return result

    def link(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        strength: float = 0.8,
        context: Optional[str] = None,
    ) -> dict[str, Any]:
        """Create a relationship between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Type of relationship.
            strength: Relationship strength (0-1).
            context: Optional context about the relationship.

        Returns:
            Dictionary with 'success' key.
        """
        edges = self.facade.link(
            source_id=source_id,
            target_ids=[target_id],
            relation_type=relation,
            strength=strength,
        )

        logger.info(f"Linked {source_id} to {target_id} with relation '{relation}'")

        return {
            "success": True,
            "edge_count": len(edges),
        }

    def check_compaction(
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
        result = self.facade.check_and_compact(
            context_percent=context_percent,
            max_importance=max_importance,
            max_nodes=max_nodes,
        )

        logger.info(f"Compaction check: triggered={result.get('triggered', False)}")

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with node counts and tier distribution.
        """
        stats = self.facade.get_stats()

        logger.info(f"Memory stats: {stats}")

        return stats

    def _serialize_node(self, node: MemoryNode) -> dict[str, Any]:
        """Serialize a MemoryNode to a dictionary.

        Args:
            node: MemoryNode to serialize.

        Returns:
            Dictionary representation of the node.
        """
        return {
            "id": str(node.id),
            "content": node.content,
            "title": node.title,
            "importance": node.importance,
            "tags": node.tags,
            "department": node.department,
            "agent_id": node.agent_id,
            "session_id": node.session_id,
            "created_at": node.created_at.isoformat() if node.created_at else None,
            "updated_at": node.updated_at.isoformat() if node.updated_at else None,
            "access_count": node.access_count,
        }


__all__ = ["GraphMemoryTools"]
