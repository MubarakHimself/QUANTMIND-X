"""Unified Graph Memory Facade for QuantMindX.

This module provides the GraphMemoryFacade class that wraps all graph-based
memory components (store, operations, tier manager, compaction) into a single
unified interface.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.memory.graph.compaction import ContextCompactionTrigger
from src.memory.graph.config import resolve_graph_memory_db_path
from src.memory.graph.operations import MemoryOperations
from src.memory.graph.reflection_executor import ReflectionExecutor
from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.tier_manager import MemoryTierManager
from src.memory.graph.types import MemoryEdge, MemoryNode, MemoryTier, SessionStatus

logger = logging.getLogger(__name__)

# Global singleton instance
_facade_instance: Optional["GraphMemoryFacade"] = None
_facade_db_path: Optional[Path] = None


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
            db_path: Path to SQLite database file. If None, uses the shared
                graph-memory database path.
            compaction_threshold: Percentage threshold for triggering compaction.
        """
        db_path = resolve_graph_memory_db_path(db_path)
        self.db_path = db_path

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
        entity_id: Optional[str] = None,
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
            entity_id: Entity identifier for entity-based grouping (e.g., strategy_id).

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
            entity_id=entity_id,
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

    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node by ID."""
        return self.store.delete_node(node_id)

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

    # Session recovery methods

    def load_committed_state(
        self,
        session_id: str,
        include_content: bool = True,
    ) -> dict[str, Any]:
        """Load all committed nodes for a session to resume work.

        This method retrieves all committed memory nodes for a session,
        enabling the agent to resume work from the last known committed state.

        Args:
            session_id: Session ID to load committed state for.
            include_content: Whether to include full content (vs just summaries).

        Returns:
            Dictionary with committed state including nodes and metadata.

        Raises:
            ValueError: If session_id is empty or None.
        """
        if not session_id:
            raise ValueError("session_id is required and cannot be empty")

        try:
            committed_nodes = self.store.query_nodes(
                session_id=session_id,
                session_status=SessionStatus.COMMITTED,
                limit=1000,
            )

            nodes_data = []
            for node in committed_nodes:
                node_data = {
                    "id": node.id,
                    "node_type": node.node_type.value,
                    "title": node.title,
                    "category": node.category.value,
                    "importance": node.importance,
                    "tags": node.tags,
                    "created_at": node.created_at_utc.isoformat() if node.created_at_utc else None,
                }

                if include_content:
                    node_data["content"] = node.content
                    node_data["summary"] = node.summary
                    node_data["evidence"] = node.evidence

                nodes_data.append(node_data)

            return {
                "session_id": session_id,
                "committed_nodes_count": len(nodes_data),
                "nodes": nodes_data,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to load committed state for session {session_id}: {e}")
            return {
                "session_id": session_id,
                "committed_nodes_count": 0,
                "nodes": [],
                "loaded_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }

    async def mark_draft_nodes_interrupted(self, interrupted_at: Optional[datetime] = None) -> int:
        """
        Mark all draft nodes as interrupted when Copilot kill switch activates.

        This prevents partial state changes from being committed to graph memory.

        Args:
            interrupted_at: Timestamp when the interruption occurred

        Returns:
            Number of nodes marked as interrupted
        """
        if interrupted_at is None:
            interrupted_at = datetime.now(timezone.utc)

        try:
            # Find all draft nodes
            draft_nodes = self.store.get_nodes_by_session_status(SessionStatus.DRAFT)
            count = 0

            for node in draft_nodes:
                # Update node to mark as interrupted - use tags to track this
                if not node.tags:
                    node.tags = []
                node.tags.append("interrupted")
                node.updated_at_utc = datetime.now(timezone.utc)
                self.store.update_node(node)
                count += 1

            logger.info(f"Marked {count} draft nodes as interrupted at {interrupted_at.isoformat()}")
            return count

        except Exception as e:
            logger.error(f"Failed to mark draft nodes as interrupted: {e}")
            return 0

    # === Session Workspace Isolation Methods ===

    def get_draft_nodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Get draft nodes, optionally filtered by session_id.

        Args:
            session_id: Optional session ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of draft MemoryNode objects.
        """
        return self.store.get_nodes_by_session_status(
            session_status=SessionStatus.DRAFT,
            session_id=session_id,
            limit=limit,
        )

    def get_committed_nodes(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Get committed nodes, optionally filtered by session_id.

        Args:
            session_id: Optional session ID to filter by.
            limit: Maximum number of results.

        Returns:
            List of committed MemoryNode objects.
        """
        return self.store.get_nodes_by_session_status(
            session_status=SessionStatus.COMMITTED,
            session_id=session_id,
            limit=limit,
        )

    def commit_session(
        self,
        session_id: str,
        department: Optional[str] = None,
        entity_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Commit all draft nodes for a session.

        Args:
            session_id: Session ID to commit nodes for.
            department: Optional department to filter nodes.
            entity_id: Entity identifier to check for conflicts (e.g., strategy_id).

        Returns:
            Dictionary with commit results: node_count, committed_at, session_id.
            If conflicts detected, returns status='pending_review' with conflict info.
        """
        # Check for conflicts if entity_id provided (AC #3: concurrent conflict detection)
        if entity_id:
            conflicts = self.store.detect_conflicts(
                strategy_id=entity_id,
                exclude_session_id=session_id,
            )
            if conflicts:
                return {
                    "status": "pending_review",
                    "session_id": session_id,
                    "entity_id": entity_id,
                    "conflict_count": len(conflicts),
                    "conflicts": [
                        {
                            "id": str(c.id),
                            "session_id": c.session_id,
                            "title": c.title,
                        }
                        for c in conflicts
                    ],
                    "message": "Conflicts detected - requires DeptHead review",
                }

        return self.commit_session_with_validation(session_id=session_id, department=department, entity_id=entity_id, force_commit=False)

    def commit_session_with_validation(
        self,
        session_id: str,
        department: Optional[str] = None,
        entity_id: Optional[str] = None,
        force_commit: bool = False,
    ) -> dict[str, Any]:
        """Commit draft nodes for a session with validation.

        Only nodes that pass validation will be committed unless force_commit=True.
        When force_commit=True, all draft nodes are committed regardless of validation.

        Args:
            session_id: Session ID to commit nodes for.
            department: Optional department to filter nodes.
            entity_id: Entity identifier to check for conflicts (e.g., strategy_id).
            force_commit: If True, commit all draft nodes regardless of validation.

        Returns:
            Dictionary with commit results: node_count, committed_at, session_id.
        """
        # Check for conflicts if entity_id provided
        if entity_id:
            conflicts = self.store.detect_conflicts(
                strategy_id=entity_id,
                exclude_session_id=session_id,
            )
            if conflicts:
                return {
                    "status": "pending_review",
                    "session_id": session_id,
                    "entity_id": entity_id,
                    "conflict_count": len(conflicts),
                    "conflicts": [
                        {
                            "id": str(c.id),
                            "session_id": c.session_id,
                            "title": c.title,
                        }
                        for c in conflicts
                    ],
                    "message": "Conflicts detected - requires DeptHead review",
                }

        # Use ReflectionExecutor to commit nodes
        executor = ReflectionExecutor(self.store)
        result = executor.execute(session_id=session_id, force_commit=force_commit)
        return {
            "session_id": session_id,
            "node_count": result.get("committed_count", 0),
            "committed_at_utc": result.get("reflection_time", datetime.now(timezone.utc).isoformat()),
            "department": department,
        }

    def get_commit_log(
        self,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get commit log entries.

        Args:
            session_id: Optional filter by session_id.
            limit: Maximum number of results.

        Returns:
            List of commit log entries.
        """
        return self.store.get_commit_log(session_id=session_id, limit=limit)

    def detect_conflicts(
        self,
        strategy_id: str,
        exclude_session_id: str,
    ) -> list[MemoryNode]:
        """Detect concurrent writes to the same strategy namespace.

        Args:
            strategy_id: Strategy identifier to check for conflicts.
            exclude_session_id: Session ID to exclude (the current session).

        Returns:
            List of conflicting nodes from other sessions.
        """
        return self.store.detect_conflicts(
            strategy_id=strategy_id,
            exclude_session_id=exclude_session_id,
        )

    def query_session_isolated(
        self,
        session_id: str,
        query: Optional[str] = None,
        include_committed: bool = True,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Query nodes with session isolation.

        Returns nodes that are:
        - In the specified session_id with status 'draft' (private to session)
        - OR have status 'committed' (visible to all)

        Args:
            session_id: Session ID for filtering draft nodes.
            query: Optional text search query.
            include_committed: Whether to include committed nodes (default True).
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects visible to the session.
        """
        # Get draft nodes for this session
        draft_nodes = self.store.query_nodes(
            session_id=session_id,
            session_status=SessionStatus.DRAFT,
            limit=limit,
        )

        if not include_committed:
            return draft_nodes

        # Also get committed nodes
        committed_nodes = self.store.query_nodes(
            session_status=SessionStatus.COMMITTED,
            limit=limit,
        )

        # Combine and deduplicate by ID
        node_dict = {str(node.id): node for node in draft_nodes}
        for node in committed_nodes:
            if str(node.id) not in node_dict:
                node_dict[str(node.id)] = node

        # Sort by importance and return
        result = list(node_dict.values())
        result.sort(key=lambda n: n.importance, reverse=True)
        return result[:limit]

    def get_daily_opinion_summary(self) -> dict[str, list[dict[str, Any]]]:
        """Fetch all OPINION nodes created in the last 24 hours and group by department.

        Used by the morning digest and daily reporting system.

        Returns:
            A dict keyed by department name, each value being a list of opinion
            dicts containing: id, title, content, action, reasoning, confidence,
            agent_role, created_at.
            Example::

                {
                    "research": [{"id": "...", "title": "...", ...}],
                    "development": [...],
                    "risk": [...],
                    ...
                }
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        try:
            all_opinion_nodes = self.store.query_nodes(
                node_type=MemoryNodeType.OPINION,
                limit=1000,
            )
        except Exception as e:
            logger.error(f"Failed to fetch OPINION nodes for daily summary: {e}")
            return {}

        # Filter to last 24 hours in Python (store doesn't expose a created_at filter)
        recent_opinions = [
            node
            for node in all_opinion_nodes
            if node.created_at_utc and node.created_at_utc >= cutoff
        ]

        summary: dict[str, list[dict[str, Any]]] = {}
        for node in recent_opinions:
            dept = node.department or "unknown"
            summary.setdefault(dept, [])
            summary[dept].append(
                {
                    "id": str(node.id),
                    "title": node.title,
                    "content": node.content,
                    "action": node.action,
                    "reasoning": node.reasoning,
                    "confidence": node.confidence,
                    "agent_role": node.agent_role,
                    "created_at": node.created_at_utc.isoformat() if node.created_at_utc else None,
                }
            )

        return summary

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
    global _facade_instance, _facade_db_path

    resolved_path = resolve_graph_memory_db_path(db_path)
    if _facade_instance is None or _facade_db_path != resolved_path:
        if _facade_instance is not None and _facade_db_path != resolved_path:
            logger.warning(
                "Reinitializing GraphMemoryFacade for new db_path: %s -> %s",
                _facade_db_path,
                resolved_path,
            )
        _facade_instance = GraphMemoryFacade(
            db_path=resolved_path,
            compaction_threshold=compaction_threshold,
        )
        _facade_db_path = resolved_path

    return _facade_instance
