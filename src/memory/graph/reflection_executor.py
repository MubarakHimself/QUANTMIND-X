"""ReflectionExecutor for Graph Memory System.

This module implements the ReflectionExecutor that processes session memories
after a session settles, extracts episodic and semantic memories, and promotes
draft nodes to committed status.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

from sqlalchemy.exc import SQLAlchemyError

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.operations import MemoryOperations
from src.memory.graph.types import (
    MemoryCategory,
    MemoryNode,
    MemoryNodeType,
    RelationType,
    SessionStatus,
)

logger = logging.getLogger(__name__)


class ReflectionExecutor:
    """Extracts and commits session memories after session settles.

    The ReflectionExecutor runs after a session settles and:
    1. Processes session memories (draft nodes)
    2. Extracts episodic and semantic memories
    3. Validates memories meet quality criteria
    4. Promotes validated memories from draft to committed
    5. Creates appropriate edges between memories
    """

    # Minimum confidence threshold for memory promotion
    MIN_CONFIDENCE_THRESHOLD = 0.5

    # Minimum importance for semantic memory extraction
    SEMANTIC_MEMORY_IMPORTANCE = 0.6

    def __init__(self, store: GraphMemoryStore) -> None:
        """Initialize the ReflectionExecutor.

        Args:
            store: GraphMemoryStore instance for persistence.
        """
        self.store = store
        self.operations = MemoryOperations(store)

    def execute(
        self,
        session_id: str,
        force_commit: bool = False,
    ) -> dict[str, Any]:
        """Execute reflection on a session's draft memories.

        Args:
            session_id: Session ID to reflect on.
            force_commit: If True, commit all draft nodes regardless of validation.

        Returns:
            Dictionary with reflection results.
        """
        logger.info(f"Starting reflection for session {session_id}")

        # Get all draft nodes for this session
        draft_nodes = self.store.query_nodes(
            session_id=session_id,
            session_status=SessionStatus.DRAFT,
            limit=1000,
        )

        if not draft_nodes:
            logger.info(f"No draft nodes found for session {session_id}")
            return {
                "session_id": session_id,
                "draft_count": 0,
                "episodic_extracted": 0,
                "semantic_extracted": 0,
                "committed_count": 0,
            }

        # Categorize nodes
        episodic_memories = []
        semantic_memories = []
        opinion_nodes = []

        for node in draft_nodes:
            if node.node_type == MemoryNodeType.OPINION:
                opinion_nodes.append(node)
            elif node.category == MemoryCategory.EXPERIENTIAL:
                episodic_memories.append(node)
            elif node.category == MemoryCategory.FACTUAL:
                semantic_memories.append(node)

        logger.info(
            f"Session {session_id}: {len(episodic_memories)} episodic, "
            f"{len(semantic_memories)} semantic, {len(opinion_nodes)} opinions"
        )

        # Extract semantic memories from high-importance nodes
        semantic_extracted = self._extract_semantic_memories(
            draft_nodes, session_id
        )

        # Validate and commit nodes
        committed_count = self._promote_to_committed(
            draft_nodes,
            force=force_commit,
        )

        result = {
            "session_id": session_id,
            "draft_count": len(draft_nodes),
            "episodic_extracted": len(episodic_memories),
            "semantic_extracted": semantic_extracted,
            "opinion_count": len(opinion_nodes),
            "committed_count": committed_count,
            "reflection_time": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Reflection complete for session {session_id}: {result}")
        return result

    def _extract_semantic_memories(
        self,
        nodes: list[MemoryNode],
        session_id: str,
    ) -> int:
        """Extract semantic memories from high-importance nodes.

        Args:
            nodes: List of draft nodes to analyze.
            session_id: Session ID for context.

        Returns:
            Number of semantic memories extracted.
        """
        extracted_count = 0

        for node in nodes:
            # Skip if already has embedding
            if node.embedding:
                continue

            # Only extract semantic memories from important nodes
            if node.importance < self.SEMANTIC_MEMORY_IMPORTANCE:
                continue

            # Create semantic memory entry
            # Tag nodes for embedding pipeline to process later

            # Mark as semantic if importance is high enough
            if node.category == MemoryCategory.FACTUAL:
                node.tags = node.tags + ["semantic_memory"]
                self.store.update_node(node)
                extracted_count += 1

        return extracted_count

    def _promote_to_committed(
        self,
        nodes: list[MemoryNode],
        force: bool = False,
    ) -> int:
        """Promote draft nodes to committed status.

        Args:
            nodes: List of draft nodes to promote.
            force: If True, skip validation.

        Returns:
            Number of nodes promoted.
        """
        promoted = 0

        for node in nodes:
            # Validate node before promotion
            if not force and not self._validate_for_promotion(node):
                logger.debug(
                    f"Node {node.id} failed validation, not promoting"
                )
                continue

            # Check for mandatory SUPPORTED_BY edges on opinions
            if node.node_type == MemoryNodeType.OPINION:
                if not self._has_supported_by_edges(node.id):
                    logger.warning(
                        f"OPINION node {node.id} missing SUPPORTED_BY edges"
                    )
                    if not force:
                        continue

            # Promote to committed
            node.session_status = SessionStatus.COMMITTED
            self.store.update_node(node)
            promoted += 1

        return promoted

    def _validate_for_promotion(self, node: MemoryNode) -> bool:
        """Validate a node meets quality criteria for promotion.

        Args:
            node: Node to validate.

        Returns:
            True if node passes validation.
        """
        # Check minimum importance
        if node.importance < 0.3:
            return False

        # Check has content
        if not node.content or len(node.content.strip()) < 10:
            return False

        # For opinion nodes, check required fields AND mandatory SUPPORTED_BY edges
        if node.node_type == MemoryNodeType.OPINION:
            logger.debug(f"OPINION node validation: action={node.action}, reasoning={node.reasoning}, confidence={node.confidence}")
            if not node.action or not node.reasoning:
                logger.debug(f"OPINION node {node.id} failed: missing action or reasoning")
                return False
            if node.confidence is not None and node.confidence < self.MIN_CONFIDENCE_THRESHOLD:
                logger.debug(f"OPINION node {node.id} failed: confidence {node.confidence} < {self.MIN_CONFIDENCE_THRESHOLD}")
                return False
            # Check for mandatory SUPPORTED_BY edges
            has_edges = self._has_supported_by_edges(str(node.id))
            logger.debug(f"OPINION node {node.id} SUPPORTED_BY edges check: {has_edges}")
            if not has_edges:
                logger.debug(f"OPINION node {node.id} failed: missing SUPPORTED_BY edges")
                return False

        return True

    def _has_supported_by_edges(self, node_id: str) -> bool:
        """Check if an OPINION node has SUPPORTED_BY edges.

        Args:
            node_id: Node ID to check.

        Returns:
            True if node has at least one SUPPORTED_BY edge.
        """
        edges = self.store.get_edges(node_id)

        for edge in edges:
            if edge.relation_type == RelationType.SUPPORTED_BY:
                return True

        return False

    def get_session_summary(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Get a summary of a session's draft memories.

        Args:
            session_id: Session ID to summarize.

        Returns:
            Dictionary with session memory summary.
        """
        draft_nodes = self.store.query_nodes(
            session_id=session_id,
            session_status=SessionStatus.DRAFT,
            limit=1000,
        )

        committed_nodes = self.store.query_nodes(
            session_id=session_id,
            session_status=SessionStatus.COMMITTED,
            limit=1000,
        )

        # Count by type
        type_counts = {}
        category_counts = {}

        for node in draft_nodes:
            type_counts[node.node_type.value] = type_counts.get(
                node.node_type.value, 0
            ) + 1
            category_counts[node.category.value] = category_counts.get(
                node.category.value, 0
            ) + 1

        return {
            "session_id": session_id,
            "draft_count": len(draft_nodes),
            "committed_count": len(committed_nodes),
            "draft_by_type": type_counts,
            "draft_by_category": category_counts,
            "has_opinions": any(
                n.node_type == MemoryNodeType.OPINION for n in draft_nodes
            ),
        }

    def cleanup_stale_drafts(
        self,
        threshold_hours: int = 24,
        archive: bool = True,
    ) -> dict[str, Any]:
        """Clean up stale draft nodes older than threshold.

        Args:
            threshold_hours: Age threshold in hours for stale drafts.
            archive: If True, archive stale nodes. If False, delete them.

        Returns:
            Dictionary with cleanup results.
        """
        logger.info(f"Cleaning up stale drafts older than {threshold_hours} hours")

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=threshold_hours)

        # Get all draft nodes
        all_draft_nodes = self.store.query_nodes(
            session_status=SessionStatus.DRAFT,
            limit=10000,
        )

        stale_nodes = [
            node for node in all_draft_nodes
            if node.created_at_utc and node.created_at_utc < cutoff_time
        ]

        archived_count = 0
        deleted_count = 0

        for node in stale_nodes:
            try:
                if archive:
                    # Add archive tag and move to cold tier
                    node.tags = node.tags + ["archived", "stale"]
                    self.store.update_node(node)
                    archived_count += 1
                else:
                    # Delete the node
                    self.store.delete_node(node.id)
                    deleted_count += 1
            except (SQLAlchemyError, RuntimeError, KeyError) as e:
                # Catch specific exceptions - continue processing other nodes on failure
                logger.warning(f"Failed to cleanup node {node.id}: {e}")

        result = {
            "total_draft_nodes": len(all_draft_nodes),
            "stale_nodes_found": len(stale_nodes),
            "archived_count": archived_count,
            "deleted_count": deleted_count,
            "threshold_hours": threshold_hours,
            "cutoff_time": cutoff_time.isoformat(),
        }

        logger.info(f"Stale draft cleanup complete: {result}")
        return result

    def recover_session(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        """Recover session state from committed memories.

        This method loads all committed nodes for a session to enable
        resuming work from the last known committed state.

        Args:
            session_id: Session ID to recover.

        Returns:
            Dictionary with recovery results including committed nodes.

        Raises:
            ValueError: If session_id is empty or None.
        """
        if not session_id:
            raise ValueError("session_id is required and cannot be empty")

        logger.info(f"Recovering session {session_id} from committed state")

        try:
            committed_nodes = self.store.query_nodes(
                session_id=session_id,
                session_status=SessionStatus.COMMITTED,
                limit=1000,
            )

            result = {
                "session_id": session_id,
                "committed_nodes_count": len(committed_nodes),
                "committed_nodes": [
                    {
                        "id": node.id,
                        "node_type": node.node_type.value,
                        "title": node.title,
                        "content": node.content[:200] if node.content else None,
                        "category": node.category.value,
                        "importance": node.importance,
                    }
                    for node in committed_nodes
                ],
                "recovery_time": datetime.now(timezone.utc).isoformat(),
            }

            logger.info(f"Session recovery complete: {len(committed_nodes)} nodes loaded")
            return result
        except Exception as e:
            logger.error(f"Failed to recover session {session_id}: {e}")
            return {
                "session_id": session_id,
                "committed_nodes_count": 0,
                "committed_nodes": [],
                "recovery_time": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }


def create_reflection_executor(
    db_path: str,
) -> ReflectionExecutor:
    """Create a ReflectionExecutor instance.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        ReflectionExecutor instance.
    """
    store = GraphMemoryStore(db_path)
    return ReflectionExecutor(store)


__all__ = ["ReflectionExecutor", "create_reflection_executor"]
