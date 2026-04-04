"""Context-Aware Compaction Trigger for Graph-Based Memory System.

This module provides the ContextCompactionTrigger class for managing memory
compaction based on context usage, preserving important relationships through
memory anchors.
"""
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
    RelationType,
)

logger = logging.getLogger(__name__)


class ContextCompactionTrigger:
    """Trigger for context-aware memory compaction.

    Manages memory compaction based on context usage percentage, preserving
    important relationships through memory anchors that survive compaction.

    Attributes:
        DEFAULT_COMPACTION_THRESHOLD: Default threshold percentage (50.0%)
        DEFAULT_SUMMARY_MAX_TOKENS: Default max tokens for summaries (500)
    """

    DEFAULT_COMPACTION_THRESHOLD: float = 50.0
    DEFAULT_SUMMARY_MAX_TOKENS: int = 500

    def __init__(
        self,
        store: GraphMemoryStore,
        threshold: float = DEFAULT_COMPACTION_THRESHOLD,
        summary_max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
    ) -> None:
        """Initialize the compaction trigger.

        Args:
            store: The GraphMemoryStore instance to manage.
            threshold: Context percentage threshold for triggering compaction.
            summary_max_tokens: Maximum tokens for generated summaries.
        """
        self.store = store
        self.threshold = threshold
        self.summary_max_tokens = summary_max_tokens

    def should_compact(self, context_percent: float) -> bool:
        """Determine if compaction should be triggered.

        Args:
            context_percent: Current context usage percentage.

        Returns:
            True if context_percent >= threshold, False otherwise.
        """
        result = context_percent >= self.threshold
        logger.debug(
            f"Compaction check: {context_percent}% >= {self.threshold}% = {result}"
        )
        return result

    def get_compaction_candidates(
        self,
        max_importance: float = 0.5,
        max_results: int = 100,
        exclude_recent_hours: int = 24,
    ) -> list[MemoryNode]:
        """Get nodes eligible for compaction.

        Finds nodes with low importance and low access count that are
        candidates for compaction.

        Args:
            max_importance: Maximum importance threshold (nodes below this
                          are candidates).
            max_results: Maximum number of candidates to return.
            exclude_recent_hours: Exclude nodes accessed within this many hours.

        Returns:
            List of MemoryNode objects eligible for compaction.
        """
        all_nodes = self.store.query_nodes(min_importance=0.0, limit=max_results * 2)

        # Calculate cutoff time for recent access exclusion
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=exclude_recent_hours)

        candidates = []
        for node in all_nodes:
            # Skip already compacted nodes
            if node.compaction_level >= 2:
                continue

            # Skip nodes with importance above threshold
            if node.importance >= max_importance:
                continue

            # Skip recently accessed nodes
            if node.last_accessed_utc:
                last_accessed = node.last_accessed_utc
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                if last_accessed >= cutoff_time:
                    continue

            # Prioritize nodes with low access count
            candidates.append(node)

        # Sort by importance (lowest first) then by access_count (lowest first)
        candidates.sort(key=lambda n: (n.importance, n.access_count))

        return candidates[:max_results]

    def create_memory_anchor(
        self,
        original_id: str,
        summary: Optional[str] = None,
    ) -> Optional[MemoryNode]:
        """Create an anchor node that preserves the original's relationships.

        The anchor links to the original node via original_id, preserves
        the importance, and takes over the edges from the original node.

        Args:
            original_id: ID of the original node to anchor.
            summary: Optional summary of the original node's content.
                     If not provided, will be generated from the original.

        Returns:
            The created anchor MemoryNode, or None if original not found.
        """
        original = self.store.get_node(original_id)
        if original is None:
            logger.warning(f"Original node not found: {original_id}")
            return None

        # Generate summary if not provided
        if summary is None:
            summary = self._generate_summary(original.content)

        # Create anchor node
        anchor = MemoryNode(
            node_type=MemoryNodeType.WORLD,
            category=original.category,
            title=f"[ANCHOR] {original.title[:100]}",
            content=summary,
            summary=summary,
            importance=original.importance,
            compaction_level=1,
            original_id=uuid.UUID(original_id),
            tier=original.tier,
            department=original.department,
            agent_id=original.agent_id,
            session_id=original.session_id,
            tags=original.tags.copy(),
        )

        # Save anchor to store
        anchor = self.store.create_node(anchor)

        # Transfer edges from original to anchor
        edges = self.store.get_edges(original_id)
        for edge in edges:
            # Create new edge with anchor as source or target
            if str(edge.source_id) == original_id:
                new_edge = MemoryEdge(
                    relation_type=edge.relation_type,
                    source_id=anchor.id,
                    target_id=edge.target_id,
                    strength=edge.strength,
                    bidirectional=edge.bidirectional,
                    pattern=edge.pattern,
                    context=edge.context,
                )
                self.store.create_edge(new_edge)

            if str(edge.target_id) == original_id:
                new_edge = MemoryEdge(
                    relation_type=edge.relation_type,
                    source_id=edge.source_id,
                    target_id=anchor.id,
                    strength=edge.strength,
                    bidirectional=edge.bidirectional,
                    pattern=edge.pattern,
                    context=edge.context,
                )
                self.store.create_edge(new_edge)

        logger.info(f"Created memory anchor {anchor.id} for original {original_id}")
        return anchor

    def compact_and_anchor(
        self,
        max_importance: float = 0.5,
        max_nodes: int = 100,
        create_anchors: bool = True,
    ) -> dict:
        """Perform compaction by creating anchors and marking originals as compacted.

        Gets compaction candidates, optionally creates anchor nodes to preserve
        relationships, and marks original nodes as compacted.

        Args:
            max_importance: Maximum importance threshold for candidates.
            max_nodes: Maximum number of nodes to compact.
            create_anchors: Whether to create anchor nodes (preserves relationships).

        Returns:
            Dictionary with compaction results including counts.
        """
        # Get candidates
        candidates = self.get_compaction_candidates(
            max_importance=max_importance,
            max_results=max_nodes,
        )

        anchors_created = 0
        originals_compacted = 0

        for node in candidates[:max_nodes]:
            if create_anchors:
                # Create anchor to preserve relationships
                anchor = self.create_memory_anchor(str(node.id))
                if anchor is not None:
                    anchors_created += 1

            # Mark original as compacted (level 2 = fully compressed)
            node.compaction_level = 2
            self.store.update_node(node)
            originals_compacted += 1

        logger.info(
            f"Compaction complete: {anchors_created} anchors created, "
            f"{originals_compacted} nodes compacted"
        )

        return {
            "compacted_count": originals_compacted,
            "anchors_created": anchors_created,
            "max_importance": max_importance,
            "max_nodes": max_nodes,
        }

    def _generate_summary(self, content: str) -> str:
        """Generate a simple summary of the content.

        Takes the first 2 sentences from the content, capped at 200 characters.

        Args:
            content: The content to summarize.

        Returns:
            A summary string (max 200 characters).
        """
        if not content:
            return ""

        # Simple sentence splitting on period
        sentences = content.split(".")
        summary_parts = []
        char_count = 0

        for sentence in sentences[:2]:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back if there's more content
            if char_count + len(sentence) + 1 <= 200:
                if summary_parts:
                    summary_parts.append(".")
                summary_parts.append(sentence)
                char_count += len(sentence) + 1
            else:
                # Truncate last sentence if needed
                remaining = 200 - char_count - 1
                if remaining > 10:
                    if summary_parts:
                        summary_parts.append(".")
                    summary_parts.append(sentence[:remaining])
                break

        result = "".join(summary_parts).strip()

        # If result is empty or very short, just truncate the content
        if len(result) < 10 and len(content) > 200:
            result = content[:197] + "..."

        return result
