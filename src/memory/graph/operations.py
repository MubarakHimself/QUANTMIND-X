"""Memory operations for graph-based memory system.

This module implements the Retain/Recall/Reflect operations based on the
Hindsight paper for the graph-based memory system.
"""
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    RelationType,
)

logger = logging.getLogger(__name__)

# Keywords for content categorization (includes variants)
EXPERIENTIAL_KEYWORDS = {
    "executed",
    "traded",
    "bought",
    "sold",
    "closed",
    "opened",
    "order",
    "entered",
    "exited",
    "filled",
    "cancelled",
    "modified",
    "set",
    "adjusted",
    "execute",
    "buy",
    "sell",
}

SUBJECTIVE_KEYWORDS = {
    "prefer",
    "prefers",
    "preferred",
    "thinking",
    "think",
    "believe",
    "believes",
    "believed",
    "feel",
    "feels",
    "feeling",
    "opinion",
    "opinions",
    "might",
    "probably",
    "would",
    "should",
    "could",
    "maybe",
    "seems",
    "appears",
    "hope",
    "wish",
}


class MemoryOperations:
    """Memory operations implementing Retain/Recall/Reflect based on Hindsight.

    This class provides the core memory operations:
    - RETAIN: Store new memories
    - RECALL: Retrieve memories
    - REFLECT: Synthesize answers from memories
    """

    def __init__(self, store: GraphMemoryStore) -> None:
        """Initialize memory operations.

        Args:
            store: GraphMemoryStore instance for persistence.
        """
        self.store = store

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
        category: Optional[MemoryCategory] = None,
    ) -> MemoryNode:
        """Store a new memory (RETAIN operation).

        Args:
            content: The memory content to store.
            source: Source of the memory (e.g., 'user_preference', 'trade_execution').
            department: Department associated with this memory.
            agent_id: Agent that created this memory.
            session_id: Session where this memory was created.
            importance: Importance score (0-1).
            tags: List of tags for categorization.
            related_to: List of node IDs this memory relates to.
            category: Optional explicit category (auto-categorized if not provided).

        Returns:
            The created MemoryNode.
        """
        # Auto-categorize if not provided
        if category is None:
            category = self._categorize_content(content)

        # Determine node type based on category and source
        node_type = self._determine_node_type(category, source)

        # Extract title from content
        title = self._extract_title(content)

        # Create the memory node
        node = MemoryNode(
            node_type=node_type,
            category=category,
            title=title,
            content=content,
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            importance=importance,
            tags=tags or [],
            created_by=agent_id,
            relevance_score=importance,
        )

        # Store the node
        created_node = self.store.create_node(node)

        # Create edges to related nodes
        if related_to:
            self.link_nodes(created_node.id, related_to)

        self._log_operation(
            "retain",
            {
                "node_id": str(created_node.id),
                "category": category.value,
                "department": department,
            },
        )

        return created_node

    def recall(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        tags: Optional[list[str]] = None,
        node_type: Optional[MemoryNodeType] = None,
        min_importance: float = 0.0,
        limit: int = 10,
        traverse: bool = False,
        traverse_depth: int = 2,
    ) -> list[MemoryNode]:
        """Retrieve memories (RECALL operation).

        Args:
            query: Text search query.
            department: Filter by department.
            agent_id: Filter by agent ID.
            session_id: Filter by session ID.
            role: Filter by role.
            tags: Filter by tags (any match).
            node_type: Filter by node type.
            min_importance: Minimum importance score.
            limit: Maximum number of results.
            traverse: Whether to perform graph traversal.
            traverse_depth: Depth for graph traversal.

        Returns:
            List of MemoryNode objects sorted by relevance.
        """
        # Extract keywords from query for flexible matching
        query_keywords = self._extract_keywords(query) if query else set()

        # Query the store without text query first (for other filters)
        nodes = self.store.query_nodes(
            query=None,  # Handle query separately for keyword matching
            department=department,
            agent_id=agent_id,
            session_id=session_id,
            role=role,
            tags=tags,
            node_type=node_type,
            min_importance=min_importance,
            limit=limit * 3,  # Get more to filter
        )

        # Filter by query keywords if provided
        if query_keywords:
            matched_nodes = []
            for node in nodes:
                content_words = self._extract_keywords(node.content)
                title_words = self._extract_keywords(node.title)
                # Check if any query keyword matches content or title
                # Use both exact match and substring match
                matched = False
                for kw in query_keywords:
                    if kw in content_words or kw in title_words:
                        matched = True
                        break
                    # Also check substring match (e.g., "facts" matches "fact")
                    if not matched:
                        for cw in content_words:
                            if kw in cw or cw in kw:
                                matched = True
                                break
                    if not matched:
                        for tw in title_words:
                            if kw in tw or tw in kw:
                                matched = True
                                break
                if matched:
                    matched_nodes.append(node)
            nodes = matched_nodes

        # Perform graph traversal if requested
        if traverse and nodes:
            nodes = self._traverse_graph(nodes, depth=traverse_depth)

        # Update access statistics
        now = datetime.now(timezone.utc)
        for node in nodes:
            node.access_count += 1
            node.last_accessed = now
            self.store.update_node(node)

        # Sort by relevance score and importance
        nodes.sort(key=lambda n: (n.relevance_score, n.importance), reverse=True)

        self._log_operation(
            "recall",
            {
                "query": query,
                "department": department,
                "results_count": len(nodes),
            },
        )

        return nodes[:limit]

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
        # Retrieve relevant memories
        memories = self.recall(
            query=query,
            department=department,
            agent_id=agent_id,
            limit=5,
        )

        if not memories:
            return {
                "answer": "No relevant memories found.",
                "sources": [],
                "synthesized": False,
            }

        # Synthesize answer (simple concatenation for now - no LLM)
        answer = self._synthesize_answer(query, memories, context)

        self._log_operation(
            "reflect",
            {
                "query": query,
                "department": department,
                "memories_used": len(memories),
            },
        )

        return {
            "answer": answer,
            "sources": [str(node.id) for node in memories],
            "synthesized": True,
        }

    def link_nodes(
        self,
        source_id: str,
        target_ids: list[str],
        relation_type: str = RelationType.RELATED_TO,
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
        edges = []
        for target_id in target_ids:
            edge = MemoryEdge(
                relation_type=relation_type,
                source_id=source_id,
                target_id=target_id,
                strength=strength,
            )
            created_edge = self.store.create_edge(edge)
            edges.append(created_edge)
            logger.debug(f"Linked {source_id} to {target_id}")

        return edges

    def _categorize_content(self, content: str) -> MemoryCategory:
        """Categorize content based on keywords.

        Args:
            content: The content to categorize.

        Returns:
            MemoryCategory based on keyword analysis.
        """
        content_lower = content.lower()
        words = set(re.findall(r"\w+", content_lower))

        # Check for experiential keywords
        if words & EXPERIENTIAL_KEYWORDS:
            return MemoryCategory.EXPERIENTIAL

        # Check for subjective keywords
        if words & SUBJECTIVE_KEYWORDS:
            return MemoryCategory.SUBJECTIVE

        # Default to factual
        return MemoryCategory.FACTUAL

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text for matching.

        Args:
            text: Text to extract keywords from.

        Returns:
            Set of lowercase keywords.
        """
        if not text:
            return set()
        # Extract words, convert to lowercase
        words = re.findall(r"\w+", text.lower())
        # Filter out common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "must", "shall",
                      "can", "need", "dare", "ought", "used", "to", "of", "in",
                      "for", "on", "with", "at", "by", "from", "as", "into",
                      "through", "during", "before", "after", "above", "below",
                      "between", "under", "again", "further", "then", "once",
                      "that", "this", "these", "those", "what", "which", "who",
                      "whom", "whose", "where", "when", "why", "how", "all",
                      "each", "every", "both", "few", "more", "most", "other",
                      "some", "such", "no", "nor", "not", "only", "own", "same",
                      "so", "than", "too", "very", "just", "and", "but", "or",
                      "if", "because", "until", "while", "about", "against",
                      "it", "its", "i", "you", "he", "she", "we", "they", "me",
                      "him", "her", "us", "them", "my", "your", "his", "our",
                      "their", "mine", "yours", "hers", "ours", "theirs", "am"}
        return {w for w in words if w not in stop_words and len(w) > 1}

    def _extract_title(self, content: str) -> str:
        """Extract title from content.

        Uses the first sentence or first N characters as the title.

        Args:
            content: The content to extract title from.

        Returns:
            Extracted title string.
        """
        # Find first sentence
        sentence_end = re.search(r"[.!?]\s", content)
        if sentence_end:
            title = content[: sentence_end.start()].strip()
        else:
            title = content.strip()

        # Truncate if too long
        if len(title) > 100:
            title = title[:97] + "..."

        return title

    def _determine_node_type(
        self,
        category: MemoryCategory,
        source: str,
    ) -> MemoryNodeType:
        """Determine appropriate node type based on category and source.

        Args:
            category: Memory category.
            source: Source of the memory.

        Returns:
            Appropriate MemoryNodeType.
        """
        source_lower = source.lower()

        # Map specific sources to node types
        if "trade" in source_lower or "execution" in source_lower:
            return MemoryNodeType.EPISODIC
        if "preference" in source_lower or "opinion" in source_lower:
            return MemoryNodeType.OPINION
        if "conversation" in source_lower or "chat" in source_lower:
            return MemoryNodeType.CONVERSATION
        if "decision" in source_lower:
            return MemoryNodeType.DECISION

        # Default based on category
        if category == MemoryCategory.EXPERIENTIAL:
            return MemoryNodeType.EPISODIC
        elif category == MemoryCategory.SUBJECTIVE:
            return MemoryNodeType.OPINION
        else:
            return MemoryNodeType.BANK

    def _synthesize_answer(
        self,
        query: str,
        memories: list[MemoryNode],
        context: Optional[str] = None,
    ) -> str:
        """Synthesize answer from memories (simple concatenation).

        Args:
            query: The original query.
            memories: Retrieved memories.
            context: Additional context.

        Returns:
            Synthesized answer string.
        """
        if not memories:
            return "No relevant information found."

        # Simple synthesis: combine relevant content
        parts = []

        if context:
            parts.append(f"Context: {context}")

        parts.append("Relevant memories:")
        for i, memory in enumerate(memories, 1):
            parts.append(f"{i}. {memory.content}")

        return "\n".join(parts)

    def _traverse_graph(
        self,
        nodes: list[MemoryNode],
        depth: int = 2,
    ) -> list[MemoryNode]:
        """Perform graph traversal to find related nodes.

        Args:
            nodes: Starting nodes.
            depth: Traversal depth.

        Returns:
            List of nodes including traversed ones.
        """
        visited = {node.id for node in nodes}
        result = list(nodes)

        current_level = list(nodes)

        for _ in range(depth - 1):
            next_level = []
            for node in current_level:
                edges = self.store.get_edges(str(node.id))
                for edge in edges:
                    # Get the other node in the edge
                    neighbor_id = (
                        edge.target_id
                        if str(edge.source_id) == str(node.id)
                        else edge.source_id
                    )
                    if neighbor_id not in visited:
                        neighbor = self.store.get_node(str(neighbor_id))
                        if neighbor:
                            visited.add(neighbor_id)
                            next_level.append(neighbor)
                            result.append(neighbor)

            current_level = next_level
            if not current_level:
                break

        return result

    def _log_operation(
        self,
        operation: str,
        details: dict[str, Any],
    ) -> None:
        """Log memory operation for tracking.

        Args:
            operation: Operation name (retain, recall, reflect).
            details: Operation details.
        """
        logger.info(f"Memory operation: {operation} - {details}")
