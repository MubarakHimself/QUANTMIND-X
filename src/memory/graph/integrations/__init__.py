"""Department Mail Integration for Graph-Based Memory System.

This module provides integration between the DepartmentMail service
and the graph-based memory system, allowing messages to be stored
and retrieved from the memory graph.
"""
import logging
from typing import Any, Optional

from src.agents.departments.department_mail import (
    DepartmentMessage,
    MessageType,
)
from src.memory.graph.facade import GraphMemoryFacade

logger = logging.getLogger(__name__)


class DepartmentMailIntegration:
    """Integration layer between DepartmentMail and GraphMemory.

    This class provides methods to:
    - Convert department mail messages to memory nodes
    - Convert message threads to linked memory nodes
    - Link related messages in the graph
    - Search mail memories
    """

    def __init__(self, facade: GraphMemoryFacade) -> None:
        """Initialize the integration.

        Args:
            facade: GraphMemoryFacade instance for storing memories.
        """
        self.facade = facade

    def convert_message_to_memory(
        self,
        message: DepartmentMessage,
        importance: float = 0.5,
    ) -> str:
        """Convert a department mail message to a memory node.

        Args:
            message: The DepartmentMessage to convert.
            importance: Importance score for the memory (0-1).

        Returns:
            The ID of the created memory node.
        """
        # Build content from message
        content_parts = [
            f"Subject: {message.subject}",
            f"From: {message.from_dept}",
            f"To: {message.to_dept}",
            f"Type: {message.type.value}",
            f"Priority: {message.priority.value}",
            "",
            f"Message: {message.body}",
        ]

        if message.workflow_id:
            content_parts.extend([
                "",
                f"Workflow ID: {message.workflow_id}",
            ])

        if message.gate_id:
            content_parts.extend([
                f"Gate ID: {message.gate_id}",
            ])

        content = "\n".join(content_parts)

        # Build tags
        tags = [
            "department_mail",
            f"msg-{message.id}",
            message.type.value,
            message.priority.value,
        ]

        if message.workflow_id:
            tags.append(f"workflow:{message.workflow_id}")

        if message.gate_id:
            tags.append(f"gate:{message.gate_id}")

        # Store in graph memory
        node_id = self.facade.retain(
            content=content,
            source="department_mail",
            department=message.from_dept,
            importance=importance,
            tags=tags,
        )

        logger.info(f"Converted message {message.id} to memory node {node_id}")

        return node_id

    def convert_thread_to_memory(
        self,
        messages: list[DepartmentMessage],
        thread_id: str,
    ) -> list[str]:
        """Convert a message thread to linked memory nodes.

        Creates a node for each message and links them in chronological order.

        Args:
            messages: List of messages in the thread (should be ordered by time).
            thread_id: Unique identifier for the thread.

        Returns:
            List of created node IDs.
        """
        if not messages:
            return []

        # Sort messages by timestamp
        sorted_messages = sorted(messages, key=lambda m: m.timestamp)

        node_ids = []
        previous_node_id = None

        for i, message in enumerate(sorted_messages):
            # Determine importance based on position and type
            if i == 0:
                importance = 0.7  # First message is usually important
            elif i == len(sorted_messages) - 1:
                importance = 0.6  # Last message (response) is important
            else:
                importance = 0.5

            # Add thread-specific tags
            base_tags = [f"thread:{thread_id}"]

            # Store in graph memory
            node_id = self.facade.retain(
                content=self._build_message_content(message),
                source="department_mail_thread",
                department=message.from_dept,
                importance=importance,
                tags=base_tags,
            )

            node_ids.append(node_id)

            # Link to previous message
            if previous_node_id:
                self.facade.link(
                    source_id=previous_node_id,
                    target_ids=[node_id],
                    relation_type="next_in_thread",
                    strength=0.9,
                )

            previous_node_id = node_id

        logger.info(
            f"Converted thread {thread_id} with {len(node_ids)} messages"
        )

        return node_ids

    def link_messages(
        self,
        source_id: str,
        target_id: str,
        relation: str = "related_to",
        context: Optional[str] = None,
    ) -> list[Any]:
        """Link two message nodes in the graph.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Type of relation (e.g., 'replies_to', 'references').
            context: Optional context about the relationship.

        Returns:
            List of created edges.
        """
        # Determine relation type based on relation string
        relation_type = self._map_relation_type(relation)

        edges = self.facade.link(
            source_id=source_id,
            target_ids=[target_id],
            relation_type=relation_type,
            strength=0.8,
        )

        logger.info(f"Linked {source_id} to {target_id} with relation '{relation}'")

        return edges

    def search_mail_memory(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[Any]:
        """Search memories created from department mail.

        Args:
            query: Text search query.
            department: Filter by department.
            message_type: Filter by message type.
            min_importance: Minimum importance score.
            limit: Maximum number of results.

        Returns:
            List of matching MemoryNode objects.
        """
        # Build tags filter
        tags = ["department_mail"]

        if message_type:
            tags.append(message_type.value)

        # Search using recall
        results = self.facade.recall(
            query=query,
            department=department,
            tags=tags if len(tags) > 1 else None,
            min_importance=min_importance,
            limit=limit,
        )

        logger.info(
            f"Searched mail memories: query={query}, dept={department}, "
            f"type={message_type}, results={len(results)}"
        )

        return results

    def _build_message_content(self, message: DepartmentMessage) -> str:
        """Build content string for a message.

        Args:
            message: The message to build content for.

        Returns:
            Formatted content string.
        """
        parts = [
            f"Subject: {message.subject}",
            f"From: {message.from_dept}",
            f"To: {message.to_dept}",
            f"Type: {message.type.value}",
            "",
            f"Message: {message.body}",
        ]

        return "\n".join(parts)

    def _map_relation_type(self, relation: str) -> str:
        """Map relation string to graph relation type.

        Args:
            relation: Relation string.

        Returns:
            Mapped relation type.
        """
        relation_map = {
            "replies_to": "replies_to",
            "references": "references",
            "related_to": "related_to",
            "forwarded_from": "forwarded_from",
            "delegates_to": "delegates_to",
        }

        return relation_map.get(relation, "related_to")


__all__ = ["DepartmentMailIntegration"]
