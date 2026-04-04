"""Memory Tier Manager for Hot/Warm/Cold tier management.

This module provides the MemoryTierManager class for managing memory nodes
across Hot, Warm, and Cold storage tiers based on access patterns and age.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.memory.graph.store import GraphMemoryStore
from src.memory.graph.types import MemoryNode, MemoryTier

logger = logging.getLogger(__name__)


class MemoryTierManager:
    """Manager for memory tier operations.

    Provides methods to query, move, and compact memory nodes across
    Hot, Warm, and Cold storage tiers.
    """

    # Tier thresholds
    HOT_THRESHOLD_HOURS: int = 1
    WARM_THRESHOLD_DAYS: int = 30

    def __init__(self, store: GraphMemoryStore) -> None:
        """Initialize the tier manager.

        Args:
            store: The GraphMemoryStore instance to manage.
        """
        self.store = store

    def _make_aware(self, dt: Optional[datetime]) -> Optional[datetime]:
        """Ensure datetime is timezone-aware (UTC)."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def get_hot_nodes(self, limit: int = 50) -> list[MemoryNode]:
        """Get nodes in the hot tier.

        Hot tier contains nodes accessed within the last hour.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the hot tier.
        """
        threshold = datetime.now(timezone.utc) - timedelta(hours=self.HOT_THRESHOLD_HOURS)
        nodes = self.store.query_nodes(limit=limit * 2)

        hot_nodes = []
        for node in nodes:
            if node.tier == MemoryTier.HOT:
                hot_nodes.append(node)
                if len(hot_nodes) >= limit:
                    break

        # Also include nodes that were recently accessed
        if len(hot_nodes) < limit:
            for node in nodes:
                last_accessed = self._make_aware(node.last_accessed_utc)
                if last_accessed and last_accessed >= threshold:
                    if node not in hot_nodes:
                        hot_nodes.append(node)
                        if len(hot_nodes) >= limit:
                            break

        logger.debug(f"Retrieved {len(hot_nodes)} hot nodes")
        return hot_nodes

    def get_warm_nodes(self, limit: int = 100) -> list[MemoryNode]:
        """Get nodes in the warm tier.

        Warm tier contains nodes not in hot or cold tiers.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the warm tier.
        """
        all_nodes = self.store.query_nodes(limit=limit * 2)

        warm_nodes = []
        for node in all_nodes:
            if node.tier == MemoryTier.WARM:
                warm_nodes.append(node)
                if len(warm_nodes) >= limit:
                    break

        logger.debug(f"Retrieved {len(warm_nodes)} warm nodes")
        return warm_nodes

    def get_cold_nodes(self, limit: int = 100) -> list[MemoryNode]:
        """Get nodes in the cold tier.

        Cold tier contains archived, rarely accessed nodes.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the cold tier.
        """
        all_nodes = self.store.query_nodes(limit=limit * 2)

        cold_nodes = []
        for node in all_nodes:
            if node.tier == MemoryTier.COLD:
                cold_nodes.append(node)
                if len(cold_nodes) >= limit:
                    break

        logger.debug(f"Retrieved {len(cold_nodes)} cold nodes")
        return cold_nodes

    def get_nodes_by_tier(self, tier: MemoryTier, limit: int = 100) -> list[MemoryNode]:
        """Get nodes by specific tier.

        Args:
            tier: The MemoryTier to filter by.
            limit: Maximum number of nodes to return.

        Returns:
            List of MemoryNode objects in the specified tier.
        """
        if tier == MemoryTier.HOT:
            return self.get_hot_nodes(limit)
        elif tier == MemoryTier.WARM:
            return self.get_warm_nodes(limit)
        elif tier == MemoryTier.COLD:
            return self.get_cold_nodes(limit)

        logger.warning(f"Unknown tier: {tier}")
        return []

    def move_to_hot(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the hot tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        node = self.store.get_node(str(node_id))
        if node is None:
            logger.warning(f"Node not found: {node_id}")
            return None

        node.tier = MemoryTier.HOT
        node.last_accessed_utc = datetime.now(timezone.utc)
        node.access_count += 1

        updated = self.store.update_node(node)
        logger.info(f"Moved node {node_id} to hot tier")
        return updated

    def move_to_warm(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the warm tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        node = self.store.get_node(str(node_id))
        if node is None:
            logger.warning(f"Node not found: {node_id}")
            return None

        node.tier = MemoryTier.WARM

        updated = self.store.update_node(node)
        logger.info(f"Moved node {node_id} to warm tier")
        return updated

    def move_to_cold(self, node_id: str) -> Optional[MemoryNode]:
        """Move a node to the cold tier.

        Args:
            node_id: ID of the node to move.

        Returns:
            The updated node, or None if not found.
        """
        node = self.store.get_node(str(node_id))
        if node is None:
            logger.warning(f"Node not found: {node_id}")
            return None

        node.tier = MemoryTier.COLD

        updated = self.store.update_node(node)
        logger.info(f"Moved node {node_id} to cold tier")
        return updated

    def promote_hot(self, node_id: str) -> Optional[MemoryNode]:
        """Promote a node to hot tier (mark as accessed).

        This method moves a node from cold to hot tier and marks it as recently accessed.

        Args:
            node_id: ID of the node to promote.

        Returns:
            The updated node, or None if not found.
        """
        node = self.store.get_node(str(node_id))
        if node is None:
            logger.warning(f"Node not found: {node_id}")
            return None

        node.tier = MemoryTier.HOT
        node.last_accessed_utc = datetime.now(timezone.utc)
        node.access_count += 1

        updated = self.store.update_node(node)
        logger.info(f"Promoted node {node_id} to hot tier")
        return updated

    def compact_old_nodes(
        self,
        days_old: int = 30,
        max_nodes: int = 100,
        min_importance: float = 0.3,
    ) -> dict:
        """Compact old low-importance nodes to cold tier.

        Moves old nodes with low importance to cold storage to free up
        space in faster storage tiers.

        Args:
            days_old: Age threshold in days to consider a node old.
            max_nodes: Maximum number of nodes to compact.
            min_importance: Maximum importance threshold (nodes with importance
                          below this value will be moved to cold).

        Returns:
            Dictionary with compaction results.
        """
        threshold_date = datetime.now(timezone.utc) - timedelta(days=days_old)

        # Query nodes that are old and have low importance
        all_nodes = self.store.query_nodes(min_importance=0.0, limit=max_nodes * 2)

        nodes_to_compact = []
        for node in all_nodes:
            # Check if node is old enough (make datetime aware for comparison)
            created_at = self._make_aware(node.created_at_utc)
            is_old = (
                created_at < threshold_date
                if created_at
                else False
            )
            # Check if node has low importance
            has_low_importance = node.importance < min_importance

            # Only compact warm nodes that are old and have low importance
            if is_old and has_low_importance and node.tier == MemoryTier.WARM:
                nodes_to_compact.append(node)

        nodes_compacted = 0
        for node in nodes_to_compact[:max_nodes]:
            self.move_to_cold(str(node.id))
            nodes_compacted += 1

        logger.info(f"Compacted {nodes_compacted} old nodes to cold tier")
        return {
            "nodes_compacted": nodes_compacted,
            "days_old": days_old,
            "min_importance": min_importance,
        }

    def get_tier_stats(self) -> dict[MemoryTier, int]:
        """Get statistics for each tier.

        Returns:
            Dictionary mapping MemoryTier to count of nodes in that tier.
        """
        all_nodes = self.store.query_nodes(limit=10000)

        stats: dict[MemoryTier, int] = {
            MemoryTier.HOT: 0,
            MemoryTier.WARM: 0,
            MemoryTier.COLD: 0,
        }

        for node in all_nodes:
            if node.tier in stats:
                stats[node.tier] += 1

        logger.debug(f"Tier stats: {stats}")
        return stats
