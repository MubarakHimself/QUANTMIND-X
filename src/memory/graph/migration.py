"""Migration Path & Wrapper for Graph-Based Memory System.

This module provides:
1. MemoryMigrator - Migrate entries from one memory system to another
2. DualWriteWrapper - Write to both legacy and graph memory systems
"""
import logging
from typing import Any, Optional

from src.memory.graph.facade import GraphMemoryFacade

logger = logging.getLogger(__name__)


class MemoryMigrator:
    """Migrates memory entries from one system to another.

    This class handles migration from legacy memory systems to the
    graph-based memory system.
    """

    def __init__(
        self,
        source_facade: Any,
        target_facade: GraphMemoryFacade,
    ) -> None:
        """Initialize the migrator.

        Args:
            source_facade: Source memory facade (e.g., UnifiedMemoryFacade).
            target_facade: Target GraphMemoryFacade.
        """
        self.source = source_facade
        self.target = target_facade

    def migrate_entries(
        self,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Migrate entries from source to target.

        Args:
            department: Filter by department.
            agent_id: Filter by agent ID.
            session_id: Filter by session ID.
            limit: Maximum entries to migrate.

        Returns:
            Dictionary with migration results.
        """
        migrated_count = 0
        failed_count = 0

        # Retrieve entries from source
        try:
            source_results = self.source.recall(
                department=department,
                agent_id=agent_id,
                session_id=session_id,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Failed to retrieve from source: {e}")
            return {
                "migrated": 0,
                "failed": 0,
                "error": str(e),
            }

        # Migrate each entry
        for entry in source_results:
            try:
                # Extract fields from entry
                content = getattr(entry, "content", None) or getattr(entry, "value", "")
                importance = getattr(entry, "importance", 0.5)
                tags = getattr(entry, "tags", []) or []
                entry_department = getattr(entry, "department", department)
                entry_agent_id = getattr(entry, "agent_id", agent_id)
                entry_session_id = getattr(entry, "session_id", session_id)

                # Store in target
                self.target.retain(
                    content=content,
                    source="migrated",
                    department=entry_department,
                    agent_id=entry_agent_id,
                    session_id=entry_session_id,
                    importance=importance,
                    tags=tags + ["migrated"],
                )

                migrated_count += 1

            except Exception as e:
                logger.error(f"Failed to migrate entry: {e}")
                failed_count += 1

        logger.info(
            f"Migration complete: {migrated_count} migrated, {failed_count} failed"
        )

        return {
            "migrated": migrated_count,
            "failed": failed_count,
            "source_count": len(source_results),
        }

    def migrate_department_entries(
        self,
        department: str,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Migrate all entries for a specific department.

        Args:
            department: Department to migrate.
            limit: Maximum entries to migrate.

        Returns:
            Dictionary with migration results.
        """
        return self.migrate_entries(department=department, limit=limit)

    def verify_migration(self) -> dict[str, Any]:
        """Verify migration by comparing counts.

        Returns:
            Dictionary with verification results.
        """
        try:
            source_stats = self.source.get_stats()
            target_stats = self.target.get_stats()

            source_total = (
                source_stats.get("total_entries", 0) or
                source_stats.get("total_nodes", 0)
            )
            target_total = target_stats.get("total_nodes", 0)

            verified = source_total == target_total

            return {
                "verified": verified,
                "source_total": source_total,
                "target_total": target_total,
            }

        except Exception as e:
            logger.error(f"Failed to verify migration: {e}")
            return {
                "verified": False,
                "error": str(e),
            }


class DualWriteWrapper:
    """Wrapper that writes to both legacy and graph memory systems.

    This class provides a migration path by writing to both systems
    simultaneously, allowing gradual migration with fallback.
    """

    def __init__(
        self,
        legacy_facade: Any,
        graph_facade: GraphMemoryFacade,
    ) -> None:
        """Initialize the wrapper.

        Args:
            legacy_facade: Legacy memory facade (e.g., UnifiedMemoryFacade).
            graph_facade: GraphMemoryFacade.
        """
        self.legacy = legacy_facade
        self.graph = graph_facade

    def add_memory(
        self,
        key: str,
        value: str,
        namespace: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Add memory to both systems.

        Args:
            key: Memory key.
            value: Memory value.
            namespace: Namespace.
            tags: Optional tags.
            metadata: Optional metadata.

        Returns:
            Dictionary with result from both systems.
        """
        legacy_result = None
        graph_result = None

        # Write to legacy
        try:
            legacy_id = self.legacy.add_memory(
                key=key,
                value=value,
                namespace=namespace,
                tags=tags,
                metadata=metadata,
            )
            legacy_result = {"id": legacy_id, "success": True}
        except Exception as e:
            logger.warning(f"Failed to write to legacy: {e}")
            legacy_result = {"error": str(e)}

        # Write to graph
        try:
            graph_id = self.graph.retain(
                content=value,
                source="dual_write",
                importance=0.5,
                tags=tags,
            )
            graph_result = {"id": graph_id, "success": True}
        except Exception as e:
            logger.warning(f"Failed to write to graph: {e}")
            graph_result = {"error": str(e)}

        return {
            "legacy": legacy_result,
            "graph": graph_result,
        }

    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10,
    ) -> list[Any]:
        """Search memories, preferring graph but falling back to legacy.

        Args:
            query: Search query.
            namespace: Namespace filter.
            limit: Maximum results.

        Returns:
            List of search results.
        """
        # Try graph first
        try:
            graph_results = self.graph.recall(
                query=query,
                limit=limit,
            )

            if graph_results:
                return graph_results
        except Exception as e:
            logger.warning(f"Graph search failed: {e}")

        # Fallback to legacy
        try:
            legacy_result = self.legacy.search(
                query=query,
                namespace=namespace,
                limit=limit,
            )

            if hasattr(legacy_result, "entries"):
                return legacy_result.entries

            return legacy_result if isinstance(legacy_result, list) else []

        except Exception as e:
            logger.error(f"Legacy search also failed: {e}")
            return []

    def get_stats(self) -> dict[str, Any]:
        """Get combined stats from both systems.

        Returns:
            Dictionary with combined statistics.
        """
        stats = {"legacy_total": 0, "graph_total": 0}

        try:
            legacy_stats = self.legacy.get_stats()
            stats["legacy_total"] = (
                legacy_stats.get("total_entries", 0) or
                legacy_stats.get("total_nodes", 0)
            )
        except Exception as e:
            logger.warning(f"Failed to get legacy stats: {e}")

        try:
            graph_stats = self.graph.get_stats()
            stats["graph_total"] = graph_stats.get("total_nodes", 0)
        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")

        stats["combined_total"] = stats["legacy_total"] + stats["graph_total"]

        return stats


__all__ = ["MemoryMigrator", "DualWriteWrapper"]
