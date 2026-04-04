"""Migration Path & Wrapper for Graph-Based Memory System.

This module provides:
1. MemoryMigrator - Migrate entries from one memory system to another
2. DualWriteWrapper - Write to both legacy and graph memory systems
3. add_session_status_column - Add session_status column to existing databases
4. add_embedding_column - Add embedding column to existing databases
5. add_opinion_columns - Add OPINION-specific columns to existing databases
6. rename_timestamps_to_utc - Rename timestamp columns to _utc suffix
"""
import logging
import sqlite3
from typing import Any, Optional

from src.memory.graph.facade import GraphMemoryFacade

logger = logging.getLogger(__name__)


def add_session_status_column(db_path: str) -> bool:
    """Add session_status column to nodes table.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(nodes)")
        columns = [col[1] for col in cursor.fetchall()]

        if "session_status" not in columns:
            cursor.execute(
                "ALTER TABLE nodes ADD COLUMN session_status TEXT DEFAULT 'draft'"
            )
            conn.commit()
            logger.info(f"Added session_status column to {db_path}")
            conn.close()
            return True

        logger.info(f"session_status column already exists in {db_path}")
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add session_status column: {e}")
        return False


def add_embedding_column(db_path: str) -> bool:
    """Add embedding column to nodes table.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(nodes)")
        columns = [col[1] for col in cursor.fetchall()]

        if "embedding" not in columns:
            cursor.execute(
                "ALTER TABLE nodes ADD COLUMN embedding BLOB"
            )
            conn.commit()
            logger.info(f"Added embedding column to {db_path}")
            conn.close()
            return True

        logger.info(f"embedding column already exists in {db_path}")
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add embedding column: {e}")
        return False


def add_opinion_columns(db_path: str) -> bool:
    """Add OPINION-specific columns to nodes table.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if column already exists
        cursor.execute("PRAGMA table_info(nodes)")
        columns = [col[1] for col in cursor.fetchall()]

        opinion_columns = {
            "action": "TEXT",
            "reasoning": "TEXT",
            "confidence": "REAL",
            "alternatives_considered": "TEXT",
            "constraints_applied": "TEXT",
            "agent_role": "TEXT",
        }

        for col_name, col_type in opinion_columns.items():
            if col_name not in columns:
                cursor.execute(
                    f"ALTER TABLE nodes ADD COLUMN {col_name} {col_type}"
                )
                logger.info(f"Added {col_name} column to {db_path}")

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to add opinion columns: {e}")
        return False


def rename_timestamps_to_utc(db_path: str) -> bool:
    """Rename timestamp columns to _utc suffix by recreating the nodes and edges tables.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check current column names in nodes
        cursor.execute("PRAGMA table_info(nodes)")
        node_columns = {col[1] for col in cursor.fetchall()}

        if "created_at_utc" not in node_columns and "created_at" in node_columns:
            logger.info(f"Migrating nodes timestamp columns in {db_path}")
            # Map old → new column names for nodes
            old_to_new = {
                "last_accessed": "last_accessed_utc",
                "created_at": "created_at_utc",
                "updated_at": "updated_at_utc",
                "expires_at": "expires_at_utc",
                "event_time": "event_time_utc",
            }
            # Build SELECT list: rename old cols, keep others, add any missing new cols as NULL
            new_cols = [
                "last_accessed_utc", "created_at_utc", "updated_at_utc",
                "expires_at_utc", "event_time_utc",
                "id", "node_type", "category", "title", "content", "summary",
                "evidence", "department", "agent_id", "session_id", "role", "tags",
                "importance", "relevance_score", "access_count", "created_by",
                "tier", "compaction_level", "original_id", "entity_id",
                "session_status", "embedding", "action", "reasoning", "confidence",
                "alternatives_considered", "constraints_applied", "agent_role",
            ]
            select_parts = []
            for col in new_cols:
                old_name = {v: k for k, v in old_to_new.items()}.get(col, col)
                if col in old_to_new.values():
                    old_name = {v: k for k, v in old_to_new.items()}[col]
                    select_parts.append(f"{old_name} AS {col}")
                elif old_name in node_columns:
                    select_parts.append(col)
                else:
                    select_parts.append(f"NULL AS {col}")

            cursor.execute(f"""
                CREATE TABLE nodes_new AS
                SELECT {', '.join(select_parts)}
                FROM nodes
            """)
            cursor.execute("DROP TABLE nodes")
            cursor.execute("ALTER TABLE nodes_new RENAME TO nodes")
            conn.commit()
            logger.info(f"Nodes timestamp columns migrated in {db_path}")

        # Check edges table
        cursor.execute("PRAGMA table_info(edges)")
        edge_columns = {col[1] for col in cursor.fetchall()}

        if "created_at_utc" not in edge_columns and "created_at" in edge_columns:
            logger.info(f"Migrating edges timestamp columns in {db_path}")
            cursor.execute("""
                CREATE TABLE edges_new AS
                SELECT
                    id, relation_type, source_id, target_id, strength,
                    bidirectional, pattern, context,
                    created_at AS created_at_utc,
                    traversal_count,
                    last_traversed AS last_traversed_utc
                FROM edges
            """)
            cursor.execute("DROP TABLE edges")
            cursor.execute("ALTER TABLE edges_new RENAME TO edges")
            conn.commit()
            logger.info(f"Edges timestamp columns migrated in {db_path}")

        conn.close()
        return True

    except Exception as e:
        logger.error(f"Failed to migrate timestamps: {e}")
        return False


def migrate_graph_memory_db(db_path: str) -> bool:
    """Run all migrations for graph memory database.

    Args:
        db_path: Path to SQLite database file.

    Returns:
        True if all migrations successful, False otherwise.
    """
    success = True
    success = add_session_status_column(db_path) and success
    success = add_embedding_column(db_path) and success
    success = add_opinion_columns(db_path) and success
    success = rename_timestamps_to_utc(db_path) and success

    if success:
        logger.info(f"Completed all migrations for {db_path}")
    else:
        logger.error(f"Some migrations failed for {db_path}")

    return success


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


__all__ = [
    "MemoryMigrator",
    "DualWriteWrapper",
    "add_session_status_column",
    "add_embedding_column",
    "add_opinion_columns",
    "rename_timestamps_to_utc",
    "migrate_graph_memory_db",
]
