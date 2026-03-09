"""Graph-based memory store for QuantMindX memory system.

This module provides the GraphMemoryStore class for storing and querying
memory nodes and edges in a SQLite database.
"""
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
)

logger = logging.getLogger(__name__)


class GraphMemoryStore:
    """Store for graph-based memory nodes and edges.

    Provides persistence for MemoryNode and MemoryEdge objects using SQLite.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize the store.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        """Create database tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT DEFAULT '',
                evidence TEXT DEFAULT '',
                department TEXT,
                agent_id TEXT,
                session_id TEXT,
                role TEXT,
                tags TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.0,
                relevance_score REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_by TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT,
                event_time TEXT,
                tier TEXT DEFAULT 'hot',
                compaction_level INTEGER DEFAULT 0,
                original_id TEXT,
                entity_id TEXT
            )
        """)

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                relation_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                bidirectional INTEGER DEFAULT 0,
                pattern TEXT,
                context TEXT,
                created_at TEXT NOT NULL,
                traversal_count INTEGER DEFAULT 0,
                last_traversed TEXT,
                FOREIGN KEY (source_id) REFERENCES nodes(id),
                FOREIGN KEY (target_id) REFERENCES nodes(id)
            )
        """)

        # Create indexes for common queries
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_department
            ON nodes(department)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_agent_id
            ON nodes(agent_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_session_id
            ON nodes(session_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_importance
            ON nodes(importance)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_source
            ON edges(source_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_edges_target
            ON edges(target_id)
        """)

        self._conn.commit()

    def create_node(self, node: MemoryNode) -> MemoryNode:
        """Create a new memory node.

        Args:
            node: MemoryNode to create.

        Returns:
            The created MemoryNode with updated timestamps.
        """
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO nodes (
                id, node_type, category, title, content, summary, evidence,
                department, agent_id, session_id, role, tags, importance,
                relevance_score, access_count, last_accessed, created_by,
                created_at, updated_at, expires_at, event_time, tier,
                compaction_level, original_id, entity_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(node.id),
                node.node_type.value,
                node.category.value,
                node.title,
                node.content,
                node.summary,
                node.evidence,
                node.department,
                node.agent_id,
                node.session_id,
                node.role,
                ",".join(node.tags) if node.tags else "",
                node.importance,
                node.relevance_score,
                node.access_count,
                node.last_accessed.isoformat() if node.last_accessed else None,
                node.created_by,
                node.created_at.isoformat(),
                now,
                node.expires_at.isoformat() if node.expires_at else None,
                node.event_time.isoformat() if node.event_time else None,
                node.tier.value,
                node.compaction_level,
                str(node.original_id) if node.original_id else None,
                node.entity_id,
            ),
        )
        self._conn.commit()
        node.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Created node: {node.id}")
        return node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Get a node by ID.

        Args:
            node_id: UUID of the node.

        Returns:
            MemoryNode if found, None otherwise.
        """
        cursor = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?",
            (node_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._row_to_node(row)

    def update_node(self, node: MemoryNode) -> MemoryNode:
        """Update an existing node.

        Args:
            node: MemoryNode to update.

        Returns:
            The updated MemoryNode.
        """
        now = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            UPDATE nodes SET
                node_type = ?, category = ?, title = ?, content = ?,
                summary = ?, evidence = ?, department = ?, agent_id = ?,
                session_id = ?, role = ?, tags = ?, importance = ?,
                relevance_score = ?, access_count = ?, last_accessed = ?,
                created_by = ?, updated_at = ?, expires_at = ?, event_time = ?,
                tier = ?, compaction_level = ?, original_id = ?, entity_id = ?
            WHERE id = ?
            """,
            (
                node.node_type.value,
                node.category.value,
                node.title,
                node.content,
                node.summary,
                node.evidence,
                node.department,
                node.agent_id,
                node.session_id,
                node.role,
                ",".join(node.tags) if node.tags else "",
                node.importance,
                node.relevance_score,
                node.access_count,
                node.last_accessed.isoformat() if node.last_accessed else None,
                node.created_by,
                now,
                node.expires_at.isoformat() if node.expires_at else None,
                node.event_time.isoformat() if node.event_time else None,
                node.tier.value,
                node.compaction_level,
                str(node.original_id) if node.original_id else None,
                node.entity_id,
                str(node.id),
            ),
        )
        self._conn.commit()
        node.updated_at = datetime.now(timezone.utc)
        logger.debug(f"Updated node: {node.id}")
        return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges.

        Args:
            node_id: UUID of the node.

        Returns:
            True if deleted, False if not found.
        """
        # Delete edges first
        self._conn.execute(
            "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
            (node_id, node_id),
        )
        cursor = self._conn.execute(
            "DELETE FROM nodes WHERE id = ?",
            (node_id,),
        )
        self._conn.commit()
        logger.debug(f"Deleted node: {node_id}")
        return cursor.rowcount > 0

    def create_edge(self, edge: MemoryEdge) -> MemoryEdge:
        """Create a new memory edge.

        Args:
            edge: MemoryEdge to create.

        Returns:
            The created MemoryEdge.
        """
        self._conn.execute(
            """
            INSERT INTO edges (
                id, relation_type, source_id, target_id, strength,
                bidirectional, pattern, context, created_at,
                traversal_count, last_traversed
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(edge.id),
                edge.relation_type,
                str(edge.source_id),
                str(edge.target_id),
                edge.strength,
                1 if edge.bidirectional else 0,
                edge.pattern,
                edge.context,
                edge.created_at.isoformat(),
                edge.traversal_count,
                edge.last_traversed.isoformat() if edge.last_traversed else None,
            ),
        )
        self._conn.commit()
        logger.debug(f"Created edge: {edge.id}")
        return edge

    def get_edges(self, node_id: str) -> list[MemoryEdge]:
        """Get all edges connected to a node.

        Args:
            node_id: UUID of the node.

        Returns:
            List of MemoryEdge objects.
        """
        cursor = self._conn.execute(
            """
            SELECT * FROM edges
            WHERE source_id = ? OR target_id = ?
            """,
            (node_id, node_id),
        )
        return [self._row_to_edge(row) for row in cursor.fetchall()]

    def query_nodes(
        self,
        query: Optional[str] = None,
        department: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        tags: Optional[list[str]] = None,
        node_type: Optional[MemoryNodeType] = None,
        min_importance: float = 0.0,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Query memory nodes with filters.

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

        Returns:
            List of matching MemoryNode objects.
        """
        sql_parts = ["1=1"]
        params: list[Any] = []

        if query:
            sql_parts.append("(content LIKE ? OR title LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])

        if department:
            sql_parts.append("department = ?")
            params.append(department)

        if agent_id:
            sql_parts.append("agent_id = ?")
            params.append(agent_id)

        if session_id:
            sql_parts.append("session_id = ?")
            params.append(session_id)

        if role:
            sql_parts.append("role = ?")
            params.append(role)

        if node_type:
            sql_parts.append("node_type = ?")
            params.append(node_type.value)

        if min_importance > 0:
            sql_parts.append("importance >= ?")
            params.append(min_importance)

        sql = f"SELECT * FROM nodes WHERE {' AND '.join(sql_parts)} ORDER BY importance DESC, relevance_score DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        nodes = [self._row_to_node(row) for row in cursor.fetchall()]

        # Filter by tags if specified
        if tags:
            nodes = [
                node
                for node in nodes
                if any(tag in node.tags for tag in tags)
            ]

        return nodes

    def _row_to_node(self, row: sqlite3.Row) -> MemoryNode:
        """Convert a database row to a MemoryNode."""
        return MemoryNode(
            id=row["id"],
            node_type=MemoryNodeType(row["node_type"]),
            category=MemoryCategory(row["category"]),
            title=row["title"],
            content=row["content"],
            summary=row["summary"] or "",
            evidence=row["evidence"] or "",
            department=row["department"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            role=row["role"],
            tags=row["tags"].split(",") if row["tags"] else [],
            importance=row["importance"],
            relevance_score=row["relevance_score"],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"]) if row["last_accessed"] else None,
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            event_time=datetime.fromisoformat(row["event_time"]) if row["event_time"] else None,
            tier=MemoryTier(row["tier"]),
            compaction_level=row["compaction_level"],
            original_id=row["original_id"] if row["original_id"] else None,
            entity_id=row["entity_id"],
        )

    def _row_to_edge(self, row: sqlite3.Row) -> MemoryEdge:
        """Convert a database row to a MemoryEdge."""
        return MemoryEdge(
            id=row["id"],
            relation_type=row["relation_type"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            strength=row["strength"],
            bidirectional=bool(row["bidirectional"]),
            pattern=row["pattern"],
            context=row["context"],
            created_at=datetime.fromisoformat(row["created_at"]),
            traversal_count=row["traversal_count"],
            last_traversed=datetime.fromisoformat(row["last_traversed"]) if row["last_traversed"] else None,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug(f"Closed database: {self.db_path}")
