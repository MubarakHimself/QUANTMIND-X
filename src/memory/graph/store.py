"""Graph-based memory store for QuantMindX memory system.

This module provides the GraphMemoryStore class for storing and querying
memory nodes and edges in a SQLite database.
"""
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.memory.graph.types import (
    MemoryCategory,
    MemoryEdge,
    MemoryNode,
    MemoryNodeType,
    MemoryTier,
    SessionStatus,
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
                last_accessed_utc TEXT,
                created_by TEXT,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                expires_at_utc TEXT,
                event_time_utc TEXT,
                tier TEXT DEFAULT 'hot',
                compaction_level INTEGER DEFAULT 0,
                original_id TEXT,
                entity_id TEXT,
                session_status TEXT DEFAULT 'draft',
                embedding BLOB,
                -- OPINION-specific fields
                action TEXT,
                reasoning TEXT,
                confidence REAL,
                alternatives_considered TEXT,
                constraints_applied TEXT,
                agent_role TEXT
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
                created_at_utc TEXT NOT NULL,
                traversal_count INTEGER DEFAULT 0,
                last_traversed_utc TEXT,
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
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_node_type
            ON nodes(node_type)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_tier
            ON nodes(tier)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_session_status
            ON nodes(session_status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_nodes_session_status_session
            ON nodes(session_status, session_id)
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
                relevance_score, access_count, last_accessed_utc, created_by,
                created_at_utc, updated_at_utc, expires_at_utc, event_time_utc, tier,
                compaction_level, original_id, entity_id, session_status, embedding,
                action, reasoning, confidence, alternatives_considered, constraints_applied, agent_role
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(node.tags) if node.tags else "[]",
                node.importance,
                node.relevance_score,
                node.access_count,
                node.last_accessed_utc.isoformat() if node.last_accessed_utc else None,
                node.created_by,
                node.created_at_utc.isoformat(),
                now,
                node.expires_at_utc.isoformat() if node.expires_at_utc else None,
                node.event_time_utc.isoformat() if node.event_time_utc else None,
                node.tier.value,
                node.compaction_level,
                str(node.original_id) if node.original_id else None,
                node.entity_id,
                node.session_status,
                node.embedding,
                node.action,
                node.reasoning,
                node.confidence,
                node.alternatives_considered,
                node.constraints_applied,
                node.agent_role,
            ),
        )
        self._conn.commit()
        node.updated_at_utc = datetime.now(timezone.utc)
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
                relevance_score = ?, access_count = ?, last_accessed_utc = ?,
                created_by = ?, updated_at_utc = ?, expires_at_utc = ?, event_time_utc = ?,
                tier = ?, compaction_level = ?, original_id = ?, entity_id = ?,
                session_status = ?, embedding = ?,
                action = ?, reasoning = ?, confidence = ?,
                alternatives_considered = ?, constraints_applied = ?, agent_role = ?
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
                json.dumps(node.tags) if node.tags else "[]",
                node.importance,
                node.relevance_score,
                node.access_count,
                node.last_accessed_utc.isoformat() if node.last_accessed_utc else None,
                node.created_by,
                now,
                node.expires_at_utc.isoformat() if node.expires_at_utc else None,
                node.event_time_utc.isoformat() if node.event_time_utc else None,
                node.tier.value,
                node.compaction_level,
                str(node.original_id) if node.original_id else None,
                node.entity_id,
                node.session_status,
                node.embedding,
                node.action,
                node.reasoning,
                node.confidence,
                node.alternatives_considered,
                node.constraints_applied,
                node.agent_role,
                str(node.id),
            ),
        )
        self._conn.commit()
        node.updated_at_utc = datetime.now(timezone.utc)
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
                bidirectional, pattern, context, created_at_utc,
                traversal_count, last_traversed_utc
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
                edge.created_at_utc.isoformat(),
                edge.traversal_count,
                edge.last_traversed_utc.isoformat() if edge.last_traversed_utc else None,
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
        session_status: Optional[str] = None,
        has_embedding: bool = False,
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
            session_status: Filter by session_status ('draft' or 'committed').
            has_embedding: Filter to only nodes with embeddings.
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

        if session_status:
            sql_parts.append("session_status = ?")
            params.append(session_status)

        if has_embedding:
            sql_parts.append("embedding IS NOT NULL")

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

    def get_nodes_by_session_status(
        self,
        session_status: str,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryNode]:
        """Get nodes by session_status filter.

        Args:
            session_status: Filter by session_status ('draft' or 'committed').
            session_id: Optional additional filter by session_id.
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects matching the filter.
        """
        sql_parts = ["session_status = ?"]
        params: list[Any] = [session_status]

        if session_id:
            sql_parts.append("session_id = ?")
            params.append(session_id)

        sql = f"SELECT * FROM nodes WHERE {' AND '.join(sql_parts)} ORDER BY created_at_utc DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        return [self._row_to_node(row) for row in cursor.fetchall()]

    def commit_session(self, session_id: str, department: Optional[str] = None) -> dict[str, Any]:
        """Commit all draft nodes for a session (change status to committed).

        Args:
            session_id: Session ID to commit nodes for.
            department: Optional department to filter nodes.

        Returns:
            Dictionary with commit results: node_count, committed_at, session_id.
        """
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)

        # Build WHERE clause for finding draft nodes
        where_parts = ["session_status = ?", "session_id = ?"]
        where_params: list[Any] = [SessionStatus.DRAFT, session_id]

        if department:
            where_parts.append("department = ?")
            where_params.append(department)

        # Get count first
        count_sql = f"SELECT COUNT(*) as count FROM nodes WHERE {' AND '.join(where_parts)}"
        cursor = self._conn.execute(count_sql, where_params)
        node_count = cursor.fetchone()["count"]

        # Update all matching nodes to committed
        # UPDATE needs: new_session_status, new_updated_at, then the WHERE conditions
        update_params: list[Any] = [SessionStatus.COMMITTED, now.isoformat()] + where_params
        update_sql = f"UPDATE nodes SET session_status = ?, updated_at_utc = ? WHERE {' AND '.join(where_parts)}"
        self._conn.execute(update_sql, update_params)
        self._conn.commit()

        logger.info(f"Committed {node_count} nodes for session {session_id}")

        return {
            "session_id": session_id,
            "node_count": node_count,
            "committed_at_utc": now.isoformat(),
            "department": department,
        }

    def get_commit_log(self, session_id: Optional[str] = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get commit log entries (committed nodes).

        Args:
            session_id: Optional filter by session_id.
            limit: Maximum number of results.

        Returns:
            List of commit log entries (nodes with session_status='committed').
        """
        sql_parts = ["session_status = ?"]
        params: list[Any] = [SessionStatus.COMMITTED]

        if session_id:
            sql_parts.append("session_id = ?")
            params.append(session_id)

        sql = f"SELECT * FROM nodes WHERE {' AND '.join(sql_parts)} ORDER BY updated_at_utc DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.execute(sql, params)
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "title": row["title"],
                "department": row["department"],
                "committed_at_utc": row["updated_at_utc"],
            }
            for row in cursor.fetchall()
        ]

    def detect_conflicts(
        self,
        strategy_id: str,
        exclude_session_id: str,
    ) -> list[dict[str, Any]]:
        """Detect concurrent writes to the same strategy namespace.

        Args:
            strategy_id: Strategy identifier to check for conflicts.
            exclude_session_id: Session ID to exclude (the current session).

        Returns:
            List of conflicting nodes from other sessions.
        """
        # Find draft nodes with the same strategy/entity_id but different session
        cursor = self._conn.execute(
            """
            SELECT * FROM nodes
            WHERE session_status = ?
            AND entity_id = ?
            AND session_id != ?
            ORDER BY created_at_utc DESC
            LIMIT 50
            """,
            (SessionStatus.DRAFT, strategy_id, exclude_session_id),
        )
        return [self._row_to_node(row) for row in cursor.fetchall()]

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
            tags=json.loads(row["tags"]) if row["tags"] else [],
            importance=row["importance"],
            relevance_score=row["relevance_score"],
            access_count=row["access_count"],
            last_accessed_utc=datetime.fromisoformat(row["last_accessed_utc"]) if row["last_accessed_utc"] else None,
            created_by=row["created_by"],
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
            updated_at_utc=datetime.fromisoformat(row["updated_at_utc"]),
            expires_at_utc=datetime.fromisoformat(row["expires_at_utc"]) if row["expires_at_utc"] else None,
            event_time_utc=datetime.fromisoformat(row["event_time_utc"]) if row["event_time_utc"] else None,
            tier=MemoryTier(row["tier"]),
            compaction_level=row["compaction_level"],
            original_id=row["original_id"] if row["original_id"] else None,
            entity_id=row["entity_id"],
            session_status=row["session_status"] or "draft",
            embedding=row["embedding"],
            action=row["action"],
            reasoning=row["reasoning"],
            confidence=row["confidence"],
            alternatives_considered=row["alternatives_considered"],
            constraints_applied=row["constraints_applied"],
            agent_role=row["agent_role"],
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
            created_at_utc=datetime.fromisoformat(row["created_at_utc"]),
            traversal_count=row["traversal_count"],
            last_traversed_utc=datetime.fromisoformat(row["last_traversed_utc"]) if row["last_traversed_utc"] else None,
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
        logger.debug(f"Closed database: {self.db_path}")

    def search_by_embedding(
        self,
        query_embedding: bytes,
        min_similarity: float = 0.5,
        limit: int = 10,
        session_status: Optional[str] = None,
    ) -> list[tuple[MemoryNode, float]]:
        """Search nodes by vector embedding similarity.

        Args:
            query_embedding: Query embedding as bytes.
            min_similarity: Minimum cosine similarity threshold.
            limit: Maximum number of results.
            session_status: Filter by session_status if provided.

        Returns:
            List of (MemoryNode, similarity_score) tuples.
        """
        # Get all nodes with embeddings
        sql_parts = ["embedding IS NOT NULL"]
        params: list[Any] = []

        if session_status:
            sql_parts.append("session_status = ?")
            params.append(session_status)

        sql = f"SELECT * FROM nodes WHERE {' AND '.join(sql_parts)}"
        cursor = self._conn.execute(sql, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Compute similarities
        query_vec = np.frombuffer(query_embedding, dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        results = []
        for row in rows:
            node = self._row_to_node(row)
            if node.embedding:
                node_vec = np.frombuffer(node.embedding, dtype=np.float32)
                node_vec = node_vec / np.linalg.norm(node_vec)

                similarity = float(np.dot(query_vec, node_vec))

                if similarity >= min_similarity:
                    results.append((node, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:limit]

    def get_all_embeddings(self) -> list[tuple[str, bytes]]:
        """Get all node embeddings for similarity search.

        Returns:
            List of (node_id, embedding_bytes) tuples.
        """
        cursor = self._conn.execute(
            "SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL"
        )

        return [(row[0], row[1]) for row in cursor.fetchall() if row[1]]
