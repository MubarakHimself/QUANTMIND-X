"""
QuantMindX Memory Manager - Core Memory System

Inspired by the openclaw architecture, this module provides a comprehensive
memory management system with SQLite persistence, vector embeddings support,
and temporal decay for memory relevance.

Features:
- SQLite backend with vector similarity search via sqlite-vec
- Multiple embedding providers (OpenAI, Z.AI, local sentence-transformers)
- Memory sources: "memory" (files) and "sessions" (conversations)
- Dirty tracking for sync operations
- Temporal decay for time-based relevance
- Full-text search (FTS) support
"""

import asyncio
import aiosqlite
import sqlite3
import logging
import os
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MemorySource(str, Enum):
    """Memory source types."""
    MEMORY = "memory"  # File-based memory
    SESSION = "session"  # Conversation/session-based memory


class SyncStatus(str, Enum):
    """Sync status for dirty tracking."""
    CLEAN = "clean"  # In sync with storage
    DIRTY = "dirty"  # Pending sync


@dataclass
class MemoryEntry:
    """
    A single memory entry with metadata and embedding support.
    
    Attributes:
        id: Unique identifier (hash-based or UUID)
        source: Memory source (MEMORY or SESSION)
        content: The actual content/text
        embedding: Vector embedding (list of floats)
        metadata: Additional metadata dictionary
        created_at: Creation timestamp (UTC)
        updated_at: Last update timestamp (UTC)
        accessed_at: Last access timestamp (UTC)
        access_count: Number of times accessed
        importance: Importance score (0.0 to 1.0)
        tags: List of tags for categorization
        sync_status: Sync status for dirty tracking
    """
    id: str
    source: MemorySource
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    sync_status: SyncStatus = SyncStatus.CLEAN
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Generate checksum if not provided."""
        if self.checksum is None:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """Compute content checksum for integrity verification."""
        content_bytes = f"{self.content}{json.dumps(self.metadata, sort_keys=True)}".encode()
        return hashlib.sha256(content_bytes).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source": self.source.value,
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "tags": self.tags,
            "sync_status": self.sync_status.value,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create instance from dictionary."""
        return cls(
            id=data["id"],
            source=MemorySource(data["source"]),
            content=data["content"],
            embedding=data.get("embedding"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            accessed_at=datetime.fromisoformat(data["accessed_at"]),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            sync_status=SyncStatus(data.get("sync_status", "clean")),
            checksum=data.get("checksum"),
        )
    
    def mark_dirty(self) -> None:
        """Mark entry as dirty (pending sync)."""
        self.sync_status = SyncStatus.DIRTY
        self.updated_at = datetime.now(timezone.utc)
    
    def mark_clean(self) -> None:
        """Mark entry as clean (synced)."""
        self.sync_status = SyncStatus.CLEAN
    
    def touch(self) -> None:
        """Update access timestamp and count."""
        self.accessed_at = datetime.now(timezone.utc)
        self.access_count += 1


@dataclass
class MemoryStats:
    """Memory statistics."""
    total_entries: int
    entries_by_source: Dict[str, int]
    total_size_bytes: int
    dirty_entries: int
    oldest_entry: Optional[datetime]
    newest_entry: Optional[datetime]
    average_importance: float


class MemoryManager:
    """
    Core memory manager with SQLite persistence and vector search.
    
    This manager provides:
    - Async SQLite operations with aiosqlite
    - Vector similarity search via sqlite-vec
    - Full-text search via FTS5
    - Dirty tracking for sync operations
    - Automatic temporal decay calculation
    
    Example:
        >>> manager = MemoryManager(
        ...     db_path="/path/to/memory.db",
        ...     embedding_dim=1536
        ... )
        >>> await manager.initialize()
        >>> 
        >>> # Add memory with embedding
        >>> entry = await manager.add_memory(
        ...     source=MemorySource.MEMORY,
        ...     content="EURUSD shows mean reversion behavior",
        ...     embedding=[0.1, 0.2, ...],  # 1536 dimensions
        ...     importance=0.8,
        ...     tags=["trading", "forex"]
        ... )
        >>> 
        >>> # Search by similarity
        >>> results = await manager.search_similar(
        ...     query_embedding=[0.1, 0.2, ...],
        ...     limit=10
        ... )
    """
    
    DEFAULT_DB_PATH = Path("data/memory.db")
    EMBEDDING_DIM_DEFAULT = 1536  # OpenAI text-embedding-3-small
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedding_dim: int = EMBEDDING_DIM_DEFAULT,
        enable_fts: bool = True,
        enable_vec: bool = True,
    ):
        """
        Initialize memory manager.
        
        Args:
            db_path: Path to SQLite database (default: data/memory.db)
            embedding_dim: Dimension of embeddings (default: 1536 for OpenAI)
            enable_fts: Enable full-text search (default: True)
            enable_vec: Enable vector similarity search (default: True)
        """
        self.db_path = db_path or self.DEFAULT_DB_PATH
        self.embedding_dim = embedding_dim
        self.enable_fts = enable_fts
        self.enable_vec = enable_vec
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized = False
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"MemoryManager initialized: db={self.db_path}, "
            f"dim={embedding_dim}, fts={enable_fts}, vec={enable_vec}"
        )
    
    async def initialize(self) -> None:
        """Initialize database schema and connections."""
        if self._initialized:
            return
        
        await self._create_schema()
        self._initialized = True
        logger.info("MemoryManager database initialized")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        self._initialized = False
        logger.info("MemoryManager closed")
    
    def _get_db(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._db is None:
            self._db = aiosqlite.connect(self.db_path)
        return self._db
    
    async def _create_schema(self) -> None:
        """Create database schema with tables and indexes."""
        db = self._get_db()
        
        # Main memory table
        await db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5,
                tags TEXT,
                sync_status TEXT DEFAULT 'clean',
                checksum TEXT
            )
        """)
        
        # Create indexes
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_source 
            ON memories(source)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_importance 
            ON memories(importance DESC)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_created_at 
            ON memories(created_at DESC)
        """)
        
        await db.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_sync_status 
            ON memories(sync_status)
        """)
        
        # Full-text search table
        if self.enable_fts:
            await db.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts 
                USING fts5(content, id)
            """)
            
            # Trigger to keep FTS in sync
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_insert 
                AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content, id)
                    VALUES (new.rowid, new.content, new.id);
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_delete 
                AFTER DELETE ON memories BEGIN
                    DELETE FROM memories_fts WHERE rowid = old.rowid;
                END
            """)
            
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_fts_update 
                AFTER UPDATE ON memories BEGIN
                    UPDATE memories_fts 
                    SET content = new.content 
                    WHERE rowid = new.rowid;
                END
            """)
        
        # Vector similarity table (sqlite-vec)
        if self.enable_vec:
            # Check if sqlite-vec is available
            try:
                # Create vec0 virtual table for vector search
                await db.execute(f"""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec 
                    USING vec0(
                        embedding float[{self.embedding_dim}],
                        id TEXT PRIMARY KEY
                    )
                """)
                
                # Triggers to keep vec table in sync
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_vec_insert 
                    AFTER INSERT ON memories BEGIN
                        INSERT INTO memories_vec(embedding, id)
                        VALUES (new.embedding, new.id);
                    END
                """)
                
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_vec_delete 
                    AFTER DELETE ON memories BEGIN
                        DELETE FROM memories_vec WHERE id = old.id;
                    END
                """)
                
                await db.execute("""
                    CREATE TRIGGER IF NOT EXISTS memories_vec_update 
                    AFTER UPDATE ON memories BEGIN
                        UPDATE memories_vec 
                        SET embedding = new.embedding 
                        WHERE id = new.id;
                    END
                """)
                
                logger.info("Vector search enabled with sqlite-vec")
                
            except aiosqlite.Error as e:
                logger.warning(f"Could not create vec0 table: {e}. Vector search disabled.")
                self.enable_vec = False
        
        await db.commit()
    
    async def add_memory(
        self,
        source: MemorySource,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        entry_id: Optional[str] = None,
    ) -> MemoryEntry:
        """
        Add a new memory entry.
        
        Args:
            source: Memory source type
            content: Memory content
            embedding: Vector embedding (optional)
            metadata: Additional metadata
            importance: Importance score (0.0 to 1.0)
            tags: Tags for categorization
            entry_id: Custom ID (auto-generated if not provided)
            
        Returns:
            Created MemoryEntry
        """
        await self.initialize()
        
        # Generate ID if not provided
        if entry_id is None:
            entry_id = self._generate_id(content)
        
        # Create entry
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            id=entry_id,
            source=source,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
            accessed_at=now,
            importance=importance,
            tags=tags or [],
            sync_status=SyncStatus.CLEAN,
        )
        
        # Serialize for storage
        embedding_blob = self._serialize_embedding(embedding) if embedding else None
        
        db = self._get_db()
        await db.execute(
            """
            INSERT OR REPLACE INTO memories 
            (id, source, content, embedding, metadata, created_at, updated_at, 
             accessed_at, access_count, importance, tags, sync_status, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.source.value,
                entry.content,
                embedding_blob,
                json.dumps(entry.metadata),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.accessed_at.isoformat(),
                entry.access_count,
                entry.importance,
                json.dumps(entry.tags),
                entry.sync_status.value,
                entry.checksum,
            ),
        )
        await db.commit()
        
        logger.debug(f"Added memory: {entry.id} (source={source})")
        return entry
    
    async def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID.
        
        Args:
            entry_id: Memory identifier
            
        Returns:
            MemoryEntry or None if not found
        """
        await self.initialize()
        
        db = self._get_db()
        cursor = await db.execute(
            "SELECT * FROM memories WHERE id = ?",
            (entry_id,)
        )
        row = await cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_entry(row)
    
    async def update_memory(
        self,
        entry_id: str,
        content: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[MemoryEntry]:
        """
        Update an existing memory entry.
        
        Args:
            entry_id: Memory identifier
            content: New content (optional)
            embedding: New embedding (optional)
            metadata: New metadata (optional)
            importance: New importance (optional)
            tags: New tags (optional)
            
        Returns:
            Updated MemoryEntry or None if not found
        """
        await self.initialize()
        
        entry = await self.get_memory(entry_id)
        if entry is None:
            return None
        
        # Update fields
        if content is not None:
            entry.content = content
        if embedding is not None:
            entry.embedding = embedding
        if metadata is not None:
            entry.metadata = metadata
        if importance is not None:
            entry.importance = importance
        if tags is not None:
            entry.tags = tags
        
        entry.mark_dirty()
        entry.checksum = entry._compute_checksum()
        
        # Save to database
        embedding_blob = self._serialize_embedding(entry.embedding) if entry.embedding else None
        
        db = self._get_db()
        await db.execute(
            """
            UPDATE memories SET
                content = ?, embedding = ?, metadata = ?, importance = ?,
                tags = ?, sync_status = ?, checksum = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                entry.content,
                embedding_blob,
                json.dumps(entry.metadata),
                entry.importance,
                json.dumps(entry.tags),
                entry.sync_status.value,
                entry.checksum,
                entry.updated_at.isoformat(),
                entry.id,
            ),
        )
        await db.commit()
        
        return entry
    
    async def delete_memory(self, entry_id: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            entry_id: Memory identifier
            
        Returns:
            True if deleted, False if not found
        """
        await self.initialize()
        
        db = self._get_db()
        cursor = await db.execute(
            "DELETE FROM memories WHERE id = ?",
            (entry_id,)
        )
        await db.commit()
        
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Deleted memory: {entry_id}")
        
        return deleted
    
    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 10,
        source: Optional[MemorySource] = None,
        min_importance: float = 0.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memories by vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            limit: Maximum results
            source: Filter by source (optional)
            min_importance: Minimum importance threshold
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        await self.initialize()
        
        if not self.enable_vec:
            logger.warning("Vector search not enabled")
            return []
        
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, got {len(query_embedding)}"
            )
        
        # Build query with sqlite-vec
        query_blob = self._serialize_embedding(query_embedding)
        
        sql = """
            SELECT
                m.*,
                v.distance
            FROM memories_vec v
            JOIN memories m ON m.id = v.id
            WHERE v.embedding MATCH ?
            AND m.importance >= ?
        """
        
        params = [query_blob, min_importance]
        
        if source:
            sql += " AND m.source = ?"
            params.append(source.value)
        
        sql += " ORDER BY v.distance ASC LIMIT ?"
        params.append(limit)
        
        db = self._get_db()
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            # Convert distance to similarity (cosine: 0 = identical, 2 = opposite)
            similarity = 1.0 - (row["distance"] / 2.0) if row["distance"] is not None else 0.0
            results.append((entry, similarity))
        
        return results
    
    async def search_fts(
        self,
        query: str,
        limit: int = 10,
        source: Optional[MemorySource] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memories using full-text search.
        
        Args:
            query: FTS query string
            limit: Maximum results
            source: Filter by source (optional)
            
        Returns:
            List of (MemoryEntry, rank) tuples
        """
        await self.initialize()
        
        if not self.enable_fts:
            logger.warning("Full-text search not enabled")
            return []
        
        sql = """
            SELECT m.*, fts.rank
            FROM memories_fts fts
            JOIN memories m ON m.id = fts.id
            WHERE memories_fts MATCH ?
        """
        
        params = [query]
        
        if source:
            sql += " AND m.source = ?"
            params.append(source.value)
        
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(limit)
        
        db = self._get_db()
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            entry = self._row_to_entry(row)
            rank = row.get("rank", 0.0)
            results.append((entry, -rank))  # Negative rank: higher is better
        
        return results
    
    async def get_dirty_entries(self) -> List[MemoryEntry]:
        """
        Get all entries marked as dirty (pending sync).
        
        Returns:
            List of dirty MemoryEntry objects
        """
        await self.initialize()
        
        db = self._get_db()
        cursor = await db.execute(
            "SELECT * FROM memories WHERE sync_status = ?",
            (SyncStatus.DIRTY.value,)
        )
        rows = await cursor.fetchall()
        
        return [self._row_to_entry(row) for row in rows]
    
    async def mark_clean(self, entry_id: str) -> bool:
        """
        Mark a dirty entry as clean (synced).
        
        Args:
            entry_id: Memory identifier
            
        Returns:
            True if updated, False if not found
        """
        await self.initialize()
        
        db = self._get_db()
        cursor = await db.execute(
            "UPDATE memories SET sync_status = ? WHERE id = ?",
            (SyncStatus.CLEAN.value, entry_id)
        )
        await db.commit()
        
        return cursor.rowcount > 0
    
    async def get_stats(self) -> MemoryStats:
        """
        Get memory statistics.
        
        Returns:
            MemoryStats object
        """
        await self.initialize()
        
        db = self._get_db()
        
        # Total entries
        cursor = await db.execute("SELECT COUNT(*) FROM memories")
        total = (await cursor.fetchone())[0]
        
        # By source
        cursor = await db.execute(
            "SELECT source, COUNT(*) FROM memories GROUP BY source"
        )
        by_source = {row[0]: row[1] for row in await cursor.fetchall()}
        
        # Total size
        cursor = await db.execute(
            "SELECT SUM(LENGTH(content)) FROM memories"
        )
        size = (await cursor.fetchone())[0] or 0
        
        # Dirty entries
        cursor = await db.execute(
            "SELECT COUNT(*) FROM memories WHERE sync_status = ?",
            (SyncStatus.DIRTY.value,)
        )
        dirty = (await cursor.fetchone())[0]
        
        # Date range
        cursor = await db.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM memories"
        )
        min_date, max_date = await cursor.fetchone()
        
        oldest = datetime.fromisoformat(min_date) if min_date else None
        newest = datetime.fromisoformat(max_date) if max_date else None
        
        # Average importance
        cursor = await db.execute("SELECT AVG(importance) FROM memories")
        avg_importance = (await cursor.fetchone())[0] or 0.0
        
        return MemoryStats(
            total_entries=total,
            entries_by_source=by_source,
            total_size_bytes=size,
            dirty_entries=dirty,
            oldest_entry=oldest,
            newest_entry=newest,
            average_importance=avg_importance,
        )
    
    async def cleanup_old_entries(
        self,
        older_than: timedelta,
        min_importance: float = 0.3,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Clean up old, low-importance entries.
        
        Args:
            older_than: Age threshold for cleanup
            min_importance: Minimum importance to preserve
            dry_run: If True, don't actually delete
            
        Returns:
            List of deleted entry IDs
        """
        await self.initialize()
        
        cutoff = datetime.now(timezone.utc) - older_than
        cutoff_str = cutoff.isoformat()
        
        db = self._get_db()
        
        # Find candidates
        cursor = await db.execute(
            """
            SELECT id FROM memories
            WHERE created_at < ? AND importance < ?
            ORDER BY importance ASC, created_at ASC
            """,
            (cutoff_str, min_importance)
        )
        candidates = [row[0] for row in await cursor.fetchall()]
        
        if not dry_run:
            for entry_id in candidates:
                await self.delete_memory(entry_id)
        
        logger.info(
            f"Cleanup: {'would delete' if dry_run else 'deleted'} "
            f"{len(candidates)} entries older than {older_than}"
        )
        
        return candidates
    
    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content hash."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"mem_{timestamp}_{content_hash}"
    
    def _serialize_embedding(self, embedding: List[float]) -> bytes:
        """Serialize embedding to bytes for storage."""
        import struct
        # Pack floats as little-endian bytes
        return struct.pack(f"{len(embedding)}f", *embedding)
    
    def _deserialize_embedding(self, data: bytes) -> List[float]:
        """Deserialize embedding bytes to list of floats."""
        import struct
        # Unpack bytes as floats
        fmt = f"{len(data) // 4}f"
        return list(struct.unpack(fmt, data))
    
    def _row_to_entry(self, row: Dict[str, Any]) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        embedding = None
        if row["embedding"]:
            embedding = self._deserialize_embedding(row["embedding"])
        
        return MemoryEntry(
            id=row["id"],
            source=MemorySource(row["source"]),
            content=row["content"],
            embedding=embedding,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            accessed_at=datetime.fromisoformat(row["accessed_at"]),
            access_count=row["access_count"],
            importance=row["importance"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            sync_status=SyncStatus(row["sync_status"]),
            checksum=row["checksum"],
        )


# Convenience functions for common operations

async def create_memory_manager(
    db_path: Optional[Path] = None,
    embedding_dim: int = 1536,
) -> MemoryManager:
    """
    Create and initialize a MemoryManager.
    
    Args:
        db_path: Path to SQLite database
        embedding_dim: Embedding dimension
        
    Returns:
        Initialized MemoryManager
    """
    manager = MemoryManager(db_path=db_path, embedding_dim=embedding_dim)
    await manager.initialize()
    return manager
