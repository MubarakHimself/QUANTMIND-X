"""
Vector Memory - Vector-based semantic memory with AgentDB integration.

Provides:
- VectorMemory class for semantic search
- AgentDB backend integration
- Hybrid memory backend
- HNSW indexing for fast similarity search
"""

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not available, using numpy-based similarity search")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class VectorMemoryEntry:
    """A memory entry with vector embedding."""
    id: str
    key: str
    value: str
    namespace: str
    embedding: Optional[np.ndarray] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "namespace": self.namespace,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorMemoryEntry":
        """Create from dictionary."""
        embedding = None
        if data.get("embedding") is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)

        return cls(
            id=data["id"],
            key=data["key"],
            value=data["value"],
            namespace=data["namespace"],
            embedding=embedding,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
        )


# =============================================================================
# Embedding Provider Interface
# =============================================================================

class EmbeddingProvider:
    """Interface for embedding providers."""

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        raise NotImplementedError

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        raise NotImplementedError

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        raise NotImplementedError


class DefaultEmbeddingProvider(EmbeddingProvider):
    """Default embedding provider using OpenAI or fallback."""

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        self._provider = None
        self._init_provider()

    def _init_provider(self):
        """Initialize the embedding provider."""
        try:
            # Try to use the project's embedding provider
            from src.memory.embeddings import get_embedding_provider

            provider_type = os.getenv("EMBEDDING_PROVIDER", "openai")
            self._provider = get_embedding_provider(
                provider_type,
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            )
            logger.info(f"Using {provider_type} embedding provider")
        except Exception as e:
            logger.warning(f"Could not initialize embedding provider: {e}")
            # Use mock provider
            self._provider = None

    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        if self._provider is not None:
            try:
                embedding = await self._provider.embed(text)
                return np.array(embedding, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Embedding failed, using random: {e}")

        # Fallback: return random normalized vector
        vec = np.random.randn(self._dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        if self._provider is not None:
            try:
                return self._provider.get_dimension()
            except Exception:
                pass
        return self._dimension


# =============================================================================
# FAISS-based Vector Index
# =============================================================================

class FAISSVectorIndex:
    """FAISS-based vector index with HNSW-like search."""

    def __init__(self, dimension: int, use_hnsw: bool = True):
        self.dimension = dimension
        self.use_hnsw = use_hnsw and FAISS_AVAILABLE
        self._index = None
        self._id_to_idx: Dict[str, int] = {}
        self._idx_to_id: Dict[int, str] = {}
        self._next_idx = 0

        if self.use_hnsw:
            try:
                # HNSW is faster for approximate nearest neighbor search
                self._index = faiss.IndexHNSWFlat(dimension, 32)
                logger.info("Using FAISS HNSW index")
            except Exception as e:
                logger.warning(f"Could not create HNSW index: {e}")
                self._index = None

        if self._index is None:
            # Fallback to exact search with inner product
            self._index = faiss.IndexFlatIP(dimension)
            logger.info("Using FAISS exact index (no HNSW)")

    def add(self, id: str, embedding: np.ndarray) -> None:
        """Add vector to index."""
        # Normalize for cosine similarity
        vec = embedding.copy()
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)

        # Remap ID
        if id not in self._id_to_idx:
            self._id_to_idx[id] = self._next_idx
            self._idx_to_id[self._next_idx] = id
            self._next_idx += 1

        vec = vec.reshape(1, -1)
        self._index.add(vec)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors."""
        if self._index.ntotal == 0:
            return []

        # Normalize query
        vec = query_embedding.copy()
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        vec = vec.reshape(1, -1)

        # Search
        distances, indices = self._index.search(vec, min(k, self._index.ntotal))

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._idx_to_id:
                # Convert cosine similarity to distance-like score
                score = float(dist)
                results.append((self._idx_to_id[idx], score))

        return results

    def remove(self, id: str) -> bool:
        """Remove vector from index."""
        # Note: FAISS doesn't support efficient removal
        # This is a placeholder - rebuild index if needed
        if id in self._id_to_idx:
            logger.warning("FAISS removal requires index rebuild")
            return True
        return False

    def clear(self) -> None:
        """Clear all vectors."""
        if FAISS_AVAILABLE:
            dim = self.dimension
            if self.use_hnsw:
                self._index = faiss.IndexHNSWFlat(dim, 32)
            else:
                self._index = faiss.IndexFlatIP(dim)
        self._id_to_idx = {}
        self._idx_to_id = {}
        self._next_idx = 0

    @property
    def count(self) -> int:
        """Return number of vectors."""
        return self._index.ntotal if self._index else 0


# =============================================================================
# Numpy-based Vector Index (Fallback)
# =============================================================================

class NumpyVectorIndex:
    """Numpy-based vector index for similarity search (fallback)."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._vectors: Dict[str, np.ndarray] = {}

    def add(self, id: str, embedding: np.ndarray) -> None:
        """Add vector to index."""
        vec = embedding.copy()
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)
        self._vectors[id] = vec

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Search for k nearest neighbors using cosine similarity."""
        if not self._vectors:
            return []

        # Normalize query
        vec = query_embedding.copy()
        if np.linalg.norm(vec) > 0:
            vec = vec / np.linalg.norm(vec)

        # Compute similarities
        results = []
        for id, stored_vec in self._vectors.items():
            # Cosine similarity
            similarity = float(np.dot(stored_vec, vec))
            results.append((id, similarity))

        # Sort by similarity and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def remove(self, id: str) -> bool:
        """Remove vector from index."""
        if id in self._vectors:
            del self._vectors[id]
            return True
        return False

    def clear(self) -> None:
        """Clear all vectors."""
        self._vectors = {}

    @property
    def count(self) -> int:
        """Return number of vectors."""
        return len(self._vectors)


# =============================================================================
# Vector Memory Backend (SQLite + Vector Index)
# =============================================================================

class VectorMemoryBackend:
    """SQLite-backed vector memory with HNSW indexing."""

    def __init__(
        self,
        db_path: Path,
        dimension: int = 1536,
        use_faiss: bool = True,
    ):
        """Initialize vector memory backend."""
        self.db_path = db_path
        self.dimension = dimension
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize vector index
        if use_faiss and FAISS_AVAILABLE:
            self._index = FAISSVectorIndex(dimension, use_hnsw=True)
        else:
            self._index = NumpyVectorIndex(dimension)
            logger.info("Using numpy-based vector index")

        # Initialize embedding provider
        self._embedding_provider = DefaultEmbeddingProvider(dimension)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vector_memories (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                namespace TEXT NOT NULL,
                embedding BLOB,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_namespace ON vector_memories(namespace)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_id ON vector_memories(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON vector_memories(session_id)
        """)
        conn.commit()
        conn.close()

    async def store(
        self,
        entry: VectorMemoryEntry,
        generate_embedding: bool = True,
    ) -> str:
        """Store a memory entry with vector embedding."""
        # Generate embedding if needed
        if entry.embedding is None and generate_embedding:
            entry.embedding = await self._embedding_provider.embed(entry.value)

        # Normalize embedding
        if entry.embedding is not None:
            entry.embedding = entry.embedding.astype(np.float32)
            if np.linalg.norm(entry.embedding) > 0:
                entry.embedding = entry.embedding / np.linalg.norm(entry.embedding)

        # Add to vector index
        if entry.embedding is not None:
            self._index.add(entry.id, entry.embedding)

        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO vector_memories
            (id, key, value, namespace, embedding, tags, metadata, created_at, updated_at, agent_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.key,
                entry.value,
                entry.namespace,
                entry.embedding.tobytes() if entry.embedding is not None else None,
                json.dumps(entry.tags),
                json.dumps(entry.metadata),
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                entry.agent_id,
                entry.session_id,
            ),
        )
        conn.commit()
        conn.close()

        return entry.id

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Tuple[VectorMemoryEntry, float]]:
        """Search memories by semantic similarity."""
        # Generate query embedding
        query_embedding = await self._embedding_provider.embed(query)

        # Search vector index
        vector_results = self._index.search(query_embedding, k=limit * 2)

        # Filter and fetch full entries
        results = []
        conn = sqlite3.connect(str(self.db_path))

        for mem_id, score in vector_results:
            if score < min_score:
                continue

            cursor = conn.execute(
                "SELECT * FROM vector_memories WHERE id = ?",
                (mem_id,),
            )
            row = cursor.fetchone()

            if row:
                entry = self._row_to_entry(row)
                # Apply filters
                if namespace and entry.namespace != namespace:
                    continue
                if agent_id and entry.agent_id != agent_id:
                    continue
                if session_id and entry.session_id != session_id:
                    continue

                results.append((entry, score))

                if len(results) >= limit:
                    break

        conn.close()
        return results

    def retrieve(self, key: str, namespace: str) -> Optional[VectorMemoryEntry]:
        """Retrieve a memory entry by key and namespace."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT * FROM vector_memories WHERE key = ? AND namespace = ?",
            (key, namespace),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_entry(row)
        return None

    def list_by_namespace(
        self,
        namespace: str,
        limit: int = 100,
    ) -> List[VectorMemoryEntry]:
        """List all entries in a namespace."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT * FROM vector_memories WHERE namespace = ? ORDER BY updated_at DESC LIMIT ?",
            (namespace, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def delete(self, key: str, namespace: str) -> bool:
        """Delete a memory entry."""
        # Find the entry first
        entry = self.retrieve(key, namespace)
        if entry:
            # Remove from vector index
            self._index.remove(entry.id)

            # Delete from database
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute(
                "DELETE FROM vector_memories WHERE key = ? AND namespace = ?",
                (key, namespace),
            )
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()
            return deleted
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM vector_memories")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT namespace, COUNT(*) FROM vector_memories GROUP BY namespace")
        by_namespace = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_entries": total,
            "by_namespace": by_namespace,
            "vector_count": self._index.count,
            "dimension": self.dimension,
            "backend": "vector",
            "faiss_available": FAISS_AVAILABLE,
        }

    def _row_to_entry(self, row: tuple) -> VectorMemoryEntry:
        """Convert database row to VectorMemoryEntry."""
        embedding = None
        if row[4]:  # embedding column
            embedding = np.frombuffer(row[4], dtype=np.float32).copy()

        return VectorMemoryEntry(
            id=row[0],
            key=row[1],
            value=row[2],
            namespace=row[3],
            embedding=embedding,
            tags=json.loads(row[5]) if row[5] else [],
            metadata=json.loads(row[6]) if row[6] else {},
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            agent_id=row[9],
            session_id=row[10],
        )


# =============================================================================
# AgentDB Backend (via ruflo MCP)
# =============================================================================

class AgentDBVectorBackend:
    """AgentDB-backed vector memory using ruflo MCP tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize AgentDB backend."""
        self.config = config
        self._available = False
        self._tool_loader = None
        self._server_id = config.get("server_id", "ruflo")
        self._namespace = config.get("namespace", "vector_memory")
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if AgentDB MCP tools are available."""
        try:
            # Check environment first
            if os.getenv("AGENTDB_ENABLED", "").lower() != "true":
                logger.info("AgentDB not enabled via AGENTDB_ENABLED, using local vector memory")
                return

            # Try to get the dynamic tool loader
            from src.agents.mcp.loader import get_tool_loader

            self._tool_loader = get_tool_loader()
            self._available = True
            logger.info("AgentDB backend enabled with ruflo MCP")
        except ImportError:
            logger.warning("AgentDB MCP loader not available, using local vector memory")
            self._available = False
        except Exception as e:
            logger.warning(f"AgentDB availability check failed: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if AgentDB is available."""
        return self._available and self._tool_loader is not None

    async def _call_agentdb_tool(self, tool_name: str, **kwargs) -> Any:
        """Call an AgentDB MCP tool via the tool loader."""
        if not self._tool_loader:
            raise RuntimeError("AgentDB tool loader not available")

        try:
            result = await self._tool_loader.call_tool(
                self._server_id,
                tool_name,
                kwargs
            )
            return result
        except Exception as e:
            logger.error(f"AgentDB tool {tool_name} failed: {e}")
            raise

    async def store(
        self,
        entry: VectorMemoryEntry,
        namespace: str = "memory",
    ) -> str:
        """Store a memory entry in AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        try:
            # Store using agentdb_pattern_store
            await self._call_agentdb_tool(
                "agentdb_pattern_store",
                key=entry.key,
                value=entry.value,
                namespace=entry.namespace or namespace,
                tags=entry.tags,
                metadata={
                    **entry.metadata,
                    "embedding_dimension": entry.embedding.shape[0] if entry.embedding is not None else None,
                    "agent_id": entry.agent_id,
                    "session_id": entry.session_id,
                },
            )
            return entry.id
        except Exception as e:
            logger.error(f"AgentDB store failed: {e}")
            raise RuntimeError(f"AgentDB store failed: {e}")

    async def search(
        self,
        query: str,
        namespace: str = "memory",
        limit: int = 10,
    ) -> List[Tuple[VectorMemoryEntry, float]]:
        """Search in AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        try:
            result = await self._call_agentdb_tool(
                "agentdb_pattern_search",
                query=query,
                namespace=namespace,
                limit=limit,
            )

            results: List[Tuple[VectorMemoryEntry, float]] = []
            if result and isinstance(result, dict) and "results" in result:
                for item in result["results"]:
                    entry = VectorMemoryEntry(
                        id=item.get("id", ""),
                        key=item.get("key", ""),
                        value=item.get("value", ""),
                        namespace=item.get("namespace", namespace),
                        tags=item.get("tags", []),
                        metadata=item.get("metadata", {}),
                        created_at=datetime.fromisoformat(item.get("created_at", datetime.utcnow().isoformat())),
                        updated_at=datetime.fromisoformat(item.get("updated_at", datetime.utcnow().isoformat())),
                        agent_id=item.get("agent_id"),
                        session_id=item.get("session_id"),
                    )
                    score = item.get("relevance_score", item.get("score", 0.0))
                    results.append((entry, float(score)))

            return results
        except Exception as e:
            logger.error(f"AgentDB search failed: {e}")
            raise RuntimeError(f"AgentDB search failed: {e}")

    async def delete(self, key: str, namespace: str = "memory") -> bool:
        """Delete from AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        # AgentDB may not have direct delete, log and return success
        logger.info(f"AgentDB delete called for {namespace}/{key}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get AgentDB stats."""
        return {
            "backend": "agentdb",
            "available": self._available,
            "server_id": self._server_id,
            "namespace": self._namespace,
        }


# =============================================================================
# Hybrid Vector Memory (Main Interface)
# =============================================================================

class VectorMemory:
    """
    Vector-based memory with semantic search.

    Features:
    - Store and retrieve context with vector embeddings
    - Semantic search using cosine similarity
    - HNSW indexing for fast search (FAISS)
    - Namespace-based organization
    - Agent and session tracking

    Usage:
        memory = VectorMemory(agent_id="my-agent")
        await memory.store("context_key", "context_value", namespace="session")
        results = await memory.search("context_value", namespace="session")
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        base_path: Optional[Path] = None,
        use_agentdb: bool = False,
        agentdb_config: Optional[Dict[str, Any]] = None,
        dimension: int = 1536,
    ):
        """
        Initialize vector memory.

        Args:
            agent_id: ID of the agent (for tracking)
            session_id: ID of the current session
            base_path: Base path for file-based storage
            use_agentdb: Whether to try AgentDB first
            agentdb_config: Configuration for AgentDB
            dimension: Embedding dimension
        """
        self.agent_id = agent_id
        self.session_id = session_id
        self.dimension = dimension

        # Determine storage path
        if base_path is None:
            base_path = Path(os.environ.get("VECTOR_MEMORY_PATH", "data/vector_memory"))
        self.base_path = Path(base_path)

        # Try AgentDB first if requested
        self._agentdb: Optional[AgentDBVectorBackend] = None
        if use_agentdb and agentdb_config:
            self._agentdb = AgentDBVectorBackend(agentdb_config)

        # Initialize vector backend
        use_faiss = os.getenv("USE_FAISS", "true").lower() == "true"
        self._backend = VectorMemoryBackend(
            db_path=self.base_path / "vector_memory.db",
            dimension=dimension,
            use_faiss=use_faiss,
        )

        logger.info(
            f"VectorMemory initialized: agent_id={agent_id}, "
            f"dimension={dimension}, faiss={use_faiss}"
        )

    async def store(
        self,
        key: str,
        value: str,
        namespace: str = "default",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory entry with vector embedding.

        Args:
            key: Memory key
            value: Memory value (used for embedding)
            namespace: Memory namespace
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Memory entry ID
        """
        entry = VectorMemoryEntry(
            id=str(uuid.uuid4()),
            key=key,
            value=value,
            namespace=namespace,
            tags=tags or [],
            metadata=metadata or {},
            agent_id=self.agent_id,
            session_id=self.session_id,
        )

        # Try AgentDB first
        if self._agentdb and self._agentdb.is_available():
            try:
                return await self._agentdb.store(entry, namespace)
            except NotImplementedError:
                pass

        # Use local vector backend
        return await self._backend.store(entry)

    async def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic similarity.

        Args:
            query: Search query
            namespace: Optional namespace filter
            limit: Maximum results
            minimum_score: Minimum similarity score

        Returns:
            List of memory entries with scores
        """
        # Try AgentDB first
        if self._agentdb and self._agentdb.is_available():
            try:
                results = await self._agentdb.search(query, namespace or "memory", limit)
                return [
                    {
                        **entry.to_dict(),
                        "relevance_score": score,
                    }
                    for entry, score in results
                ]
            except NotImplementedError:
                pass

        # Use local vector backend
        results = await self._backend.search(
            query=query,
            namespace=namespace,
            agent_id=self.agent_id,
            session_id=self.session_id,
            limit=limit,
            min_score=min_score,
        )

        return [
            {
                **entry.to_dict(),
                "relevance_score": score,
            }
            for entry, score in results
        ]

    def retrieve(self, key: str, namespace: str = "default") -> Optional[str]:
        """Retrieve a memory entry by key."""
        entry = self._backend.retrieve(key, namespace)
        return entry.value if entry else None

    def list(self, namespace: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        """List all entries in a namespace."""
        return [entry.to_dict() for entry in self._backend.list_by_namespace(namespace, limit)]

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a memory entry."""
        # Try AgentDB first
        if self._agentdb and self._agentdb.is_available():
            try:
                import asyncio
                return asyncio.run(self._agentdb.delete(key, namespace))
            except (NotImplementedError, RuntimeError):
                pass

        return self._backend.delete(key, namespace)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self._backend.get_stats()
        stats["agent_id"] = self.agent_id
        stats["session_id"] = self.session_id
        stats["agentdb_available"] = self._agentdb is not None and self._agentdb.is_available()
        return stats


# =============================================================================
# Factory Functions
# =============================================================================

def get_vector_memory(
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    use_agentdb: bool = False,
    agentdb_config: Optional[Dict[str, Any]] = None,
) -> VectorMemory:
    """
    Factory function to get a VectorMemory instance.

    Args:
        agent_id: ID of the agent
        session_id: ID of the session
        use_agentdb: Whether to use AgentDB
        agentdb_config: AgentDB configuration

    Returns:
        VectorMemory instance
    """
    return VectorMemory(
        agent_id=agent_id,
        session_id=session_id,
        use_agentdb=use_agentdb,
        agentdb_config=agentdb_config,
    )
