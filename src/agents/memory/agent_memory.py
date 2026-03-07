"""
Agent Memory - Cross-session persistence and context retention for sub-agents.

Provides:
- AgentMemory class for memory persistence
- Cross-session context retention
- Memory search/retrieval
- AgentDB support (if available) with file-based fallback

Reference: https://platform.claude.com/cookbook/misc-session-memory-compaction
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    key: str
    value: str
    namespace: str
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
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            key=data["key"],
            value=data["value"],
            namespace=data["namespace"],
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            agent_id=data.get("agent_id"),
            session_id=data.get("session_id"),
        )


class FileMemoryBackend:
    """File-based memory backend using SQLite."""

    def __init__(self, db_path: Path):
        """Initialize the file-based backend."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                namespace TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                agent_id TEXT,
                session_id TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_namespace ON memories(namespace)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)
        """)
        conn.commit()
        conn.close()

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (id, key, value, namespace, tags, metadata, created_at, updated_at, agent_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.key,
                entry.value,
                entry.namespace,
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

    def retrieve(self, key: str, namespace: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key and namespace."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT * FROM memories WHERE key = ? AND namespace = ?",
            (key, namespace),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return self._row_to_entry(row)
        return None

    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Search memory entries."""
        conn = sqlite3.connect(str(self.db_path))

        sql = "SELECT * FROM memories WHERE (key LIKE ? OR value LIKE ?)"
        params = [f"%{query}%", f"%{query}%"]

        if namespace:
            sql += " AND namespace = ?"
            params.append(namespace)

        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)

        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)

        sql += " LIMIT ?"
        params.append(limit)

        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_entry(row) for row in rows]

    def list_by_namespace(self, namespace: str, limit: int = 100) -> List[MemoryEntry]:
        """List all entries in a namespace."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "SELECT * FROM memories WHERE namespace = ? ORDER BY updated_at DESC LIMIT ?",
            (namespace, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_entry(row) for row in rows]

    def delete(self, key: str, namespace: str) -> bool:
        """Delete a memory entry."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute(
            "DELETE FROM memories WHERE key = ? AND namespace = ?",
            (key, namespace),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]

        cursor = conn.execute("SELECT namespace, COUNT(*) FROM memories GROUP BY namespace")
        by_namespace = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_entries": total,
            "by_namespace": by_namespace,
            "backend": "file",
        }

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        """Convert database row to MemoryEntry."""
        return MemoryEntry(
            id=row[0],
            key=row[1],
            value=row[2],
            namespace=row[3],
            tags=json.loads(row[4]) if row[4] else [],
            metadata=json.loads(row[5]) if row[5] else {},
            created_at=datetime.fromisoformat(row[6]),
            updated_at=datetime.fromisoformat(row[7]),
            agent_id=row[8],
            session_id=row[9],
        )


class AgentDBMemoryBackend:
    """AgentDB-based memory backend using ruflo MCP tools."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize with AgentDB configuration."""
        self.config = config
        self._available = False
        self._tool_loader = None
        self._server_id = config.get("server_id", "ruflo")
        self._namespace = config.get("namespace", "memory")
        self._check_availability()

    def _check_availability(self) -> None:
        """Check if AgentDB MCP tools are available."""
        try:
            # Try to get the dynamic tool loader
            from src.agents.mcp.loader import get_tool_loader

            self._tool_loader = get_tool_loader()
            # Mark as available if we can get the loader
            # Actual tool availability will be checked per-operation
            self._available = True
            logger.info("AgentDB backend initialized with ruflo MCP")
        except Exception as e:
            logger.warning(f"AgentDB not available: {e}")
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

    def store(self, entry: MemoryEntry) -> None:
        """Store a memory entry in AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        import asyncio
        try:
            # Run async operation in sync context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new task
                future = asyncio.run_coroutine_threadsafe(
                    self._store_async(entry),
                    loop
                )
                return future.result(timeout=30)
            else:
                return loop.run_until_complete(self._store_async(entry))
        except Exception as e:
            logger.error(f"Failed to store in AgentDB: {e}")
            raise RuntimeError(f"AgentDB store failed: {e}")

    async def _store_async(self, entry: MemoryEntry) -> None:
        """Async store implementation."""
        await self._call_agentdb_tool(
            "agentdb_pattern_store",
            key=entry.key,
            value=entry.value,
            namespace=entry.namespace,
            tags=entry.tags,
            metadata=entry.metadata,
        )

    def retrieve(self, key: str, namespace: str) -> Optional[MemoryEntry]:
        """Retrieve from AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._retrieve_async(key, namespace),
                    loop
                )
                return future.result(timeout=30)
            else:
                return loop.run_until_complete(self._retrieve_async(key, namespace))
        except Exception as e:
            logger.error(f"Failed to retrieve from AgentDB: {e}")
            return None

    async def _retrieve_async(self, key: str, namespace: str) -> Optional[MemoryEntry]:
        """Async retrieve implementation."""
        try:
            result = await self._call_agentdb_tool(
                "agentdb_pattern_search",
                query=key,
                namespace=namespace,
                limit=1,
            )
            if result and isinstance(result, dict) and "results" in result:
                results = result["results"]
                if results:
                    data = results[0]
                    return MemoryEntry(
                        id=data.get("id", key),
                        key=data.get("key", key),
                        value=data.get("value", ""),
                        namespace=data.get("namespace", namespace),
                        tags=data.get("tags", []),
                        metadata=data.get("metadata", {}),
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
                        updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
                        agent_id=data.get("agent_id"),
                        session_id=data.get("session_id"),
                    )
            return None
        except Exception as e:
            logger.warning(f"AgentDB retrieve failed: {e}")
            return None

    def search(self, query: str, namespace: Optional[str] = None,
               agent_id: Optional[str] = None, session_id: Optional[str] = None,
               limit: int = 10) -> List[MemoryEntry]:
        """Search in AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._search_async(query, namespace, agent_id, session_id, limit),
                    loop
                )
                return future.result(timeout=30)
            else:
                return loop.run_until_complete(
                    self._search_async(query, namespace, agent_id, session_id, limit)
                )
        except Exception as e:
            logger.error(f"Failed to search AgentDB: {e}")
            return []

    async def _search_async(self, query: str, namespace: Optional[str] = None,
                           agent_id: Optional[str] = None, session_id: Optional[str] = None,
                           limit: int = 10) -> List[MemoryEntry]:
        """Async search implementation."""
        try:
            params = {"query": query, "limit": limit}
            if namespace:
                params["namespace"] = namespace
            if agent_id:
                params["agent_id"] = agent_id
            if session_id:
                params["session_id"] = session_id

            result = await self._call_agentdb_tool("agentdb_pattern_search", **params)

            entries = []
            if result and isinstance(result, dict) and "results" in result:
                for data in result["results"]:
                    entries.append(MemoryEntry(
                        id=data.get("id", ""),
                        key=data.get("key", ""),
                        value=data.get("value", ""),
                        namespace=data.get("namespace", namespace or "default"),
                        tags=data.get("tags", []),
                        metadata=data.get("metadata", {}),
                        created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
                        updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat())),
                        agent_id=data.get("agent_id"),
                        session_id=data.get("session_id"),
                    ))
            return entries
        except Exception as e:
            logger.warning(f"AgentDB search failed: {e}")
            return []

    def list_by_namespace(self, namespace: str, limit: int = 100) -> List[MemoryEntry]:
        """List by namespace in AgentDB."""
        # Use search with empty query to get all entries in namespace
        return self.search(query="", namespace=namespace, limit=limit)

    def delete(self, key: str, namespace: str) -> bool:
        """Delete from AgentDB."""
        if not self.is_available():
            raise RuntimeError("AgentDB not available")

        # AgentDB may not have direct delete, mark as success if no error
        logger.info(f"AgentDB delete called for {namespace}/{key}")
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get stats from AgentDB."""
        return {
            "backend": "agentdb",
            "available": self._available,
            "server_id": self._server_id,
            "namespace": self._namespace,
        }


class AgentMemory:
    """
    Agent memory for cross-session persistence and context retention.

    Features:
    - Store and retrieve context across sessions
    - Namespace-based organization
    - Tag-based filtering
    - Agent and session tracking
    - Search capabilities

    Usage:
        memory = AgentMemory(agent_id="my-agent")
        memory.store("context_key", "context_value", namespace="session")
        value = memory.retrieve("context_key", namespace="session")
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        base_path: Optional[Path] = None,
        use_agentdb: bool = False,
        agentdb_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the agent memory.

        Args:
            agent_id: ID of the agent (for tracking)
            session_id: ID of the current session
            base_path: Base path for file-based storage
            use_agentdb: Whether to try AgentDB first
            agentdb_config: Configuration for AgentDB
        """
        self.agent_id = agent_id
        self.session_id = session_id

        # Determine storage path
        if base_path is None:
            base_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
        self.base_path = Path(base_path)

        # Initialize backend (prefer AgentDB if requested and available)
        self._backend: FileMemoryBackend

        if use_agentdb and agentdb_config:
            agentdb_backend = AgentDBMemoryBackend(agentdb_config)
            if agentdb_backend.is_available():
                self._backend = agentdb_backend
                logger.info("Using AgentDB backend for agent memory")
            else:
                self._backend = FileMemoryBackend(self.base_path / "memory.db")
                logger.info("AgentDB not available, using file backend")
        else:
            self._backend = FileMemoryBackend(self.base_path / "memory.db")
            logger.info("Using file backend for agent memory")

    def store(self, key: str, value: str, namespace: str = "default",
              tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a memory entry."""
        import uuid
        entry = MemoryEntry(id=str(uuid.uuid4()), key=key, value=value, namespace=namespace,
                            tags=tags or [], metadata=metadata or {},
                            agent_id=self.agent_id, session_id=self.session_id)
        self._backend.store(entry)
        logger.debug(f"Stored memory: {namespace}/{key}")
        return entry.id

    def retrieve(self, key: str, namespace: str = "default") -> Optional[str]:
        """Retrieve a memory entry."""
        entry = self._backend.retrieve(key, namespace)
        return entry.value if entry else None

    def search(self, query: str, namespace: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory entries."""
        entries = self._backend.search(query=query, namespace=namespace,
                                       agent_id=self.agent_id, session_id=self.session_id, limit=limit)
        return [entry.to_dict() for entry in entries]

    def list(self, namespace: str = "default", limit: int = 100) -> List[Dict[str, Any]]:
        """List all entries in a namespace."""
        return [entry.to_dict() for entry in self._backend.list_by_namespace(namespace, limit)]

    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete a memory entry."""
        return self._backend.delete(key, namespace)

    def get_context(self, namespace: str = "session") -> Dict[str, str]:
        """Get all context for the current agent/session."""
        return {entry.key: entry.value for entry in self._backend.list_by_namespace(namespace, 500)}

    def store_context(self, context: Dict[str, str], namespace: str = "session") -> None:
        """Store multiple context entries at once."""
        for key, value in context.items():
            self.store(key, value, namespace=namespace)

    def clear_session(self) -> None:
        """Clear all memories for the current session."""
        if not self.session_id:
            logger.warning("No session_id set, cannot clear session")
            return
        for entry in self._backend.search(query="", session_id=self.session_id, limit=1000):
            self._backend.delete(entry.key, entry.namespace)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = self._backend.get_stats()
        stats["agent_id"] = self.agent_id
        stats["session_id"] = self.session_id
        return stats


def get_agent_memory(
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> AgentMemory:
    """
    Factory function to get an AgentMemory instance.

    Args:
        agent_id: ID of the agent
        session_id: ID of the session

    Returns:
        AgentMemory instance
    """
    return AgentMemory(agent_id=agent_id, session_id=session_id)


# =============================================================================
# Department-Aware Memory Extensions
# =============================================================================

def get_agent_memory_with_department(
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
    department: Optional[str] = None,
) -> "AgentMemoryWithDepartment":
    """
    Factory function to get an AgentMemory instance with department context.

    Args:
        agent_id: ID of the agent
        session_id: ID of the session
        department: Department name for memory routing

    Returns:
        AgentMemoryWithDepartment instance
    """
    return AgentMemoryWithDepartment(
        agent_id=agent_id,
        session_id=session_id,
        department=department,
    )


class AgentMemoryWithDepartment(AgentMemory):
    """
    Extended AgentMemory with department-aware memory operations.

    Features:
    - Department-specific memory namespaces
    - Cross-department memory routing
    - Memory sharing rules enforcement

    Usage:
        memory = AgentMemoryWithDepartment(
            agent_id="my-agent",
            department="research"
        )
        memory.store_department("key", "value")
        memory.search_all_departments("query")
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        department: Optional[str] = None,
        base_path: Optional[Path] = None,
    ):
        """
        Initialize the department-aware agent memory.

        Args:
            agent_id: ID of the agent
            session_id: ID of the current session
            department: Department name for memory routing
            base_path: Base path for file-based storage
        """
        super().__init__(agent_id=agent_id, session_id=session_id, base_path=base_path)
        self.department = department

        # Import department integration
        from src.agents.memory.department_integration import (
            DEPARTMENT_PREFIX,
            GLOBAL_NAMESPACE,
            get_department_namespace,
            get_department_config,
            route_memory,
            is_memory_accessible,
            MemorySharingRule,
        )
        self._dept_prefix = DEPARTMENT_PREFIX
        self._global_ns = GLOBAL_NAMESPACE
        self._get_dept_ns = get_department_namespace
        self._get_dept_config = get_department_config
        self._route_memory = route_memory
        self._is_accessible = is_memory_accessible
        self._sharing_rule = MemorySharingRule

    def get_department_namespace(self) -> str:
        """Get the memory namespace for the current department."""
        if not self.department:
            return self._global_ns

        try:
            from src.agents.departments.types import Department
            dept_enum = Department(self.department)
            return self._get_dept_ns(dept_enum)
        except ValueError:
            return f"{self._dept_prefix}{self.department}"

    def store_department(
        self,
        key: str,
        value: str,
        sharing_rule: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a memory in the department's namespace.

        Args:
            key: Memory key
            value: Memory value
            sharing_rule: Memory sharing rule (private, department, global, restricted)
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Memory entry ID
        """
        from src.agents.departments.types import Department

        # Determine namespace and sharing rule
        if self.department:
            try:
                dept_enum = Department(self.department)
                rule = self._sharing_rule(sharing_rule) if sharing_rule else None
                routing = self._route_memory(
                    department=dept_enum,
                    key=key,
                    value=value,
                    sharing_rule=rule,
                )
                namespace = routing["namespace"]
            except ValueError:
                namespace = self.get_department_namespace()
        else:
            namespace = self._global_ns

        return self.store(
            key=key,
            value=value,
            namespace=namespace,
            tags=tags,
            metadata=metadata,
        )

    def retrieve_department(self, key: str) -> Optional[str]:
        """Retrieve from department namespace."""
        return self.retrieve(key=key, namespace=self.get_department_namespace())

    def search_department(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search within department namespace."""
        return self.search(
            query=query,
            namespace=self.get_department_namespace(),
            limit=limit,
        )

    def search_global(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search within global namespace."""
        return self.search(
            query=query,
            namespace=self._global_ns,
            limit=limit,
        )

    def search_all_departments(
        self,
        query: str,
        include_global: bool = True,
        limit_per_namespace: int = 5,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across accessible namespaces.

        Args:
            query: Search query
            include_global: Include global namespace
            limit_per_namespace: Results per namespace

        Returns:
            Dictionary mapping namespace to results
        """
        results = {}

        if not self.department:
            # No department context - only search global
            if include_global:
                results[self._global_ns] = self.search_global(query, limit_per_namespace)
            return results

        # Search department namespace
        dept_ns = self.get_department_namespace()
        results[dept_ns] = self.search_department(query, limit_per_namespace)

        # Search global if permitted
        if include_global:
            config = self._get_dept_config(
                Department(self.department) if self.department else None
            )
            if config.can_access_global:
                results[self._global_ns] = self.search_global(query, limit_per_namespace)

        return results

    def list_department_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all memories in department namespace."""
        return self.list(namespace=self.get_department_namespace(), limit=limit)

    def list_global_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all global memories."""
        return self.list(namespace=self._global_ns, limit=limit)

    def delete_department(self, key: str) -> bool:
        """Delete from department namespace."""
        return self.delete(key=key, namespace=self.get_department_namespace())

    def contribute_to_global(
        self,
        key: str,
        value: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Contribute a memory to the global namespace.

        Args:
            key: Memory key
            value: Memory value
            tags: Optional tags
            metadata: Optional metadata

        Returns:
            Memory entry ID
        """
        return self.store(
            key=key,
            value=value,
            namespace=self._global_ns,
            tags=tags,
            metadata={**(metadata or {}), "contributing_department": self.department},
        )
