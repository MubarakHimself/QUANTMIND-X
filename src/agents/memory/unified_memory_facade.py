"""
Unified Memory Facade - Integration Layer for Department and Agent Memory Systems

This module provides a unified interface that integrates:
1. DepartmentMemoryManager (Markdown-based) - data/departments/{dept}/
2. AgentMemoryWithDepartment (SQLite-based) - data/agent_memory/memory.db

Features:
- Single API for both memory systems
- Cross-system search
- Sync mechanism from DepartmentMemoryManager to AgentMemory
- Unified statistics and management
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.types import Department
from src.agents.memory.agent_memory import (
    AgentMemory,
    AgentMemoryWithDepartment,
    get_agent_memory,
    get_agent_memory_with_department,
)
from src.agents.memory.department_integration import (
    DEPARTMENT_PREFIX,
    GLOBAL_NAMESPACE,
    MemorySharingRule,
    get_department_config,
    search_across_namespaces,
)

# Import GraphMemoryFacade for delegation
try:
    from src.memory.graph.facade import GraphMemoryFacade
except ImportError:
    GraphMemoryFacade = None  # Graph memory not available

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class UnifiedMemoryEntry:
    """A unified memory entry that can come from either system."""
    id: str
    key: str
    value: str
    namespace: str
    source: str  # "department" or "agent_memory"
    department: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    relevance_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "key": self.key,
            "value": self.value,
            "namespace": self.namespace,
            "source": self.source,
            "department": self.department,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "relevance_score": self.relevance_score,
        }


@dataclass
class UnifiedSearchResult:
    """Results from unified search across both memory systems."""
    entries: List[UnifiedMemoryEntry]
    total: int
    query: str
    sources: Dict[str, int]  # Source name -> count
    elapsed_ms: float


@dataclass
class UnifiedMemoryStats:
    """Unified statistics from both memory systems."""
    # Department memory (Markdown) stats
    department_memory: Dict[str, Any]

    # Agent memory (SQLite) stats
    agent_memory: Dict[str, Any]

    # Combined totals
    total_entries: int
    sources: List[str]

    # Sync status
    last_sync: Optional[datetime] = None
    sync_status: str = "not_synced"  # not_synced, syncing, synced, error


# =============================================================================
# Unified Memory Facade
# =============================================================================


class UnifiedMemoryFacade:
    """
    Unified facade that integrates DepartmentMemoryManager and AgentMemoryWithDepartment.

    This class provides a single interface for both memory systems, enabling:
    - Cross-system search
    - Unified CRUD operations
    - Sync from DepartmentMemoryManager to AgentMemory
    - Combined statistics

    Usage:
        facade = UnifiedMemoryFacade(department="research")
        facade.add_memory("key", "value", namespace="research")
        results = facade.search("query")
        stats = facade.get_stats()
    """

    def __init__(
        self,
        department: Optional[Union[str, Department]] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        base_path: Optional[Path] = None,
        auto_initialize: bool = True,
    ):
        """
        Initialize the unified memory facade.

        Args:
            department: Department name or enum
            agent_id: Agent identifier for AgentMemory
            session_id: Session identifier for AgentMemory
            base_path: Base path for memory storage
            auto_initialize: Whether to auto-initialize memory systems
        """
        # Handle department conversion
        if department is None:
            self._department: Optional[Department] = None
            self._department_str: Optional[str] = None
        elif isinstance(department, str):
            try:
                self._department = Department(department)
                self._department_str = department
            except ValueError:
                self._department = None
                self._department_str = department
        else:
            self._department = department
            self._department_str = department.value if hasattr(department, 'value') else str(department)

        self._agent_id = agent_id
        self._session_id = session_id

        # Determine base paths
        if base_path is None:
            base_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
        self._base_path = Path(base_path)

        # Initialize DepartmentMemoryManager (Markdown-based)
        if self._department and auto_initialize:
            try:
                self._dept_memory: Optional[DepartmentMemoryManager] = DepartmentMemoryManager(
                    department=self._department,
                    base_path=Path("data/departments"),
                    auto_initialize=True
                )
            except Exception as e:
                logger.warning(f"Failed to initialize DepartmentMemoryManager: {e}")
                self._dept_memory = None
        else:
            self._dept_memory = None

        # Initialize AgentMemoryWithDepartment (SQLite-based)
        if self._department_str:
            self._agent_memory: Optional[AgentMemoryWithDepartment] = get_agent_memory_with_department(
                agent_id=agent_id,
                session_id=session_id,
                department=self._department_str,
            )
        else:
            # Direct instantiation to support base_path
            self._agent_memory = AgentMemory(
                agent_id=agent_id,
                session_id=session_id,
                base_path=self._base_path,
            )

        # Sync tracking
        self._last_sync: Optional[datetime] = None

        # Initialize GraphMemoryFacade for graph-based memory
        self._graph_memory: Optional[GraphMemoryFacade] = None
        if GraphMemoryFacade is not None and auto_initialize:
            try:
                graph_db_path = Path(os.environ.get(
                    "GRAPH_MEMORY_PATH",
                    "data/graph_memory/memory.db"
                ))
                graph_db_path.parent.mkdir(parents=True, exist_ok=True)
                self._graph_memory = GraphMemoryFacade(db_path=graph_db_path)
                logger.info(f"Initialized GraphMemoryFacade for unified facade")
            except Exception as e:
                logger.warning(f"Failed to initialize GraphMemoryFacade: {e}")
                self._graph_memory = None

        logger.info(
            f"Initialized UnifiedMemoryFacade for department={self._department_str}, "
            f"agent_id={agent_id}"
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def department(self) -> Optional[str]:
        """Get the department name."""
        return self._department_str

    @property
    def agent_memory(self) -> AgentMemory:
        """Get the underlying AgentMemory instance."""
        return self._agent_memory

    @property
    def dept_memory(self) -> Optional[DepartmentMemoryManager]:
        """Get the underlying DepartmentMemoryManager instance."""
        return self._dept_memory

    @property
    def graph_memory(self) -> Optional[GraphMemoryFacade]:
        """Get the underlying GraphMemoryFacade instance."""
        return self._graph_memory

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def add_memory(
        self,
        key: str,
        value: str,
        namespace: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sync_to_agent: bool = True,
        sync_to_graph: bool = True,
    ) -> str:
        """
        Add a memory entry to the appropriate system.

        Args:
            key: Memory key
            value: Memory value
            namespace: Target namespace (defaults to department or global)
            tags: Optional tags
            metadata: Optional metadata
            sync_to_agent: Whether to also sync to AgentMemory
            sync_to_graph: Whether to also sync to graph memory

        Returns:
            Memory entry ID
        """
        # Use AgentMemory for storage (primary)
        if self._department_str and namespace is None:
            # Use department-aware storage
            if hasattr(self._agent_memory, 'store_department'):
                entry_id = self._agent_memory.store_department(
                    key=key,
                    value=value,
                    tags=tags,
                    metadata=metadata,
                )
            else:
                entry_id = self._agent_memory.store(
                    key=key,
                    value=value,
                    namespace=f"{DEPARTMENT_PREFIX}{self._department_str}",
                    tags=tags,
                    metadata=metadata,
                )
        else:
            entry_id = self._agent_memory.store(
                key=key,
                value=value,
                namespace=namespace or GLOBAL_NAMESPACE,
                tags=tags,
                metadata=metadata,
            )

        # Optionally add to DepartmentMemoryManager (Markdown)
        if sync_to_agent and self._dept_memory and self._department:
            try:
                self._dept_memory.add_memory(
                    category=namespace or "general",
                    content=f"{key}: {value}",
                    tags=tags,
                )
            except Exception as e:
                logger.warning(f"Failed to sync to DepartmentMemoryManager: {e}")

        # Also write to graph memory if enabled
        if sync_to_graph and self._graph_memory:
            try:
                self._graph_memory.retain(
                    content=f"{key}: {value}",
                    source="unified_memory_facade",
                    department=self._department_str,
                    agent_id=self._agent_id,
                    session_id=self._session_id,
                    importance=0.5,
                    tags=tags or [],
                )
            except Exception as e:
                logger.warning(f"Failed to sync to graph memory: {e}")

        logger.info(f"Added memory: {namespace or 'global'}/{key}")
        return entry_id

    def retrieve(
        self,
        key: str,
        namespace: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieve a memory entry.

        Args:
            key: Memory key
            namespace: Namespace to search in

        Returns:
            Memory value or None if not found
        """
        # Try AgentMemory first
        ns = namespace or (f"{DEPARTMENT_PREFIX}{self._department_str}" if self._department_str else GLOBAL_NAMESPACE)
        value = self._agent_memory.retrieve(key=key, namespace=ns)

        if value:
            return value

        # Fallback to DepartmentMemoryManager
        if self._dept_memory and namespace:
            try:
                results = self._dept_memory.search(query=key, category=namespace)
                if results:
                    # Extract value from first match
                    return results[0]
            except Exception as e:
                logger.warning(f"Failed to retrieve from DepartmentMemoryManager: {e}")

        return None

    def update(
        self,
        key: str,
        value: str,
        namespace: Optional[str] = None,
    ) -> bool:
        """
        Update a memory entry (adds new entry with same key).

        Args:
            key: Memory key
            value: New value
            namespace: Namespace

        Returns:
            True if successful
        """
        # Add with same key (AgentMemory uses INSERT OR REPLACE)
        self.add_memory(key=key, value=value, namespace=namespace, sync_to_agent=False)
        return True

    def delete(
        self,
        key: str,
        namespace: Optional[str] = None,
    ) -> bool:
        """
        Delete a memory entry.

        Args:
            key: Memory key
            namespace: Namespace

        Returns:
            True if deleted, False if not found
        """
        ns = namespace or (f"{DEPARTMENT_PREFIX}{self._department_str}" if self._department_str else GLOBAL_NAMESPACE)
        return self._agent_memory.delete(key=key, namespace=ns)

    # =========================================================================
    # Search Operations
    # =========================================================================

    def search(
        self,
        query: str,
        namespace: Optional[str] = None,
        limit: int = 10,
        include_all_sources: bool = True,
        prefer_graph: bool = True,
    ) -> UnifiedSearchResult:
        """
        Search across memory systems.

        Args:
            query: Search query
            namespace: Optional namespace filter
            limit: Maximum results per source
            include_all_sources: Whether to search both systems
            prefer_graph: If True, query graph first, fallback to legacy

        Returns:
            UnifiedSearchResult with combined results
        """
        import time
        start_time = time.time()

        entries: List[UnifiedMemoryEntry] = []
        sources: Dict[str, int] = {"agent_memory": 0, "department_memory": 0, "graph_memory": 0}

        # Try graph memory first if available and prefer_graph is True
        if prefer_graph and self._graph_memory:
            try:
                graph_results = self._graph_memory.recall(
                    query=query,
                    department=self._department_str,
                    agent_id=self._agent_id,
                    limit=limit,
                )

                for node in graph_results:
                    entries.append(UnifiedMemoryEntry(
                        id=str(node.id),
                        key=node.title,
                        value=node.content,
                        namespace="graph_memory",
                        source="graph_memory",
                        department=node.department,
                        tags=node.tags,
                        metadata={},
                        created_at=node.created_at,
                        updated_at=node.updated_at,
                        agent_id=node.agent_id,
                        session_id=node.session_id,
                        relevance_score=node.relevance_score,
                    ))
                    sources["graph_memory"] += 1

                # If we have graph results and don't want all sources, return early
                if not include_all_sources and entries:
                    elapsed_ms = (time.time() - start_time) * 1000
                    return UnifiedSearchResult(
                        entries=entries[:limit],
                        total=len(entries),
                        query=query,
                        sources=sources,
                        elapsed_ms=elapsed_ms,
                    )

            except Exception as e:
                logger.warning(f"GraphMemory search failed: {e}")

        # Search AgentMemory (SQLite)
        try:
            if namespace:
                agent_results = self._agent_memory.search(
                    query=query,
                    namespace=namespace,
                    limit=limit,
                )
            elif self._department_str and hasattr(self._agent_memory, 'search_department'):
                agent_results = self._agent_memory.search_department(query, limit=limit)
            else:
                agent_results = self._agent_memory.search(
                    query=query,
                    namespace=None,
                    limit=limit,
                )

            for entry in agent_results:
                entries.append(UnifiedMemoryEntry(
                    id=entry.get("id", ""),
                    key=entry.get("key", ""),
                    value=entry.get("value", ""),
                    namespace=entry.get("namespace", ""),
                    source="agent_memory",
                    department=self._department_str,
                    tags=entry.get("tags", []),
                    metadata=entry.get("metadata", {}),
                    created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat()))
                        if isinstance(entry.get("created_at"), str) else entry.get("created_at", datetime.utcnow()),
                    updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat()))
                        if isinstance(entry.get("updated_at"), str) else entry.get("updated_at", datetime.utcnow()),
                    agent_id=entry.get("agent_id"),
                    session_id=entry.get("session_id"),
                ))
                sources["agent_memory"] += 1
        except Exception as e:
            logger.warning(f"AgentMemory search failed: {e}")

        # Search DepartmentMemoryManager (Markdown)
        if include_all_sources and self._dept_memory:
            try:
                dept_results = self._dept_memory.search(
                    query=query,
                    category=namespace,
                )

                for i, result in enumerate(dept_results[:limit]):
                    entries.append(UnifiedMemoryEntry(
                        id=f"dept_{self._department_str}_{i}",
                        key=query,
                        value=result,
                        namespace=self._department_str or "department",
                        source="department_memory",
                        department=self._department_str,
                        tags=[],
                        metadata={"category": namespace},
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    ))
                    sources["department_memory"] += 1
            except Exception as e:
                logger.warning(f"DepartmentMemoryManager search failed: {e}")

        elapsed_ms = (time.time() - start_time) * 1000

        return UnifiedSearchResult(
            entries=entries[:limit],
            total=len(entries),
            query=query,
            sources=sources,
            elapsed_ms=elapsed_ms,
        )

    def search_all_departments(
        self,
        query: str,
        include_global: bool = True,
        limit_per_namespace: int = 5,
    ) -> Dict[str, List[UnifiedMemoryEntry]]:
        """
        Search across all accessible departments and global memory.

        Args:
            query: Search query
            include_global: Include global namespace
            limit_per_namespace: Results per namespace

        Returns:
            Dictionary mapping namespace to results
        """
        results: Dict[str, List[UnifiedMemoryEntry]] = {}

        if not self._department:
            # No department context - only search global
            if include_global:
                search_result = self.search(query=query, namespace=GLOBAL_NAMESPACE, limit=limit_per_namespace)
                results[GLOBAL_NAMESPACE] = search_result.entries
            return results

        # Get accessible namespaces for this department
        accessible_namespaces = search_across_namespaces(
            department=self._department,
            query=query,
            include_global=include_global,
        )

        # Search each accessible namespace
        for namespace in accessible_namespaces:
            search_result = self.search(query=query, namespace=namespace, limit=limit_per_namespace)
            if search_result.entries:
                results[namespace] = search_result.entries

        return results

    # =========================================================================
    # List Operations
    # =========================================================================

    def list_memories(
        self,
        namespace: Optional[str] = None,
        limit: int = 100,
    ) -> List[UnifiedMemoryEntry]:
        """
        List all memories in a namespace.

        Args:
            namespace: Namespace to list (defaults to department or global)
            limit: Maximum results

        Returns:
            List of memory entries
        """
        ns = namespace or (f"{DEPARTMENT_PREFIX}{self._department_str}" if self._department_str else GLOBAL_NAMESPACE)

        entries = self._agent_memory.list(namespace=ns, limit=limit)

        return [
            UnifiedMemoryEntry(
                id=entry.get("id", ""),
                key=entry.get("key", ""),
                value=entry.get("value", ""),
                namespace=entry.get("namespace", ""),
                source="agent_memory",
                department=self._department_str,
                tags=entry.get("tags", []),
                metadata=entry.get("metadata", {}),
                created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("created_at"), str) else entry.get("created_at", datetime.utcnow()),
                updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("updated_at"), str) else entry.get("updated_at", datetime.utcnow()),
                agent_id=entry.get("agent_id"),
                session_id=entry.get("session_id"),
            )
            for entry in entries
        ]

    def list_department_memories(self, limit: int = 100) -> List[UnifiedMemoryEntry]:
        """List all memories in department namespace."""
        if hasattr(self._agent_memory, 'list_department_memories'):
            entries = self._agent_memory.list_department_memories(limit=limit)
        else:
            entries = self._agent_memory.list(
                namespace=f"{DEPARTMENT_PREFIX}{self._department_str}",
                limit=limit,
            )

        return [
            UnifiedMemoryEntry(
                id=entry.get("id", ""),
                key=entry.get("key", ""),
                value=entry.get("value", ""),
                namespace=entry.get("namespace", ""),
                source="agent_memory",
                department=self._department_str,
                tags=entry.get("tags", []),
                metadata=entry.get("metadata", {}),
                created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("created_at"), str) else entry.get("created_at", datetime.utcnow()),
                updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("updated_at"), str) else entry.get("updated_at", datetime.utcnow()),
                agent_id=entry.get("agent_id"),
                session_id=entry.get("session_id"),
            )
            for entry in entries
        ]

    def list_global_memories(self, limit: int = 100) -> List[UnifiedMemoryEntry]:
        """List all global memories."""
        if hasattr(self._agent_memory, 'list_global_memories'):
            entries = self._agent_memory.list_global_memories(limit=limit)
        else:
            entries = self._agent_memory.list(namespace=GLOBAL_NAMESPACE, limit=limit)

        return [
            UnifiedMemoryEntry(
                id=entry.get("id", ""),
                key=entry.get("key", ""),
                value=entry.get("value", ""),
                namespace=entry.get("namespace", ""),
                source="agent_memory",
                department=None,
                tags=entry.get("tags", []),
                metadata=entry.get("metadata", {}),
                created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("created_at"), str) else entry.get("created_at", datetime.utcnow()),
                updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat()))
                    if isinstance(entry.get("updated_at"), str) else entry.get("updated_at", datetime.utcnow()),
                agent_id=entry.get("agent_id"),
                session_id=entry.get("session_id"),
            )
            for entry in entries
        ]

    # =========================================================================
    # Department-Specific Operations
    # =========================================================================

    def add_daily_log(
        self,
        log_date: date,
        content: str,
        category: Optional[str] = None,
    ) -> None:
        """
        Add an entry to department daily log.

        Args:
            log_date: Date for the log entry
            content: Log content
            category: Optional category
        """
        if self._dept_memory:
            self._dept_memory.add_daily_log(date=log_date, content=content, category=category)
        else:
            # Fallback to agent memory
            self.add_memory(
                key=f"daily_log_{log_date.isoformat()}",
                value=content,
                namespace=f"{DEPARTMENT_PREFIX}{self._department_str}_logs",
            )

    def get_daily_log(self, log_date: date) -> Optional[str]:
        """
        Get a specific daily log.

        Args:
            log_date: Date to retrieve

        Returns:
            Daily log content or None
        """
        if self._dept_memory:
            return self._dept_memory.get_daily_log(date=log_date)
        return None

    def get_recent_logs(self, days: int = 7) -> Dict[str, str]:
        """
        Get recent daily logs.

        Args:
            days: Number of recent days

        Returns:
            Dictionary mapping date strings to log contents
        """
        if self._dept_memory:
            return self._dept_memory.get_recent_logs(days=days)
        return {}

    # =========================================================================
    # Sync Operations
    # =========================================================================

    def sync_from_department_memory(
        self,
        categories: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Sync memories from DepartmentMemoryManager to AgentMemory.

        This reads all memories from the Markdown-based DepartmentMemoryManager
        and stores them in the SQLite-based AgentMemory.

        Args:
            categories: Optional list of categories to sync (default: all)
            force: Whether to force re-sync of existing entries

        Returns:
            Sync result with counts
        """
        if not self._dept_memory:
            return {"status": "no_department_memory", "synced": 0}

        synced_count = 0
        skipped_count = 0
        error_count = 0

        try:
            # Read department memory content
            content = self._dept_memory.read_memory()

            # Parse and sync entries (basic parsing)
            import re
            # Find all ## category - timestamp sections
            pattern = r"## ([^-]+) - ([\dT:.Z+-]+)\n\n(.*?)(?=\n---\n|\Z)"
            matches = re.findall(pattern, content, re.DOTALL)

            for match in matches:
                category = match[0].strip()
                timestamp = match[1]
                entry_content = match[2].strip()

                # Filter by category if specified
                if categories and category.lower() not in [c.lower() for c in categories]:
                    skipped_count += 1
                    continue

                # Check if entry already exists in AgentMemory
                existing = self._agent_memory.retrieve(
                    key=f"dept_{category}_{timestamp}",
                    namespace=f"{DEPARTMENT_PREFIX}{self._department_str}",
                )

                if existing and not force:
                    skipped_count += 1
                    continue

                # Store in AgentMemory
                try:
                    self._agent_memory.store(
                        key=f"dept_{category}_{timestamp}",
                        value=entry_content,
                        namespace=f"{DEPARTMENT_PREFIX}{self._department_str}",
                        tags=[category, "synced_from_department"],
                        metadata={
                            "original_category": category,
                            "synced_at": datetime.utcnow().isoformat(),
                            "source": "department_memory",
                        },
                    )
                    synced_count += 1
                except Exception as e:
                    logger.error(f"Failed to sync entry {category}_{timestamp}: {e}")
                    error_count += 1

            self._last_sync = datetime.utcnow()

            return {
                "status": "success",
                "synced": synced_count,
                "skipped": skipped_count,
                "errors": error_count,
                "last_sync": self._last_sync.isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to sync from department memory: {e}")
            return {
                "status": "error",
                "error": str(e),
                "synced": synced_count,
                "skipped": skipped_count,
                "errors": error_count,
            }

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> UnifiedMemoryStats:
        """
        Get unified statistics from both memory systems.

        Returns:
            UnifiedMemoryStats with combined information
        """
        # Get DepartmentMemoryManager stats
        dept_stats: Dict[str, Any] = {}
        if self._dept_memory:
            try:
                dept_stats = self._dept_memory.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get department memory stats: {e}")
                dept_stats = {"error": str(e)}

        # Get AgentMemory stats
        agent_stats: Dict[str, Any] = {}
        try:
            agent_stats = self._agent_memory.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get agent memory stats: {e}")
            agent_stats = {"error": str(e)}

        # Get GraphMemory stats
        graph_stats: Dict[str, Any] = {}
        if self._graph_memory:
            try:
                graph_stats = self._graph_memory.get_stats()
            except Exception as e:
                logger.warning(f"Failed to get graph memory stats: {e}")
                graph_stats = {"error": str(e)}

        # Calculate totals
        total_entries = (
            dept_stats.get("total_entries", 0) +
            agent_stats.get("total_entries", 0) +
            graph_stats.get("total_nodes", 0)
        )

        # Build sources list
        sources = list(set(
            list(dept_stats.get("categories", [])) +
            list(agent_stats.get("by_namespace", {}).keys()) +
            ["graph_memory"]
        ))

        return UnifiedMemoryStats(
            department_memory=dept_stats,
            agent_memory=agent_stats,
            total_entries=total_entries,
            sources=sources,
            last_sync=self._last_sync,
            sync_status="synced" if self._last_sync else "not_synced",
        )

    def get_all_department_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all departments.

        Returns:
            Dictionary mapping department names to their stats
        """
        all_stats: Dict[str, Dict[str, Any]] = {}

        # Iterate over all known departments
        from src.agents.departments.types import Department

        for dept in Department:
            try:
                facade = UnifiedMemoryFacade(department=dept.value)
                all_stats[dept.value] = facade.get_stats().department_memory
            except Exception as e:
                logger.warning(f"Failed to get stats for department {dept.value}: {e}")
                all_stats[dept.value] = {"error": str(e)}

        return all_stats


# =============================================================================
# Factory Functions
# =============================================================================


def get_unified_memory(
    department: Optional[str] = None,
    agent_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> UnifiedMemoryFacade:
    """
    Factory function to get a UnifiedMemoryFacade instance.

    Args:
        department: Department name
        agent_id: Agent identifier
        session_id: Session identifier

    Returns:
        UnifiedMemoryFacade instance
    """
    return UnifiedMemoryFacade(
        department=department,
        agent_id=agent_id,
        session_id=session_id,
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "UnifiedMemoryFacade",
    "UnifiedMemoryEntry",
    "UnifiedSearchResult",
    "UnifiedMemoryStats",
    "get_unified_memory",
]
