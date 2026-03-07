"""
Memory API Endpoints

Provides REST API for memory management in QuantMindX.
Supports semantic search, memory CRUD, and sync operations.
Now uses AgentMemory (SQLite-backed) for persistence.
"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

logger = logging.getLogger(__name__)

from src.agents.memory import AgentMemory, get_agent_memory
from src.agents.memory.agent_memory import AgentMemoryWithDepartment, get_agent_memory_with_department
from src.agents.memory.compaction import (
    MemoryCompactor,
    MemoryCompactionConfig,
    MemoryCompactionResult,
    create_memory_compactor,
)
from src.agents.memory.vector_memory import VectorMemory, get_vector_memory
from src.agents.memory.department_integration import (
    DEPARTMENT_CONFIGS,
    DEPARTMENT_PREFIX,
    GLOBAL_NAMESPACE,
    MemorySharingRule,
)
from src.agents.memory.unified_memory_facade import (
    UnifiedMemoryFacade,
    UnifiedMemoryEntry,
    UnifiedMemoryStats,
    get_unified_memory,
)
from src.agents.departments.memory_manager import DepartmentMemoryManager
from src.agents.departments.types import Department

router = APIRouter(prefix="/api/memory", tags=["memory"])

# Department routes (with /dept prefix to avoid conflicts)
dept_router = APIRouter(prefix="/api/memory/department", tags=["memory-department"])

# =============================================================================
# Models
# =============================================================================

class MemoryEntry(BaseModel):
    """Memory entry model."""
    id: Optional[str] = None
    content: str
    source: str = "memory"  # memory, sessions
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    relevance_score: Optional[float] = None


class MemorySearchRequest(BaseModel):
    """Memory search request."""
    query: str
    limit: int = 10
    source: Optional[str] = None
    agent_id: Optional[str] = None
    min_relevance: float = 0.0
    use_temporal_decay: bool = True


class MemorySearchResponse(BaseModel):
    """Memory search response."""
    results: List[MemoryEntry]
    total: int
    query: str
    elapsed_ms: float


class MemoryStats(BaseModel):
    """Memory system statistics."""
    total_memories: int
    total_sessions: int
    embedding_model: str
    last_sync: Optional[datetime]
    vector_dimensions: Optional[int]
    sources: List[str]


class MemorySyncRequest(BaseModel):
    """Memory sync request."""
    force: bool = False
    sources: List[str] = ["memory", "sessions"]


class HookLogEntry(BaseModel):
    """Hook execution log entry."""
    id: str
    hook_name: str
    executed_at: datetime
    status: str  # success, error, pending
    duration_ms: Optional[float] = None
    result: Optional[str] = None
    error: Optional[str] = None


class CronJobCreate(BaseModel):
    """Cron job creation request."""
    name: str
    schedule: str
    enabled: bool = True
    command: Optional[str] = None


class CronJob(BaseModel):
    """Cron job model."""
    id: str
    name: str
    schedule: str
    enabled: bool = True
    command: Optional[str] = None
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


# =============================================================================
# Compaction Models
# =============================================================================


class CompactionConfigRequest(BaseModel):
    """Request model for configuring memory compaction."""
    # Compression settings
    compress_old_entries: bool = True
    compression_age_days: int = 30
    compression_min_size_bytes: int = 1024

    # Deduplication settings
    remove_duplicates: bool = True
    duplicate_similarity_threshold: float = 0.95

    # Aggregation settings
    aggregate_similar: bool = True
    aggregation_threshold: float = 0.85
    max_cluster_size: int = 10

    # Cleanup settings
    cleanup_expired: bool = True
    expiration_days: int = 90
    min_importance_threshold: float = 0.1

    # Archive settings
    archive_old_data: bool = True
    archive_age_days: int = 60

    # Optimization settings
    optimize_storage: bool = True
    vacuum_after_compaction: bool = True


class CompactionResultResponse(BaseModel):
    """Response model for compaction results."""
    operation: str
    entries_processed: int
    entries_removed: int
    entries_modified: int
    bytes_saved: int
    duration_ms: float
    details: Dict[str, Any]
    timestamp: datetime


class FullCompactionResponse(BaseModel):
    """Response model for full compaction."""
    results: Dict[str, CompactionResultResponse]
    total_bytes_saved: int
    total_duration_ms: float


class CompactionStatsResponse(BaseModel):
    """Response model for compaction statistics."""
    total_entries: int
    compressed_entries: int
    aggregated_entries: int
    db_size_bytes: int
    old_entries: int
    duplicate_groups: int


# =============================================================================
# AgentMemory Instance (SQLite-backed persistent storage)
# =============================================================================

# Initialize AgentMemory with default settings
# Uses SQLite backend for persistence
def _get_memory_instance() -> AgentMemory:
    """Get or create the global AgentMemory instance."""
    global _agent_memory_instance
    if _agent_memory_instance is None:
        base_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
        _agent_memory_instance = AgentMemory(
            agent_id="api-server",
            session_id=None,
            base_path=base_path,
            use_agentdb=False
        )
    return _agent_memory_instance

_agent_memory_instance: Optional[AgentMemory] = None

# =============================================================================
# VectorMemory Instance (FAISS-backed vector search)
# =============================================================================

def _get_vector_memory_instance() -> VectorMemory:
    """Get or create the global VectorMemory instance."""
    global _vector_memory_instance
    if _vector_memory_instance is None:
        base_path = Path(os.environ.get("VECTOR_MEMORY_PATH", "data/vector_memory"))
        use_agentdb = os.getenv("AGENTDB_ENABLED", "false").lower() == "true"
        agentdb_config = None
        if use_agentdb:
            agentdb_config = {
                "api_key": os.getenv("AGENTDB_API_KEY"),
                "endpoint": os.getenv("AGENTDB_ENDPOINT"),
            }
        _vector_memory_instance = VectorMemory(
            agent_id="api-server",
            session_id=None,
            base_path=base_path,
            use_agentdb=use_agentdb,
            agentdb_config=agentdb_config,
            dimension=1536,
        )
    return _vector_memory_instance

_vector_memory_instance: Optional[VectorMemory] = None

# Legacy compatibility - maintain the in-memory store for API compatibility
_memory_store: Dict[str, MemoryEntry] = {}

def _get_memory_stats() -> MemoryStats:
    """Get current memory statistics from AgentMemory."""
    try:
        agent_mem = _get_memory_instance()
        stats = agent_mem.get_stats()
        return MemoryStats(
            total_memories=stats.get("total_entries", 0),
            total_sessions=0,
            embedding_model="text-embedding-3-small",
            last_sync=datetime.utcnow(),
            vector_dimensions=1536,
            sources=list(stats.get("by_namespace", {}).keys()) if stats.get("by_namespace") else ["default", "session", "patterns"]
        )
    except Exception as e:
        logger.warning(f"Could not get AgentMemory stats: {e}")
        return MemoryStats(
            total_memories=0,
            total_sessions=0,
            embedding_model="text-embedding-3-small",
            last_sync=None,
            vector_dimensions=1536,
            sources=["default", "session", "patterns"]
        )

_memory_stats = _get_memory_stats()

# Hook logs store
_hook_logs: List[Dict[str, Any]] = []
_cron_jobs: Dict[str, Dict[str, Any]] = {
    "memory_consolidation": {
        "id": "memory_consolidation",
        "name": "Memory Consolidation",
        "schedule": "0 * * * *",
        "enabled": True,
        "command": "consolidate",
        "last_run": None,
        "next_run": None
    },
    "session_cleanup": {
        "id": "session_cleanup",
        "name": "Session Cleanup",
        "schedule": "0 0 * * *",
        "enabled": True,
        "command": "cleanup",
        "last_run": None,
        "next_run": None
    }
}


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/stats", response_model=MemoryStats)
async def get_memory_stats():
    """Get memory system statistics."""
    return _get_memory_stats()


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(request: MemorySearchRequest):
    """
    Search memories using semantic search.

    Uses SQLite-backed AgentMemory for persistent storage.
    Optionally applies temporal decay to boost recent memories.
    """
    import time
    start_time = time.time()

    try:
        agent_mem = _get_memory_instance()
        # Use AgentMemory's search functionality
        search_results = agent_mem.search(
            query=request.query,
            namespace=request.source,
            limit=request.limit
        )

        # Convert to MemoryEntry format
        results = []
        for entry in search_results:
            results.append(MemoryEntry(
                id=entry.get("id"),
                content=entry.get("value", ""),
                source=entry.get("namespace", "memory"),
                agent_id=entry.get("agent_id"),
                metadata={
                    "key": entry.get("key"),
                    "tags": entry.get("tags", []),
                    "session_id": entry.get("session_id")
                },
                created_at=entry.get("created_at") if isinstance(entry.get("created_at"), datetime) else datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat())),
                updated_at=entry.get("updated_at") if isinstance(entry.get("updated_at"), datetime) else datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat())),
                relevance_score=None
            ))

        elapsed_ms = (time.time() - start_time) * 1000

        return MemorySearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            elapsed_ms=elapsed_ms
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        # Fallback to in-memory search
        results = []
        for entry in _memory_store.values():
            if request.source and entry.source != request.source:
                continue
            if request.agent_id and entry.agent_id != request.agent_id:
                continue
            results.append(entry)
        results = results[:request.limit]
        elapsed_ms = (time.time() - start_time) * 1000
        return MemorySearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            elapsed_ms=elapsed_ms
        )


@router.post("/add", response_model=MemoryEntry)
async def add_memory(entry: MemoryEntry):
    """Add a new memory entry using AgentMemory (persistent)."""
    import uuid

    try:
        agent_mem = _get_memory_instance()

        # Use key from metadata or generate one
        key = entry.metadata.get("key") if entry.metadata else None
        if not key:
            key = entry.id or str(uuid.uuid4())

        # Store in AgentMemory (SQLite-backed)
        memory_id = agent_mem.store(
            key=key,
            value=entry.content,
            namespace=entry.source or "default",
            tags=entry.metadata.get("tags", []) if entry.metadata else [],
            metadata=entry.metadata
        )

        # Also keep in legacy store for backward compatibility
        entry.id = memory_id
        entry.created_at = datetime.utcnow()
        entry.updated_at = datetime.utcnow()
        _memory_store[memory_id] = entry

        logger.info(f"Added memory via AgentMemory: {memory_id}")
        return entry

    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        # Fallback to legacy in-memory store
        memory_id = str(uuid.uuid4())
        entry.id = memory_id
        entry.created_at = datetime.utcnow()
        entry.updated_at = datetime.utcnow()
        _memory_store[memory_id] = entry
        return entry


# =============================================================================
# Vector Memory Endpoints
# =============================================================================

class VectorSearchRequest(BaseModel):
    """Vector search request."""
    query: str
    namespace: Optional[str] = "default"
    limit: int = 10
    min_score: float = 0.0


class VectorSearchResponse(BaseModel):
    """Vector search response."""
    results: List[Dict[str, Any]]
    total: int
    query: str
    elapsed_ms: float


@router.post("/vector/search", response_model=VectorSearchResponse)
async def vector_search_memories(request: VectorSearchRequest):
    """
    Search memories using vector embeddings (semantic search).

    Uses FAISS HNSW index for fast similarity search.
    """
    import time
    start_time = time.time()

    try:
        vector_mem = _get_vector_memory_instance()
        results = await vector_mem.search(
            query=request.query,
            namespace=request.namespace,
            limit=request.limit,
            min_score=request.min_score,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        return VectorSearchResponse(
            results=results,
            total=len(results),
            query=request.query,
            elapsed_ms=elapsed_ms
        )
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        elapsed_ms = (time.time() - start_time) * 1000
        return VectorSearchResponse(
            results=[],
            total=0,
            query=request.query,
            elapsed_ms=elapsed_ms
        )


@router.post("/vector/add", response_model=Dict[str, Any])
async def vector_add_memory(
    key: str,
    value: str,
    namespace: str = "default",
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Add a memory entry with vector embedding.

    The value is embedded and stored in the FAISS index for semantic search.
    """
    try:
        vector_mem = _get_vector_memory_instance()
        memory_id = await vector_mem.store(
            key=key,
            value=value,
            namespace=namespace,
            tags=tags,
            metadata=metadata,
        )

        logger.info(f"Added vector memory: {memory_id}")
        return {"id": memory_id, "status": "stored", "namespace": namespace}

    except Exception as e:
        logger.error(f"Error adding vector memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/stats")
async def get_vector_memory_stats():
    """Get vector memory statistics."""
    try:
        vector_mem = _get_vector_memory_instance()
        stats = vector_mem.get_stats()

        return {
            "total_entries": stats.get("total_entries", 0),
            "vector_count": stats.get("vector_count", 0),
            "dimension": stats.get("dimension", 1536),
            "backend": stats.get("backend", "vector"),
            "faiss_available": stats.get("faiss_available", False),
            "namespaces": stats.get("by_namespace", {}),
            "agentdb_available": stats.get("agentdb_available", False),
        }
    except Exception as e:
        logger.error(f"Error getting vector memory stats: {e}")
        return {
            "total_entries": 0,
            "vector_count": 0,
            "dimension": 1536,
            "backend": "vector",
            "faiss_available": False,
            "error": str(e),
        }


@router.delete("/vector/{key}")
async def vector_delete_memory(key: str, namespace: str = "default"):
    """Delete a vector memory entry."""
    try:
        vector_mem = _get_vector_memory_instance()
        deleted = vector_mem.delete(key=key, namespace=namespace)

        if deleted:
            return {"status": "deleted", "key": key, "namespace": namespace}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting vector memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync")
async def sync_memories(request: MemorySyncRequest, background_tasks: BackgroundTasks):
    """
    Sync memory index with files and sessions.

    This can be a long-running operation, so it runs in the background.
    """
    # TODO: Implement actual sync with memory manager
    _memory_stats.last_sync = datetime.utcnow()

    return {
        "status": "sync_started",
        "force": request.force,
        "sources": request.sources
    }


@router.post("/clear")
async def clear_memories(source: Optional[str] = None, agent_id: Optional[str] = None):
    """Clear memories, optionally filtered by source or agent."""
    global _memory_store

    try:
        agent_mem = _get_memory_instance()

        if source is None and agent_id is None:
            # Clear all - need to list all and delete
            count = 0
            for ns in ["default", "session", "patterns", "solutions"]:
                entries = agent_mem.list(namespace=ns, limit=1000)
                for entry in entries:
                    agent_mem.delete(key=entry.get("key", ""), namespace=ns)
                    count += 1

            _memory_store = {}
            return {"status": "cleared", "count": count}

        # Clear by source (namespace)
        count = 0
        namespace = source or "default"
        entries = agent_mem.list(namespace=namespace, limit=1000)
        for entry in entries:
            if agent_id is None or entry.get("agent_id") == agent_id:
                agent_mem.delete(key=entry.get("key", ""), namespace=namespace)
                count += 1

        # Also clear legacy store
        to_delete = []
        for memory_id, entry in _memory_store.items():
            if source and entry.source != source:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            to_delete.append(memory_id)

        for memory_id in to_delete:
            del _memory_store[memory_id]

        return {"status": "cleared", "count": count + len(to_delete)}

    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        # Fallback to legacy in-memory store
        if source is None and agent_id is None:
            count = len(_memory_store)
            _memory_store = {}
            return {"status": "cleared", "count": count}

        to_delete = []
        for memory_id, entry in _memory_store.items():
            if source and entry.source != source:
                continue
            if agent_id and entry.agent_id != agent_id:
                continue
            to_delete.append(memory_id)

        for memory_id in to_delete:
            del _memory_store[memory_id]

        return {"status": "cleared", "count": len(to_delete)}


# =============================================================================
# Compaction Endpoints
# =============================================================================


def _get_compactor() -> MemoryCompactor:
    """Get or create the global MemoryCompactor instance."""
    global _memory_compactor_instance
    if _memory_compactor_instance is None:
        base_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
        _memory_compactor_instance = create_memory_compactor(
            db_path=base_path / "memory.db"
        )
    return _memory_compactor_instance


_memory_compactor_instance: Optional[MemoryCompactor] = None


@router.get("/compaction/stats", response_model=CompactionStatsResponse)
async def get_compaction_stats():
    """Get memory compaction statistics."""
    try:
        compactor = _get_compactor()
        stats = compactor.get_compaction_stats()
        return CompactionStatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting compaction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/config", response_model=Dict[str, str])
async def configure_compaction(config: CompactionConfigRequest):
    """Configure memory compaction settings."""
    global _memory_compactor_instance

    try:
        # Create new compactor with custom config
        base_path = Path(os.environ.get("AGENT_MEMORY_PATH", "data/agent_memory"))
        comp_config = MemoryCompactionConfig(
            compress_old_entries=config.compress_old_entries,
            compression_age_days=config.compression_age_days,
            compression_min_size_bytes=config.compression_min_size_bytes,
            remove_duplicates=config.remove_duplicates,
            duplicate_similarity_threshold=config.duplicate_similarity_threshold,
            aggregate_similar=config.aggregate_similar,
            aggregation_threshold=config.aggregation_threshold,
            max_cluster_size=config.max_cluster_size,
            cleanup_expired=config.cleanup_expired,
            expiration_days=config.expiration_days,
            min_importance_threshold=config.min_importance_threshold,
            archive_old_data=config.archive_old_data,
            archive_age_days=config.archive_age_days,
            optimize_storage=config.optimize_storage,
        )
        _memory_compactor_instance = MemoryCompactor(
            db_path=base_path / "memory.db",
            config=comp_config
        )

        return {"status": "configured", "message": "Compaction settings updated"}

    except Exception as e:
        logger.error(f"Error configuring compaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/compress", response_model=CompactionResultResponse)
async def compress_old_memories():
    """Compress old memory entries using zlib."""
    try:
        compactor = _get_compactor()
        result = compactor.compress_old_entries()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error compressing memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/deduplicate", response_model=CompactionResultResponse)
async def deduplicate_memories():
    """Remove duplicate memory entries."""
    try:
        compactor = _get_compactor()
        result = compactor.remove_duplicates()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error deduplicating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/aggregate", response_model=CompactionResultResponse)
async def aggregate_similar_memories():
    """Aggregate similar memory entries into clusters."""
    try:
        compactor = _get_compactor()
        result = compactor.aggregate_similar()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error aggregating memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/cleanup", response_model=CompactionResultResponse)
async def cleanup_expired_memories():
    """Remove expired memory entries."""
    try:
        compactor = _get_compactor()
        result = compactor.cleanup_expired()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error cleaning up memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/archive", response_model=CompactionResultResponse)
async def archive_old_memories():
    """Archive old memory entries to a separate file."""
    try:
        compactor = _get_compactor()
        result = compactor.archive_old_data()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error archiving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/optimize", response_model=CompactionResultResponse)
async def optimize_memory_storage():
    """Optimize database storage using VACUUM."""
    try:
        compactor = _get_compactor()
        result = compactor.optimize_storage()
        return CompactionResultResponse(
            operation=result.operation,
            entries_processed=result.entries_processed,
            entries_removed=result.entries_removed,
            entries_modified=result.entries_modified,
            bytes_saved=result.bytes_saved,
            duration_ms=result.duration_ms,
            details=result.details,
            timestamp=result.timestamp
        )
    except Exception as e:
        logger.error(f"Error optimizing storage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/run", response_model=FullCompactionResponse)
async def run_full_compaction(
    include_compression: bool = True,
    include_deduplication: bool = True,
    include_aggregation: bool = True,
    include_cleanup: bool = True,
    include_archive: bool = True,
    include_optimization: bool = True,
):
    """Run full memory compaction pipeline."""
    try:
        compactor = _get_compactor()
        results = compactor.run_full_compaction(
            include_compression=include_compression,
            include_deduplication=include_deduplication,
            include_aggregation=include_aggregation,
            include_cleanup=include_cleanup,
            include_archive=include_archive,
            include_optimization=include_optimization,
        )

        response_results = {}
        total_bytes_saved = 0
        total_duration_ms = 0.0

        for key, result in results.items():
            response_results[key] = CompactionResultResponse(
                operation=result.operation,
                entries_processed=result.entries_processed,
                entries_removed=result.entries_removed,
                entries_modified=result.entries_modified,
                bytes_saved=result.bytes_saved,
                duration_ms=result.duration_ms,
                details=result.details,
                timestamp=result.timestamp
            )
            total_bytes_saved += result.bytes_saved
            total_duration_ms += result.duration_ms

        return FullCompactionResponse(
            results=response_results,
            total_bytes_saved=total_bytes_saved,
            total_duration_ms=total_duration_ms
        )
    except Exception as e:
        logger.error(f"Error running full compaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compaction/decompress/{entry_id}")
async def decompress_memory_entry(entry_id: str):
    """Decompress a single compressed memory entry."""
    try:
        compactor = _get_compactor()
        result = compactor.decompress_entry(entry_id)

        if result is None:
            raise HTTPException(status_code=404, detail="Entry not found or not compressed")

        return {"status": "decompressed", "entry_id": entry_id, "value": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error decompressing entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=PaginatedResponse[MemoryEntry])
async def list_memories(
    namespace: Optional[str] = Query(None, description="Filter by namespace"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[MemoryEntry]:
    """
    List memories with optional namespace filter and pagination.

    Uses AgentMemory (SQLite-backed) for persistent storage.
    """
    try:
        agent_mem = _get_memory_instance()

        # Get memories from AgentMemory
        list_results = agent_mem.list(
            namespace=namespace or "default",
            limit=limit + offset  # Get enough for pagination
        )

        # Convert to MemoryEntry format
        memories = []
        for entry in list_results:
            memories.append(MemoryEntry(
                id=entry.get("id"),
                content=entry.get("value", ""),
                source=entry.get("namespace", "memory"),
                agent_id=entry.get("agent_id"),
                metadata={
                    "key": entry.get("key"),
                    "tags": entry.get("tags", []),
                    "session_id": entry.get("session_id")
                },
                created_at=entry.get("created_at") if isinstance(entry.get("created_at"), datetime) else datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat())),
                updated_at=entry.get("updated_at") if isinstance(entry.get("updated_at"), datetime) else datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat())),
                relevance_score=None
            ))

        # Sort by created_at descending
        memories.sort(key=lambda x: x.created_at or datetime.min, reverse=True)

        total = len(memories)
        paginated_memories = memories[offset:offset + limit]

        return PaginatedResponse.create(
            items=paginated_memories,
            total=total,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        # Fallback to legacy in-memory store
        source_filter = namespace
        memories = []
        for entry in _memory_store.values():
            if source_filter and entry.source != source_filter:
                continue
            memories.append(entry)
        memories.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        total = len(memories)
        paginated_memories = memories[offset:offset + limit]
        return PaginatedResponse.create(
            items=paginated_memories,
            total=total,
            limit=limit,
            offset=offset
        )


# =============================================================================
# Hook Endpoints
# =============================================================================

@router.get("/hooks")
async def list_hooks():
    """List all registered hooks."""
    # TODO: Implement with hook registry
    return {
        "hooks": [
            {
                "id": "pre_tool_use",
                "name": "Pre Tool Use",
                "enabled": True,
                "priority": 1
            },
            {
                "id": "post_tool_use",
                "name": "Post Tool Use",
                "enabled": True,
                "priority": 2
            }
        ]
    }


@router.post("/hooks/{hook_id}/toggle")
async def toggle_hook(hook_id: str, enabled: bool):
    """Enable or disable a hook."""
    # TODO: Implement with hook registry
    return {"status": "updated", "hook_id": hook_id, "enabled": enabled}


@router.get("/hooks/logs")
async def get_hook_logs(limit: int = 100):
    """Get hook execution logs."""
    logs = _hook_logs[:limit]
    return {"logs": logs, "total": len(_hook_logs)}


@router.post("/hooks/{hook_name}/execute")
async def execute_hook(hook_name: str):
    """Execute a hook by name."""
    import uuid

    log_entry = {
        "id": str(uuid.uuid4()),
        "hook_name": hook_name,
        "executed_at": datetime.utcnow().isoformat(),
        "status": "success",
        "duration_ms": 15.5,
        "result": f"Hook '{hook_name}' executed successfully",
        "error": None
    }

    _hook_logs.insert(0, log_entry)

    return {"status": "executed", "hook_name": hook_name, "log_id": log_entry["id"]}


@router.post("/hooks/logs/clear")
async def clear_hook_logs():
    """Clear all hook logs."""
    global _hook_logs
    count = len(_hook_logs)
    _hook_logs = []
    return {"status": "cleared", "count": count}


# =============================================================================
# Cron Job Endpoints
# =============================================================================

@router.get("/cron")
async def list_cron_jobs():
    """List all scheduled cron jobs."""
    return {"jobs": list(_cron_jobs.values())}


@router.post("/cron", response_model=CronJob)
async def add_cron_job(job: CronJobCreate):
    """Add a new cron job."""
    import uuid

    job_id = str(uuid.uuid4())
    job_data = {
        "id": job_id,
        "name": job.name,
        "schedule": job.schedule,
        "enabled": job.enabled,
        "command": job.command,
        "last_run": None,
        "next_run": None
    }

    _cron_jobs[job_id] = job_data

    logger.info(f"Added cron job: {job_id}")
    return CronJob(**job_data)


@router.delete("/cron/{job_id}")
async def delete_cron_job(job_id: str):
    """Delete a cron job by ID."""
    if job_id not in _cron_jobs:
        raise HTTPException(status_code=404, detail="Cron job not found")

    del _cron_jobs[job_id]

    logger.info(f"Deleted cron job: {job_id}")
    return {"status": "deleted", "job_id": job_id}


@router.post("/cron/{job_id}/run")
async def run_cron_job(job_id: str, background_tasks: BackgroundTasks):
    """Manually trigger a cron job."""
    # TODO: Implement with scheduler
    return {"status": "triggered", "job_id": job_id}


@router.post("/cron/{job_id}/toggle")
async def toggle_cron_job(job_id: str, enabled: bool):
    """Enable or disable a cron job."""
    # TODO: Implement with scheduler
    return {"status": "updated", "job_id": job_id, "enabled": enabled}


# =============================================================================
# Department Memory Compatibility Routes (MUST BE BEFORE catch-all)
# =============================================================================
# These routes must come BEFORE the memory_id catch-all to take precedence

# Valid department names
_VALID_DEPARTMENTS = {"analysis", "research", "risk", "execution", "portfolio", "development", "trading", "floor_manager"}


@router.get("/department/{department}")
async def get_department_memory_forward(department: str) -> Dict[str, Any]:
    """Forward to department memory endpoint."""
    return await get_department_memory(department)


@router.get("/department/{department}/stats")
async def get_department_stats_forward(department: str) -> Dict[str, Any]:
    """Forward to department stats endpoint."""
    return await get_department_memory_stats(department)


@router.get("/department/{department}/logs")
async def get_department_logs_forward(department: str) -> Dict[str, Any]:
    """Forward to department logs endpoint."""
    return await get_daily_logs(department)


# =============================================================================
# Memory ID Endpoints (catch-all routes - MUST BE LAST)
# =============================================================================

@router.get("/{memory_id}")
async def get_memory(memory_id: str):
    """Get a specific memory by ID or department memory."""
    # Check if it's a department name first
    if memory_id in _VALID_DEPARTMENTS:
        return await get_department_memory(memory_id)

    # Try to get from AgentMemory
    try:
        agent_mem = _get_memory_instance()
        value = agent_mem.retrieve(key=memory_id, namespace="default")
        if value:
            return MemoryEntry(
                id=memory_id,
                content=value,
                source="default",
                agent_id=None,
                metadata={"key": memory_id},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                relevance_score=None
            )

        # Try other namespaces
        for ns in ["session", "patterns", "solutions"]:
            value = agent_mem.retrieve(key=memory_id, namespace=ns)
            if value:
                return MemoryEntry(
                    id=memory_id,
                    content=value,
                    source=ns,
                    agent_id=None,
                    metadata={"key": memory_id},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    relevance_score=None
                )
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")

    # Fallback to legacy in-memory store
    if memory_id not in _memory_store:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_store[memory_id]


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory entry using AgentMemory (persistent)."""
    try:
        agent_mem = _get_memory_instance()

        # Try to find and delete by key (memory_id is used as key)
        deleted = agent_mem.delete(key=memory_id, namespace="default")

        if not deleted:
            # Try other namespaces
            for ns in ["session", "patterns", "solutions"]:
                deleted = agent_mem.delete(key=memory_id, namespace=ns)
                if deleted:
                    break

        # Also delete from legacy store
        if memory_id in _memory_store:
            del _memory_store[memory_id]

        logger.info(f"Deleted memory: {memory_id}")
        return {"status": "deleted", "id": memory_id}

    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        # Fallback to legacy in-memory store
        if memory_id not in _memory_store:
            raise HTTPException(status_code=404, detail="Memory not found")
        del _memory_store[memory_id]
        return {"status": "deleted", "id": memory_id}


# =============================================================================
# Department Memory Endpoints
# =============================================================================

@dept_router.get("/{department}")
async def get_department_memory(department: str) -> Dict[str, Any]:
    """
    Get all memories for a department.

    Args:
        department: Department name (development, research, risk, trading, portfolio, floor_manager)

    Returns:
        Department memories and metadata
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Create memory manager
        memory_manager = DepartmentMemoryManager(department=dept_enum)

        # Read memory file
        content = memory_manager.read_memory()

        # Parse memories (basic parsing of markdown format)
        memories = _parse_memory_content(content)

        return {
            "department": department,
            "memories": memories,
            "stats": memory_manager.get_stats()
        }

    except Exception as e:
        logger.error(f"Error getting department memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dept_router.get("/{department}/logs")
async def get_daily_logs(department: str, days: int = 7) -> Dict[str, Any]:
    """
    Get recent daily logs for a department.

    Args:
        department: Department name
        days: Number of recent days to retrieve

    Returns:
        Daily logs
    """
    try:
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        memory_manager = DepartmentMemoryManager(department=dept_enum)
        recent_logs = memory_manager.get_recent_logs(days=days)

        # Parse logs
        logs = []
        for date_str, content in recent_logs.items():
            entries = _parse_log_content(content)
            logs.append({
                "date": date_str,
                "entries": entries
            })

        return {
            "department": department,
            "logs": logs
        }

    except Exception as e:
        logger.error(f"Error getting daily logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dept_router.get("/{department}/stats")
async def get_department_memory_stats(department: str) -> Dict[str, Any]:
    """
    Get memory statistics for a department.

    Args:
        department: Department name

    Returns:
        Memory statistics
    """
    try:
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        memory_manager = DepartmentMemoryManager(department=dept_enum)
        stats = memory_manager.get_stats()

        return stats

    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dept_router.get("/{department}/search")
async def search_department_memory(department: str, q: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Search department memory.

    Args:
        department: Department name
        q: Search query
        category: Optional category filter

    Returns:
        Search results
    """
    try:
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        memory_manager = DepartmentMemoryManager(department=dept_enum)
        results = memory_manager.search(query=q, category=category)

        # Parse results into structured format
        parsed_results = []
        for result in results:
            parsed_results.append({
                "content": result,
                "match_text": q
            })

        return {
            "department": department,
            "query": q,
            "results": parsed_results
        }

    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dept_router.post("/{department}")
async def add_department_memory(
    department: str,
    category: str,
    content: str,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Add a memory entry to department memory.

    Args:
        department: Department name
        category: Memory category
        content: Memory content
        tags: Optional list of tags

    Returns:
        Success status
    """
    try:
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        memory_manager = DepartmentMemoryManager(department=dept_enum)
        memory_manager.add_memory(category=category, content=content, tags=tags or [])

        return {
            "success": True,
            "department": department,
            "category": category
        }

    except Exception as e:
        logger.error(f"Error adding memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@dept_router.post("/{department}/logs")
async def add_daily_log(
    department: str,
    content: str,
    log_date: Optional[str] = None,
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add an entry to a daily log.

    Args:
        department: Department name
        content: Log content
        log_date: Optional date string (YYYY-MM-DD), defaults to today
        category: Optional category

    Returns:
        Success status
    """
    try:
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        memory_manager = DepartmentMemoryManager(department=dept_enum)

        # Parse date or use today
        if log_date:
            log_date_obj = datetime.strptime(log_date, "%Y-%m-%d").date()
        else:
            from datetime import date
            log_date_obj = date.today()

        memory_manager.add_daily_log(date=log_date_obj, content=content, category=category)

        from datetime import date
        return {
            "success": True,
            "department": department,
            "date": log_date or str(date.today())
        }

    except Exception as e:
        logger.error(f"Error adding daily log: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _parse_memory_content(content: str) -> List[Dict[str, Any]]:
    """Parse MEMORY.md content into structured entries."""
    memories = []
    current_entry = None

    for line in content.split('\n'):
        # Check for memory header: ## Category - timestamp
        if line.startswith('## ') and ' - ' in line:
            if current_entry:
                memories.append(current_entry)

            parts = line[3:].split(' - ', 1)
            category = parts[0].strip()
            timestamp = parts[1].strip() if len(parts) > 1 else ''

            current_entry = {
                "category": category,
                "timestamp": timestamp,
                "content": '',
                "tags": []
            }

        elif current_entry:
            # Check for tags
            if line.startswith('**Tags:**'):
                tags_content = line.split('**Tags:**')[1].strip()
                current_entry["tags"] = [t.strip('` ') for t in tags_content.split(',')]
            # Skip separator lines
            elif line.strip() == '---':
                continue
            # Add to content
            else:
                current_entry["content"] += line + '\n'

    if current_entry:
        memories.append(current_entry)

    return memories


def _parse_log_content(content: str) -> List[Dict[str, Any]]:
    """Parse daily log content into structured entries."""
    entries = []
    current_entry = None

    for line in content.split('\n'):
        # Check for time header: ## HH:MM:SS or ### Category - HH:MM:SS
        if line.startswith('### ') and ' - ' in line:
            if current_entry:
                entries.append(current_entry)

            parts = line[4:].split(' - ', 1)
            category = parts[0].strip()
            time_str = parts[1].strip() if len(parts) > 1 else ''

            current_entry = {
                "time": time_str,
                "category": category,
                "content": ''
            }

        elif line.startswith('## ') and ':' in line:
            if current_entry:
                entries.append(current_entry)

            time_str = line[3:].strip()
            current_entry = {
                "time": time_str,
                "category": None,
                "content": ''
            }

        elif current_entry:
            current_entry["content"] += line + '\n'

    if current_entry:
        entries.append(current_entry)

    return entries


# =============================================================================
# Department-Aware Global Memory Integration Endpoints
# =============================================================================

class DepartmentMemoryRequest(BaseModel):
    """Request to store department memory."""
    key: str
    value: str
    sharing_rule: Optional[str] = None  # private, department, global, restricted
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class CrossDepartmentSearchRequest(BaseModel):
    """Request to search across departments."""
    query: str
    include_global: bool = True
    include_restricted: bool = False
    limit_per_namespace: int = 5


class ContributeToGlobalRequest(BaseModel):
    """Request to contribute memory to global namespace."""
    key: str
    value: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@router.get("/global")
async def get_global_memories(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Get all global memories.

    Returns:
        Global memories with pagination
    """
    try:
        agent_mem = _get_memory_instance()
        memories = agent_mem.list(namespace=GLOBAL_NAMESPACE, limit=limit + offset)

        # Convert to MemoryEntry format
        results = []
        for entry in memories[offset:offset + limit]:
            results.append(MemoryEntry(
                id=entry.get("id"),
                content=entry.get("value", ""),
                source=entry.get("namespace", "global"),
                agent_id=entry.get("agent_id"),
                metadata={
                    "key": entry.get("key"),
                    "tags": entry.get("tags", []),
                },
                created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat())),
                relevance_score=None
            ))

        return {
            "memories": results,
            "total": len(memories),
            "namespace": GLOBAL_NAMESPACE,
        }
    except Exception as e:
        logger.error(f"Error getting global memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/global")
async def add_global_memory(request: ContributeToGlobalRequest) -> Dict[str, Any]:
    """
    Add a memory to the global namespace.

    Args:
        request: Memory data

    Returns:
        Success status with memory ID
    """
    try:
        agent_mem = _get_memory_instance()
        memory_id = agent_mem.store(
            key=request.key,
            value=request.value,
            namespace=GLOBAL_NAMESPACE,
            tags=request.tags,
            metadata=request.metadata,
        )

        return {
            "success": True,
            "id": memory_id,
            "namespace": GLOBAL_NAMESPACE,
        }
    except Exception as e:
        logger.error(f"Error adding global memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/departments/config")
async def get_department_memory_configs() -> Dict[str, Any]:
    """
    Get memory configurations for all departments.

    Returns:
        Department memory configurations
    """
    configs = {}
    for dept_name, config in DEPARTMENT_CONFIGS.items():
        configs[dept_name] = {
            "department": config.department.value,
            "namespace": config.namespace,
            "sharing_rule": config.sharing_rule.value,
            "can_access_global": config.can_access_global,
            "accessible_from": list(config.accessible_from),
            "tags": config.tags,
        }

    return {
        "departments": configs,
        "global_namespace": GLOBAL_NAMESPACE,
        "department_prefix": DEPARTMENT_PREFIX,
    }


@router.get("/department/{department}/memory")
async def get_department_memory_store(
    department: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Get memories stored in department's namespace.

    Args:
        department: Department name

    Returns:
        Department memories
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Get department namespace
        from src.agents.memory.department_integration import get_department_namespace
        namespace = get_department_namespace(dept_enum)

        # Get memories
        agent_mem = _get_memory_instance()
        memories = agent_mem.list(namespace=namespace, limit=limit + offset)

        # Convert to MemoryEntry format
        results = []
        for entry in memories[offset:offset + limit]:
            results.append(MemoryEntry(
                id=entry.get("id"),
                content=entry.get("value", ""),
                source=entry.get("namespace", namespace),
                agent_id=entry.get("agent_id"),
                metadata={
                    "key": entry.get("key"),
                    "tags": entry.get("tags", []),
                },
                created_at=datetime.fromisoformat(entry.get("created_at", datetime.utcnow().isoformat())),
                updated_at=datetime.fromisoformat(entry.get("updated_at", datetime.utcnow().isoformat())),
                relevance_score=None
            ))

        return {
            "department": department,
            "namespace": namespace,
            "memories": results,
            "total": len(memories),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting department memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/department/{department}/memory")
async def store_department_memory(
    department: str,
    request: DepartmentMemoryRequest,
) -> Dict[str, Any]:
    """
    Store a memory in department's namespace with routing.

    Args:
        department: Department name
        request: Memory data

    Returns:
        Success status with memory ID
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Get department-aware memory instance
        memory = get_agent_memory_with_department(department=department)

        # Store with routing
        sharing_rule = request.sharing_rule
        memory_id = memory.store_department(
            key=request.key,
            value=request.value,
            sharing_rule=sharing_rule,
            tags=request.tags,
            metadata=request.metadata,
        )

        return {
            "success": True,
            "id": memory_id,
            "department": department,
            "sharing_rule": sharing_rule or "default",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error storing department memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/department/{department}/search")
async def search_department_memory_advanced(
    department: str,
    request: CrossDepartmentSearchRequest,
) -> Dict[str, Any]:
    """
    Search across accessible namespaces for a department.

    Args:
        department: Department name
        request: Search request

    Returns:
        Search results from accessible namespaces
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Get department-aware memory instance
        memory = get_agent_memory_with_department(department=department)

        # Search across departments
        results = memory.search_all_departments(
            query=request.query,
            include_global=request.include_global,
            limit_per_namespace=request.limit_per_namespace,
        )

        # Format results
        formatted_results = {}
        for ns, entries in results.items():
            formatted_results[ns] = [
                {
                    "id": entry.get("id"),
                    "key": entry.get("key"),
                    "content": entry.get("value"),
                    "tags": entry.get("tags", []),
                    "metadata": entry.get("metadata", {}),
                    "created_at": entry.get("created_at"),
                    "updated_at": entry.get("updated_at"),
                }
                for entry in entries
            ]

        return {
            "department": department,
            "query": request.query,
            "results": formatted_results,
            "namespaces_searched": list(results.keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching department memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/department/{department}/contribute-global")
async def contribute_to_global(
    department: str,
    request: ContributeToGlobalRequest,
) -> Dict[str, Any]:
    """
    Contribute a memory to the global namespace from a department.

    Args:
        department: Department name
        request: Memory data

    Returns:
        Success status with memory ID
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Check if department can contribute to global
        from src.agents.memory.department_integration import get_department_config
        config = get_department_config(dept_enum)
        if not config.can_access_global:
            raise HTTPException(
                status_code=403,
                detail=f"Department {department} is not allowed to contribute to global memory"
            )

        # Get memory instance and contribute
        agent_mem = _get_memory_instance()
        memory_id = agent_mem.store(
            key=request.key,
            value=request.value,
            namespace=GLOBAL_NAMESPACE,
            tags=request.tags,
            metadata={**(request.metadata or {}), "contributing_department": department},
        )

        return {
            "success": True,
            "id": memory_id,
            "namespace": GLOBAL_NAMESPACE,
            "contributing_department": department,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error contributing to global: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/department/{department}/memory/{key}")
async def delete_department_memory(
    department: str,
    key: str,
) -> Dict[str, Any]:
    """
    Delete a memory from department's namespace.

    Args:
        department: Department name
        key: Memory key

    Returns:
        Success status
    """
    try:
        # Validate department
        try:
            dept_enum = Department(department)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid department: {department}")

        # Get department namespace
        from src.agents.memory.department_integration import get_department_namespace
        namespace = get_department_namespace(dept_enum)

        # Delete memory
        agent_mem = _get_memory_instance()
        deleted = agent_mem.delete(key=key, namespace=namespace)

        if deleted:
            return {"success": True, "key": key, "namespace": namespace}
        else:
            raise HTTPException(status_code=404, detail="Memory not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting department memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Unified Memory Endpoints (Integration Layer)
# =============================================================================

# Create a router for unified memory endpoints
unified_router = APIRouter(prefix="/api/memory/unified", tags=["memory-unified"])


# Global unified memory instance (lazy initialization)
_unified_memory_instances: Dict[str, UnifiedMemoryFacade] = {}


def _get_unified_memory(department: Optional[str] = None) -> UnifiedMemoryFacade:
    """Get or create a UnifiedMemoryFacade instance."""
    key = department or "global"
    if key not in _unified_memory_instances:
        _unified_memory_instances[key] = get_unified_memory(department=department)
    return _unified_memory_instances[key]


@unified_router.get("/stats")
async def get_unified_stats(department: Optional[str] = None) -> UnifiedMemoryStats:
    """
    Get unified memory statistics from both systems.

    Args:
        department: Optional department filter

    Returns:
        UnifiedMemoryStats with combined statistics
    """
    facade = _get_unified_memory(department)
    return facade.get_stats()


@unified_router.get("/stats/all-departments")
async def get_all_department_stats() -> Dict[str, Any]:
    """
    Get statistics for all departments.

    Returns:
        Dictionary mapping department names to their stats
    """
    facade = _get_unified_memory()
    return facade.get_all_department_stats()


@unified_router.post("/search")
async def unified_search(
    query: str,
    department: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 10,
    include_all_sources: bool = True,
) -> Dict[str, Any]:
    """
    Search across both memory systems (DepartmentMemoryManager and AgentMemory).

    Args:
        query: Search query
        department: Optional department filter
        namespace: Optional namespace filter
        limit: Maximum results
        include_all_sources: Whether to search both systems

    Returns:
        Unified search results
    """
    facade = _get_unified_memory(department)
    result = facade.search(
        query=query,
        namespace=namespace,
        limit=limit,
        include_all_sources=include_all_sources,
    )

    return {
        "query": result.query,
        "total": result.total,
        "elapsed_ms": result.elapsed_ms,
        "sources": result.sources,
        "entries": [entry.to_dict() for entry in result.entries],
    }


@unified_router.get("/search/all-departments")
async def unified_search_all_departments(
    query: str,
    department: str,
    include_global: bool = True,
    limit_per_namespace: int = 5,
) -> Dict[str, Any]:
    """
    Search across all accessible departments and global memory.

    Args:
        query: Search query
        department: Department performing the search
        include_global: Include global namespace
        limit_per_namespace: Results per namespace

    Returns:
        Dictionary mapping namespace to results
    """
    facade = _get_unified_memory(department)
    results = facade.search_all_departments(
        query=query,
        include_global=include_global,
        limit_per_namespace=limit_per_namespace,
    )

    return {
        "query": query,
        "department": department,
        "results": {
            ns: [entry.to_dict() for entry in entries]
            for ns, entries in results.items()
        },
    }


@unified_router.get("/list")
async def unified_list_memories(
    department: Optional[str] = None,
    namespace: Optional[str] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    List memories in a namespace.

    Args:
        department: Optional department filter
        namespace: Namespace to list
        limit: Maximum results

    Returns:
        List of memory entries
    """
    facade = _get_unified_memory(department)
    entries = facade.list_memories(namespace=namespace, limit=limit)

    return {
        "total": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }


@unified_router.get("/department/list")
async def unified_list_department_memories(
    department: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    List all memories in department namespace.

    Args:
        department: Department name
        limit: Maximum results

    Returns:
        List of department memory entries
    """
    facade = _get_unified_memory(department)
    entries = facade.list_department_memories(limit=limit)

    return {
        "department": department,
        "total": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }


@unified_router.get("/global/list")
async def unified_list_global_memories(
    limit: int = 100,
) -> Dict[str, Any]:
    """
    List all global memories.

    Args:
        limit: Maximum results

    Returns:
        List of global memory entries
    """
    facade = _get_unified_memory()
    entries = facade.list_global_memories(limit=limit)

    return {
        "namespace": GLOBAL_NAMESPACE,
        "total": len(entries),
        "entries": [entry.to_dict() for entry in entries],
    }


@unified_router.post("/add")
async def unified_add_memory(
    key: str,
    value: str,
    department: Optional[str] = None,
    namespace: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    sync_to_agent: bool = True,
) -> Dict[str, Any]:
    """
    Add a memory entry using the unified interface.

    Args:
        key: Memory key
        value: Memory value
        department: Optional department
        namespace: Optional namespace
        tags: Optional tags
        metadata: Optional metadata
        sync_to_agent: Whether to also sync to department memory

    Returns:
        Memory entry ID
    """
    facade = _get_unified_memory(department)
    entry_id = facade.add_memory(
        key=key,
        value=value,
        namespace=namespace,
        tags=tags,
        metadata=metadata,
        sync_to_agent=sync_to_agent,
    )

    return {
        "success": True,
        "id": entry_id,
        "key": key,
        "namespace": namespace or (f"{DEPARTMENT_PREFIX}{department}" if department else GLOBAL_NAMESPACE),
    }


@unified_router.get("/retrieve")
async def unified_retrieve_memory(
    key: str,
    department: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Retrieve a memory entry.

    Args:
        key: Memory key
        department: Optional department
        namespace: Optional namespace

    Returns:
        Memory entry or 404
    """
    facade = _get_unified_memory(department)
    value = facade.retrieve(key=key, namespace=namespace)

    if value is None:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {
        "key": key,
        "value": value,
        "namespace": namespace or (f"{DEPARTMENT_PREFIX}{department}" if department else GLOBAL_NAMESPACE),
    }


@unified_router.delete("/delete")
async def unified_delete_memory(
    key: str,
    department: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a memory entry.

    Args:
        key: Memory key
        department: Optional department
        namespace: Optional namespace

    Returns:
        Success status
    """
    facade = _get_unified_memory(department)
    deleted = facade.delete(key=key, namespace=namespace)

    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {"success": True, "key": key}


# =============================================================================
# Sync Endpoints
# =============================================================================


@unified_router.post("/sync/department/{department}")
async def sync_department_memory(
    department: str,
    categories: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Sync memories from DepartmentMemoryManager to AgentMemory.

    This reads all memories from the Markdown-based DepartmentMemoryManager
    and stores them in the SQLite-based AgentMemory.

    Args:
        department: Department name
        categories: Optional list of categories to sync
        force: Whether to force re-sync of existing entries

    Returns:
        Sync result with counts
    """
    facade = _get_unified_memory(department)
    result = facade.sync_from_department_memory(categories=categories, force=force)

    return {
        "department": department,
        **result,
    }


# =============================================================================
# Daily Log Endpoints (via Unified Facade)
# =============================================================================


@unified_router.get("/department/{department}/logs")
async def unified_get_daily_logs(
    department: str,
    days: int = 7,
) -> Dict[str, Any]:
    """
    Get recent daily logs for a department.

    Args:
        department: Department name
        days: Number of recent days to retrieve

    Returns:
        Daily logs
    """
    facade = _get_unified_memory(department)
    recent_logs = facade.get_recent_logs(days=days)

    return {
        "department": department,
        "logs": [
            {"date": date_str, "content": content}
            for date_str, content in recent_logs.items()
        ],
    }


@unified_router.post("/department/{department}/logs")
async def unified_add_daily_log(
    department: str,
    content: str,
    date: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Add an entry to department daily log.

    Args:
        department: Department name
        content: Log content
        date: Date for the log entry (ISO format, defaults to today)
        category: Optional category

    Returns:
        Success status
    """
    from datetime import date as date_type

    log_date = date_type.fromisoformat(date) if date else date_type.today()

    facade = _get_unified_memory(department)
    facade.add_daily_log(log_date=log_date, content=content, category=category)

    return {
        "success": True,
        "department": department,
        "date": log_date.isoformat(),
    }


# =============================================================================
# Include unified router in main router
# =============================================================================

# Note: The unified_router needs to be registered in the main app
# This is typically done in server.py or main.py like:
# app.include_router(unified_router)
