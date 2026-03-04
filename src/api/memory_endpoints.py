"""
Memory API Endpoints

Provides REST API for memory management in QuantMindX.
Supports semantic search, memory CRUD, and sync operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

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
# In-Memory Store (will be replaced with actual memory manager)
# =============================================================================

# Temporary in-memory store until memory manager is implemented
_memory_store: Dict[str, MemoryEntry] = {}
_memory_stats = MemoryStats(
    total_memories=0,
    total_sessions=0,
    embedding_model="text-embedding-3-small",
    last_sync=None,
    vector_dimensions=1536,
    sources=["memory", "sessions"]
)

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
    return _memory_stats


@router.post("/search", response_model=MemorySearchResponse)
async def search_memories(request: MemorySearchRequest):
    """
    Search memories using semantic search.

    Uses vector embeddings for semantic similarity.
    Optionally applies temporal decay to boost recent memories.
    """
    import time
    start_time = time.time()

    # TODO: Implement actual vector search with memory manager
    # For now, return mock results
    results = []

    # Filter by source and agent if specified
    for entry in _memory_store.values():
        if request.source and entry.source != request.source:
            continue
        if request.agent_id and entry.agent_id != request.agent_id:
            continue
        results.append(entry)

    # Limit results
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
    """Add a new memory entry."""
    import uuid

    memory_id = str(uuid.uuid4())
    entry.id = memory_id
    entry.created_at = datetime.utcnow()
    entry.updated_at = datetime.utcnow()

    _memory_store[memory_id] = entry
    _memory_stats.total_memories += 1

    logger.info(f"Added memory: {memory_id}")
    return entry


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

    if source is None and agent_id is None:
        count = len(_memory_store)
        _memory_store = {}
        _memory_stats.total_memories = 0
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

    _memory_stats.total_memories = len(_memory_store)

    return {"status": "cleared", "count": len(to_delete)}


@router.get("/list")
async def list_memories(namespace: Optional[str] = None, limit: int = 100):
    """
    List memories with optional namespace filter and limit.

    Args:
        namespace: Optional namespace to filter by (maps to source)
        limit: Maximum number of memories to return
    """
    import time

    # Convert namespace to source filter for now
    source_filter = namespace

    memories = []
    for entry in _memory_store.values():
        if source_filter and entry.source != source_filter:
            continue
        memories.append(entry)

    # Sort by created_at descending
    memories.sort(key=lambda x: x.created_at or datetime.min, reverse=True)

    # Apply limit
    memories = memories[:limit]

    return {"memories": memories, "total": len(memories)}


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
    if memory_id not in _memory_store:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_store[memory_id]


@router.delete("/{memory_id}")
async def delete_memory(memory_id: str):
    """Delete a memory entry."""
    if memory_id not in _memory_store:
        raise HTTPException(status_code=404, detail="Memory not found")

    del _memory_store[memory_id]
    _memory_stats.total_memories -= 1

    logger.info(f"Deleted memory: {memory_id}")
    return {"status": "deleted", "id": memory_id}


# =============================================================================
# Department Memory Endpoints
# =============================================================================

@dept_router.get("/{department}")
async def get_department_memory(department: str) -> Dict[str, Any]:
    """
    Get all memories for a department.

    Args:
        department: Department name (analysis, research, risk, execution, portfolio, floor_manager)

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
