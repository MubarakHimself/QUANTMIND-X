"""
Memory Tools for QuantMind Agents.

This module provides tools for agents to interact with the QuantMind memory system,
supporting department-based memory operations using SQLite with optional embeddings.
"""

import logging
from typing import Dict, Any, List, Optional

from src.memory.memory_manager import (
    MemoryManager,
    MemorySource,
    create_memory_manager,
)

logger = logging.getLogger(__name__)


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = await create_memory_manager(
            db_path=".quantmind/memory.db",
            enable_embeddings=False,  # No embeddings due to low compute
        )
    return _memory_manager


# =============================================================================
# Core Memory Tools
# =============================================================================

async def add_memory(
    content: str,
    department: str = "research",
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add a memory entry for a department.

    Stores knowledge, findings, or insights that can be retrieved later.

    Args:
        content: The content to remember
        department: Department to associate memory with
        importance: Importance score 0-1 (default: 0.5)
        tags: Optional tags for categorization
        metadata: Optional metadata for the memory

    Returns:
        Dictionary containing success status and memory_id
    """
    logger.info(f"Adding memory for {department}: {content[:50]}...")

    manager = await get_memory_manager()

    full_metadata = metadata or {}
    full_metadata["department"] = department

    try:
        entry = await manager.add_memory(
            source=MemorySource.MEMORY,
            content=content,
            metadata=full_metadata,
            importance=importance,
            tags=tags or [department],
        )

        return {
            "success": True,
            "memory_id": entry.id,
            "message": f"Memory added to {department}",
            "content": content,
            "department": department
        }

    except Exception as e:
        logger.error(f"Failed to add memory: {e}")
        return {"success": False, "error": str(e)}


async def search_memories(
    query: str,
    department: str = "all",
    limit: int = 10
) -> Dict[str, Any]:
    """
    Search memories using full-text search.

    Args:
        query: Search query
        department: Department to search (default: all)
        limit: Maximum results (default: 10)

    Returns:
        Dictionary containing search results
    """
    logger.info(f"Searching memories: {query}")

    manager = await get_memory_manager()

    try:
        # Returns List[Tuple[MemoryEntry, float]]
        results = await manager.search_fts(query=query, limit=limit)

        # Filter by department
        filtered = []
        for entry, score in results:
            if department == "all" or entry.metadata.get("department") == department:
                filtered.append({
                    "content": entry.content,
                    "memory_id": entry.id,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                    "tags": entry.tags,
                    "metadata": entry.metadata,
                    "score": score
                })

        return {
            "success": True,
            "results": filtered[:limit],
            "total": len(filtered),
            "query": query
        }

    except Exception as e:
        logger.error(f"Failed to search memories: {e}")
        return {"success": False, "error": str(e), "results": []}


async def get_memory(memory_id: str) -> Dict[str, Any]:
    """
    Get a specific memory by ID.

    Args:
        memory_id: ID of the memory to retrieve

    Returns:
        Dictionary containing the memory entry
    """
    manager = await get_memory_manager()

    try:
        entry = await manager.get_memory(memory_id)
        if entry:
            return {
                "success": True,
                "memory": {
                    "id": entry.id,
                    "content": entry.content,
                    "department": entry.metadata.get("department"),
                    "tags": entry.tags,
                    "importance": entry.importance,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None
                }
            }
        return {"success": False, "error": "Memory not found"}

    except Exception as e:
        logger.error(f"Failed to get memory: {e}")
        return {"success": False, "error": str(e)}


async def delete_memory(memory_id: str) -> Dict[str, Any]:
    """
    Delete a specific memory.

    Args:
        memory_id: ID of the memory to delete

    Returns:
        Dictionary containing deletion status
    """
    logger.info(f"Deleting memory: {memory_id}")
    manager = await get_memory_manager()

    try:
        deleted = await manager.delete_memory(memory_id)
        return {
            "success": deleted,
            "message": "Memory deleted" if deleted else "Memory not found"
        }

    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        return {"success": False, "error": str(e)}


async def get_all_memories(
    department: str = "all",
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get all memories, optionally filtered by department.

    Args:
        department: Department to filter by (default: all)
        limit: Maximum results (default: 100)

    Returns:
        Dictionary containing all memories
    """
    manager = await get_memory_manager()

    try:
        # Search for all memories in the department
        if department == "all":
            query = "*"
        else:
            query = department

        # Returns List[Tuple[MemoryEntry, float]]
        results = await manager.search_fts(query=query, limit=limit)

        memories = []
        for entry, score in results:
            memories.append({
                "id": entry.id,
                "content": entry.content,
                "department": entry.metadata.get("department"),
                "tags": entry.tags,
                "importance": entry.importance,
                "created_at": entry.created_at.isoformat() if entry.created_at else None
            })

        return {
            "success": True,
            "memories": memories,
            "total": len(memories)
        }

    except Exception as e:
        logger.error(f"Failed to get memories: {e}")
        return {"success": False, "error": str(e), "memories": []}


# =============================================================================
# Tool Registry for Claude Agent SDK
# =============================================================================

MEMORY_TOOLS = {
    "add_memory": {
        "function": add_memory,
        "description": "Add a memory entry for a department",
        "parameters": {
            "content": {"type": "string", "required": True, "description": "Content to remember"},
            "department": {"type": "string", "required": False, "default": "research", "description": "Department (research, development, trading, risk, portfolio)"},
            "importance": {"type": "number", "required": False, "default": 0.5},
            "tags": {"type": "array", "required": False},
            "metadata": {"type": "object", "required": False}
        }
    },
    "search_memories": {
        "function": search_memories,
        "description": "Search memories using full-text search",
        "parameters": {
            "query": {"type": "string", "required": True},
            "department": {"type": "string", "required": False, "default": "all"},
            "limit": {"type": "integer", "required": False, "default": 10}
        }
    },
    "get_memory": {
        "function": get_memory,
        "description": "Get a specific memory by ID",
        "parameters": {
            "memory_id": {"type": "string", "required": True}
        }
    },
    "delete_memory": {
        "function": delete_memory,
        "description": "Delete a specific memory",
        "parameters": {
            "memory_id": {"type": "string", "required": True}
        }
    },
    "get_all_memories": {
        "function": get_all_memories,
        "description": "Get all memories, optionally filtered by department",
        "parameters": {
            "department": {"type": "string", "required": False, "default": "all"},
            "limit": {"type": "integer", "required": False, "default": 100}
        }
    }
}


def get_memory_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a memory tool by name."""
    return MEMORY_TOOLS.get(name)


def list_memory_tools() -> List[str]:
    """List all available memory tools."""
    return list(MEMORY_TOOLS.keys())


async def invoke_memory_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Invoke a memory tool by name."""
    tool = get_memory_tool(name)
    if not tool:
        raise ValueError(f"Unknown memory tool: {name}")

    func = tool["function"]
    return await func(**kwargs)
