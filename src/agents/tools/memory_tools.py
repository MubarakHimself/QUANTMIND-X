"""
Memory Tools for QuantMind Agents.

This module provides tools for agents to interact with the LangMem memory system,
supporting semantic, episodic, and procedural memory operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Global memory manager instance
_memory_manager = None


def get_memory_manager():
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        from src.memory.langmem_manager import LangMemManager
        _memory_manager = LangMemManager()
    return _memory_manager


# =============================================================================
# Semantic Memory Tools (Facts and Concepts)
# =============================================================================

async def add_semantic_memory(
    content: str,
    agent_name: str = "analyst",
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add a semantic memory (fact or concept).
    
    Semantic memories store factual knowledge and concepts that can be
    retrieved and used across sessions.
    
    Args:
        content: The fact or concept to remember
        agent_name: Agent to associate memory with (default: "analyst")
        importance: Importance score 0-1 (default: 0.5)
        metadata: Optional metadata for the memory
        
    Returns:
        Dictionary containing:
        - success: Whether the memory was added
        - memory_id: ID of the created memory
        - message: Status message
    """
    logger.info(f"Adding semantic memory for {agent_name}: {content[:50]}...")
    
    manager = get_memory_manager()
    
    try:
        entry = await manager.add_semantic_memory(
            agent_name=agent_name,
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        return {
            "success": True,
            "memory_id": entry.id,
            "message": f"Semantic memory added successfully",
            "content": content,
            "importance": importance
        }
        
    except Exception as e:
        logger.error(f"Failed to add semantic memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add semantic memory"
        }


async def search_semantic_memory(
    query: str,
    agent_name: str = "analyst",
    limit: int = 10,
    min_relevance: float = 0.0
) -> Dict[str, Any]:
    """
    Search semantic memories by query.
    
    Retrieves factual knowledge and concepts matching the query.
    
    Args:
        query: Search query
        agent_name: Agent to search memories for (default: "analyst")
        limit: Maximum results to return (default: 10)
        min_relevance: Minimum relevance threshold (default: 0.0)
        
    Returns:
        Dictionary containing:
        - results: List of matching memories with relevance scores
        - total: Total number of results
        - query: Original query
    """
    logger.info(f"Searching semantic memory for {agent_name}: {query}")
    
    manager = get_memory_manager()
    
    try:
        results = await manager.search_semantic_memory(
            agent_name=agent_name,
            query=query,
            limit=limit
        )
        
        return {
            "success": True,
            "results": [
                {
                    "content": r.entry.content,
                    "relevance": r.relevance,
                    "memory_id": r.entry.id,
                    "created_at": r.entry.created_at,
                    "importance": r.entry.importance
                }
                for r in results
                if r.relevance >= min_relevance
            ],
            "total": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Failed to search semantic memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query
        }


# =============================================================================
# Episodic Memory Tools (Events and Experiences)
# =============================================================================

async def add_episodic_memory(
    content: str,
    agent_name: str = "analyst",
    importance: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add an episodic memory (event or experience).
    
    Episodic memories store events, experiences, and interactions that
    occurred during agent operation.
    
    Args:
        content: Description of the event or experience
        agent_name: Agent to associate memory with (default: "analyst")
        importance: Importance score 0-1 (default: 0.5)
        metadata: Optional metadata (e.g., timestamp, context)
        
    Returns:
        Dictionary containing:
        - success: Whether the memory was added
        - memory_id: ID of the created memory
        - message: Status message
    """
    logger.info(f"Adding episodic memory for {agent_name}: {content[:50]}...")
    
    manager = get_memory_manager()
    
    # Add timestamp to metadata
    if metadata is None:
        metadata = {}
    metadata["event_timestamp"] = datetime.now().isoformat()
    
    try:
        entry = await manager.add_episodic_memory(
            agent_name=agent_name,
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        return {
            "success": True,
            "memory_id": entry.id,
            "message": "Episodic memory added successfully",
            "content": content,
            "importance": importance
        }
        
    except Exception as e:
        logger.error(f"Failed to add episodic memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add episodic memory"
        }


async def search_episodic_memory(
    query: str,
    agent_name: str = "analyst",
    limit: int = 10,
    min_relevance: float = 0.0
) -> Dict[str, Any]:
    """
    Search episodic memories by query.
    
    Retrieves events and experiences matching the query.
    
    Args:
        query: Search query
        agent_name: Agent to search memories for (default: "analyst")
        limit: Maximum results to return (default: 10)
        min_relevance: Minimum relevance threshold (default: 0.0)
        
    Returns:
        Dictionary containing:
        - results: List of matching memories with relevance scores
        - total: Total number of results
        - query: Original query
    """
    logger.info(f"Searching episodic memory for {agent_name}: {query}")
    
    manager = get_memory_manager()
    
    try:
        results = await manager.search_episodic_memory(
            agent_name=agent_name,
            query=query,
            limit=limit
        )
        
        return {
            "success": True,
            "results": [
                {
                    "content": r.entry.content,
                    "relevance": r.relevance,
                    "memory_id": r.entry.id,
                    "created_at": r.entry.created_at,
                    "importance": r.entry.importance,
                    "metadata": r.entry.metadata
                }
                for r in results
                if r.relevance >= min_relevance
            ],
            "total": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Failed to search episodic memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query
        }


# =============================================================================
# Procedural Memory Tools (Skills and Procedures)
# =============================================================================

async def add_procedural_memory(
    content: str,
    agent_name: str = "analyst",
    importance: float = 0.7,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Add a procedural memory (skill or procedure).
    
    Procedural memories store how-to knowledge, skills, and procedures
    that can be applied to similar situations.
    
    Args:
        content: Description of the skill or procedure
        agent_name: Agent to associate memory with (default: "analyst")
        importance: Importance score 0-1 (default: 0.7, higher for procedures)
        metadata: Optional metadata (e.g., steps, prerequisites)
        
    Returns:
        Dictionary containing:
        - success: Whether the memory was added
        - memory_id: ID of the created memory
        - message: Status message
    """
    logger.info(f"Adding procedural memory for {agent_name}: {content[:50]}...")
    
    manager = get_memory_manager()
    
    try:
        entry = await manager.add_procedural_memory(
            agent_name=agent_name,
            content=content,
            importance=importance,
            metadata=metadata
        )
        
        return {
            "success": True,
            "memory_id": entry.id,
            "message": "Procedural memory added successfully",
            "content": content,
            "importance": importance
        }
        
    except Exception as e:
        logger.error(f"Failed to add procedural memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add procedural memory"
        }


async def search_procedural_memory(
    query: str,
    agent_name: str = "analyst",
    limit: int = 10,
    min_relevance: float = 0.0
) -> Dict[str, Any]:
    """
    Search procedural memories by query.
    
    Retrieves skills and procedures matching the query.
    
    Args:
        query: Search query
        agent_name: Agent to search memories for (default: "analyst")
        limit: Maximum results to return (default: 10)
        min_relevance: Minimum relevance threshold (default: 0.0)
        
    Returns:
        Dictionary containing:
        - results: List of matching memories with relevance scores
        - total: Total number of results
        - query: Original query
    """
    logger.info(f"Searching procedural memory for {agent_name}: {query}")
    
    manager = get_memory_manager()
    
    try:
        results = await manager.search_procedural_memory(
            agent_name=agent_name,
            query=query,
            limit=limit
        )
        
        return {
            "success": True,
            "results": [
                {
                    "content": r.entry.content,
                    "relevance": r.relevance,
                    "memory_id": r.entry.id,
                    "created_at": r.entry.created_at,
                    "importance": r.entry.importance,
                    "metadata": r.entry.metadata
                }
                for r in results
                if r.relevance >= min_relevance
            ],
            "total": len(results),
            "query": query
        }
        
    except Exception as e:
        logger.error(f"Failed to search procedural memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query
        }


# =============================================================================
# General Memory Tools
# =============================================================================

async def get_all_memories(
    agent_name: str = "analyst",
    memory_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get all memories for an agent.
    
    Args:
        agent_name: Agent to get memories for (default: "analyst")
        memory_type: Filter by memory type (optional)
        
    Returns:
        Dictionary containing all memories
    """
    logger.info(f"Getting all memories for {agent_name}")
    
    manager = get_memory_manager()
    
    try:
        memories = await manager.get_all_memories(
            agent_name=agent_name,
            memory_type=memory_type
        )
        
        return {
            "success": True,
            "memories": [m.to_dict() for m in memories],
            "total": len(memories),
            "agent_name": agent_name,
            "memory_type": memory_type
        }
        
    except Exception as e:
        logger.error(f"Failed to get memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "memories": [],
            "total": 0
        }


async def delete_memory(
    memory_id: str
) -> Dict[str, Any]:
    """
    Delete a specific memory.
    
    Args:
        memory_id: ID of the memory to delete
        
    Returns:
        Dictionary containing deletion status
    """
    logger.info(f"Deleting memory: {memory_id}")
    
    manager = get_memory_manager()
    
    try:
        deleted = await manager.delete_memory(memory_id)
        
        return {
            "success": deleted,
            "memory_id": memory_id,
            "message": "Memory deleted" if deleted else "Memory not found"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete memory: {e}")
        return {
            "success": False,
            "error": str(e),
            "memory_id": memory_id
        }


async def clear_memories(
    agent_name: str
) -> Dict[str, Any]:
    """
    Clear all memories for an agent.
    
    Args:
        agent_name: Agent to clear memories for
        
    Returns:
        Dictionary containing number of memories cleared
    """
    logger.info(f"Clearing memories for {agent_name}")
    
    manager = get_memory_manager()
    
    try:
        count = await manager.clear_memories(agent_name)
        
        return {
            "success": True,
            "cleared_count": count,
            "agent_name": agent_name,
            "message": f"Cleared {count} memories"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear memories: {e}")
        return {
            "success": False,
            "error": str(e),
            "cleared_count": 0
        }


# =============================================================================
# Tool Registry for LangGraph Integration
# =============================================================================

MEMORY_TOOLS = {
    # Semantic memory tools
    "add_semantic_memory": {
        "function": add_semantic_memory,
        "description": "Add a semantic memory (fact or concept)",
        "parameters": {
            "content": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "importance": {"type": "number", "required": False, "default": 0.5},
            "metadata": {"type": "object", "required": False}
        }
    },
    "search_semantic_memory": {
        "function": search_semantic_memory,
        "description": "Search semantic memories by query",
        "parameters": {
            "query": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "limit": {"type": "integer", "required": False, "default": 10},
            "min_relevance": {"type": "number", "required": False, "default": 0.0}
        }
    },
    
    # Episodic memory tools
    "add_episodic_memory": {
        "function": add_episodic_memory,
        "description": "Add an episodic memory (event or experience)",
        "parameters": {
            "content": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "importance": {"type": "number", "required": False, "default": 0.5},
            "metadata": {"type": "object", "required": False}
        }
    },
    "search_episodic_memory": {
        "function": search_episodic_memory,
        "description": "Search episodic memories by query",
        "parameters": {
            "query": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "limit": {"type": "integer", "required": False, "default": 10},
            "min_relevance": {"type": "number", "required": False, "default": 0.0}
        }
    },
    
    # Procedural memory tools
    "add_procedural_memory": {
        "function": add_procedural_memory,
        "description": "Add a procedural memory (skill or procedure)",
        "parameters": {
            "content": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "importance": {"type": "number", "required": False, "default": 0.7},
            "metadata": {"type": "object", "required": False}
        }
    },
    "search_procedural_memory": {
        "function": search_procedural_memory,
        "description": "Search procedural memories by query",
        "parameters": {
            "query": {"type": "string", "required": True},
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "limit": {"type": "integer", "required": False, "default": 10},
            "min_relevance": {"type": "number", "required": False, "default": 0.0}
        }
    },
    
    # General memory tools
    "get_all_memories": {
        "function": get_all_memories,
        "description": "Get all memories for an agent",
        "parameters": {
            "agent_name": {"type": "string", "required": False, "default": "analyst"},
            "memory_type": {"type": "string", "required": False}
        }
    },
    "delete_memory": {
        "function": delete_memory,
        "description": "Delete a specific memory",
        "parameters": {
            "memory_id": {"type": "string", "required": True}
        }
    },
    "clear_memories": {
        "function": clear_memories,
        "description": "Clear all memories for an agent",
        "parameters": {
            "agent_name": {"type": "string", "required": True}
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
