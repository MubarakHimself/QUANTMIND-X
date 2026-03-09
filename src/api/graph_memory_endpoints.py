"""
Graph Memory API Endpoints

Provides REST API for graph-based memory management in QuantMindX.
Supports memory operations (retain, recall, reflect, link) and tier management.
"""

import logging
from typing import Optional, List, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.memory.graph.facade import GraphMemoryFacade, get_graph_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph-memory", tags=["graph-memory"])

# Singleton instance
_facade: Optional[GraphMemoryFacade] = None


def _get_facade() -> GraphMemoryFacade:
    """Get or create the graph memory facade singleton."""
    global _facade
    if _facade is None:
        _facade = get_graph_memory()
    return _facade


# =============================================================================
# Request/Response Models
# =============================================================================


class RetainRequest(BaseModel):
    """Request model for retaining a memory."""
    content: str
    source: str = "unknown"
    department: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    importance: float = 0.5
    tags: Optional[List[str]] = None
    related_to: Optional[List[str]] = None


class RecallRequest(BaseModel):
    """Request model for recalling memories."""
    query: str
    department: Optional[str] = None
    agent_id: Optional[str] = None
    tags: Optional[List[str]] = None
    node_types: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    min_importance: float = 0.0
    limit: int = 50
    cursor: Optional[str] = None


class ReflectRequest(BaseModel):
    """Request model for reflecting on memories."""
    query: str
    department: Optional[str] = None
    agent_id: Optional[str] = None
    context: Optional[str] = None


class LinkRequest(BaseModel):
    """Request model for linking memory nodes."""
    source_id: str
    target_ids: List[str]
    relation_type: str = "related_to"
    strength: float = 0.8


class NodeResponse(BaseModel):
    """Response model for a memory node."""
    id: str
    content: str
    node_type: str
    category: str
    department: Optional[str]
    agent_id: Optional[str]
    session_id: Optional[str]
    importance: float
    tags: List[str]
    tier: str
    created_at: str
    updated_at: str
    access_count: int
    relevance_score: Optional[float]


class EdgeResponse(BaseModel):
    """Response model for a memory edge."""
    id: str
    source_id: str
    target_id: str
    relation_type: str
    strength: float
    created_at: str


class StatsResponse(BaseModel):
    """Response model for memory stats."""
    total_nodes: int
    hot: int
    warm: int
    cold: int


class CompactionResponse(BaseModel):
    """Response model for compaction status."""
    should_compact: bool
    current_percent: float
    threshold_percent: float
    hot_count: int
    warm_count: int
    cold_count: int
    threshold: float


# =============================================================================
# Memory Operations
# =============================================================================


@router.post("/retain", response_model=dict)
async def retain_memory(request: RetainRequest):
    """Store a new memory in the graph."""
    try:
        facade = _get_facade()
        node_id = facade.retain(
            content=request.content,
            source=request.source,
            department=request.department,
            agent_id=request.agent_id,
            session_id=request.session_id,
            importance=request.importance,
            tags=request.tags,
            related_to=request.related_to,
        )
        return {"success": True, "node_id": node_id}
    except Exception as e:
        logger.error(f"Error retaining memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recall")
async def recall_memories(request: RecallRequest):
    """Recall memories based on query and filters."""
    try:
        facade = _get_facade()
        results = facade.recall(
            query=request.query,
            department=request.department,
            agent_id=request.agent_id,
            tags=request.tags,
            node_types=request.node_types,
            categories=request.categories,
            min_importance=request.min_importance,
            limit=request.limit,
            cursor=request.cursor,
        )

        # Convert to response format
        nodes = []
        for node in results:
            nodes.append(NodeResponse(
                id=node.id,
                content=node.content,
                node_type=node.node_type.value if hasattr(node.node_type, 'value') else str(node.node_type),
                category=node.category.value if hasattr(node.category, 'value') else str(node.category),
                department=node.department,
                agent_id=node.agent_id,
                session_id=node.session_id,
                importance=node.importance,
                tags=node.tags,
                tier=node.tier.value if hasattr(node.tier, 'value') else str(node.tier),
                created_at=node.created_at.isoformat() if hasattr(node.created_at, 'isoformat') else str(node.created_at),
                updated_at=node.updated_at.isoformat() if hasattr(node.updated_at, 'isoformat') else str(node.updated_at),
                access_count=node.access_count,
                relevance_score=node.relevance_score,
            ))

        return {"success": True, "nodes": nodes, "count": len(nodes)}
    except Exception as e:
        logger.error(f"Error recalling memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reflect")
async def reflect_on_memories(request: ReflectRequest):
    """Synthesize answer from memories (REFLECT operation)."""
    try:
        facade = _get_facade()
        result = facade.reflect(
            query=request.query,
            department=request.department,
            agent_id=request.agent_id,
            context=request.context,
        )
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Error reflecting on memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/link", response_model=dict)
async def link_memories(request: LinkRequest):
    """Create relationship edges between memory nodes."""
    try:
        facade = _get_facade()
        edges = facade.link(
            source_id=request.source_id,
            target_ids=request.target_ids,
            relation_type=request.relation_type,
            strength=request.strength,
        )
        return {
            "success": True,
            "edges_created": len(edges),
            "edge_ids": [e.id for e in edges]
        }
    except Exception as e:
        logger.error(f"Error linking memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Stats & Status
# =============================================================================


@router.get("/stats", response_model=StatsResponse)
async def get_memory_stats():
    """Get memory statistics."""
    try:
        facade = _get_facade()
        stats = facade.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compaction", response_model=CompactionResponse)
async def get_compaction_status(context_percent: float = Query(0, ge=0, le=100)):
    """Check compaction status."""
    try:
        facade = _get_facade()
        should_compact = facade.should_compact(context_percent)
        stats = facade.get_stats()

        return CompactionResponse(
            should_compact=should_compact,
            current_percent=context_percent,
            threshold_percent=50.0,
            hot_count=stats.get("hot", 0),
            warm_count=stats.get("warm", 0),
            cold_count=stats.get("cold", 0),
            threshold=50.0,
        )
    except Exception as e:
        logger.error(f"Error checking compaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compact")
async def trigger_compaction():
    """Trigger manual compaction of old nodes."""
    try:
        facade = _get_facade()
        result = facade.check_and_compact()
        return {"success": True, **result}
    except Exception as e:
        logger.error(f"Error triggering compaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Tier Management
# =============================================================================


@router.get("/nodes/hot", response_model=List[NodeResponse])
async def get_hot_nodes(limit: int = Query(50, ge=1, le=100)):
    """Get hot (recent) memory nodes."""
    try:
        facade = _get_facade()
        nodes = facade.get_hot_nodes(limit=limit)

        return [
            NodeResponse(
                id=n.id,
                content=n.content,
                node_type=n.node_type.value if hasattr(n.node_type, 'value') else str(n.node_type),
                category=n.category.value if hasattr(n.category, 'value') else str(n.category),
                department=n.department,
                agent_id=n.agent_id,
                session_id=n.session_id,
                importance=n.importance,
                tags=n.tags,
                tier=n.tier.value if hasattr(n.tier, 'value') else str(n.tier),
                created_at=n.created_at.isoformat() if hasattr(n.created_at, 'isoformat') else str(n.created_at),
                updated_at=n.updated_at.isoformat() if hasattr(n.updated_at, 'isoformat') else str(n.updated_at),
                access_count=n.access_count,
                relevance_score=n.relevance_score,
            )
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"Error getting hot nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/warm", response_model=List[NodeResponse])
async def get_warm_nodes(limit: int = Query(100, ge=1, le=200)):
    """Get warm (recent) memory nodes."""
    try:
        facade = _get_facade()
        nodes = facade.get_warm_nodes(limit=limit)

        return [
            NodeResponse(
                id=n.id,
                content=n.content,
                node_type=n.node_type.value if hasattr(n.node_type, 'value') else str(n.node_type),
                category=n.category.value if hasattr(n.category, 'value') else str(n.category),
                department=n.department,
                agent_id=n.agent_id,
                session_id=n.session_id,
                importance=n.importance,
                tags=n.tags,
                tier=n.tier.value if hasattr(n.tier, 'value') else str(n.tier),
                created_at=n.created_at.isoformat() if hasattr(n.created_at, 'isoformat') else str(n.created_at),
                updated_at=n.updated_at.isoformat() if hasattr(n.updated_at, 'isoformat') else str(n.updated_at),
                access_count=n.access_count,
                relevance_score=n.relevance_score,
            )
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"Error getting warm nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nodes/cold", response_model=List[NodeResponse])
async def get_cold_nodes(limit: int = Query(100, ge=1, le=200)):
    """Get cold (archived) memory nodes."""
    try:
        facade = _get_facade()
        nodes = facade.get_cold_nodes(limit=limit)

        return [
            NodeResponse(
                id=n.id,
                content=n.content,
                node_type=n.node_type.value if hasattr(n.node_type, 'value') else str(n.node_type),
                category=n.category.value if hasattr(n.category, 'value') else str(n.category),
                department=n.department,
                agent_id=n.agent_id,
                session_id=n.session_id,
                importance=n.importance,
                tags=n.tags,
                tier=n.tier.value if hasattr(n.tier, 'value') else str(n.tier),
                created_at=n.created_at.isoformat() if hasattr(n.created_at, 'isoformat') else str(n.created_at),
                updated_at=n.updated_at.isoformat() if hasattr(n.updated_at, 'isoformat') else str(n.updated_at),
                access_count=n.access_count,
                relevance_score=n.relevance_score,
            )
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"Error getting cold nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes/{node_id}/move-to-hot")
async def move_node_to_hot(node_id: str):
    """Move a node to hot tier."""
    try:
        facade = _get_facade()
        node = facade.move_to_hot(node_id)
        if node:
            return {"success": True, "node_id": node_id, "new_tier": "hot"}
        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving node to hot: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes/{node_id}/move-to-warm")
async def move_node_to_warm(node_id: str):
    """Move a node to warm tier."""
    try:
        facade = _get_facade()
        node = facade.move_to_warm(node_id)
        if node:
            return {"success": True, "node_id": node_id, "new_tier": "warm"}
        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving node to warm: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/nodes/{node_id}/move-to-cold")
async def move_node_to_cold(node_id: str):
    """Move a node to cold tier."""
    try:
        facade = _get_facade()
        node = facade.move_to_cold(node_id)
        if node:
            return {"success": True, "node_id": node_id, "new_tier": "cold"}
        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error moving node to cold: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Node Deletion
# =============================================================================


@router.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete a memory node."""
    try:
        facade = _get_facade()
        success = facade.delete_node(node_id)
        if success:
            return {"success": True, "node_id": node_id}
        raise HTTPException(status_code=404, detail="Node not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting node: {e}")
        raise HTTPException(status_code=500, detail=str(e))
