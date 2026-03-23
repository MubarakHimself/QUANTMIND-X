"""
Reasoning Log API Endpoints

Provides REST API endpoints for retrieving agent reasoning transparency logs.
Enables querying of OPINION nodes from the memory graph to provide reasoning chains.

**Validates: Story 10.2 - Agent Reasoning Transparency Log & API**
**FR78: Copilot explains its reasoning for any past decision**
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime

from src.memory.graph.facade import GraphMemoryFacade, get_graph_memory
from src.memory.graph.types import MemoryNodeType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/audit/reasoning", tags=["audit", "reasoning"])

# Singleton instance
_facade: Optional[GraphMemoryFacade] = None


def _get_facade() -> GraphMemoryFacade:
    """Get or create the graph memory facade singleton."""
    global _facade
    if _facade is None:
        import os
        db_path = os.environ.get("GRAPH_MEMORY_DB", "data/graph_memory.db")
        _facade = get_graph_memory(db_path=db_path)
    return _facade


# =============================================================================
# Request/Response Models
# =============================================================================


class OpinionNodeResponse(BaseModel):
    """Response model for an OPINION node."""
    node_id: str
    action: Optional[str] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    alternatives_considered: Optional[str] = None
    constraints_applied: Optional[str] = None
    agent_role: Optional[str] = None
    created_at_utc: Optional[str] = None


class ReasoningChainItem(BaseModel):
    """A single item in a reasoning chain."""
    hypothesis: str
    evidence_sources: List[str] = []
    confidence_score: float
    sub_agents: List[str] = []
    decision_timestamp_utc: str


class DepartmentReasoningResponse(BaseModel):
    """Response model for department reasoning query."""
    department: str
    reasoning_chain: List[ReasoningChainItem]
    total_decisions: int


class ReasoningLogResponse(BaseModel):
    """Response model for reasoning log query by decision_id."""
    decision_id: str
    context_at_decision: Optional[dict] = None
    model_used: Optional[str] = None
    prompt_summary: Optional[str] = None
    response_summary: Optional[str] = None
    action_taken: Optional[str] = None
    opinion_nodes: List[OpinionNodeResponse] = []


# =============================================================================
# API Endpoints
# =============================================================================


@router.get("/{decision_id}", response_model=ReasoningLogResponse)
async def get_reasoning_by_decision_id(decision_id: str):
    """
    Get full reasoning chain for a specific decision.

    Query Memory Graph for OPINION nodes matching the decision_id.
    Returns context, model used, prompt/response summaries, and opinion nodes.
    """
    facade = _get_facade()

    # Query for OPINION nodes
    # decision_id is stored in metadata or matched via session_id + timestamp
    nodes = facade.store.query_nodes(
        node_type=MemoryNodeType.OPINION,
        limit=100
    )

    # Filter by decision_id if stored in metadata
    matching_nodes = []
    for node in nodes:
        metadata = node.metadata or {}
        # Check if decision_id matches in various possible locations
        node_decision_id = (
            metadata.get("decision_id") or
            node.session_id  # Fallback: use session_id as decision context
        )
        if str(node_decision_id) == decision_id or node_decision_id is None:
            # Include all OPINION nodes if no specific decision_id filter
            # This allows searching by session context
            if str(node_decision_id) == decision_id:
                matching_nodes.append(node)
            elif decision_id in (node.session_id or ""):
                matching_nodes.append(node)

    # If no exact match, try broader search with session_id pattern
    if not matching_nodes:
        # Search by partial session match
        nodes = facade.store.query_nodes(
            node_type=MemoryNodeType.OPINION,
            session_id=decision_id,
            limit=10
        )
        matching_nodes = nodes

    # Build response
    opinion_nodes = []
    for node in matching_nodes:
        opinion_nodes.append(OpinionNodeResponse(
            node_id=str(node.id),
            action=node.action,
            reasoning=node.reasoning,
            confidence=node.confidence,
            alternatives_considered=node.alternatives_considered,
            constraints_applied=node.constraints_applied,
            agent_role=node.agent_role,
            created_at_utc=node.created_at.isoformat() if node.created_at else None
        ))

    # Extract context and summaries from first node if available
    context_at_decision = None
    model_used = None
    prompt_summary = None
    response_summary = None
    action_taken = None

    if matching_nodes:
        first_node = matching_nodes[0]
        metadata = first_node.metadata or {}
        context_at_decision = metadata.get("context", {})
        model_used = metadata.get("model_used", "unknown")
        # Summarize from content - first 200 chars as prompt summary
        prompt_summary = (first_node.content[:200] + "...") if first_node.content else None
        # Response summary from action + reasoning
        action_taken = first_node.action
        if first_node.reasoning:
            response_summary = first_node.reasoning[:200] + "..." if len(first_node.reasoning) > 200 else first_node.reasoning

    return ReasoningLogResponse(
        decision_id=decision_id,
        context_at_decision=context_at_decision,
        model_used=model_used,
        prompt_summary=prompt_summary,
        response_summary=response_summary,
        action_taken=action_taken,
        opinion_nodes=opinion_nodes
    )


@router.get("/department/{department}", response_model=DepartmentReasoningResponse)
async def get_department_reasoning(
    department: str,
    start_date: Optional[str] = Query(None, description="Start date ISO format"),
    end_date: Optional[str] = Query(None, description="End date ISO format"),
    limit: int = Query(50, ge=1, le=500, description="Maximum number of results")
):
    """
    Get reasoning chain for a specific department.

    Query OPINION nodes filtered by agent_role (department).
    Returns hypothesis chain, evidence sources, confidence scores, and contributing sub-agents.
    """
    from datetime import datetime

    facade = _get_facade()

    # Map department to role
    department_role_map = {
        "research": "research",
        "development": "development",
        "risk": "risk",
        "trading": "trading",
        "portfolio": "portfolio"
    }

    role = department_role_map.get(department.lower(), department.lower())

    # Parse date filters
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError:
            pass
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError:
            pass

    # Query for OPINION nodes filtered by role/agent_role
    nodes = facade.store.query_nodes(
        node_type=MemoryNodeType.OPINION,
        role=role,
        limit=limit
    )

    # Also search by agent_role in metadata
    if not nodes:
        nodes = facade.store.query_nodes(
            node_type=MemoryNodeType.OPINION,
            limit=limit
        )
        # Filter by agent_role
        nodes = [n for n in nodes if (n.agent_role or "").lower() == role.lower()]

    # Filter by date range if specified
    if start_dt or end_dt:
        filtered_nodes = []
        for node in nodes:
            if node.created_at:
                if start_dt and node.created_at < start_dt:
                    continue
                if end_dt and node.created_at > end_dt:
                    continue
            filtered_nodes.append(node)
        nodes = filtered_nodes

    # Build reasoning chain from opinion nodes
    reasoning_chain = []
    for node in nodes:
        # Extract evidence sources from metadata or content
        metadata = node.metadata or {}
        evidence_sources = metadata.get("evidence_sources", [])
        if isinstance(evidence_sources, str):
            evidence_sources = [evidence_sources]

        # Extract sub-agents from metadata
        sub_agents = metadata.get("sub_agents", [])
        if isinstance(sub_agents, str):
            sub_agents = [sub_agents]

        # Build hypothesis from reasoning + action
        hypothesis = node.reasoning or node.action or "No reasoning recorded"

        reasoning_chain.append(ReasoningChainItem(
            hypothesis=hypothesis,
            evidence_sources=evidence_sources or [],
            confidence_score=node.confidence or 0.0,
            sub_agents=sub_agents or [],
            decision_timestamp_utc=node.created_at.isoformat() if node.created_at else ""
        ))

    return DepartmentReasoningResponse(
        department=department,
        reasoning_chain=reasoning_chain,
        total_decisions=len(reasoning_chain)
    )