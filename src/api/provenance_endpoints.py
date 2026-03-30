"""
Provenance Chain API Endpoints

REST API for querying EA provenance and origin tracking.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["provenance-chain"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ProvenanceNode(BaseModel):
    """A node in the provenance chain."""
    stage: str = Field(..., description="Stage name (source, research, dev, review, approval)")
    timestamp: str = Field(..., description="When this stage was completed")
    actor: str = Field(..., description="Who performed this action")
    status: str = Field(..., description="Status: pending, in_progress, completed, failed")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")


class ProvenanceChain(BaseModel):
    """Complete provenance chain for an EA."""
    strategy_id: str
    version_tag: str
    chain: List[ProvenanceNode]
    source_url: Optional[str] = Field(None, description="Source URL (MQL5/YouTube)")
    total_stages: int


class ProvenanceQueryRequest(BaseModel):
    """Natural language query for provenance."""
    query: str = Field(..., description="User query about EA origin")
    strategy_id: str = Field(..., description="Strategy to query")


class ProvenanceQueryResponse(BaseModel):
    """Response to provenance query."""
    answer: str = Field(..., description="Natural language answer")
    chain: Optional[ProvenanceChain] = Field(None, description="Referenced provenance chain")
    confidence: float = Field(..., description="Confidence score")


# ============================================================================
# Provenance Data Service (Extend from Story 8.4)
# ============================================================================


class ProvenanceService:
    """Service to retrieve and construct provenance chains."""

    async def get_provenance_chain(
        self,
        strategy_id: str,
        version_tag: Optional[str] = None,
    ) -> ProvenanceChain:
        """
        Get provenance chain for a strategy version.

        In production, this would query:
        - Strategy version metadata (Story 8.4)
        - Research department records (Epic 6)
        - Code review records (Story 7-3)
        - Approval gate records (Story 8-5)
        """
        raise NotImplementedError(
            "Provenance chain not available. Provenance tracking not wired to production."
        )

    async def query_provenance(
        self,
        strategy_id: str,
        query: str,
    ) -> ProvenanceQueryResponse:
        """
        Answer natural language questions about EA origin.

        In production, this would use LLM to answer based on provenance chain.
        """
        raise NotImplementedError(
            "Provenance query not available. Provenance tracking not wired to production."
        )


# Singleton instance
_provenance_service = ProvenanceService()


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/{strategy_id}/provenance")
async def get_provenance_chain(
    strategy_id: str,
    version_tag: Optional[str] = Query(None, description="Version tag (default: latest)"),
) -> ProvenanceChain:
    """
    Get provenance chain for a strategy.

    GET /api/strategies/{strategy_id}/provenance

    Returns: source → Research → Dev → review → approval timeline.
    """
    logger.info(f"Getting provenance for strategy {strategy_id}")
    try:
        return await _provenance_service.get_provenance_chain(strategy_id, version_tag)
    except NotImplementedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.post("/provenance/query")
async def query_provenance(
    request: ProvenanceQueryRequest,
) -> ProvenanceQueryResponse:
    """
    Query provenance with natural language.

    POST /api/strategies/provenance/query

    Body: { "strategy_id": "...", "query": "what's the origin?" }

    Returns natural language answer with referenced chain.
    """
    logger.info(f"Provenance query for {request.strategy_id}: {request.query}")
    try:
        return await _provenance_service.query_provenance(
            request.strategy_id,
            request.query,
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/{strategy_id}/versions/{version_tag}/provenance")
async def get_version_provenance(
    strategy_id: str,
    version_tag: str,
) -> ProvenanceChain:
    """
    Get provenance for a specific version.

    GET /api/strategies/{strategy_id}/versions/{version_tag}/provenance
    """
    try:
        return await _provenance_service.get_provenance_chain(strategy_id, version_tag)
    except NotImplementedError as e:
        raise HTTPException(status_code=503, detail=str(e))