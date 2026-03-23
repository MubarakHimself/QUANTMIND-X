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
        # Mock implementation - returns simulated provenance
        # In production: query strategy_versions, research_scores, code_reviews tables

        if version_tag is None:
            version_tag = "1.0.0"

        # Build provenance chain based on Alpha Forge pipeline stages
        chain = [
            ProvenanceNode(
                stage="source",
                timestamp=datetime.utcnow().isoformat(),
                actor="Video Ingest (YouTube)",
                status="completed",
                details={
                    "url": "https://youtube.com/watch?v=example",
                    "source_type": "youtube",
                    "ingested_at": datetime.utcnow().isoformat(),
                },
            ),
            ProvenanceNode(
                stage="research",
                timestamp=datetime.utcnow().isoformat(),
                actor="Research Department",
                status="completed",
                details={
                    "score": 0.85,
                    "hypothesis_id": "hyp-123",
                    "validation": "passed",
                    "tags": ["trend-following", "moving-average"],
                },
            ),
            ProvenanceNode(
                stage="dev",
                timestamp=datetime.utcnow().isoformat(),
                actor="Development Department",
                status="completed",
                details={
                    "build_id": "dev-456",
                    "compilation": "success",
                    "variant_type": "vanilla",
                    "lines_of_code": 1250,
                },
            ),
            ProvenanceNode(
                stage="review",
                timestamp=datetime.utcnow().isoformat(),
                actor="Code Review Agent",
                status="completed",
                details={
                    "review_id": "rev-789",
                    "findings": 0,
                    "approved": True,
                    "reviewer": "claude-sonnet-4",
                },
            ),
            ProvenanceNode(
                stage="approval",
                timestamp=datetime.utcnow().isoformat(),
                actor="Human Approver",
                status="completed",
                details={
                    "approver": "mubarak",
                    "approval_type": "manual",
                    "notes": "Ready for deployment",
                },
            ),
        ]

        return ProvenanceChain(
            strategy_id=strategy_id,
            version_tag=version_tag,
            chain=chain,
            source_url="https://www.mql5.com/en/articles/example",
            total_stages=len(chain),
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
        # Get the provenance chain
        chain = await self.get_provenance_chain(strategy_id)

        # Simple pattern matching for demo
        # In production: use LLM to generate answer
        query_lower = query.lower()

        if "origin" in query_lower or "source" in query_lower:
            if chain.source_url:
                answer = f"The EA originated from {chain.source_url}. It was first ingested through the Video Ingest pipeline and scored {chain.chain[1].details.get('score', 'N/A')} by Research."
            else:
                answer = "The EA was created internally in the Development Department."
        elif "research" in query_lower or "score" in query_lower:
            research_node = next((n for n in chain.chain if n.stage == "research"), None)
            if research_node:
                answer = f"Research scored this EA at {research_node.details.get('score', 'N/A')}. Validation status: {research_node.details.get('validation', 'N/A')}."
            else:
                answer = "No research data available."
        elif "review" in query_lower or "approved" in query_lower:
            review_node = next((n for n in chain.chain if n.stage == "review"), None)
            if review_node:
                approved = review_node.details.get("approved", False)
                findings = review_node.details.get("findings", 0)
                answer = f"Code review: {'Approved' if approved else 'Not approved'} with {findings} findings."
            else:
                answer = "No review data available."
        else:
            answer = f"I can tell you about this EA's origin. It went through {chain.total_stages} stages: source → research → dev → review → approval. Would you like details on a specific stage?"

        return ProvenanceQueryResponse(
            answer=answer,
            chain=chain,
            confidence=0.9,
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
    return await _provenance_service.get_provenance_chain(strategy_id, version_tag)


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
    return await _provenance_service.query_provenance(
        request.strategy_id,
        request.query,
    )


@router.get("/{strategy_id}/versions/{version_tag}/provenance")
async def get_version_provenance(
    strategy_id: str,
    version_tag: str,
) -> ProvenanceChain:
    """
    Get provenance for a specific version.

    GET /api/strategies/{strategy_id}/versions/{version_tag}/provenance
    """
    return await _provenance_service.get_provenance_chain(strategy_id, version_tag)