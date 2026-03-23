"""
Alpha Forge Template API Endpoints

REST API for strategy template library management.
"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.mql5.templates.schema import (
    StrategyTemplate,
    TemplateMatchResult,
    EventType,
    StrategyTypeTemplate,
    RiskProfile,
)
from src.mql5.templates.storage import get_template_storage
from src.mql5.templates.matcher import get_template_matcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/alpha-forge", tags=["alpha-forge"])


# ============================================================================
# Request/Response Models
# ============================================================================


class TemplateCreateRequest(BaseModel):
    """Request model for creating a new template."""
    name: str = Field(..., description="Template name")
    strategy_type: StrategyTypeTemplate = Field(
        default=StrategyTypeTemplate.NEWS_EVENT_BREAKOUT,
        description="Strategy type"
    )
    applicable_events: List[str] = Field(
        default=[],
        description="List of applicable event types"
    )
    applicable_symbols: List[str] = Field(
        default=[],
        description="List of applicable trading symbols"
    )
    risk_profile: RiskProfile = Field(
        default=RiskProfile.CONSERVATIVE,
        description="Risk profile"
    )
    avg_deployment_time: int = Field(
        default=11,
        description="Average deployment time in minutes"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Template parameters (EA input parameters)"
    )
    lot_sizing_multiplier: float = Field(
        default=0.5,
        description="Lot sizing multiplier (0.5 = conservative)"
    )
    auto_expiry_hours: int = Field(
        default=24,
        description="Auto-expiry in hours"
    )
    is_islamic_compliant: bool = Field(
        default=True,
        description="Islamic compliance flag"
    )


class TemplateUpdateRequest(BaseModel):
    """Request model for updating a template."""
    name: Optional[str] = None
    strategy_type: Optional[StrategyTypeTemplate] = None
    applicable_events: Optional[List[str]] = None
    applicable_symbols: Optional[List[str]] = None
    risk_profile: Optional[RiskProfile] = None
    avg_deployment_time: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None
    lot_sizing_multiplier: Optional[float] = None
    auto_expiry_hours: Optional[int] = None
    is_islamic_compliant: Optional[bool] = None
    is_active: Optional[bool] = None


class TemplateMatchRequest(BaseModel):
    """Request model for template matching."""
    event_type: str = Field(..., description="News event type")
    affected_symbols: List[str] = Field(
        default=[],
        description="List of affected trading symbols"
    )
    impact_tier: str = Field(
        default="HIGH",
        description="Impact tier: HIGH, MEDIUM, or LOW"
    )


class TemplateResponse(BaseModel):
    """Response model for template."""
    id: str
    name: str
    strategy_type: str
    applicable_events: List[str]
    applicable_symbols: List[str]
    risk_profile: str
    avg_deployment_time: int
    parameters: Dict[str, Any]
    lot_sizing_multiplier: float
    auto_expiry_hours: int
    is_islamic_compliant: bool
    created_at: Optional[str]
    updated_at: Optional[str]
    author: str
    is_active: bool


class TemplateListResponse(BaseModel):
    """Response model for template list."""
    templates: List[TemplateResponse]
    total: int


class TemplateMatchResponse(BaseModel):
    """Response model for template matching."""
    matches: List[Dict[str, Any]]
    total: int
    event_type: str
    impact_tier: str


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/templates", response_model=TemplateListResponse)
async def get_templates(
    active_only: bool = Query(True, description="Return only active templates"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
) -> TemplateListResponse:
    """
    Get all strategy templates.

    GET /api/alpha-forge/templates

    Returns templates with: name, strategy_type, applicable_events,
    risk_profile, avg_deployment_time.
    """
    storage = get_template_storage()

    # Seed default templates if empty
    templates = storage.seed_default_templates()

    # Apply filters
    if active_only:
        templates = [t for t in templates if t.is_active]

    if event_type:
        templates = [t for t in templates if event_type in t.applicable_events]

    if symbol:
        templates = [t for t in templates if symbol in t.applicable_symbols]

    return TemplateListResponse(
        templates=[TemplateResponse(**t.to_dict()) for t in templates],
        total=len(templates),
    )


@router.get("/templates/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str) -> TemplateResponse:
    """Get a specific template by ID."""
    storage = get_template_storage()
    template = storage.get(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return TemplateResponse(**template.to_dict())


@router.post("/templates", response_model=TemplateResponse, status_code=201)
async def create_template(request: TemplateCreateRequest) -> TemplateResponse:
    """Create a new strategy template."""
    storage = get_template_storage()

    template = StrategyTemplate(
        name=request.name,
        strategy_type=request.strategy_type,
        applicable_events=request.applicable_events,
        applicable_symbols=request.applicable_symbols,
        risk_profile=request.risk_profile,
        avg_deployment_time=request.avg_deployment_time,
        parameters=request.parameters,
        lot_sizing_multiplier=request.lot_sizing_multiplier,
        auto_expiry_hours=request.auto_expiry_hours,
        is_islamic_compliant=request.is_islamic_compliant,
    )

    saved = storage.save(template)
    return TemplateResponse(**saved.to_dict())


@router.put("/templates/{template_id}", response_model=TemplateResponse)
async def update_template(
    template_id: str,
    request: TemplateUpdateRequest,
) -> TemplateResponse:
    """Update an existing template."""
    storage = get_template_storage()
    template = storage.get(template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Apply updates
    update_data = request.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if hasattr(template, key):
            setattr(template, key, value)

    template.updated_at = datetime.now()
    saved = storage.save(template)

    return TemplateResponse(**saved.to_dict())


@router.delete("/templates/{template_id}")
async def delete_template(template_id: str) -> Dict[str, str]:
    """Delete a template."""
    storage = get_template_storage()
    success = storage.delete(template_id)

    if not success:
        raise HTTPException(status_code=404, detail="Template not found")

    return {"status": "deleted", "template_id": template_id}


@router.post("/templates/match", response_model=TemplateMatchResponse)
async def match_templates(request: TemplateMatchRequest) -> TemplateMatchResponse:
    """
    Match templates against a news event.

    Returns ranked templates with confidence scores.
    """
    matcher = get_template_matcher()

    matches = matcher.get_top_matches(
        event_type=request.event_type,
        affected_symbols=request.affected_symbols,
        impact_tier=request.impact_tier,
        min_confidence=0.3,
        limit=5,
    )

    return TemplateMatchResponse(
        matches=[m.to_dict() for m in matches],
        total=len(matches),
        event_type=request.event_type,
        impact_tier=request.impact_tier,
    )


@router.get("/templates/match/simulate")
async def simulate_template_match(
    event_type: str = Query(..., description="Event type"),
    symbols: str = Query(..., description="Comma-separated symbols"),
    impact: str = Query("HIGH", description="Impact tier"),
) -> TemplateMatchResponse:
    """
    Simulate template matching (for testing/demo).

    Query parameters:
    - event_type: HIGH_IMPACT_NEWS, CENTRAL_BANK, GEOPOLITICAL, etc.
    - symbols: comma-separated list (e.g., "EURUSD,GBPUSD")
    - impact: HIGH, MEDIUM, or LOW
    """
    affected_symbols = [s.strip() for s in symbols.split(",")]

    matcher = get_template_matcher()
    matches = matcher.get_top_matches(
        event_type=event_type,
        affected_symbols=affected_symbols,
        impact_tier=impact,
    )

    return TemplateMatchResponse(
        matches=[m.to_dict() for m in matches],
        total=len(matches),
        event_type=event_type,
        impact_tier=impact,
    )


# ============================================================================
# Fast-Track Deployment Endpoints
# ============================================================================


class FastTrackDeployRequest(BaseModel):
    """Request model for fast-track deployment."""
    template_id: str = Field(..., description="Template ID to deploy")
    symbol: str = Field(..., description="Trading symbol")
    strategy_name: Optional[str] = Field(None, description="Optional custom strategy name")


class FastTrackDeployResponse(BaseModel):
    """Response model for fast-track deployment."""
    workflow_id: str
    status: str
    strategy_id: Optional[str] = None
    message: str
    estimated_time: int  # minutes


@router.post("/fast-track/deploy", response_model=FastTrackDeployResponse)
async def deploy_fast_track(request: FastTrackDeployRequest) -> FastTrackDeployResponse:
    """
    Deploy a strategy using the fast-track workflow.

    POST /api/alpha-forge/fast-track/deploy

    This triggers the FastTrackFlow which:
    1. Generates TRD from template (with fast-track settings)
    2. Compiles MQL5 code
    3. Runs SIT gate only (no full backtest)
    4. Deploys to live trading

    Target: 11-15 minute deployment
    """
    # Get template
    storage = get_template_storage()
    template = storage.get(request.template_id)

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    if not template.is_active:
        raise HTTPException(status_code=400, detail="Template is not active")

    # Run fast-track flow
    try:
        from flows.fast_track_flow import fast_track_flow

        result = fast_track_flow(
            template_id=template.id,
            template_data=template.to_dict(),
            symbol=request.symbol,
            strategy_name=request.strategy_name,
        )

        return FastTrackDeployResponse(
            workflow_id=result.get("workflow_id", ""),
            status=result.get("status", "unknown"),
            strategy_id=result.get("strategy_id"),
            message=f"Fast-track deployment initiated for {template.name}",
            estimated_time=template.avg_deployment_time,
        )

    except Exception as e:
        logger.error(f"Fast-track deployment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Fast-track deployment failed: {str(e)}"
        )


@router.get("/fast-track/status/{workflow_id}")
async def get_fast_track_status(workflow_id: str) -> Dict[str, Any]:
    """Get fast-track workflow status."""
    from flows.database import get_workflow_database

    db = get_workflow_database()

    # Get workflow from database
    # Note: In production, would query the actual workflow run
    return {
        "workflow_id": workflow_id,
        "status": "running",  # Placeholder
        "message": "Use Prefect dashboard for detailed status"
    }