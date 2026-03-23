"""
EA Deployment API Endpoints

FastAPI router for triggering EA deployments to live MT5 trading.
Implements the deployment pipeline REST interface.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/deployment", tags=["deployment"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class DeploymentRequest(BaseModel):
    """Request model for EA deployment."""
    strategy_id: str
    ea_name: str
    ea_params: Optional[Dict[str, Any]] = None
    source_dir: Optional[str] = "src/mql5/Experts"


class DeploymentStatusRequest(BaseModel):
    """Request model for deployment status check."""
    deployment_id: str


class DeploymentListRequest(BaseModel):
    """Request model for listing deployments."""
    strategy_id: Optional[str] = None
    status: Optional[str] = None
    limit: int = 50
    offset: int = 0


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class DeploymentResponse(BaseModel):
    """Response model for deployment."""
    deployment_id: str
    strategy_id: str
    status: str
    status_detail: str
    started_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None


# In-memory deployment tracking (would use database in production)
_deployments: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# DEPLOYMENT ENDPOINTS
# =============================================================================

@router.post("/deploy", response_model=DeploymentResponse)
async def deploy_ea(
    request: DeploymentRequest,
    background_tasks: BackgroundTasks
) -> DeploymentResponse:
    """
    Trigger EA deployment to live MT5 trading.

    This endpoint starts the deployment pipeline which:
    1. Checks deployment window (Friday 22:00 - Sunday 22:00 UTC)
    2. Transfers EA file to Cloudzy server
    3. Registers EA on MT5 terminal
    4. Injects configuration parameters
    5. Verifies health check (ACTIVE state + tick received)
    6. Registers ZMQ stream for position updates

    Args:
        request: Deployment request with strategy_id, ea_name, and optional params

    Returns:
        Deployment response with deployment_id and initial status
    """
    logger.info(f"Deployment request received: {request.strategy_id} / {request.ea_name}")

    # Generate deployment ID
    import time
    deployment_id = f"deploy_{request.strategy_id}_{int(time.time())}"

    # Store initial deployment record
    _deployments[deployment_id] = {
        "deployment_id": deployment_id,
        "strategy_id": request.strategy_id,
        "ea_name": request.ea_name,
        "ea_params": request.ea_params,
        "source_dir": request.source_dir,
        "status": "pending",
        "status_detail": "Deployment queued",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None
    }

    # Trigger the deployment in background
    background_tasks.add_task(
        _run_deployment,
        deployment_id,
        request.strategy_id,
        request.ea_name,
        request.ea_params,
        request.source_dir
    )

    return DeploymentResponse(
        deployment_id=deployment_id,
        strategy_id=request.strategy_id,
        status="pending",
        status_detail="Deployment queued",
        started_at=_deployments[deployment_id]["started_at"]
    )


async def _run_deployment(
    deployment_id: str,
    strategy_id: str,
    ea_name: str,
    ea_params: Optional[Dict[str, Any]],
    source_dir: str
):
    """Run the actual deployment in async context."""
    try:
        # Import the deployment flow
        from flows.assembled.ea_deployment_flow import ea_deployment_pipeline

        # Update status
        _deployments[deployment_id]["status"] = "running"
        _deployments[deployment_id]["status_detail"] = "Deployment in progress"

        # Run the deployment pipeline
        result = await asyncio.to_thread(
            ea_deployment_pipeline,
            strategy_id=strategy_id,
            ea_name=ea_name,
            ea_params=ea_params,
            source_dir=source_dir
        )

        # Update deployment record with result
        if result.get("status") == "deployed":
            _deployments[deployment_id]["status"] = "deployed"
            _deployments[deployment_id]["status_detail"] = "Deployment successful"
            _deployments[deployment_id]["completed_at"] = result.get("completed_at")
            _deployments[deployment_id]["result"] = result
        else:
            _deployments[deployment_id]["status"] = "failed"
            _deployments[deployment_id]["status_detail"] = f"Deployment failed: {result.get('reason', 'Unknown')}"
            _deployments[deployment_id]["error"] = result.get("reason")

    except Exception as e:
        logger.error(f"Deployment error: {e}")
        _deployments[deployment_id]["status"] = "error"
        _deployments[deployment_id]["status_detail"] = f"Deployment error: {str(e)}"
        _deployments[deployment_id]["error"] = str(e)


@router.get("/status/{deployment_id}", response_model=DeploymentResponse)
async def get_deployment_status(deployment_id: str) -> DeploymentResponse:
    """
    Get deployment status by deployment ID.

    Args:
        deployment_id: The deployment identifier

    Returns:
        Current deployment status
    """
    if deployment_id not in _deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment = _deployments[deployment_id]

    return DeploymentResponse(
        deployment_id=deployment["deployment_id"],
        strategy_id=deployment["strategy_id"],
        status=deployment["status"],
        status_detail=deployment["status_detail"],
        started_at=deployment["started_at"],
        completed_at=deployment.get("completed_at"),
        error=deployment.get("error")
    )


@router.get("/list")
async def list_deployments(
    strategy_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """
    List deployments with optional filtering.

    Args:
        strategy_id: Filter by strategy ID
        status: Filter by deployment status (pending, running, deployed, failed, error)
        limit: Maximum number of results
        offset: Offset for pagination

    Returns:
        List of deployments with pagination info
    """
    deployments = list(_deployments.values())

    # Apply filters
    if strategy_id:
        deployments = [d for d in deployments if d["strategy_id"] == strategy_id]

    if status:
        deployments = [d for d in deployments if d["status"] == status]

    # Apply pagination
    total = len(deployments)
    deployments = deployments[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "deployments": [
            {
                "deployment_id": d["deployment_id"],
                "strategy_id": d["strategy_id"],
                "status": d["status"],
                "status_detail": d["status_detail"],
                "started_at": d["started_at"],
                "completed_at": d.get("completed_at")
            }
            for d in deployments
        ]
    }


@router.post("/cancel/{deployment_id}")
async def cancel_deployment(deployment_id: str) -> Dict[str, Any]:
    """
    Cancel a pending or running deployment.

    Args:
        deployment_id: The deployment identifier

    Returns:
        Cancellation result
    """
    if deployment_id not in _deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment = _deployments[deployment_id]

    if deployment["status"] in ["pending", "running"]:
        deployment["status"] = "cancelled"
        deployment["status_detail"] = "Deployment cancelled by user"
        deployment["completed_at"] = datetime.utcnow().isoformat()

        return {
            "status": "cancelled",
            "deployment_id": deployment_id,
            "cancelled_at": deployment["completed_at"]
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel deployment in status: {deployment['status']}"
        )


# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@router.get("/health/deployment-window")
async def check_deployment_window() -> Dict[str, Any]:
    """
    Check if current time is within deployment window.

    Deployment window: Friday 22:00 - Sunday 22:00 UTC

    Returns:
        Deployment window status
    """
    try:
        from flows.alpha_forge_flow import check_deployment_window

        result = check_deployment_window()
        return {
            "in_window": result.get("in_window"),
            "current_time": result.get("current_time"),
            "window_info": result.get("window_info")
        }
    except Exception as e:
        logger.error(f"Deployment window check error: {e}")
        return {
            "in_window": False,
            "error": str(e)
        }


# =============================================================================
# WEBHOOK FOR DEPLOYMENT TRIGGER (from approval gate)
# =============================================================================

@router.post("/webhook/approval-trigger")
async def approval_trigger_webhook(
    strategy_id: str,
    approval_status: str
) -> Dict[str, Any]:
    """
    Webhook endpoint to trigger deployment after human approval.
    Called from approval gate when FR79 approval is granted.

    Args:
        strategy_id: Strategy that was approved
        approval_status: Approval status (approved, rejected)

    Returns:
        Trigger result
    """
    logger.info(f"Approval trigger webhook: {strategy_id} -> {approval_status}")

    if approval_status != "approved":
        return {
            "status": "skipped",
            "reason": f"Approval status is {approval_status}, not approved"
        }

    # In production, would look up the EA name from strategy
    # For now, construct from strategy_id
    ea_name = f"QuantMindX_{strategy_id}"

    deployment_id = f"deploy_{strategy_id}_{int(datetime.utcnow().timestamp())}"

    _deployments[deployment_id] = {
        "deployment_id": deployment_id,
        "strategy_id": strategy_id,
        "ea_name": ea_name,
        "ea_params": None,
        "source_dir": "src/mql5/Experts",
        "status": "pending",
        "status_detail": "Triggered by approval webhook",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "trigger": "approval_webhook"
    }

    return {
        "status": "triggered",
        "deployment_id": deployment_id,
        "strategy_id": strategy_id,
        "message": "Deployment triggered by approval"
    }


# =============================================================================
# ROUTER REGISTRATION INFO
# =============================================================================

def get_router() -> APIRouter:
    """Return the API router for registration."""
    return router