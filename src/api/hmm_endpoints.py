"""
HMM API Endpoints
=================

REST API endpoints for HMM regime detection system management.
Provides endpoints for model status, synchronization, mode management,
and shadow log retrieval.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hmm", tags=["hmm"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SyncRequest(BaseModel):
    """Request to sync model from Contabo."""
    version: Optional[str] = Field(None, description="Specific version to sync (null for latest)")
    verify_checksum: bool = Field(True, description="Verify checksum after transfer")


class SyncResponse(BaseModel):
    """Response from sync request."""
    success: bool
    message: str
    progress_url: Optional[str] = None


class ModeChangeRequest(BaseModel):
    """Request to change deployment mode."""
    mode: str = Field(..., description="Target mode: ising_only, hmm_shadow, hmm_hybrid_20, hmm_hybrid_50, hmm_hybrid_80, hmm_only")
    approval_token: Optional[str] = Field(None, description="Approval token (required for restricted modes)")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Current performance metrics")


class ApprovalTokenRequest(BaseModel):
    """Request to generate approval token."""
    target_mode: str = Field(..., description="Target mode for transition")
    requester: str = Field("api", description="Who is requesting the token")


class ApprovalTokenResponse(BaseModel):
    """Response with approval token."""
    token: str
    target_mode: str
    expires_at: str
    message: str


class HMMStatusResponse(BaseModel):
    """HMM system status response."""
    model_loaded: bool
    model_version: Optional[str]
    deployment_mode: str
    hmm_weight: float
    shadow_mode_active: bool
    contabo_version: Optional[str]
    cloudzy_version: Optional[str]
    version_mismatch: bool
    agreement_metrics: Dict[str, Any]
    last_sync: Optional[str]
    sync_status: Optional[str]


class ShadowLogResponse(BaseModel):
    """Shadow log entry response."""
    id: int
    timestamp: str
    symbol: str
    timeframe: str
    ising_regime: str
    ising_confidence: float
    hmm_regime: str
    hmm_state: int
    hmm_confidence: float
    agreement: bool
    decision_source: str


class ModelMetricsResponse(BaseModel):
    """Model metrics response."""
    symbol: Optional[str]
    timeframe: Optional[str]
    version: str
    n_states: int
    log_likelihood: Optional[float]
    state_distribution: Optional[Dict[str, float]]
    transition_matrix: Optional[List[List[float]]]
    training_samples: int
    training_date: Optional[str]
    validation_status: str


# =============================================================================
# Helper Functions
# =============================================================================

def get_hmm_sensor():
    """Get HMM sensor instance."""
    try:
        from src.risk.physics.hmm_sensor import create_hmm_sensor
        return create_hmm_sensor()
    except Exception as e:
        logger.error(f"Failed to get HMM sensor: {e}")
        return None


def get_deployment_manager():
    """Get deployment manager instance."""
    try:
        from src.router.hmm_deployment import get_deployment_manager
        return get_deployment_manager()
    except Exception as e:
        logger.error(f"Failed to get deployment manager: {e}")
        return None


def get_version_control():
    """Get version control instance."""
    try:
        from src.router.hmm_version_control import get_version_control
        return get_version_control()
    except Exception as e:
        logger.error(f"Failed to get version control: {e}")
        return None


def get_sentinel():
    """Get sentinel instance."""
    try:
        from src.router.sentinel import get_sentinel
        return get_sentinel()
    except Exception as e:
        logger.error(f"Failed to get sentinel: {e}")
        return None


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get("/status", response_model=HMMStatusResponse)
async def get_hmm_status():
    """
    Get comprehensive HMM system status.

    Returns current model versions on both servers, deployment mode,
    agreement metrics, and sync status.
    """
    hmm_sensor = get_hmm_sensor()
    deployment_mgr = get_deployment_manager()
    version_ctrl = get_version_control()
    sentinel = get_sentinel()

    # Get model info
    model_info = {}
    if hmm_sensor:
        model_info = hmm_sensor.get_model_info()

    # Get deployment state
    deployment_state = {}
    current_mode = "ising_only"
    hmm_weight = 0.0
    if deployment_mgr:
        state = deployment_mgr.get_current_state()
        deployment_state = state.to_dict()
        current_mode = deployment_state.get("mode", "ising_only")
        hmm_weight = deployment_mgr.get_hmm_weight_for_mode()

    # Get version comparison
    version_comparison = {}
    sync_status = None
    sync_message = None
    if version_ctrl:
        version_mismatch = version_ctrl.check_version_mismatch()
        version_info = version_ctrl.get_version_info()
        contabo_info = version_info.get("contabo", {})
        cloudzy_info = version_info.get("cloudzy", {})
        version_comparison = {
            "remote_version": contabo_info.get("version"),
            "local_version": cloudzy_info.get("version"),
            "mismatch": version_mismatch,
            "local_deployed": cloudzy_info.get("deployed_date")
        }
        sync_progress = version_ctrl.get_sync_progress()
        sync_status = sync_progress.get("status")
        sync_message = sync_progress.get("message")

    # Get agreement metrics
    agreement_metrics = {}
    if sentinel:
        agreement_metrics = sentinel.get_agreement_metrics()

    return HMMStatusResponse(
        model_loaded=model_info.get("loaded", False),
        model_version=model_info.get("version"),
        deployment_mode=current_mode,
        hmm_weight=hmm_weight,
        shadow_mode_active=current_mode == "hmm_shadow",
        contabo_version=version_comparison.get("remote_version"),
        cloudzy_version=version_comparison.get("local_version"),
        version_mismatch=version_comparison.get("mismatch", True),
        agreement_metrics=agreement_metrics,
        last_sync=version_comparison.get("local_deployed"),
        sync_status=sync_status
    )


@router.get("/metrics/{symbol}/{timeframe}", response_model=ModelMetricsResponse)
async def get_model_metrics(symbol: str, timeframe: str):
    """
    Get detailed metrics for a specific model.

    Args:
        symbol: Trading symbol (e.g., EURUSD)
        timeframe: Timeframe (e.g., H1, M5)
        
    Returns:
        ModelMetricsResponse with training metrics, state distribution,
        transition matrix, and validation status.
    """
    hmm_sensor = get_hmm_sensor()

    if not hmm_sensor or not hmm_sensor.is_model_loaded():
        raise HTTPException(status_code=404, detail="No HMM model loaded")

    model_info = hmm_sensor.get_model_info()
    metrics = model_info.get("metrics", {})

    return ModelMetricsResponse(
        symbol=symbol.upper(),
        timeframe=timeframe.upper(),
        version=model_info.get("version", "unknown"),
        n_states=model_info.get("n_states", 4),
        log_likelihood=metrics.get("log_likelihood"),
        state_distribution=metrics.get("state_distribution"),
        transition_matrix=metrics.get("transition_matrix"),
        training_samples=metrics.get("training_samples", 0),
        training_date=metrics.get("training_date") or model_info.get("last_load_time"),
        validation_status=metrics.get("validation_status", "unknown")
    )


# =============================================================================
# Sync Endpoints
# =============================================================================

@router.post("/sync", response_model=SyncResponse)
async def sync_model(request: SyncRequest, background_tasks: BackgroundTasks):
    """
    Trigger model sync from Contabo to Cloudzy.

    Sync progress is broadcast via WebSocket for real-time UI updates.
    """
    version_ctrl = get_version_control()

    if not version_ctrl:
        raise HTTPException(status_code=500, detail="Version control not available")

    # Check if sync is already in progress
    progress = version_ctrl.get_sync_progress()
    if progress.get("status") in ('in_progress', 'connecting', 'downloading', 'verifying', 'deploying'):
        raise HTTPException(status_code=409, detail=f"Sync already in progress: {progress.get('message')}")

    # Start sync in background - synchronous function that calls sync_model (which is sync)
    def run_sync():
        import asyncio
        from src.api.websocket_endpoints import broadcast_hmm_sync_progress

        def progress_callback(p):
            # Broadcast progress via WebSocket using asyncio.run for async broadcast
            try:
                asyncio.run(broadcast_hmm_sync_progress(
                    status=p.get("status"),
                    progress=p.get("progress"),
                    message=p.get("message"),
                    error=p.get("error")
                ))
            except RuntimeError:
                # If event loop is already running, create a new task
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(broadcast_hmm_sync_progress(
                        status=p.get("status"),
                        progress=p.get("progress"),
                        message=p.get("message"),
                        error=p.get("error")
                    ))

        version_ctrl.add_progress_callback(progress_callback)
        # Call sync_model synchronously with correct parameter names
        version_ctrl.sync_model(
            version=request.version,
            model_type="universal",  # Default model type
            verify_checksum=request.verify_checksum
        )

    background_tasks.add_task(run_sync)

    return SyncResponse(
        success=True,
        message="Sync started in background",
        progress_url="/api/hmm/sync/progress"
    )


@router.get("/sync/progress")
async def get_sync_progress():
    """Get current sync progress."""
    version_ctrl = get_version_control()

    if not version_ctrl:
        raise HTTPException(status_code=500, detail="Version control not available")

    progress = version_ctrl.get_sync_progress()

    return {
        "status": progress.get("status"),
        "progress": progress.get("progress"),
        "message": progress.get("message"),
        "error": progress.get("error"),
        "started_at": progress.get("timestamp"),
        "completed_at": None
    }


# =============================================================================
# Mode Management Endpoints
# =============================================================================

@router.post("/mode")
async def change_mode(request: ModeChangeRequest):
    """
    Change HMM deployment mode.

    Restricted modes require an approval token.
    """
    deployment_mgr = get_deployment_manager()

    if not deployment_mgr:
        raise HTTPException(status_code=500, detail="Deployment manager not available")

    from src.router.hmm_deployment import DeploymentMode

    # Validate mode
    try:
        target_mode = DeploymentMode(request.mode)
    except ValueError:
        valid_modes = [m.value for m in DeploymentMode]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Valid modes: {valid_modes}"
        )

    # Check transition validity
    can_transition, reason = deployment_mgr.can_transition_to(target_mode)
    if not can_transition:
        raise HTTPException(status_code=400, detail=reason)

    # Perform transition
    success = deployment_mgr.transition_to(
        target_mode,
        approval_token=request.approval_token,
        performance_metrics=request.metrics
    )

    if not success:
        raise HTTPException(status_code=400, detail="Transition failed")

    # Get HMM weight for the target mode
    hmm_weight = DeploymentMode.get_hmm_weight(target_mode)

    # Update sentinel mode
    sentinel = get_sentinel()
    if sentinel:
        sentinel.set_mode(
            shadow_mode=(target_mode == DeploymentMode.HMM_SHADOW),
            hmm_weight=hmm_weight
        )

    # Broadcast mode change via WebSocket
    from src.api.websocket_endpoints import broadcast_hmm_mode_change
    current_state = deployment_mgr.get_current_state()
    await broadcast_hmm_mode_change(
        previous_mode=None,  # Previous mode not tracked in current state
        new_mode=target_mode.value,
        hmm_weight=hmm_weight
    )

    return {
        "success": True,
        "message": f"Transitioned to {target_mode.value}",
        "current_mode": target_mode.value,
        "hmm_weight": hmm_weight
    }


@router.post("/approval-token", response_model=ApprovalTokenResponse)
async def generate_approval_token(request: ApprovalTokenRequest):
    """
    Generate an approval token for mode transition.

    Token expires after configured duration (default 24 hours).
    """
    deployment_mgr = get_deployment_manager()

    if not deployment_mgr:
        raise HTTPException(status_code=500, detail="Deployment manager not available")

    from src.router.hmm_deployment import DeploymentMode

    # Validate mode
    try:
        target_mode = DeploymentMode(request.target_mode)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid target mode")

    # Check if approval is required
    if not DeploymentMode.requires_approval(target_mode):
        raise HTTPException(
            status_code=400,
            detail=f"Approval not required for {target_mode.value}"
        )

    # Generate token using request_approval
    token = deployment_mgr.request_approval(target_mode, request.requester)

    # Get token expiry from pending approvals
    token_data = deployment_mgr._pending_approvals.get(token, {})
    expires_at = token_data.get("requested_at", "unknown")

    return ApprovalTokenResponse(
        token=token,
        target_mode=target_mode.value,
        expires_at=expires_at,
        message=f"Token generated for {target_mode.value}."
    )


@router.post("/rollback")
async def rollback_mode():
    """
    Rollback to previous deployment mode.

    Can be used for manual rollback when issues are detected.
    """
    deployment_mgr = get_deployment_manager()

    if not deployment_mgr:
        raise HTTPException(status_code=500, detail="Deployment manager not available")

    success = deployment_mgr.rollback()

    if not success:
        raise HTTPException(status_code=400, detail="Rollback failed")

    # Update sentinel
    sentinel = get_sentinel()
    if sentinel:
        current_state = deployment_mgr.get_current_state()
        hmm_weight = DeploymentMode.get_hmm_weight(current_state.mode)
        sentinel.set_mode(
            shadow_mode=(current_state.mode.value == "hmm_shadow"),
            hmm_weight=hmm_weight
        )

    current_state = deployment_mgr.get_current_state()
    return {
        "success": True,
        "message": "Rollback completed",
        "current_mode": current_state.mode.value
    }


# =============================================================================
# Shadow Mode Endpoints
# =============================================================================

@router.post("/shadow-mode/toggle")
async def toggle_shadow_mode(enabled: bool = Query(..., description="Enable or disable shadow mode")):
    """
    Enable or disable HMM shadow mode.

    Shadow mode runs HMM in parallel with Ising for validation.
    """
    deployment_mgr = get_deployment_manager()
    sentinel = get_sentinel()

    if not deployment_mgr:
        raise HTTPException(status_code=500, detail="Deployment manager not available")

    from src.router.hmm_deployment import DeploymentMode

    if enabled:
        target_mode = DeploymentMode.HMM_SHADOW
    else:
        target_mode = DeploymentMode.ISING_ONLY

    # Check transition
    can_transition, reason = deployment_mgr.can_transition_to(target_mode)
    if not can_transition:
        raise HTTPException(status_code=400, detail=reason)

    # Perform transition
    success = deployment_mgr.transition_to(target_mode)

    if not success:
        raise HTTPException(status_code=400, detail="Transition failed")

    # Update sentinel
    if sentinel:
        sentinel.set_mode(shadow_mode=enabled, hmm_weight=0.0)

    return {
        "success": True,
        "message": f"Shadow mode {'enabled' if enabled else 'disabled'}",
        "shadow_mode": enabled
    }


@router.get("/shadow-mode/logs")
async def get_shadow_logs(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    timeframe: Optional[str] = Query(None, description="Filter by timeframe"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of logs"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """
    Retrieve shadow mode logs with optional filtering.

    Logs show Ising vs HMM predictions for comparison.
    Queries the database for persistent storage.
    """
    sentinel = get_sentinel()

    if not sentinel:
        raise HTTPException(status_code=500, detail="Sentinel not available")

    # Query from database with filters and pagination
    logs, total = sentinel.get_shadow_logs_from_db(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        offset=offset
    )

    # Get agreement metrics
    metrics = sentinel.get_agreement_metrics()

    return {
        "logs": logs,
        "total": total,
        "offset": offset,
        "limit": limit,
        "agreement_metrics": metrics
    }


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models")
async def list_available_models():
    """List all available HMM model versions."""
    version_ctrl = get_version_control()

    if not version_ctrl:
        raise HTTPException(status_code=500, detail="Version control not available")

    models = version_ctrl.list_available_models()

    return {
        "models": models,
        "total": len(models)
    }


@router.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Manually trigger HMM training on Contabo.

    Training runs on the training server (Contabo).
    Returns job ID for status tracking.
    """
    # This would typically call a training job API on Contabo
    # For now, return a placeholder response

    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "success": True,
        "message": "Training job submitted",
        "job_id": job_id,
        "status_url": f"/api/hmm/train/{job_id}/status"
    }


@router.get("/train/{job_id}/status")
async def get_training_status(job_id: str):
    """Get status of a training job."""
    # Placeholder - would check actual training job status
    return {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Training job not found or not implemented"
    }


# =============================================================================
# Prediction Endpoint
# =============================================================================

@router.get("/predict/{symbol}/{timeframe}")
async def get_current_prediction(symbol: str, timeframe: str):
    """
    Get current HMM prediction for a symbol/timeframe.

    Returns the most recent prediction from the cache.
    """
    hmm_sensor = get_hmm_sensor()

    if not hmm_sensor or not hmm_sensor.is_model_loaded():
        raise HTTPException(status_code=404, detail="HMM model not loaded")

    # Get model info and reading
    model_info = hmm_sensor.get_model_info()
    reading = hmm_sensor.get_reading()
    confidence = hmm_sensor.get_regime_confidence()

    return {
        "symbol": symbol.upper(),
        "timeframe": timeframe.upper(),
        "model_version": model_info.get("version"),
        "reading": reading,
        "confidence": confidence,
        "cache_hits": model_info.get("cache_hits", 0),
        "cache_misses": model_info.get("cache_misses", 0)
    }
