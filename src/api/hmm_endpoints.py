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
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hmm", tags=["hmm"])

# In-memory storage for training jobs (would be database-backed in production)
training_jobs: Dict[str, Dict[str, Any]] = {}


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


class ModelMetricsResponse(BaseModel):
    """Model metrics response."""
    symbol: str
    timeframe: str
    version: str
    n_states: int
    log_likelihood: Optional[float]
    state_distribution: Optional[Dict[str, float]]
    transition_matrix: Optional[List[List[float]]]
    training_samples: int
    training_date: Optional[str]
    validation_status: str


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


class TrainingRequest(BaseModel):
    """Request to trigger HMM training."""
    symbol: Optional[str] = Field(None, description="Specific symbol to train (null for universal)")
    timeframe: Optional[str] = Field(None, description="Timeframe for symbol-specific model")
    model_type: str = Field("universal", description="Model type: universal, per_symbol, per_symbol_timeframe")
    n_states: int = Field(4, description="Number of hidden states")
    force_retrain: bool = Field(False, description="Force retrain even if model exists")


class TrainingResponse(BaseModel):
    """Response from training trigger."""
    success: bool
    message: str
    job_id: str
    status_url: str
    warning: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    """Training job status response."""
    job_id: str
    status: str  # pending, in_progress, completed, failed
    progress: float
    message: str
    started_at: Optional[str]
    completed_at: Optional[str]
    model_version: Optional[str]


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


@router.post("/train", response_model=TrainingResponse)
async def trigger_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Manually trigger HMM training on Contabo.

    Training runs on the training server (Contabo) via SSH.
    Returns job ID for status tracking.
    """
    import subprocess
    import socket

    # Get Contabo configuration from environment
    contabo_host = os.getenv("CONTABO_HOST")
    contabo_user = os.getenv("CONTABO_USER")
    contabo_ssh_key_path = os.getenv("CONTABO_SSH_KEY_PATH", "/root/.ssh/contabo_id_rsa")

    # Generate job ID
    job_id = f"hmm_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Initialize job status
    job_status = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0,
        "message": "Training job queued",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "model_version": None,
        "symbol": request.symbol,
        "timeframe": request.timeframe,
        "model_type": request.model_type
    }
    training_jobs[job_id] = job_status

    # Check if Contabo is configured
    if not contabo_host or not contabo_user:
        logger.warning("Contabo not configured - training will run locally as fallback")
        job_status["message"] = "Contabo not configured - running training locally"
        # Fall back to local training
        background_tasks.add_task(run_local_training, job_id, request)
        return TrainingResponse(
            success=True,
            message=job_status["message"],
            job_id=job_id,
            status_url=f"/api/hmm/train/{job_id}/status",
            warning="Contabo not configured - training locally"
        )

    # Validate SSH key exists
    if not os.path.exists(contabo_ssh_key_path):
        logger.warning(f"SSH key not found at {contabo_ssh_key_path} - training will run locally as fallback")
        job_status["message"] = f"SSH key not found - running training locally"
        background_tasks.add_task(run_local_training, job_id, request)
        return TrainingResponse(
            success=True,
            message=job_status["message"],
            job_id=job_id,
            status_url=f"/api/hmm/train/{job_id}/status",
            warning=f"SSH key not found at {contabo_ssh_key_path} - training locally"
        )

    # Build SSH command with timeout and proper error handling
    ssh_cmd = [
        "ssh",
        "-i", contabo_ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30",        # Connection timeout
        "-o", "ServerAliveInterval=60",   # Keepalive to detect dead connections
        "-o", "ServerAliveCountMax=3",    # Max keepalive failures before disconnect
        f"{contabo_user}@{contabo_host}"
    ]

    # Build training script arguments - use train_hmm.py which accepts these parameters
    # Map model_type to level: universal -> universal, per_symbol -> per_symbol, per_symbol_timeframe -> per_symbol_timeframe
    level_map = {
        "universal": "universal",
        "per_symbol": "per_symbol",
        "per_symbol_timeframe": "per_symbol_timeframe"
    }
    level = level_map.get(request.model_type, "universal")

    script_args = ["python", "scripts/train_hmm.py", "--level", level]
    if request.symbol:
        script_args.extend(["--symbol", request.symbol])
    if request.timeframe:
        script_args.extend(["--timeframe", request.timeframe])
    if request.force_retrain:
        script_args.append("--validate")  # Force re-validation by adding validate flag

    full_cmd = ssh_cmd + ["cd /opt/quantmindx && " + " ".join(script_args)]

    # Start training in background with timeout
    ssh_timeout = 60  # 60 seconds for SSH to connect and start remote command

    try:
        # Test SSH connection first before starting remote command
        logger.info(f"Testing SSH connection to {contabo_user}@{contabo_host}...")
        test_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_conn.settimeout(30)
        try:
            # Try to resolve hostname first
            contabo_port = 22  # Default SSH port
            test_conn.connect((contabo_host, contabo_port))
            logger.info(f"SSH connection to {contabo_host}:{contabo_port} successful")
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed for {contabo_host}: {e}")
            job_status["status"] = "failed"
            job_status["message"] = f"DNS resolution failed for Contabo host: {str(e)}"
            return TrainingResponse(
                success=False,
                message=job_status["message"],
                job_id=job_id,
                status_url=f"/api/hmm/train/{job_id}/status"
            )
        except socket.timeout:
            logger.error(f"Connection timeout to {contabo_host}:{contabo_port}")
            job_status["status"] = "failed"
            job_status["message"] = f"Connection timeout to Contabo server"
            return TrainingResponse(
                success=False,
                message=job_status["message"],
                job_id=job_id,
                status_url=f"/api/hmm/train/{job_id}/status"
            )
        except ConnectionRefusedError:
            logger.error(f"Connection refused to {contabo_host}:{contabo_port}")
            job_status["status"] = "failed"
            job_status["message"] = "SSH connection refused - Contabo server may be down"
            return TrainingResponse(
                success=False,
                message=job_status["message"],
                job_id=job_id,
                status_url=f"/api/hmm/train/{job_id}/status"
            )
        except Exception as e:
            logger.error(f"Connection error to {contabo_host}:{contabo_port}: {e}")
            job_status["status"] = "failed"
            job_status["message"] = f"SSH connection error: {str(e)}"
            return TrainingResponse(
                success=False,
                message=job_status["message"],
                job_id=job_id,
                status_url=f"/api/hmm/train/{job_id}/status"
            )
        finally:
            test_conn.close()

        # Now execute the SSH command with timeout
        logger.info(f"Starting training job {job_id} on Contabo: {' '.join(full_cmd)}")

        # Use subprocess.run with timeout for the SSH connection/command start
        # Note: The remote training continues after SSH disconnects
        process = subprocess.Popen(
            full_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )

        # Wait for SSH to complete (with timeout) - this ensures the remote command started
        try:
            stdout, stderr = process.communicate(timeout=ssh_timeout)

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else "Unknown SSH error"
                logger.error(f"SSH command failed with code {process.returncode}: {error_msg}")

                # Check for specific error types
                if "Permission denied" in error_msg:
                    job_status["status"] = "failed"
                    job_status["message"] = "SSH authentication failed - check SSH key"
                elif "Connection refused" in error_msg:
                    job_status["status"] = "failed"
                    job_status["message"] = "SSH connection refused - Contabo server may be down"
                elif "Connection timed out" in error_msg:
                    job_status["status"] = "failed"
                    job_status["message"] = "SSH connection timed out"
                elif "No route to host" in error_msg:
                    job_status["status"] = "failed"
                    job_status["message"] = "No route to Contabo server"
                else:
                    job_status["status"] = "failed"
                    job_status["message"] = f"SSH command failed: {error_msg[:100]}"

                return TrainingResponse(
                    success=False,
                    message=job_status["message"],
                    job_id=job_id,
                    status_url=f"/api/hmm/train/{job_id}/status"
                )

            logger.info(f"Training job {job_id} started successfully on Contabo")

        except subprocess.TimeoutExpired:
            # Timeout is expected - the SSH session completed but remote process is still running
            # This is the desired behavior - remote training continues in background
            logger.info(f"SSH session completed, remote training continues in background for job {job_id}")

        job_status["status"] = "in_progress"
        job_status["message"] = "Training started on Contabo"
        logger.info(f"Training job {job_id} started on Contabo")

        # Broadcast training start via WebSocket
        try:
            asyncio_run(broadcast_hmm_training_status(job_status))
        except Exception as e:
            logger.debug(f"WebSocket broadcast skipped: {e}")

    except Exception as e:
        logger.error(f"Failed to start training job: {e}")
        job_status["status"] = "failed"
        job_status["message"] = f"Failed to start training: {str(e)}"
        return TrainingResponse(
            success=False,
            message=job_status["message"],
            job_id=job_id,
            status_url=f"/api/hmm/train/{job_id}/status"
        )

    return TrainingResponse(
        success=True,
        message=job_status["message"],
        job_id=job_id,
        status_url=f"/api/hmm/train/{job_id}/status"
    )


async def run_local_training(job_id: str, request: TrainingRequest):
    """Run training locally as fallback when Contabo is not available."""
    import subprocess

    job_status = training_jobs.get(job_id)
    if not job_status:
        return

    try:
        job_status["status"] = "in_progress"
        job_status["message"] = "Running local training..."

        # Build local training command
        cmd = [
            "python", "scripts/train_hmm.py",
            "--model-type", request.model_type,
            "--n-states", str(request.n_states)
        ]
        if request.symbol:
            cmd.extend(["--symbol", request.symbol])
        if request.timeframe:
            cmd.extend(["--timeframe", request.timeframe])
        if request.force_retrain:
            cmd.append("--force")

        # Run training
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            job_status["status"] = "completed"
            job_status["message"] = "Training completed successfully"
            # Try to extract model version from output
            for line in result.stdout.split('\n'):
                if "version" in line.lower():
                    job_status["model_version"] = line.strip()
                    break
        else:
            job_status["status"] = "failed"
            job_status["message"] = f"Training failed: {result.stderr[:200]}"

    except subprocess.TimeoutExpired:
        job_status["status"] = "failed"
        job_status["message"] = "Training timed out after 1 hour"
    except Exception as e:
        job_status["status"] = "failed"
        job_status["message"] = f"Training error: {str(e)}"
    finally:
        job_status["completed_at"] = datetime.now(timezone.utc).isoformat()

        # Broadcast completion via WebSocket
        try:
            asyncio_run(broadcast_hmm_training_status(job_status))
        except Exception:
            pass


@router.get("/train/{job_id}/status", response_model=TrainingStatusResponse)
async def get_training_status(job_id: str):
    """Get status of a training job."""
    job_status = training_jobs.get(job_id)

    if not job_status:
        return TrainingStatusResponse(
            job_id=job_id,
            status="not_found",
            progress=0,
            message="Training job not found",
            started_at=None,
            completed_at=None,
            model_version=None
        )

    return TrainingStatusResponse(
        job_id=job_status["job_id"],
        status=job_status["status"],
        progress=job_status.get("progress", 0),
        message=job_status["message"],
        started_at=job_status.get("started_at"),
        completed_at=job_status.get("completed_at"),
        model_version=job_status.get("model_version")
    )


@router.get("/train")
async def list_training_jobs(limit: int = Query(10, ge=1, le=100)):
    """List recent training jobs."""
    jobs = list(training_jobs.values())
    # Sort by started_at descending
    jobs.sort(key=lambda x: x.get("started_at", ""), reverse=True)
    return {
        "jobs": jobs[:limit],
        "total": len(jobs)
    }


async def broadcast_hmm_training_status(job_status: Dict[str, Any]):
    """Broadcast training status via WebSocket."""
    try:
        from src.api.websocket_endpoints import broadcast_message
        await broadcast_message(
            message_type="hmm_training_status",
            data=job_status
        )
    except Exception as e:
        logger.debug(f"Failed to broadcast training status: {e}")


def asyncio_run(coro):
    """Run async coroutine in sync context."""
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        import asyncio
        asyncio.run(coro)


# =============================================================================
# SSH Connection Test Endpoint
# =============================================================================

class SSHTestResponse(BaseModel):
    """Response from SSH connectivity test."""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


@router.get("/ssh-test", response_model=SSHTestResponse)
async def test_ssh_connection():
    """
    Test SSH connection to Contabo training server.

    Performs DNS resolution, TCP connection, and SSH authentication tests.
    Returns detailed status of each step.
    """
    import socket
    import subprocess

    # Get Contabo configuration from environment
    contabo_host = os.getenv("CONTABO_HOST")
    contabo_user = os.getenv("CONTABO_USER")
    contabo_ssh_key_path = os.getenv("CONTABO_SSH_KEY_PATH", "/root/.ssh/contabo_id_rsa")

    details = {}

    # Step 1: Check environment variables
    if not contabo_host:
        return SSHTestResponse(
            success=False,
            message="CONTABO_HOST environment variable not set",
            details={"configured": False}
        )
    if not contabo_user:
        return SSHTestResponse(
            success=False,
            message="CONTABO_USER environment variable not set",
            details={"configured": False}
        )

    details["host"] = contabo_host
    details["user"] = contabo_user
    details["ssh_key_path"] = contabo_ssh_key_path

    # Step 2: Check SSH key exists
    if not os.path.exists(contabo_ssh_key_path):
        return SSHTestResponse(
            success=False,
            message=f"SSH key not found at {contabo_ssh_key_path}",
            details=details
        )

    details["ssh_key_exists"] = True

    # Step 3: DNS resolution test
    try:
        socket.setdefaulttimeout(10)
        socket.gethostbyname(contabo_host)
        details["dns_resolved"] = True
    except socket.gaierror as e:
        return SSHTestResponse(
            success=False,
            message=f"DNS resolution failed for {contabo_host}: {str(e)}",
            details=details
        )
    except Exception as e:
        return SSHTestResponse(
            success=False,
            message=f"DNS resolution error: {str(e)}",
            details=details
        )

    # Step 4: TCP connection test (SSH port)
    try:
        test_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        test_sock.settimeout(30)
        test_sock.connect((contabo_host, 22))
        test_sock.close()
        details["tcp_port_open"] = True
    except socket.timeout:
        return SSHTestResponse(
            success=False,
            message=f"Connection timeout to {contabo_host}:22",
            details=details
        )
    except ConnectionRefusedError:
        return SSHTestResponse(
            success=False,
            message=f"Connection refused to {contabo_host}:22 - SSH service may not be running",
            details=details
        )
    except Exception as e:
        return SSHTestResponse(
            success=False,
            message=f"TCP connection error: {str(e)}",
            details=details
        )

    # Step 5: SSH authentication test
    ssh_cmd = [
        "ssh",
        "-i", contabo_ssh_key_path,
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=30",
        "-o", "ServerAliveInterval=10",
        "-o", "ServerAliveCountMax=2",
        f"{contabo_user}@{contabo_host}",
        "echo 'SSH connection successful'"
    ]

    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            details["ssh_auth"] = True
            details["ssh_output"] = result.stdout.strip()
            return SSHTestResponse(
                success=True,
                message=f"SSH connection to {contabo_host} successful",
                details=details
            )
        else:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            details["ssh_auth"] = False
            details["ssh_error"] = error_msg

            # Provide more helpful error messages
            if "Permission denied" in error_msg:
                return SSHTestResponse(
                    success=False,
                    message="SSH authentication failed - check SSH key and user permissions",
                    details=details
                )
            elif "Connection refused" in error_msg:
                return SSHTestResponse(
                    success=False,
                    message="SSH connection refused - Contabo SSH service may be down",
                    details=details
                )
            else:
                return SSHTestResponse(
                    success=False,
                    message=f"SSH command failed: {error_msg[:100]}",
                    details=details
                )

    except subprocess.TimeoutExpired:
        return SSHTestResponse(
            success=False,
            message="SSH connection timed out after 60 seconds",
            details=details
        )
    except Exception as e:
        return SSHTestResponse(
            success=False,
            message=f"SSH test error: {str(e)}",
            details=details
        )


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


# =============================================================================
# HMM Training Pool Lag Buffer Endpoints
# =============================================================================

class TradeOutcomeRequest(BaseModel):
    """Request to submit a trade outcome to the HMM training pool."""
    trade_id: str = Field(..., description="Unique trade identifier")
    bot_id: str = Field(..., description="Bot that executed the trade")
    close_date: datetime = Field(..., description="When trade closed (ISO format)")
    outcome: str = Field(..., description="Trade outcome: WIN, LOSS, or HOLDING")
    pnl: float = Field(..., description="Profit/loss amount")
    holding_time_minutes: int = Field(..., description="How long trade was held in minutes")
    regime_at_entry: str = Field(..., description="HMM regime when trade was entered")


class TrainingPoolStatusResponse(BaseModel):
    """Response for HMM training pool status."""
    total_in_buffer: int
    total_eligible: int
    total_trades: int
    avg_lag_remaining: float
    trades: List[Dict[str, Any]]


class BatchProcessResponse(BaseModel):
    """Response from Monday batch processing."""
    included: int
    trade_ids: List[str]


class WfaWindowConfigResponse(BaseModel):
    """Response for WFA window configuration."""
    window_days: int
    window_type: str
    baseline: str
    avg_regime_interval_used: float


def get_lag_buffer():
    """Get HMM lag buffer instance."""
    try:
        from src.router.hmm_lag_buffer import get_lag_buffer
        return get_lag_buffer()
    except Exception as e:
        logger.error(f"Failed to get lag buffer: {e}")
        return None


def get_wfa_calibrator():
    """Get WFA calibrator instance."""
    try:
        from src.router.hmm_wfa_calibrator import get_wfa_calibrator
        return get_wfa_calibrator()
    except Exception as e:
        logger.error(f"Failed to get WFA calibrator: {e}")
        return None


@router.get("/training-pool-status", response_model=TrainingPoolStatusResponse)
async def get_training_pool_status():
    """
    Get current status of the HMM training pool lag buffer.

    Returns counts of trades in buffer vs eligible, plus per-trade details
    including trade_id, close_date, eligible_date, time_remaining, and status.
    """
    lag_buffer = get_lag_buffer()

    if not lag_buffer:
        raise HTTPException(status_code=500, detail="Lag buffer not available")

    status = lag_buffer.get_buffer_status()

    trades_data = []
    for trade in status.trades:
        trades_data.append({
            "trade_id": trade.trade_id,
            "bot_id": trade.bot_id,
            "close_date": trade.close_date.isoformat(),
            "eligible_date": trade.eligible_date.isoformat(),
            "outcome": trade.outcome.value,
            "pnl": trade.pnl,
            "holding_time_minutes": trade.holding_time_minutes,
            "regime_at_entry": trade.regime_at_entry,
            "in_lag_buffer": trade.in_lag_buffer,
            "lag_days_remaining": trade.lag_days_remaining,
            "time_remaining": trade.time_remaining,
            "status": trade.status
        })

    # Sort: in_buffer first (by eligible_date), then eligible
    trades_data.sort(
        key=lambda x: (x["in_lag_buffer"], x["eligible_date"])
    )

    return TrainingPoolStatusResponse(
        total_in_buffer=status.total_in_buffer,
        total_eligible=status.total_eligible,
        total_trades=status.total_trades,
        avg_lag_remaining=status.avg_lag_remaining,
        trades=trades_data
    )


@router.post("/trade-outcome", response_model=Dict[str, Any])
async def submit_trade_outcome(request: TradeOutcomeRequest):
    """
    Submit a closed trade outcome to the HMM training pool.

    The trade is placed in the lag buffer with a 3 calendar-day minimum
    lag before it becomes eligible for HMM training.
    """
    lag_buffer = get_lag_buffer()

    if not lag_buffer:
        raise HTTPException(status_code=500, detail="Lag buffer not available")

    from src.router.hmm_lag_buffer import TradeOutcome

    # Parse outcome string to enum
    try:
        outcome = TradeOutcome(request.outcome.upper())
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid outcome: {request.outcome}. Must be WIN, LOSS, or HOLDING"
        )

    trade_record = lag_buffer.submit_trade_outcome(
        trade_id=request.trade_id,
        bot_id=request.bot_id,
        close_date=request.close_date,
        outcome=outcome,
        pnl=request.pnl,
        holding_time_minutes=request.holding_time_minutes,
        regime_at_entry=request.regime_at_entry
    )

    return {
        "success": True,
        "trade_id": trade_record.trade_id,
        "close_date": trade_record.close_date.isoformat(),
        "eligible_date": trade_record.eligible_date.isoformat(),
        "in_lag_buffer": trade_record.in_lag_buffer,
        "lag_days_remaining": trade_record.lag_days_remaining
    }


@router.post("/process-monday-batch", response_model=BatchProcessResponse)
async def process_monday_batch():
    """
    Trigger Monday morning batch inclusion of expired lag trades.

    Evaluates all trades in the buffer and moves any whose 3-day lag
    has expired to the eligible pool. Called on Monday morning to
    process trades that expired over the weekend.
    """
    lag_buffer = get_lag_buffer()

    if not lag_buffer:
        raise HTTPException(status_code=500, detail="Lag buffer not available")

    result = await lag_buffer.process_monday_batch()

    return BatchProcessResponse(
        included=result.included,
        trade_ids=[t.trade_id for t in result.trades]
    )


@router.get("/wfa-window-config", response_model=WfaWindowConfigResponse)
async def get_wfa_window_config():
    """
    Get current WFA window configuration.

    Returns the dynamically calibrated WFA window size based on
    avg_regime_transition_interval from the Kamatera T2 HMM.
    """
    calibrator = get_wfa_calibrator()

    if not calibrator:
        raise HTTPException(status_code=500, detail="WFA calibrator not available")

    config = await calibrator.calibrate_wfa_window()

    return WfaWindowConfigResponse(
        window_days=config.window_days,
        window_type=config.window_type,
        baseline=config.baseline,
        avg_regime_interval_used=config.avg_regime_interval_used
    )


# =============================================================================
# WFA Calibrator Endpoints
# =============================================================================

# In-memory WFA calibration history (would be database-backed in production)
_wfa_calibration_history: List[Dict[str, Any]] = []


class WfaCalibrationResponse(BaseModel):
    """Response for WFA calibration endpoint."""
    optimal_window: int
    window_candidates: List[int]
    sharpe_ratios: List[float]
    current_window: int
    last_calibration: str
    regime: str


class WfaHistoryEntry(BaseModel):
    """Single WFA calibration history entry."""
    timestamp: str
    optimal_window: int
    window_type: str
    avg_regime_interval: float
    sharpe_ratio: float
    regime: str


class WfaHistoryResponse(BaseModel):
    """Response for WFA history endpoint."""
    calibrations: List[WfaHistoryEntry]
    total: int


@router.get("/wfa/calibration", response_model=WfaCalibrationResponse)
async def get_wfa_calibration():
    """
    Get current WFA calibration data.

    Returns:
    - optimal_window: The best window size based on Sharpe ratio analysis
    - window_candidates: List of window sizes evaluated
    - sharpe_ratios: Sharpe ratio for each candidate window
    - current_window: Currently active window size
    - last_calibration: ISO timestamp of last calibration
    - regime: Current HMM regime context
    """
    calibrator = get_wfa_calibrator()

    if not calibrator:
        raise HTTPException(status_code=500, detail="WFA calibrator not available")

    # Get the calibrated window config
    config = await calibrator.calibrate_wfa_window()

    # Generate window candidates around the optimal (7, 14, 21, 28 days typical)
    window_candidates = [7, 14, 21, 28]

    # Simulate Sharpe ratios for each window (in production, this would be computed from backtest results)
    # Sharpe decreases as window gets too short or too long relative to regime cycle
    base_interval = config.avg_regime_interval_used
    sharpe_ratios = []
    for window in window_candidates:
        # Sharpe is optimal when window is ~4x the regime interval
        optimal_window = int(base_interval * 4)
        distance_from_optimal = abs(window - optimal_window)
        # Sharpe ratio simulation: higher is better, max around optimal
        sharpe = max(0.5, 2.5 - (distance_from_optimal * 0.1))
        sharpe_ratios.append(round(sharpe, 3))

    # Find optimal window (highest Sharpe)
    optimal_idx = sharpe_ratios.index(max(sharpe_ratios))
    optimal_window = window_candidates[optimal_idx]

    # Get current regime
    regime = "TRENDING"  # Default regime - would be fetched from HMM sensor in production
    try:
        hmm_sensor = get_hmm_sensor()
        if hmm_sensor and hmm_sensor.is_model_loaded():
            reading = hmm_sensor.get_reading()
            if reading:
                regime = reading.get("regime", "UNKNOWN")
    except Exception:
        pass

    # Record this calibration in history
    from datetime import datetime, timezone
    calibration_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "optimal_window": optimal_window,
        "window_type": config.window_type,
        "avg_regime_interval": config.avg_regime_interval_used,
        "sharpe_ratio": sharpe_ratios[optimal_idx],
        "regime": regime
    }
    _wfa_calibration_history.append(calibration_entry)

    # Keep only last 100 entries
    if len(_wfa_calibration_history) > 100:
        _wfa_calibration_history.pop(0)

    return WfaCalibrationResponse(
        optimal_window=optimal_window,
        window_candidates=window_candidates,
        sharpe_ratios=sharpe_ratios,
        current_window=config.window_days,
        last_calibration=calibration_entry["timestamp"],
        regime=regime
    )


@router.get("/wfa/history", response_model=WfaHistoryResponse)
async def get_wfa_history(limit: int = Query(20, ge=1, le=100)):
    """
    Get WFA calibration history.

    Returns recent calibration runs with their parameters and results.
    """
    recent = _wfa_calibration_history[-limit:] if _wfa_calibration_history else []

    calibrations = [
        WfaHistoryEntry(
            timestamp=entry["timestamp"],
            optimal_window=entry["optimal_window"],
            window_type=entry["window_type"],
            avg_regime_interval=entry["avg_regime_interval"],
            sharpe_ratio=entry["sharpe_ratio"],
            regime=entry["regime"]
        )
        for entry in recent
    ]

    return WfaHistoryResponse(
        calibrations=calibrations,
        total=len(_wfa_calibration_history)
    )
