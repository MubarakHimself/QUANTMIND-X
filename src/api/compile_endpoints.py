"""
MQL5 Compilation API Endpoints

Provides REST API for MQL5 compilation workflow.
"""
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.mql5.compiler.service import (
    MQL5CompilationService,
    get_compilation_service,
    COMPILE_STATUS_PENDING,
    COMPILE_STATUS_SUCCESS,
    COMPILE_STATUS_FAILED,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response models
class CompileRequest(BaseModel):
    """Request to compile an EA."""
    strategy_id: str
    version: Optional[int] = None


class CompileResponse(BaseModel):
    """Response from compilation request."""
    success: bool
    strategy_id: str
    version: int
    compile_status: str
    mq5_path: str
    ex5_path: Optional[str] = None
    errors: list[str] = []
    warnings: list[str] = []
    auto_correction_attempts: int = 0
    escalated: bool = False
    escalation_reason: Optional[str] = None


class CompilationStatusResponse(BaseModel):
    """Response for compilation status check."""
    strategy_id: str
    version: int
    compile_status: str
    mq5_path: str
    ex5_path: Optional[str] = None
    errors: Optional[list[str]] = None
    warnings: Optional[list[str]] = None
    compile_attempts: int = 0
    last_attempt: Optional[str] = None


# Service instance
_compilation_service: Optional[MQL5CompilationService] = None


def get_service() -> MQL5CompilationService:
    """Get or create compilation service."""
    global _compilation_service
    if _compilation_service is None:
        _compilation_service = get_compilation_service()
    return _compilation_service


@router.post("/compile", response_model=CompileResponse)
async def compile_ea(
    request: CompileRequest,
    background_tasks: BackgroundTasks,
) -> CompileResponse:
    """
    Compile an MQL5 EA to EX5.

    Triggers the compilation workflow including:
    - Docker-based compilation on Contabo
    - Auto-correction (up to 2 attempts)
    - Escalation to FloorManager on failure
    """
    logger.info(f"Received compile request for {request.strategy_id} v{request.version}")

    service = get_service()

    try:
        result = service.compile_ea(
            strategy_id=request.strategy_id,
            version=request.version,
        )

        return CompileResponse(
            success=result.success,
            strategy_id=result.strategy_id,
            version=result.version,
            compile_status=result.compile_status,
            mq5_path=result.mq5_path,
            ex5_path=result.ex5_path,
            errors=result.errors,
            warnings=result.warnings,
            auto_correction_attempts=result.auto_correction_attempts,
            escalated=result.escalated_to_floor_manager,
            escalation_reason=result.escalation_reason,
        )

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compile/{strategy_id}", response_model=CompilationStatusResponse)
async def get_compile_status(
    strategy_id: str,
    version: Optional[int] = None,
) -> CompilationStatusResponse:
    """
    Get compilation status for an EA.

    Args:
        strategy_id: Strategy identifier
        version: Specific version, or None for latest
    """
    from src.strategy.output import EAOutputStorage

    storage = EAOutputStorage()
    ea = storage.get_ea(strategy_id, version)

    if not ea:
        raise HTTPException(
            status_code=404,
            detail=f"EA not found: {strategy_id} v{version}",
        )

    return CompilationStatusResponse(
        strategy_id=ea.strategy_id,
        version=ea.version,
        compile_status=ea.compile_status or COMPILE_STATUS_PENDING,
        mq5_path=ea.file_path,
        ex5_path=ea.ex5_path,
        errors=ea.compile_errors,
        warnings=ea.compile_warnings,
        compile_attempts=ea.compile_attempts,
        last_attempt=ea.compile_last_attempt.isoformat() if ea.compile_last_attempt else None,
    )


@router.get("/compile/{strategy_id}/ex5")
async def download_ex5(
    strategy_id: str,
    version: Optional[int] = None,
):
    """
    Download compiled EX5 file.

    Returns the binary EX5 file if available.
    """
    from fastapi.responses import FileResponse
    from pathlib import Path

    storage = EAOutputStorage()
    ea = storage.get_ea(strategy_id, version)

    if not ea:
        raise HTTPException(
            status_code=404,
            detail=f"EA not found: {strategy_id} v{version}",
        )

    if not ea.ex5_path:
        raise HTTPException(
            status_code=404,
            detail="EX5 file not available - compilation may have failed",
        )

    ex5_path = Path(ea.ex5_path)
    if not ex5_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"EX5 file not found: {ea.ex5_path}",
        )

    return FileResponse(
        path=str(ex5_path),
        media_type="application/octet-stream",
        filename=ex5_path.name,
    )


@router.post("/compile/{strategy_id}/escalate")
async def escalate_compilation(
    strategy_id: str,
    reason: str,
    version: Optional[int] = None,
):
    """
    Manually escalate compilation failure to FloorManager.

    Args:
        strategy_id: Strategy identifier
        reason: Reason for escalation
        version: Specific version
    """
    service = get_service()

    # Get current errors
    storage = service.ea_storage
    ea = storage.get_ea(strategy_id, version)

    if not ea:
        raise HTTPException(
            status_code=404,
            detail=f"EA not found: {strategy_id} v{version}",
        )

    escalation = service.escalate_to_floor_manager(
        strategy_id=strategy_id,
        version=ea.version,
        reason=reason,
        errors=ea.compile_errors or [],
    )

    return {
        "status": "escalated",
        "message": "Escalation sent to FloorManager",
        "escalation": escalation,
    }
