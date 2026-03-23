"""
Strategy Version Control API Endpoints

REST API for EA version control and rollback operations.
"""
import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.mql5.versions import get_version_manager, get_ea_version_storage
from src.mql5.versions.schema import EAVersionArtifacts, VariantType
from src.mql5.versions.manager import VersionManager
from src.mql5.versions.storage import EAVersionStorage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategy-versions"])


# ============================================================================
# Request/Response Models
# ============================================================================


class CreateVersionRequest(BaseModel):
    """Request model for creating a new version."""
    strategy_id: str = Field(..., description="Strategy identifier")
    author: str = Field(..., description="Creator of this version")
    source_code: Optional[str] = Field(None, description="Source code for hash")
    template_deps: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Template dependencies"
    )
    auto_increment: str = Field(
        default="patch",
        description="Which version part to increment (major/minor/patch)"
    )
    variant_type: VariantType = Field(
        default=VariantType.VANILLA,
        description="Variant classification"
    )
    improvement_cycle: int = Field(
        default=0,
        description="Workflow iteration number"
    )
    artifacts: Optional[EAVersionArtifacts] = Field(
        default=None,
        description="Linked artifacts"
    )


class UpdateArtifactsRequest(BaseModel):
    """Request model for updating version artifacts."""
    mq5_path: Optional[str] = None
    ex5_path: Optional[str] = None
    trd_id: Optional[str] = None
    backtest_result_ids: Optional[List[str]] = None


class RollbackRequest(BaseModel):
    """Request model for rollback operation."""
    target_version: str = Field(..., description="Version to rollback to")
    author: str = Field(..., description="Who initiated rollback")
    reason: Optional[str] = Field(None, description="Rollback reason")


class VersionResponse(BaseModel):
    """Response model for a version."""
    id: str
    strategy_id: str
    version_tag: str
    created_at: str
    author: str
    source_hash: str
    template_deps: Dict[str, Any]
    pin_template_version: bool
    variant_type: str
    improvement_cycle: int
    artifacts: Dict[str, Any]
    is_active: bool


class VersionListResponse(BaseModel):
    """Response model for version list."""
    strategy_id: str
    versions: List[VersionResponse]
    active_version: Optional[str]
    total: int


class VersionMetadataResponse(BaseModel):
    """Response model for version metadata."""
    version_tag: str
    created_at: str
    author: str
    source_hash: str
    variant_type: str
    improvement_cycle: int


class RollbackResponse(BaseModel):
    """Response model for rollback operation."""
    success: bool
    strategy_id: str
    from_version: str
    to_version: str
    compilation_passed: bool
    sit_validation_passed: bool
    audit_id: Optional[str]
    message: str


class HistoryResponse(BaseModel):
    """Response model for version history."""
    strategy_id: str
    versions: List[Dict[str, Any]]
    active_version: Optional[str]
    rollback_history: List[Dict[str, Any]]


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/{strategy_id}/versions", response_model=VersionListResponse)
async def list_strategy_versions(
    strategy_id: str,
    include_metadata: bool = Query(
        False,
        description="Include only metadata (timestamp, author)"
    ),
) -> VersionListResponse:
    """
    List all versions of a strategy.

    GET /api/strategies/{strategy_id}/versions

    Returns version list with optional metadata display.
    """
    storage = get_ea_version_storage()

    if include_metadata:
        metadata = storage.list_version_metadata(strategy_id)
        return VersionListResponse(
            strategy_id=strategy_id,
            versions=[
                VersionResponse(
                    id="",
                    strategy_id=strategy_id,
                    version_tag=m["version_tag"],
                    created_at=m["created_at"] or "",
                    author=m["author"],
                    source_hash=m["source_hash"],
                    template_deps={},
                    pin_template_version=False,
                    variant_type=m["variant_type"],
                    improvement_cycle=m["improvement_cycle"],
                    artifacts={},
                    is_active=False,
                )
                for m in metadata
            ],
            active_version=None,
            total=len(metadata),
        )

    versions = storage.list_versions(strategy_id)
    active = storage.get_active_version(strategy_id)

    return VersionListResponse(
        strategy_id=strategy_id,
        versions=[
            VersionResponse(
                id=v.id,
                strategy_id=v.strategy_id,
                version_tag=v.version_tag,
                created_at=v.created_at.isoformat() if v.created_at else "",
                author=v.author,
                source_hash=v.source_hash,
                template_deps=v.template_deps,
                pin_template_version=v.pin_template_version,
                variant_type=v.variant_type.value,
                improvement_cycle=v.improvement_cycle,
                artifacts=v.artifacts.model_dump(),
                is_active=active.version_tag == v.version_tag if active else False,
            )
            for v in versions
        ],
        active_version=active.version_tag if active else None,
        total=len(versions),
    )


@router.get("/{strategy_id}/versions/{version_tag}", response_model=VersionResponse)
async def get_strategy_version(
    strategy_id: str,
    version_tag: str,
) -> VersionResponse:
    """Get a specific version of a strategy."""
    storage = get_ea_version_storage()
    version = storage.get_version(strategy_id, version_tag)
    active = storage.get_active_version(strategy_id)

    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_tag} not found for strategy {strategy_id}"
        )

    return VersionResponse(
        id=version.id,
        strategy_id=version.strategy_id,
        version_tag=version.version_tag,
        created_at=version.created_at.isoformat() if version.created_at else "",
        author=version.author,
        source_hash=version.source_hash,
        template_deps=version.template_deps,
        pin_template_version=version.pin_template_version,
        variant_type=version.variant_type.value,
        improvement_cycle=version.improvement_cycle,
        artifacts=version.artifacts.model_dump(),
        is_active=active.version_tag == version.version_tag if active else False,
    )


@router.post("/{strategy_id}/versions", response_model=VersionResponse, status_code=201)
async def create_strategy_version(
    strategy_id: str,
    request: CreateVersionRequest,
) -> VersionResponse:
    """
    Create a new version for a strategy.

    POST /api/strategies/{strategy_id}/versions

    Creates a new version with semantic versioning and artifact linking.
    """
    manager = get_version_manager()

    version = manager.create_new_version(
        strategy_id=strategy_id,
        author=request.author,
        source_code=request.source_code or "",
        template_deps=request.template_deps,
        auto_increment=request.auto_increment,
        variant_type=request.variant_type,
        improvement_cycle=request.improvement_cycle,
    )

    # Update artifacts if provided
    if request.artifacts:
        storage = get_ea_version_storage()
        storage.update_artifacts(strategy_id, version.version_tag, request.artifacts)
        version = storage.get_version(strategy_id, version.version_tag)

    return VersionResponse(
        id=version.id,
        strategy_id=version.strategy_id,
        version_tag=version.version_tag,
        created_at=version.created_at.isoformat() if version.created_at else "",
        author=version.author,
        source_hash=version.source_hash,
        template_deps=version.template_deps,
        pin_template_version=version.pin_template_version,
        variant_type=version.variant_type.value,
        improvement_cycle=version.improvement_cycle,
        artifacts=version.artifacts.model_dump(),
        is_active=True,
    )


@router.put("/{strategy_id}/versions/{version_tag}/artifacts")
async def update_version_artifacts(
    strategy_id: str,
    version_tag: str,
    request: UpdateArtifactsRequest,
) -> Dict[str, Any]:
    """Update artifacts for a specific version."""
    storage = get_ea_version_storage()

    # Build artifacts object
    artifacts = EAVersionArtifacts(
        mq5_path=request.mq5_path,
        ex5_path=request.ex5_path,
        trd_id=request.trd_id,
        backtest_result_ids=request.backtest_result_ids or [],
    )

    version = storage.update_artifacts(strategy_id, version_tag, artifacts)

    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_tag} not found for strategy {strategy_id}"
        )

    return {
        "success": True,
        "strategy_id": strategy_id,
        "version_tag": version_tag,
        "artifacts": version.artifacts.model_dump(),
    }


@router.post("/{strategy_id}/rollback", response_model=RollbackResponse)
async def rollback_strategy(
    strategy_id: str,
    request: RollbackRequest,
) -> RollbackResponse:
    """
    Rollback a strategy to a previous version.

    POST /api/strategies/{strategy_id}/rollback

    Performs:
    1. Validates target version exists
    2. Restores artifacts from target version
    3. Re-compiles .mq5 -> .ex5
    4. Runs SIT validation
    5. Records audit entry

    Returns rollback status with validation results.
    """
    manager = get_version_manager()

    result = manager.rollback(
        strategy_id=strategy_id,
        target_version_tag=request.target_version,
        author=request.author,
        reason=request.reason,
    )

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Rollback failed")
        )

    return RollbackResponse(
        success=result["success"],
        strategy_id=result["strategy_id"],
        from_version=result["from_version"],
        to_version=result["to_version"],
        compilation_passed=result["compilation_passed"],
        sit_validation_passed=result["sit_validation_passed"],
        audit_id=result.get("audit_id"),
        message=f"Rolled back to version {result['to_version']}",
    )


@router.get("/{strategy_id}/history", response_model=HistoryResponse)
async def get_strategy_history(
    strategy_id: str,
) -> HistoryResponse:
    """
    Get comprehensive version history for a strategy.

    GET /api/strategies/{strategy_id}/history

    Returns: versions, active version, rollback history.
    """
    manager = get_version_manager()

    history = manager.get_version_history(strategy_id)

    return HistoryResponse(
        strategy_id=history["strategy_id"],
        versions=history["versions"],
        active_version=history["active_version"],
        rollback_history=history["rollback_history"],
    )


@router.get("/{strategy_id}/versions/{version_a}/compare/{version_b}")
async def compare_versions(
    strategy_id: str,
    version_a: str,
    version_b: str,
) -> Dict[str, Any]:
    """
    Compare two versions of a strategy.

    GET /api/strategies/{strategy_id}/versions/{version_a}/compare/{version_b}

    Returns comparison of source, template deps, artifacts, etc.
    """
    manager = get_version_manager()

    comparison = manager.compare_versions(strategy_id, version_a, version_b)

    if not comparison:
        raise HTTPException(
            status_code=404,
            detail="One or both versions not found"
        )

    return comparison


@router.get("/{strategy_id}/active", response_model=VersionResponse)
async def get_active_version(
    strategy_id: str,
) -> VersionResponse:
    """Get the active version for a strategy."""
    storage = get_ea_version_storage()
    version = storage.get_active_version(strategy_id)

    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"No active version found for strategy {strategy_id}"
        )

    return VersionResponse(
        id=version.id,
        strategy_id=version.strategy_id,
        version_tag=version.version_tag,
        created_at=version.created_at.isoformat() if version.created_at else "",
        author=version.author,
        source_hash=version.source_hash,
        template_deps=version.template_deps,
        pin_template_version=version.pin_template_version,
        variant_type=version.variant_type.value,
        improvement_cycle=version.improvement_cycle,
        artifacts=version.artifacts.model_dump(),
        is_active=True,
    )


@router.put("/{strategy_id}/versions/{version_tag}/activate")
async def activate_version(
    strategy_id: str,
    version_tag: str,
) -> Dict[str, Any]:
    """Set a version as active for a strategy."""
    storage = get_ea_version_storage()

    success = storage.set_active_version(strategy_id, version_tag)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_tag} not found for strategy {strategy_id}"
        )

    return {
        "success": True,
        "strategy_id": strategy_id,
        "active_version": version_tag,
    }