"""
GitHub EA Sync API Endpoints

Provides REST API endpoints for GitHub EA synchronization.

**Validates: Property 20: GitHub EA Sync API**
"""

import logging
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.database.models import ImportedEA, get_db_session
from src.integrations.github_ea_sync import GitHubEASync
from src.integrations.github_ea_scheduler import get_scheduler
from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/github", tags=["github", "ea-sync"])


# =============================================================================
# Request/Response Models
# =============================================================================

class SyncRequest(BaseModel):
    """Request model for triggering a sync."""
    force: bool = Field(False, description="Force sync even if already synced recently")


class SyncResponse(BaseModel):
    """Response model for sync operations."""
    status: str
    message: str
    repo_url: Optional[str] = None
    commit_hash: Optional[str] = None
    eas_found: int = 0
    eas_new: int = 0
    eas_updated: int = 0
    eas_unchanged: int = 0
    errors: List[str] = []
    synced_at: Optional[str] = None


class SyncStatusResponse(BaseModel):
    """Response model for sync status."""
    is_running: bool
    repo_url: Optional[str]
    branch: Optional[str]
    sync_interval_hours: Optional[int]
    sync_count: int
    error_count: int
    last_sync_time: Optional[str]
    last_commit_hash: Optional[str]
    next_scheduled_run: Optional[str]


class EAListItem(BaseModel):
    """Response model for EA list item."""
    id: int
    ea_filename: str
    github_path: str
    lines_of_code: int
    strategy_type: str
    status: str
    imported_at: Optional[str]
    last_synced: Optional[str]
    version: Optional[str] = None
    checksum: str


class EADetail(EAListItem):
    """Response model for detailed EA information."""
    bot_manifest_id: Optional[int] = None
    metadata: dict = {}


class ImportRequest(BaseModel):
    """Request model for importing EAs."""
    ea_ids: List[int] = Field(..., description="List of EA IDs to import")
    generate_manifest: bool = Field(True, description="Generate BotManifest for imported EAs")


class ImportResponse(BaseModel):
    """Response model for import operation."""
    imported_count: int
    imported_eas: List[dict]
    errors: List[str]


# =============================================================================
# Helper Functions
# =============================================================================

def get_sync_service() -> GitHubEASync:
    """Get the GitHub EA sync service from the scheduler."""
    scheduler = get_scheduler()
    if scheduler:
        return scheduler.sync_service
    raise HTTPException(status_code=503, detail="GitHub EA sync service not initialized")


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("/status", response_model=SyncStatusResponse)
async def get_status():
    """
    Get current sync status.
    
    Returns information about:
    - Whether the scheduler is running
    - Repository URL and branch
    - Sync interval
    - Last sync time and commit hash
    - Next scheduled run time
    """
    try:
        scheduler = get_scheduler()
        
        if not scheduler:
            return SyncStatusResponse(
                is_running=False,
                repo_url=None,
                branch=None,
                sync_interval_hours=None,
                sync_count=0,
                error_count=0,
                last_sync_time=None,
                last_commit_hash=None,
                next_scheduled_run=None
            )
        
        status = scheduler.get_status()
        
        return SyncStatusResponse(
            is_running=status['is_running'],
            repo_url=status.get('repo_url'),
            branch=status.get('branch'),
            sync_interval_hours=status.get('sync_interval_hours'),
            sync_count=status['sync_count'],
            error_count=status['error_count'],
            last_sync_time=status.get('last_sync_time'),
            last_commit_hash=status.get('last_commit_hash'),
            next_scheduled_run=status.get('next_scheduled_run')
        )
        
    except Exception as e:
        logger.error(f"Failed to get sync status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync", response_model=SyncResponse)
async def trigger_sync(
    request: SyncRequest = SyncRequest(),
    background_tasks: BackgroundTasks = None
):
    """
    Trigger a manual sync of the GitHub repository.
    
    This endpoint:
    - Pulls latest changes from the repository
    - Scans for new and updated .mq5 files
    - Parses EA metadata
    - Updates the database with changes
    """
    try:
        scheduler = get_scheduler()
        
        if not scheduler:
            raise HTTPException(
                status_code=503,
                detail="GitHub EA sync service not initialized. Check GITHUB_EA_REPO_URL environment variable."
            )
        
        # Trigger manual sync
        result = await scheduler.manual_sync()
        
        return SyncResponse(
            status=result.get('sync_status', 'unknown'),
            message="Sync completed" if result.get('sync_status') == 'success' else "Sync failed",
            repo_url=result.get('repository', {}).get('repo_url'),
            commit_hash=result.get('repository', {}).get('commit_hash'),
            eas_found=result.get('eas_found', 0),
            eas_new=result.get('eas_new', 0),
            eas_updated=result.get('eas_updated', 0),
            eas_unchanged=result.get('eas_unchanged', 0),
            errors=result.get('errors', []),
            synced_at=result.get('synced_at')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eas", response_model=PaginatedResponse[EAListItem])
async def list_eas(
    status: Optional[str] = None,
    strategy_type: Optional[str] = None,
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum number of results"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db_session)
):
    """
    List all available EAs from the GitHub repository with pagination.

    Query parameters:
    - status: Filter by status (new, updated, unchanged)
    - strategy_type: Filter by strategy type
    - limit: Maximum number of results (default: 50, max: 100)
    - offset: Pagination offset (default: 0)
    """
    try:
        query = db.query(ImportedEA)

        if status:
            query = query.filter(ImportedEA.status == status)

        if strategy_type:
            query = query.filter(ImportedEA.strategy_type.ilike(f"%{strategy_type}%"))

        # Get total count before applying limit/offset
        total = query.count()

        eas = query.order_by(ImportedEA.imported_at.desc()).offset(offset).limit(limit).all()

        items = [
            EAListItem(
                id=ea.id,
                ea_filename=ea.ea_filename,
                github_path=ea.github_path,
                lines_of_code=ea.lines_of_code,
                strategy_type=ea.strategy_type,
                status=ea.status,
                imported_at=ea.imported_at.isoformat() if ea.imported_at else None,
                last_synced=ea.last_synced.isoformat() if ea.last_synced else None,
                version=getattr(ea, 'version', '1.00'),
                checksum=ea.checksum
            )
            for ea in eas
        ]

        return PaginatedResponse.create(
            items=items,
            total=total,
            limit=limit,
            offset=offset
        )

    except Exception as e:
        logger.error(f"Failed to list EAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/eas/{ea_id}", response_model=EADetail)
async def get_ea(ea_id: int, db: Session = Depends(get_db_session)):
    """
    Get detailed information about a specific EA.
    """
    try:
        ea = db.query(ImportedEA).filter(ImportedEA.id == ea_id).first()
        
        if not ea:
            raise HTTPException(status_code=404, detail="EA not found")
        
        # Build metadata from EA data
        metadata = {
            'checksum': ea.checksum,
            'lines_of_code': ea.lines_of_code,
        }
        
        return EADetail(
            id=ea.id,
            ea_filename=ea.ea_filename,
            github_path=ea.github_path,
            lines_of_code=ea.lines_of_code,
            strategy_type=ea.strategy_type,
            status=ea.status,
            imported_at=ea.imported_at.isoformat() if ea.imported_at else None,
            last_synced=ea.last_synced.isoformat() if ea.last_synced else None,
            version=getattr(ea, 'version', '1.00'),
            checksum=ea.checksum,
            bot_manifest_id=getattr(ea, 'bot_manifest_id', None),
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/import", response_model=ImportResponse)
async def import_eas(
    request: ImportRequest,
    db: Session = Depends(get_db_session)
):
    """
    Import selected EAs and generate BotManifests.
    
    This endpoint:
    - Selects EAs by ID
    - Generates BotManifest for each EA
    - Links the BotManifest to the imported EA
    """
    try:
        imported_eas = []
        errors = []
        
        for ea_id in request.ea_ids:
            try:
                ea = db.query(ImportedEA).filter(ImportedEA.id == ea_id).first()
                
                if not ea:
                    errors.append(f"EA with ID {ea_id} not found")
                    continue
                
                if request.generate_manifest:
                    # Generate BotManifest from EA data
                    from src.database.models import BotManifest
                    
                    manifest = BotManifest(
                        name=ea.ea_filename.replace('.mq5', ''),
                        description=f"Imported from GitHub: {ea.github_path}",
                        status='draft',
                        source='github_import',
                        config={
                            'strategy_type': ea.strategy_type,
                            'github_path': ea.github_path,
                            'checksum': ea.checksum
                        }
                    )
                    
                    db.add(manifest)
                    db.flush()  # Get manifest ID
                    
                    # Link manifest to EA
                    ea.bot_manifest_id = manifest.id
                    ea.status = 'imported'
                
                imported_eas.append({
                    'id': ea.id,
                    'filename': ea.ea_filename,
                    'strategy_type': ea.strategy_type,
                    'status': ea.status
                })
                
            except Exception as e:
                errors.append(f"Failed to import EA {ea_id}: {str(e)}")
        
        db.commit()
        
        return ImportResponse(
            imported_count=len(imported_eas),
            imported_eas=imported_eas,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Import failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/imported", response_model=List[EAListItem])
async def list_imported_eas(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db_session)
):
    """
    List EAs that have been imported (have a BotManifest).
    """
    try:
        eas = db.query(ImportedEA).filter(
            ImportedEA.bot_manifest_id.isnot(None)
        ).order_by(
            ImportedEA.imported_at.desc()
        ).offset(offset).limit(limit).all()
        
        return [
            EAListItem(
                id=ea.id,
                ea_filename=ea.ea_filename,
                github_path=ea.github_path,
                lines_of_code=ea.lines_of_code,
                strategy_type=ea.strategy_type,
                status=ea.status,
                imported_at=ea.imported_at.isoformat() if ea.imported_at else None,
                last_synced=ea.last_synced.isoformat() if ea.last_synced else None,
                version=getattr(ea, 'version', '1.00'),
                checksum=ea.checksum
            )
            for ea in eas
        ]
        
    except Exception as e:
        logger.error(f"Failed to list imported EAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db_session)):
    """
    Get statistics about imported EAs.
    """
    try:
        total = db.query(ImportedEA).count()
        new = db.query(ImportedEA).filter(ImportedEA.status == 'new').count()
        updated = db.query(ImportedEA).filter(ImportedEA.status == 'updated').count()
        imported = db.query(ImportedEA).filter(ImportedEA.status == 'imported').count()
        
        # Get strategy type distribution
        from sqlalchemy import func
        strategy_counts = db.query(
            ImportedEA.strategy_type,
            func.count(ImportedEA.id)
        ).group_by(ImportedEA.strategy_type).all()
        
        return {
            'total_eas': total,
            'new_count': new,
            'updated_count': updated,
            'imported_count': imported,
            'strategy_distribution': [
                {'strategy_type': s, 'count': c}
                for s, c in strategy_counts
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Router Registration Helper
# =============================================================================

def register_github_routes(app):
    """
    Register GitHub EA sync routes with the FastAPI app.
    
    Usage:
        from src.api.github_endpoints import register_github_routes
        register_github_routes(app)
    """
    app.include_router(router)
    logger.info("GitHub EA sync routes registered")