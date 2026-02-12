"""
NPRD REST API Server.

This module provides a FastAPI-based REST API for the NPRD video processing system.

Endpoints:
- POST /jobs - Submit a new video processing job
- GET /jobs - List all jobs
- GET /jobs/{job_id} - Get job status
- GET /jobs/{job_id}/result - Get job result
- DELETE /jobs/{job_id} - Cancel a job
- POST /jobs/batch - Submit multiple jobs
- POST /jobs/playlist - Submit playlist for processing
- GET /config - Get current configuration
- GET /health - Health check
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from .models import NPRDConfig, JobOptions, JobState, TimelineOutput, JobStatus
from .processor import NPRDProcessor
from .job_queue import JobQueueManager
from .exceptions import (
    NPRDError,
    ValidationError,
    JobError,
    ConfigurationError,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for API
# ============================================================================

class JobSubmitRequest(BaseModel):
    """Request model for submitting a job."""
    url: str = Field(..., description="Video URL to process")
    provider: Optional[str] = Field(None, description="Model provider (gemini, qwen)")
    frame_interval: int = Field(30, ge=1, le=300, description="Frame extraction interval in seconds")
    audio_bitrate: str = Field("128k", description="Audio bitrate")
    audio_channels: int = Field(1, ge=1, le=2, description="Audio channels (1=mono, 2=stereo)")
    cache_enabled: bool = Field(True, description="Enable caching")
    output_dir: Optional[str] = Field(None, description="Custom output directory")
    
    @validator("url")
    def validate_url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class BatchJobSubmitRequest(BaseModel):
    """Request model for submitting multiple jobs."""
    urls: List[str] = Field(..., min_items=1, max_items=100, description="List of video URLs")
    provider: Optional[str] = Field(None, description="Model provider (gemini, qwen)")
    frame_interval: int = Field(30, ge=1, le=300, description="Frame extraction interval")
    cache_enabled: bool = Field(True, description="Enable caching")


class PlaylistJobSubmitRequest(BaseModel):
    """Request model for submitting a playlist."""
    url: str = Field(..., description="Playlist URL")
    provider: Optional[str] = Field(None, description="Model provider")
    frame_interval: int = Field(30, ge=1, le=300, description="Frame extraction interval")
    cache_enabled: bool = Field(True, description="Enable caching")
    extract_only: bool = Field(False, description="Only extract URLs, don't process")


class JobResponse(BaseModel):
    """Response model for job operations."""
    job_id: str
    status: str
    progress: int
    video_url: str
    created_at: str
    updated_at: str
    result_path: Optional[str] = None
    error: Optional[str] = None
    logs: List[str] = []


class JobSubmitResponse(BaseModel):
    """Response model for job submission."""
    job_id: str
    message: str
    status: str = "PENDING"


class BatchJobSubmitResponse(BaseModel):
    """Response model for batch job submission."""
    jobs: List[JobSubmitResponse]
    total: int
    submitted: int
    failed: int


class PlaylistInfoResponse(BaseModel):
    """Response model for playlist info."""
    url: str
    title: str
    video_count: int
    video_urls: List[str]


class PlaylistJobSubmitResponse(BaseModel):
    """Response model for playlist job submission."""
    playlist_info: PlaylistInfoResponse
    jobs: List[JobSubmitResponse]


class TimelineResponse(BaseModel):
    """Response model for timeline output."""
    meta: Dict[str, Any]
    timeline: List[Dict[str, Any]]


class ConfigResponse(BaseModel):
    """Response model for configuration."""
    gemini_api_key_set: bool
    gemini_yolo_mode: bool
    qwen_api_key_set: bool
    qwen_headless: bool
    qwen_requests_per_day: int
    cache_dir: str
    cache_max_size_gb: int
    cache_max_age_days: int
    max_concurrent_jobs: int
    job_db_path: str
    output_dir: str
    default_frame_interval: int
    default_audio_bitrate: str
    default_audio_channels: int
    max_retry_attempts: int
    base_retry_delay: float
    log_level: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: str
    code: str


# ============================================================================
# API Instance and Global State
# ============================================================================

# Global instances (initialized on startup)
_config: Optional[NPRDConfig] = None
_job_queue: Optional[JobQueueManager] = None
_processor: Optional[NPRDProcessor] = None


def get_config() -> NPRDConfig:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = NPRDConfig.from_env()
    return _config


def get_job_queue() -> JobQueueManager:
    """Get job queue manager."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueueManager(get_config())
    return _job_queue


def get_processor() -> NPRDProcessor:
    """Get NPRD processor."""
    global _processor
    if _processor is None:
        _processor = NPRDProcessor(config=get_config())
    return _processor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NPRD API server...")
    
    # Initialize components
    config = get_config()
    job_queue = get_job_queue()
    
    # Set up job processor callback
    processor = get_processor()
    job_queue.set_job_processor(
        lambda job_id, url, options: processor.process(url, job_id=job_id, options=options).timeline
    )
    
    # Start job queue
    job_queue.start()
    
    logger.info("NPRD API server started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NPRD API server...")
    job_queue.stop(wait=True)
    logger.info("NPRD API server stopped")


# Create FastAPI app
app = FastAPI(
    title="NPRD API",
    description="Video Processing Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Exception Handlers
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Validation Error", "detail": str(exc), "code": "VALIDATION_ERROR"}
    )


@app.exception_handler(JobError)
async def job_error_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Job Error", "detail": str(exc), "code": "JOB_ERROR"}
    )


@app.exception_handler(ConfigurationError)
async def config_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Configuration Error", "detail": str(exc), "code": "CONFIG_ERROR"}
    )


@app.exception_handler(NPRDError)
async def nprd_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "NPRD Error", "detail": str(exc), "code": "NPRD_ERROR"}
    )


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "NPRD API",
        "version": "1.0.0",
        "description": "Video Processing Pipeline API",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    try:
        # Check job queue
        job_queue = get_job_queue()
        components["job_queue"] = "healthy"
    except Exception as e:
        components["job_queue"] = f"unhealthy: {e}"
    
    try:
        # Check processor
        processor = get_processor()
        components["processor"] = "healthy"
    except Exception as e:
        components["processor"] = f"unhealthy: {e}"
    
    # Check cache
    try:
        config = get_config()
        if config.cache_dir.exists():
            components["cache"] = "healthy"
        else:
            components["cache"] = "unhealthy: cache directory missing"
    except Exception as e:
        components["cache"] = f"unhealthy: {e}"
    
    # Overall status
    all_healthy = all("unhealthy" not in v for v in components.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        components=components,
    )


@app.post("/jobs", response_model=JobSubmitResponse, status_code=201)
async def submit_job(request: JobSubmitRequest):
    """Submit a new video processing job."""
    try:
        options = JobOptions(
            output_dir=Path(request.output_dir) if request.output_dir else None,
            model_provider=request.provider,
            frame_interval=request.frame_interval,
            audio_bitrate=request.audio_bitrate,
            audio_channels=request.audio_channels,
            cache_enabled=request.cache_enabled,
        )
        
        job_queue = get_job_queue()
        job_id = job_queue.submit_job(request.url, options)
        
        return JobSubmitResponse(
            job_id=job_id,
            message=f"Job submitted for URL: {request.url}",
            status="PENDING",
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of jobs"),
):
    """List all jobs."""
    try:
        job_queue = get_job_queue()
        
        status_filter = None
        if status:
            try:
                status_filter = JobState(status.upper())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        jobs = job_queue.list_jobs(status=status_filter, limit=limit)
        
        return [
            JobResponse(
                job_id=job.job_id,
                status=job.status.value,
                progress=job.progress,
                video_url=job.video_url,
                created_at=job.created_at.isoformat(),
                updated_at=job.updated_at.isoformat(),
                result_path=job.result_path,
                error=job.error,
                logs=job.logs,
            )
            for job in jobs
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """Get job status."""
    try:
        job_queue = get_job_queue()
        job = job_queue.get_job_status(job_id)
        
        return JobResponse(
            job_id=job.job_id,
            status=job.status.value,
            progress=job.progress,
            video_url=job.video_url,
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            result_path=job.result_path,
            error=job.error,
            logs=job.logs,
        )
        
    except JobError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/result", response_model=TimelineResponse)
async def get_job_result(job_id: str):
    """Get job result (timeline output)."""
    try:
        job_queue = get_job_queue()
        
        # Check job status first
        job = job_queue.get_job_status(job_id)
        
        if job.status != JobState.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Job not completed. Current status: {job.status.value}"
            )
        
        # Get result
        timeline = job_queue.get_job_result(job_id)
        
        if timeline is None:
            raise HTTPException(status_code=404, detail="Result not found")
        
        return TimelineResponse(
            meta={
                "video_url": timeline.video_url,
                "title": timeline.title,
                "duration_seconds": timeline.duration_seconds,
                "processed_at": timeline.processed_at,
                "model_provider": timeline.model_provider,
                "version": timeline.version,
            },
            timeline=[clip.to_dict() for clip in timeline.timeline],
        )
        
    except HTTPException:
        raise
    except JobError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get job result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{job_id}", response_model=Dict[str, str])
async def cancel_job(job_id: str):
    """Cancel a job (if not yet completed)."""
    try:
        job_queue = get_job_queue()
        
        # Check job status
        job = job_queue.get_job_status(job_id)
        
        if job.status in [JobState.COMPLETED, JobState.FAILED]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot cancel job in {job.status.value} state"
            )
        
        # Update to failed state
        job_queue.update_job_status(
            job_id,
            JobState.FAILED,
            error="Job cancelled by user",
            log_message="Job cancelled via API"
        )
        
        return {"message": f"Job {job_id} cancelled"}
        
    except HTTPException:
        raise
    except JobError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/batch", response_model=BatchJobSubmitResponse, status_code=201)
async def submit_batch_jobs(request: BatchJobSubmitRequest):
    """Submit multiple jobs in batch."""
    try:
        job_queue = get_job_queue()
        
        jobs = []
        submitted = 0
        failed = 0
        
        for url in request.urls:
            try:
                options = JobOptions(
                    model_provider=request.provider,
                    frame_interval=request.frame_interval,
                    cache_enabled=request.cache_enabled,
                )
                
                job_id = job_queue.submit_job(url, options)
                jobs.append(JobSubmitResponse(
                    job_id=job_id,
                    message=f"Job submitted for URL: {url}",
                    status="PENDING",
                ))
                submitted += 1
                
            except Exception as e:
                logger.warning(f"Failed to submit job for {url}: {e}")
                jobs.append(JobSubmitResponse(
                    job_id="",
                    message=f"Failed: {e}",
                    status="FAILED",
                ))
                failed += 1
        
        return BatchJobSubmitResponse(
            jobs=jobs,
            total=len(request.urls),
            submitted=submitted,
            failed=failed,
        )
        
    except Exception as e:
        logger.error(f"Failed to submit batch jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/playlist", response_model=PlaylistJobSubmitResponse, status_code=201)
async def submit_playlist_jobs(request: PlaylistJobSubmitRequest):
    """Submit all videos from a playlist for processing."""
    try:
        processor = get_processor()
        job_queue = get_job_queue()
        
        # Extract playlist info
        playlist_info = processor.extract_playlist(request.url)
        
        if request.extract_only:
            return PlaylistJobSubmitResponse(
                playlist_info=PlaylistInfoResponse(
                    url=playlist_info.url,
                    title=playlist_info.title,
                    video_count=playlist_info.video_count,
                    video_urls=playlist_info.video_urls,
                ),
                jobs=[],
            )
        
        # Create jobs for each video
        options = JobOptions(
            model_provider=request.provider,
            frame_interval=request.frame_interval,
            cache_enabled=request.cache_enabled,
        )
        
        job_ids = processor.create_playlist_jobs(request.url, job_queue, options)
        
        jobs = []
        for i, job_id in enumerate(job_ids):
            if job_id:
                jobs.append(JobSubmitResponse(
                    job_id=job_id,
                    message=f"Job submitted for video {i+1}",
                    status="PENDING",
                ))
            else:
                jobs.append(JobSubmitResponse(
                    job_id="",
                    message=f"Failed to submit job for video {i+1}",
                    status="FAILED",
                ))
        
        return PlaylistJobSubmitResponse(
            playlist_info=PlaylistInfoResponse(
                url=playlist_info.url,
                title=playlist_info.title,
                video_count=playlist_info.video_count,
                video_urls=playlist_info.video_urls,
            ),
            jobs=jobs,
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to submit playlist jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config", response_model=ConfigResponse)
async def get_configuration():
    """Get current NPRD configuration."""
    try:
        config = get_config()
        
        return ConfigResponse(
            gemini_api_key_set=config.gemini_api_key is not None,
            gemini_yolo_mode=config.gemini_yolo_mode,
            qwen_api_key_set=config.qwen_api_key is not None,
            qwen_headless=config.qwen_headless,
            qwen_requests_per_day=config.qwen_requests_per_day,
            cache_dir=str(config.cache_dir),
            cache_max_size_gb=config.cache_max_size_gb,
            cache_max_age_days=config.cache_max_age_days,
            max_concurrent_jobs=config.max_concurrent_jobs,
            job_db_path=str(config.job_db_path),
            output_dir=str(config.output_dir),
            default_frame_interval=config.default_frame_interval,
            default_audio_bitrate=config.default_audio_bitrate,
            default_audio_channels=config.default_audio_channels,
            max_retry_attempts=config.max_retry_attempts,
            base_retry_delay=config.base_retry_delay,
            log_level=config.log_level,
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get job queue statistics."""
    try:
        job_queue = get_job_queue()
        
        # Count jobs by status
        all_jobs = job_queue.list_jobs(limit=10000)
        
        stats = {
            "total_jobs": len(all_jobs),
            "by_status": {},
            "active_jobs": job_queue.get_active_job_count(),
        }
        
        for status in JobState:
            count = sum(1 for j in all_jobs if j.status == status)
            stats["by_status"][status.value] = count
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Entry Point
# ============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 3000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the API server using uvicorn."""
    import uvicorn
    
    uvicorn.run(
        "src.nprd.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run_server()
