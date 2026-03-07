"""
QuantMind IDE Video Ingest Endpoints

API endpoints for video (YouTube) processing.
"""

from fastapi import APIRouter, HTTPException

from src.api.ide_handlers import VideoIngestAPIHandler
from src.api.ide_models import VideoIngestProcessRequest, VideoIngestProcessResponse, VideoIngestAuthStatus

router = APIRouter(prefix="/api/video-ingest", tags=["video-ingest"])

# Initialize handler
video_ingest_handler = VideoIngestAPIHandler()


@router.post("/process", response_model=VideoIngestProcessResponse)
async def process_video_ingest(request: VideoIngestProcessRequest):
    """
    Process a YouTube URL for strategy generation.

    Args:
        url: YouTube URL to process
        strategy_name: Name for the strategy folder
        is_playlist: Whether URL is a playlist
    """
    result = video_ingest_handler.process_video(
        url=request.url,
        strategy_name=request.strategy_name,
        is_playlist=request.is_playlist
    )
    return VideoIngestProcessResponse(**result)


@router.get("/jobs/{job_id}")
async def get_video_ingest_job(job_id: str):
    """Get video ingest job status."""
    result = video_ingest_handler.get_job_status(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found")
    return result


@router.get("/auth-status", response_model=VideoIngestAuthStatus)
async def get_video_ingest_auth_status():
    """Get authentication status for video ingest providers."""
    status = video_ingest_handler.get_auth_status()
    return VideoIngestAuthStatus(**status)
