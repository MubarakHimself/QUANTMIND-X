"""
QuantMind IDE Video Ingest Endpoints

API endpoints for video (YouTube) processing.
"""

import asyncio
import re
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.api.ide_handlers import VideoIngestAPIHandler
from src.api.ide_models import VideoIngestAuthStatus

router = APIRouter(prefix="/video-ingest", tags=["video-ingest"])

# Initialize handler
video_ingest_handler = VideoIngestAPIHandler()

# YouTube URL patterns
YOUTUBE_VIDEO_PATTERNS = [
    r"^https?://(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
    r"^https?://(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
    r"^https?://(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)",
]
YOUTUBE_PLAYLIST_PATTERN = r"^https?://(?:www\.)?youtube\.com/playlist\?list=([a-zA-Z0-9_-]+)"
YOUTUBE_PATTERNS = YOUTUBE_VIDEO_PATTERNS + [YOUTUBE_PLAYLIST_PATTERN]


def validate_youtube_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate YouTube URL and extract video ID.

    Returns:
        Tuple of (is_valid, video_id_or_error_message)
    """
    if not url or not isinstance(url, str):
        return False, "URL is required"

    # Check if it looks like a URL at all
    if not url.startswith(("http://", "https://")):
        return False, "Invalid URL format"

    # Check if it's a YouTube URL
    is_youtube = False
    for pattern in YOUTUBE_PATTERNS:
        match = re.match(pattern, url)
        if match:
            is_youtube = True
            # For watch URLs, verify video ID is not empty
            if "watch" in url and not match.group(1):
                return False, "Missing video ID"
            # For shorts URLs, verify video ID is present
            if "shorts" in url and not match.group(1):
                return False, "Missing video ID in shorts URL"
            break

    if not is_youtube:
        return False, "Only YouTube URLs are supported"

    return True, None


class VideoIngestProcessRequest(BaseModel):
    """Request model for video ingest processing."""
    url: str = Field(..., description="YouTube URL to process")
    strategy_name: str = Field(default="video_ingest", description="Name for the strategy folder")
    is_playlist: bool = Field(default=False, description="Whether URL is a playlist")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        is_valid, error = validate_youtube_url(v)
        if not is_valid:
            raise ValueError(error)
        return v

    @field_validator("strategy_name")
    @classmethod
    def validate_strategy_name(cls, v: str) -> str:
        if len(v) > 255:
            raise ValueError("Strategy name must be 255 characters or less")
        return v


class VideoIngestProcessResponse(BaseModel):
    """Response model for video ingest processing."""
    job_id: str
    status: str
    strategy_folder: Optional[str] = None
    job_ids: Optional[list[str]] = None


async def process_video_async(url: str, strategy_name: str = "video_ingest", is_playlist: bool = False) -> dict:
    """
    Async wrapper for video processing.

    Args:
        url: YouTube URL to process
        strategy_name: Name for the strategy folder
        is_playlist: Whether URL is a playlist

    Returns:
        Job result dict with job_id, status, strategy_folder
    """
    # Run sync handler method in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: video_ingest_handler.process_video(url, strategy_name, is_playlist)
    )
    return result


def get_job_status(job_id: str) -> dict:
    """
    Get job status from handler.

    Args:
        job_id: The job ID to look up

    Returns:
        Job status dict
    """
    return video_ingest_handler.get_job_status(job_id)


@router.post("/process", response_model=VideoIngestProcessResponse)
async def process_video_ingest(request: VideoIngestProcessRequest):
    """
    Process a YouTube URL for strategy generation.

    Args:
        url: YouTube URL to process
        strategy_name: Name for the strategy folder
        is_playlist: Whether URL is a playlist
    """
    try:
        result = await process_video_async(
            url=request.url,
            strategy_name=request.strategy_name,
            is_playlist=request.is_playlist
        )
        return VideoIngestProcessResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=VideoIngestProcessResponse)
async def start_video_ingest(request: VideoIngestProcessRequest):
    """Compatibility alias for older UI callers."""
    return await process_video_ingest(request)


@router.get("/jobs/{job_id}")
async def get_video_ingest_job(job_id: str):
    """Get video ingest job status."""
    result = get_job_status(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found")
    return result


@router.get("/auth-status", response_model=VideoIngestAuthStatus)
async def get_video_ingest_auth_status():
    """Get authentication status for video ingest providers."""
    status = video_ingest_handler.get_auth_status()
    return VideoIngestAuthStatus(**status)
