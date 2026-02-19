"""
Version API Endpoints for QuantMindX

Provides version information via HTTP endpoint.
Used for deployment verification and system monitoring.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from src.version import get_version_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["version"])


# ============== Models ==============

class VersionResponse(BaseModel):
    """Response model for version endpoint."""
    version: str
    env: str
    build_timestamp: str


# ============== HTTP Endpoints ==============

@router.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get current system version information.
    
    Returns version, environment, and build timestamp.
    Used for deployment verification and health checks.
    """
    info = get_version_info()
    return VersionResponse(
        version=info["version"],
        env=info["env"],
        build_timestamp=info["build_timestamp"]
    )


@router.get("/version/full")
async def get_version_full() -> Dict[str, Any]:
    """
    Get full version information including history.
    
    Returns complete version info with release history.
    """
    return get_version_info()