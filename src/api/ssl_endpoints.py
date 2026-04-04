"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/ssl", tags=["ssl"])


@router.get("/health")
async def ssl_health() -> dict[str, str]:
    return {"status": "ok", "module": "ssl", "mode": "compat"}
