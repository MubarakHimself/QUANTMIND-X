"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/svss", tags=["svss"])


@router.get("/health")
async def svss_health() -> dict[str, str]:
    return {"status": "ok", "module": "svss", "mode": "compat"}
