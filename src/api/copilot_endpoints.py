"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/copilot", tags=["copilot"])


@router.get("/health")
async def copilot_health() -> dict[str, str]:
    return {"status": "ok", "module": "copilot", "mode": "compat"}
