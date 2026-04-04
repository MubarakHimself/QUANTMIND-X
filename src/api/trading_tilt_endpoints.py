"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/trading-tilt", tags=["trading-tilt"])


@router.get("/health")
async def trading_tilt_health() -> dict[str, str]:
    return {"status": "ok", "module": "trading-tilt", "mode": "compat"}
