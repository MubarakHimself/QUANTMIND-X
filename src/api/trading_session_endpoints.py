"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/trading-session", tags=["trading-session"])


@router.get("/health")
async def trading_session_health() -> dict[str, str]:
    return {"status": "ok", "module": "trading-session", "mode": "compat"}
