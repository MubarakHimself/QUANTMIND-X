"""
Compatibility router for This module restores API startup imports for a router that is referenced by

authoritative server wiring but not yet implemented in this worktree.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/workflow-templates", tags=["workflow-templates"])


@router.get("/health")
async def workflow_templates_health() -> dict[str, str]:
    return {"status": "ok", "module": "workflow-templates", "mode": "compat"}
