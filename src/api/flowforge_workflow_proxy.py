"""
Compatibility router for FlowForge workflow proxy imports.

The current workflow API authority lives under `prefect_workflow_endpoints`.
This module restores the server import surface expected by older API startup
code until the FlowForge-specific contract is implemented properly.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/flowforge", tags=["flowforge"])


@router.get("/health")
async def flowforge_proxy_health() -> dict[str, str]:
    """Minimal health endpoint for compatibility and startup validation."""
    return {"status": "ok", "provider": "prefect-compat"}
