"""
Compatibility router for FlowForge workflow proxy imports.

The current workflow API authority lives under `prefect_workflow_endpoints`.
This module restores the server import surface expected by older API startup
code until the FlowForge-specific contract is implemented properly.
"""

from fastapi import APIRouter


router = APIRouter(prefix="/api/flowforge", tags=["flowforge"])


class _CompatPrefectClient:
    """
    Minimal compatibility surface for legacy FlowForge callers.

    The production authority for workflow cards is the coordinator/workflow DB
    merge in `prefect_workflow_endpoints`. Older code still expects a
    `get_prefect_client().list_deployments()` path, so provide a no-op client
    instead of raising import errors on every poll.
    """

    async def list_deployments(self) -> list[dict]:
        return []


def get_prefect_client() -> _CompatPrefectClient:
    """Return a compatibility Prefect client with an empty deployment list."""
    return _CompatPrefectClient()


@router.get("/health")
async def flowforge_proxy_health() -> dict[str, str]:
    """Minimal health endpoint for compatibility and startup validation."""
    return {"status": "ok", "provider": "prefect-compat"}
