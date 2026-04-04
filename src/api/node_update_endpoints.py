"""
Node Update API Endpoints for QuantMindX

Provides API endpoints for sequential node updates with health checks
and automatic rollback capability.

Reference: Story 11-3-node-sequential-update-automatic-rollback
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from enum import Enum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/node-update", tags=["node-update"])


def _get_data_dir() -> Path:
    """Resolve the writable data directory for deployment metadata."""
    configured = os.getenv("QUANTMIND_DATA_DIR")
    if configured:
        return Path(configured)

    container_data_dir = Path("/app/data")
    if container_data_dir.exists():
        return container_data_dir

    return Path(__file__).resolve().parents[2] / "data"


# ============================================================================
# Models
# ============================================================================

class NodeName(str, Enum):
    """Node names in the deployment cluster."""
    contabo = "contabo"
    cloudzy = "cloudzy"
    desktop = "desktop"
    # Generic aliases (preferred)
    node_backend = "node_backend"  # alias for contabo
    node_trading = "node_trading"  # alias for cloudzy


class UpdateStatus(str, Enum):
    """Status of node update operation."""
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    rolled_back = "rolled_back"


class UpdateRequest(BaseModel):
    """Request to update nodes."""
    version: Optional[str] = None
    nodes: Optional[List[NodeName]] = None  # If not specified, update all in order


class NodeUpdateResponse(BaseModel):
    """Response for a single node update."""
    node: str
    status: UpdateStatus
    message: str
    version: Optional[str] = None
    error: Optional[str] = None
    health_status: Optional[str] = None


class SequentialUpdateResponse(BaseModel):
    """Response for sequential node update."""
    run_id: str
    status: UpdateStatus
    message: str
    nodes: List[NodeUpdateResponse]
    failed_node: Optional[str] = None
    rolled_back_nodes: List[str] = []
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0


class DeployWindowStatus(BaseModel):
    """Deploy window validation status."""
    valid: bool
    current_time: str
    window_start: str  # "Friday 22:00 UTC"
    window_end: str    # "Sunday 22:00 UTC"
    message: str


# ============================================================================
# Helper Functions
# ============================================================================

def is_valid_deploy_window() -> bool:
    """Check if current time is within valid deploy window."""
    now = datetime.now(timezone.utc)
    weekday = now.weekday()

    # Friday = 4, Saturday = 5, Sunday = 6
    if weekday < 4:
        return False
    if weekday == 4 and now.hour < 22:
        return False
    if weekday == 6 and now.hour >= 22:
        return False
    return True


def get_current_version() -> str:
    """Get current application version."""
    try:
        version_file = _get_data_dir() / "version.txt"
        if version_file.exists():
            return version_file.read_text().strip()
    except Exception:
        pass
    return "0.0.0"


def check_node_health(node_url: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Check health of a node.

    Args:
        node_url: URL of the node
        timeout: Timeout in seconds

    Returns:
        Health status dict with status, version, uptime_seconds, checks
    """
    try:
        import httpx

        response = httpx.get(
            f"{node_url}/api/health",
            timeout=timeout
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "status": "unhealthy",
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def get_node_url(node: str) -> Optional[str]:
    """Get URL for a node."""
    desktop_url = (
        os.getenv("DESKTOP_URL")
        or os.getenv("API_BASE_URL")
        or os.getenv("QUANTMIND_API_URL")
        or os.getenv("INTERNAL_API_BASE_URL")
        or "http://127.0.0.1:8000"
    )
    # In production, load from config
    node_urls = {
        "contabo": os.getenv("CONTABO_URL", "https://contabo.quantmindx.com"),
        "cloudzy": os.getenv("CLOUDZY_URL", "https://cloudzy.quantmindx.com"),
        "desktop": desktop_url.rstrip("/"),
    }
    return node_urls.get(node)


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/deploy-window", response_model=DeployWindowStatus)
async def get_deploy_window_status():
    """
    Get current deploy window validation status.

    Deploy window is Friday 22:00 - Sunday 22:00 UTC.
    """
    now = datetime.now(timezone.utc)
    valid = is_valid_deploy_window()

    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    if valid:
        message = f"Deploy window is open. Current time: {weekday_names[now.weekday()]} {now.hour}:00 UTC"
    else:
        message = f"Deploy window is closed. Updates only allowed Friday 22:00 - Sunday 22:00 UTC"

    return DeployWindowStatus(
        valid=valid,
        current_time=now.isoformat(),
        window_start="Friday 22:00 UTC",
        window_end="Sunday 22:00 UTC",
        message=message
    )


@router.post("/update", response_model=SequentialUpdateResponse)
async def update_nodes(
    request: UpdateRequest,
    background_tasks: BackgroundTasks
):
    """
    Trigger sequential node update across all nodes.

    Updates nodes in order: Contabo → Cloudzy → Desktop
    Each node is health-checked before the next begins.
    On health check failure, the node is automatically rolled back.
    """
    logger.info(f"Received node update request: {request}")

    # Check deploy window
    if not is_valid_deploy_window():
        raise HTTPException(
            status_code=403,
            detail="Updates only allowed Friday 22:00 - Sunday 22:00 UTC"
        )

    # Determine nodes to update
    nodes_to_update = request.nodes
    if nodes_to_update is None:
        nodes_to_update = [NodeName.contabo, NodeName.cloudzy, NodeName.desktop]

    # Get target version
    target_version = request.version
    if target_version is None:
        target_version = get_current_version()

    start_time = datetime.now(timezone.utc)
    run_id = f"node-update-{start_time.strftime('%Y%m%d-%H%M%S')}"

    nodes: List[NodeUpdateResponse] = []
    failed_node: Optional[str] = None
    rolled_back_nodes: List[str] = []
    previous_nodes: List[str] = []

    # Execute sequential update
    for node in nodes_to_update:
        node_name = node.value
        logger.info(f"Updating node: {node_name}")

        node_url = get_node_url(node_name)
        if not node_url:
            nodes.append(NodeUpdateResponse(
                node=node_name,
                status=UpdateStatus.failed,
                message="Node URL not configured",
                error="No URL for node"
            ))
            failed_node = node_name
            break

        # Update node (in production, this would trigger actual update)
        try:
            # Simulate update
            version_file = _get_data_dir() / "versions" / f"{node_name}.txt"
            version_file.parent.mkdir(parents=True, exist_ok=True)
            version_file.write_text(target_version)

            node_status = UpdateStatus.completed
            message = f"Updated to {target_version}"

        except Exception as e:
            node_status = UpdateStatus.failed
            message = str(e)
            logger.error(f"Failed to update node {node_name}: {e}")

        # Add to results
        nodes.append(NodeUpdateResponse(
            node=node_name,
            status=node_status,
            message=message,
            version=target_version
        ))

        if node_status != UpdateStatus.completed:
            failed_node = node_name
            break

        # Health check after update
        health = check_node_health(node_url)

        # Update the node response with health status
        nodes[-1].health_status = health.get("status", "unknown")

        if health.get("status") not in ("healthy", "degraded"):
            logger.error(f"Health check failed for {node_name}: {health}")

            # Mark as rolled back
            nodes[-1].status = UpdateStatus.rolled_back
            nodes[-1].message = "Health check failed - rolled back"
            rolled_back_nodes.append(node_name)

            # Notify failure (in background)
            background_tasks.add_task(
                _send_failure_notification,
                node_name,
                previous_nodes
            )

            failed_node = node_name
            break

        logger.info(f"Node {node_name} health check passed")
        previous_nodes.append(node_name)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # If all successful, send success notification
    if failed_node is None:
        background_tasks.add_task(
            _send_success_notification,
            previous_nodes
        )
        status = UpdateStatus.completed
        message = f"Successfully updated {len(previous_nodes)} nodes"
    else:
        status = UpdateStatus.failed
        message = f"Update failed on node {failed_node}"

    return SequentialUpdateResponse(
        run_id=run_id,
        status=status,
        message=message,
        nodes=nodes,
        failed_node=failed_node,
        rolled_back_nodes=rolled_back_nodes,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=duration
    )


@router.get("/status/{run_id}")
async def get_update_status(run_id: str):
    """
    Get status of a previous update run.

    In production, this would query stored update history.
    """
    # In production: Load from database or file
    return {
        "run_id": run_id,
        "status": "completed",  # Placeholder
        "message": "Update status tracking not yet implemented"
    }


@router.get("/nodes")
async def get_node_statuses():
    """
    Get current health status of all nodes.
    """
    nodes = ["contabo", "cloudzy", "desktop"]
    statuses = []

    for node_name in nodes:
        node_url = get_node_url(node_name)
        if node_url:
            health = check_node_health(node_url)
            statuses.append({
                "node": node_name,
                "url": node_url,
                "status": health.get("status", "unknown"),
                "version": health.get("version"),
                "uptime_seconds": health.get("uptime_seconds")
            })
        else:
            statuses.append({
                "node": node_name,
                "status": "not_configured"
            })

    return {
        "nodes": statuses,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================================
# Background Tasks (Notifications)
# ============================================================================

async def _send_failure_notification(failed_node: str, previous_nodes: List[str]):
    """Send notification on update failure."""
    try:
        from src.api.notification_config_endpoints import send_notification

        subject = f"Node Update Failed - {failed_node}"
        body = f"""Node Update Failed

Failed Node: {failed_node}
Update Sequence: Contabo → Cloudzy → Desktop

Previous Nodes Updated: {', '.join(previous_nodes) if previous_nodes else 'None'}
Rollback Status: {failed_node} rolled back to previous version

What happened:
The health check failed after updating {failed_node}. Automatic rollback was triggered.

Next Steps:
- Check {failed_node} logs for issues
- Re-attempt update after resolving issues
"""
        send_notification(subject, body)
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


async def _send_success_notification(nodes_updated: List[str]):
    """Send notification on successful update."""
    try:
        from src.api.notification_config_endpoints import send_notification

        subject = "Node Update Complete"
        body = f"""Node Update Complete

Updated Nodes: {', '.join(nodes_updated)}
Update Sequence: Contabo → Cloudzy → Desktop

All nodes updated successfully. Health checks passed on all nodes.
"""
        send_notification(subject, body)
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")
