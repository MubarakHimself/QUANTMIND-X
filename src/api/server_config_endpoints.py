"""
Server Configuration API Endpoints

Manages server connection configurations (Cloudzy, Contabo, MT5, etc.)
for QUANTMINDX infrastructure.

Features:
- CRUD operations for server configs
- Encrypted storage of sensitive credentials
- Server connectivity testing
- Primary server designation

Story 2.2: Server Connection Configuration
"""

import logging
import uuid
import time
from contextlib import contextmanager
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import httpx
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("paramiko not available - SSH connectivity testing disabled")

from src.database.models import ServerConfig, get_db_session, db_session_scope

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/servers", tags=["servers"])


@contextmanager
def _db_session_scope():
    """Support both the FastAPI yield dependency and the existing patched tests."""
    dependency = get_db_session()

    if hasattr(dependency, "__enter__") and hasattr(dependency, "__exit__"):
        with dependency as session:
            yield session
        return

    session = next(dependency)
    try:
        yield session
    finally:
        dependency.close()


# =============================================================================
# Request/Response Models
# =============================================================================

class ServerConfigRequest(BaseModel):
    """Request model for creating/updating a server configuration."""
    name: str = Field(..., description="Display name for the server")
    server_type: str = Field(..., description="Server type: cloudzy, contabo, mt5")
    host: str = Field(..., description="Hostname or IP address")
    port: int = Field(22, description="Port number")
    username: Optional[str] = Field(None, description="Username for authentication (will be encrypted)")
    password: Optional[str] = Field(None, description="Password for authentication (will be encrypted)")
    ssh_key_path: Optional[str] = Field(None, description="Path to SSH key file")
    api_key: Optional[str] = Field(None, description="API key for the server (will be encrypted)")
    is_active: bool = Field(True, description="Whether the server is active")
    is_primary: bool = Field(False, description="Whether this is the primary server of this type")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extra configuration as JSON")


class ServerConfigResponse(BaseModel):
    """Response model for server configuration (no credentials exposed)."""
    id: str
    name: str
    server_type: str
    host: str
    port: int
    is_active: bool
    is_primary: bool
    metadata: Optional[Dict[str, Any]] = None
    created_at_utc: Optional[str] = None
    updated_at: Optional[str] = None


class ServerTestResult(BaseModel):
    """Result of server connectivity test."""
    success: bool
    latency_ms: Optional[int] = None
    error: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def mask_sensitive(value: Optional[str]) -> Optional[str]:
    """Mask sensitive value for display."""
    if not value:
        return None
    if len(value) <= 4:
        return "****"
    return value[:2] + "****" + value[-2:]


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=dict)
async def list_servers():
    """
    List all server configurations.

    Returns servers without credentials exposed.
    """
    try:
        with _db_session_scope() as db:
            servers = db.query(ServerConfig).all()

            result = []
            for s in servers:
                result.append({
                    "id": s.id,
                    "name": s.name,
                    "server_type": s.server_type,
                    "host": s.host,
                    "port": s.port,
                    "is_active": s.is_active,
                    "is_primary": s.is_primary,
                    "metadata": s.metadata_dict,
                    "created_at_utc": s.created_at_utc.isoformat() if s.created_at_utc else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                })

            return {"servers": result}
    except Exception as e:
        logger.error(f"Error listing servers: {e}")
        return {"servers": []}


@router.post("", response_model=dict)
async def create_server(config: ServerConfigRequest):
    """
    Create a new server configuration.

    If is_primary is true, removes primary flag from other servers of same type.
    """
    try:
        with _db_session_scope() as db:
            # Check if another server of this type is already primary
            if config.is_primary:
                existing_primary = db.query(ServerConfig).filter(
                    ServerConfig.server_type == config.server_type,
                    ServerConfig.is_primary == True
                ).first()
                if existing_primary:
                    existing_primary.is_primary = False

            # Create new server
            new_server = ServerConfig(
                id=str(uuid.uuid4()),
                name=config.name,
                server_type=config.server_type,
                host=config.host,
                port=config.port,
                is_active=config.is_active,
                is_primary=config.is_primary,
            )

            # Set encrypted credentials
            if config.username:
                new_server.username = config.username
            if config.password:
                new_server.password = config.password
            if config.ssh_key_path:
                new_server.ssh_key_path = config.ssh_key_path
            if config.api_key:
                new_server.api_key = config.api_key
            if config.metadata:
                new_server.metadata_dict = config.metadata

            db.add(new_server)
            db.commit()
            db.refresh(new_server)

            return {
                "success": True,
                "message": f"Server '{config.name}' created",
                "server": {
                    "id": new_server.id,
                    "name": new_server.name,
                    "server_type": new_server.server_type,
                    "is_active": new_server.is_active,
                }
            }
    except Exception as e:
        logger.error(f"Error creating server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create server: {str(e)}")


@router.put("/{server_id}", response_model=dict)
async def update_server(server_id: str, config: ServerConfigRequest):
    """
    Update a server configuration.

    If is_primary is true, removes primary flag from other servers of same type.
    If password/username/api_key not provided, existing values are preserved.
    """
    try:
        with _db_session_scope() as db:
            server = db.query(ServerConfig).filter(ServerConfig.id == server_id).first()

            if not server:
                raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found")

            # Handle primary flag
            if config.is_primary and not server.is_primary:
                # Demote other primaries of same type
                other_primaries = db.query(ServerConfig).filter(
                    ServerConfig.server_type == server.server_type,
                    ServerConfig.is_primary == True,
                    ServerConfig.id != server_id
                ).all()
                for other in other_primaries:
                    other.is_primary = False

            # Update fields
            server.name = config.name
            server.host = config.host
            server.port = config.port
            server.is_active = config.is_active
            server.is_primary = config.is_primary

            # Update credentials only if provided
            if config.username is not None:
                server.username = config.username
            if config.password is not None:
                server.password = config.password
            if config.ssh_key_path is not None:
                server.ssh_key_path = config.ssh_key_path
            if config.api_key is not None:
                server.api_key = config.api_key
            if config.metadata is not None:
                server.metadata_dict = config.metadata

            db.commit()

            return {
                "success": True,
                "message": f"Server '{config.name}' updated",
                "server": {
                    "id": server.id,
                    "name": server.name,
                    "server_type": server.server_type,
                    "is_active": server.is_active,
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update server: {str(e)}")


@router.delete("/{server_id}", response_model=dict)
async def delete_server(server_id: str):
    """
    Delete a server configuration.

    Returns 409 if the server is marked as primary.
    """
    try:
        with _db_session_scope() as db:
            server = db.query(ServerConfig).filter(ServerConfig.id == server_id).first()

            if not server:
                raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found")

            if server.is_primary:
                raise HTTPException(
                    status_code=409,
                    detail=f"Server '{server.name}' is marked as primary. Set is_primary=false first."
                )

            server_name = server.name
            db.delete(server)
            db.commit()

            return {
                "success": True,
                "message": f"Server '{server_name}' deleted"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete server: {str(e)}")


@router.post("/{server_id}/test", response_model=dict)
async def test_server(server_id: str):
    """
    Test connectivity to a server.

    Returns success status and latency.
    """
    try:
        with _db_session_scope() as db:
            server = db.query(ServerConfig).filter(ServerConfig.id == server_id).first()

            if not server:
                raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found")

            return await _test_server_connectivity(server)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error testing server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test server: {str(e)}")


async def _test_server_connectivity(server: ServerConfig) -> Dict[str, Any]:
    """Test connectivity to a specific server."""
    start_time = time.time()

    try:
        if server.server_type == "contabo":
            # Test SSH connectivity
            return await _test_ssh_connection(server, start_time)
        elif server.server_type == "cloudzy" or server.server_type == "mt5":
            # Test HTTP/API connectivity
            return await _test_http_connection(server, start_time)
        else:
            # Generic test - try HTTP on the port
            return await _test_http_connection(server, start_time)
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": str(e)[:100]
        }


async def _test_ssh_connection(server: ServerConfig, start_time: float) -> Dict[str, Any]:
    """Test SSH connection to a server."""
    if not PARAMIKO_AVAILABLE:
        return {
            "success": False,
            "latency_ms": 0,
            "error": "SSH testing not available (paramiko not installed)"
        }

    try:
        # Get credentials
        username = server.username or "root"
        password = server.password
        ssh_key_path = server.ssh_key_path

        # Try SSH connection with proper host key validation
        client = paramiko.SSHClient()

        # Load known hosts from ~/.ssh/known_hosts for MITM protection
        known_hosts_path = Path.home() / ".ssh" / "known_hosts"
        if known_hosts_path.exists():
            client.load_system_host_keys(str(known_hosts_path))

        # Use RejectPolicy by default - deny unknown hosts to prevent MITM attacks
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

        # Log warning if connecting to unknown host (host not in known_hosts)
        host_keys = client.get_host_keys()
        if server.host not in host_keys:
            logger.warning(
                f"Connecting to SSH host '{server.host}' which is not in known_hosts. "
                f"Connection will be rejected by RejectPolicy. "
                f"To allow, add the host key to ~/.ssh/known_hosts."
            )

        connect_kwargs = {
            "hostname": server.host,
            "port": server.port,
            "username": username,
            "timeout": 10,
        }

        if ssh_key_path:
            connect_kwargs["key_filename"] = ssh_key_path
        elif password:
            connect_kwargs["password"] = password

        client.connect(**connect_kwargs)
        client.close()

        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": True,
            "latency_ms": latency_ms,
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": f"SSH failed: {str(e)[:80]}"
        }


async def _test_http_connection(server: ServerConfig, start_time: float) -> Dict[str, Any]:
    """Test HTTP connection to a server."""
    try:
        # Build URL
        protocol = "https" if server.port in (443, 8443) else "http"
        url = f"{protocol}://{server.host}:{server.port}"

        # Add API key to headers if available
        headers = {}
        api_key = server.api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, follow_redirects=True)

        latency_ms = int((time.time() - start_time) * 1000)

        if response.status_code < 500:
            return {
                "success": True,
                "latency_ms": latency_ms,
            }
        else:
            return {
                "success": False,
                "latency_ms": latency_ms,
                "error": f"HTTP {response.status_code}"
            }
    except httpx.TimeoutException:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": "Connection timeout"
        }
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "latency_ms": latency_ms,
            "error": str(e)[:80]
        }


@router.get("/{server_id}", response_model=dict)
async def get_server(server_id: str):
    """
    Get a specific server by ID.
    """
    try:
        with _db_session_scope() as db:
            server = db.query(ServerConfig).filter(ServerConfig.id == server_id).first()

            if not server:
                raise HTTPException(status_code=404, detail=f"Server with ID '{server_id}' not found")

            return {
                "id": server.id,
                "name": server.name,
                "server_type": server.server_type,
                "host": server.host,
                "port": server.port,
                "is_active": server.is_active,
                "is_primary": server.is_primary,
                "metadata": server.metadata_dict,
                "created_at_utc": server.created_at_utc.isoformat() if server.created_at_utc else None,
                "updated_at": server.updated_at.isoformat() if server.updated_at else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting server: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get server: {str(e)}")


# =============================================================================
# Morning Digest & Node Health Endpoints (Story 3-7)
# =============================================================================

from datetime import datetime, timezone
import os

# Create a separate router for /api/v1/server endpoints
server_router = APIRouter(prefix="/api/v1/server", tags=["server"])


class MorningDigestResponse(BaseModel):
    """Response model for morning digest endpoint."""
    agent_activity: list = []
    pending_approvals: int = 0
    node_health: dict = {}
    critical_alerts: list = []
    market_session: dict = {}


class NodeHealthResponse(BaseModel):
    """Response model for node health endpoint."""
    contabo: dict
    cloudzy: dict


@server_router.get("/morning-digest", response_model=MorningDigestResponse)
async def get_morning_digest():
    """
    Get morning digest with overnight agent activity summary.

    Returns:
        - agent_activity: List of agent actions overnight
        - pending_approvals: Count of pending approvals
        - node_health: Current node status
        - critical_alerts: Any critical alerts
        - market_session: Market session status
    """
    # Get node health
    node_health_data = await _get_node_health_status()

    # Get pending approvals count (placeholder - would integrate with actual system)
    pending_approvals = 0
    try:
        from src.database.models import ApprovalRequest
        with _db_session_scope() as db:
            pending_count = db.query(ApprovalRequest).filter(
                ApprovalRequest.status == 'pending'
            ).count()
            pending_approvals = pending_count
    except Exception:
        pass  # If approval system not available, use 0

    # Get agent activity (placeholder - would integrate with actual system)
    agent_activity = []
    try:
        from src.database.models import AgentActivity
        with _db_session_scope() as db:
            # Get last 24 hours of activity
            yesterday = datetime.now(timezone.utc) - datetime.timedelta(hours=24)
            activities = db.query(AgentActivity).filter(
                AgentActivity.timestamp_utc >= yesterday
            ).order_by(AgentActivity.timestamp_utc.desc()).limit(10).all()

            for a in activities:
                agent_activity.append({
                    "agent_id": a.agent_id,
                    "agent_type": a.agent_type,
                    "action": a.action,
                    "timestamp_utc": a.timestamp_utc.isoformat() if a.timestamp_utc else None
                })
    except Exception:
        pass  # If agent activity not available, return empty list

    # Get critical alerts (placeholder)
    critical_alerts = []

    # Determine market sessions based on UTC time
    current_hour_utc = datetime.now(timezone.utc).hour
    market_session = {
        "tokyo": "open" if 0 <= current_hour_utc < 7 or 23 <= current_hour_utc <= 24 else "closed",
        "london": "open" if 8 <= current_hour_utc < 16 else "closed",
        "new_york": "open" if 13 <= current_hour_utc < 21 else "closed"
    }

    return MorningDigestResponse(
        agent_activity=agent_activity,
        pending_approvals=pending_approvals,
        node_health=node_health_data,
        critical_alerts=critical_alerts,
        market_session=market_session
    )


@server_router.get("/health/nodes", response_model=NodeHealthResponse)
async def get_node_health():
    """
    Get health status of Contabo and Cloudzy nodes.

    Contabo: agents/compute node
    Cloudzy: live trading node (per Story 2-6)

    Returns status, latency, and connectivity info for each node.
    """
    node_status = await _get_node_health_status()
    return NodeHealthResponse(**node_status)


async def _get_node_health_status() -> dict:
    """Get node health status for Contabo and Cloudzy."""
    contabo_status = "connected"
    contabo_latency = 0
    cloudzy_status = "connected"
    cloudzy_latency = 0

    # Check Contabo connectivity
    try:
        with _db_session_scope() as db:
            contabo = db.query(ServerConfig).filter(
                ServerConfig.server_type == "contabo",
                ServerConfig.is_active == True
            ).first()

            if contabo:
                # Try to test connectivity
                start = time.time()
                try:
                    if contabo.host:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            # Try HTTP connection to check liveness
                            protocol = "https" if contabo.port in (443, 8443) else "http"
                            url = f"{protocol}://{contabo.host}:{contabo.port}"
                            await client.get(url, follow_redirects=True)
                        contabo_status = "connected"
                        contabo_latency = int((time.time() - start) * 1000)
                    else:
                        contabo_status = "connected"  # Assume connected if host not set
                except Exception as e:
                    logger.warning(f"Contabo health check failed: {e}")
                    # If Contabo is down, it's not a critical failure - mark as disconnected
                    contabo_status = "disconnected"
            else:
                # No Contabo config found - assume connected for demo purposes
                contabo_status = "connected"
    except Exception as e:
        logger.warning(f"Error checking Contabo status: {e}")
        contabo_status = "disconnected"

    # Check Cloudzy connectivity
    try:
        with _db_session_scope() as db:
            cloudzy = db.query(ServerConfig).filter(
                ServerConfig.server_type == "cloudzy",
                ServerConfig.is_active == True
            ).first()

            if cloudzy:
                start = time.time()
                try:
                    if cloudzy.host:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            protocol = "https" if cloudzy.port in (443, 8443) else "http"
                            url = f"{protocol}://{cloudzy.host}:{cloudzy.port}"
                            await client.get(url, follow_redirects=True)
                        cloudzy_status = "connected"
                        cloudzy_latency = int((time.time() - start) * 1000)
                    else:
                        cloudzy_status = "connected"
                except Exception:
                    cloudzy_status = "disconnected"
            else:
                # No Cloudzy config - assume demo mode, mark connected
                cloudzy_status = "connected"
    except Exception as e:
        logger.warning(f"Error checking Cloudzy status: {e}")
        cloudzy_status = "disconnected"

    return {
        "contabo": {
            "status": contabo_status,
            "latency_ms": contabo_latency
        },
        "cloudzy": {
            "status": cloudzy_status,
            "latency_ms": cloudzy_latency
        }
    }
