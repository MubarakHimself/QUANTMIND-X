"""
Server Health Metrics API Endpoints

Provides detailed system metrics for Contabo and Cloudzy nodes including:
CPU, memory, disk usage, network latency, uptime, and heartbeat status.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/server/health", tags=["server-health"])


# Threshold constants
CPU_THRESHOLD = 85.0  # percentage
MEMORY_THRESHOLD = 90.0  # percentage
DISK_THRESHOLD = 90.0  # percentage
LATENCY_THRESHOLD = 500.0  # ms


# Pydantic models
class NodeMetrics(BaseModel):
    cpu: float
    memory: float
    disk: float
    latency_ms: float
    uptime_seconds: int
    last_heartbeat: str
    status: str  # healthy, warning, critical


class ServerHealthResponse(BaseModel):
    contabo: NodeMetrics
    cloudzy: NodeMetrics
    timestamp: str


def get_system_metrics() -> Dict[str, Any]:
    """
    Get current system metrics for this node.
    """
    try:
        import psutil

        # CPU usage
        cpu = round(psutil.cpu_percent(interval=0.5), 2)

        # Memory usage
        memory = round(psutil.virtual_memory().percent, 2)

        # Disk usage
        disk = round(psutil.disk_usage('/').percent, 2)

        # Uptime
        boot_time = psutil.boot_time()
        uptime_seconds = int(datetime.now(timezone.utc).timestamp() - boot_time)

        # Latency to external (simplified - measures to 8.8.8.8)
        latency_ms = 0.0
        try:
            import time
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(2)
            start = time.time()
            s.sendto(b'ping', ('8.8.8.8', 53))
            s.recvfrom(1024)
            latency_ms = round((time.time() - start) * 1000, 2)
            s.close()
        except Exception:
            pass

        # Determine status
        if cpu > CPU_THRESHOLD or memory > MEMORY_THRESHOLD or disk > DISK_THRESHOLD:
            status = "critical"
        elif cpu > CPU_THRESHOLD * 0.8 or memory > MEMORY_THRESHOLD * 0.8 or disk > DISK_THRESHOLD * 0.8:
            status = "warning"
        else:
            status = "healthy"

        return {
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "latency_ms": latency_ms,
            "uptime_seconds": uptime_seconds,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "status": status
        }

    except ImportError:
        logger.warning("psutil not available, returning default metrics")
        return {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
            "latency_ms": 0.0,
            "uptime_seconds": 0,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "status": "unknown"
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        return {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
            "latency_ms": 0.0,
            "uptime_seconds": 0,
            "last_heartbeat": datetime.now(timezone.utc).isoformat(),
            "status": "error"
        }


def get_cloudzy_metrics() -> Dict[str, Any]:
    """
    Get metrics from Cloudzy node.
    In a real deployment, this would query the Cloudzy server via API or SSH.
    Raises RuntimeError if Cloudzy is not configured/available.
    """
    # Check if Cloudzy is reachable via environment or config
    cloudzy_reachable = os.getenv("CLOUDZY_REACHABLE", "false").lower() == "true"

    if not cloudzy_reachable:
        raise RuntimeError("Cloudzy node not configured. Set CLOUDZY_REACHABLE=true to enable.")

    # In production, this would make an HTTP request to the Cloudzy node
    # or use a service registry to get the metrics
    raise NotImplementedError(
        "Cloudzy metrics collection not yet implemented. "
        "Wire to actual Cloudzy API or SSH metrics collection."
    )


@router.get("/metrics", response_model=ServerHealthResponse)
async def get_server_health_metrics():
    """
    Get health metrics for both Contabo and Cloudzy nodes.

    Returns CPU, memory, disk usage, network latency, uptime, and last heartbeat
    for each node.
    """
    # Get Contabo metrics (this server)
    contabo_metrics = get_system_metrics()

    # Get Cloudzy metrics (remote node)
    try:
        cloudzy_metrics = get_cloudzy_metrics()
    except (RuntimeError, NotImplementedError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cloudzy metrics unavailable: {e}"
        )

    return ServerHealthResponse(
        contabo=NodeMetrics(**contabo_metrics),
        cloudzy=NodeMetrics(**cloudzy_metrics),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/metrics/contabo", response_model=NodeMetrics)
async def get_contabo_metrics():
    """Get health metrics for Contabo node (this server)."""
    metrics = get_system_metrics()
    return NodeMetrics(**metrics)


@router.get("/metrics/cloudzy", response_model=NodeMetrics)
async def get_cloudzy_metrics_endpoint():
    """Get health metrics for Cloudzy node (trading server)."""
    try:
        metrics = get_cloudzy_metrics()
        return NodeMetrics(**metrics)
    except (RuntimeError, NotImplementedError) as e:
        raise HTTPException(
            status_code=503,
            detail=f"Cloudzy metrics unavailable: {e}"
        )


class ThresholdConfig(BaseModel):
    cpu: float = CPU_THRESHOLD
    memory: float = MEMORY_THRESHOLD
    disk: float = DISK_THRESHOLD
    latency: float = LATENCY_THRESHOLD


@router.get("/thresholds")
async def get_threshold_config():
    """Get current threshold configuration for alerts."""
    return ThresholdConfig()


@router.post("/thresholds")
async def update_threshold_config(config: ThresholdConfig):
    """
    Update threshold configuration for alerts.
    Note: This updates runtime thresholds only - stored in memory.
    """
    global CPU_THRESHOLD, MEMORY_THRESHOLD, DISK_THRESHOLD, LATENCY_THRESHOLD

    CPU_THRESHOLD = config.cpu
    MEMORY_THRESHOLD = config.memory
    DISK_THRESHOLD = config.disk
    LATENCY_THRESHOLD = config.latency

    return {
        "message": "Thresholds updated",
        "cpu": CPU_THRESHOLD,
        "memory": MEMORY_THRESHOLD,
        "disk": DISK_THRESHOLD,
        "latency": LATENCY_THRESHOLD
    }
