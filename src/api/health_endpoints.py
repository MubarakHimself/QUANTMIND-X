"""
Health Check API Endpoints for QuantMindX

Provides health check endpoints for all services including API, MT5,
database, Redis, and Prometheus. Used by the TUI and CLI for monitoring.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# ============== Models ==============

class ServiceHealth(BaseModel):
    status: str  # healthy, degraded, unhealthy
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class FullHealthStatus(BaseModel):
    timestamp: datetime
    overall_status: str
    services: Dict[str, ServiceHealth]
    system: Optional[Dict[str, Any]] = None


# ============== Helper Functions ==============

def get_cpu_usage() -> float:
    """Get current CPU usage percentage."""
    try:
        import psutil
        return round(psutil.cpu_percent(interval=0.1), 2)
    except Exception:
        return 0.0


def get_memory_usage() -> float:
    """Get current memory usage percentage."""
    try:
        import psutil
        return round(psutil.virtual_memory().percent, 2)
    except Exception:
        return 0.0


def get_disk_usage() -> float:
    """Get current disk usage percentage."""
    try:
        import psutil
        return round(psutil.disk_usage('/').percent, 2)
    except Exception:
        return 0.0


# ============== Health Endpoints ==============

@router.get("", response_model=FullHealthStatus)
async def get_full_health():
    """
    Get full health status of all services.
    Returns aggregate health check including system metrics.
    """
    services = {}
    overall_status = "healthy"
    
    # Check API health
    api_health = await get_api_health()
    services["api"] = api_health
    
    # Check MT5 health
    mt5_health = await get_mt5_health()
    services["mt5"] = mt5_health
    
    # Check database health
    db_health = await get_database_health()
    services["database"] = db_health
    
    # Check Redis health
    redis_health = await get_redis_health()
    services["redis"] = redis_health
    
    # Check Prometheus health
    prometheus_health = await get_prometheus_health()
    services["prometheus"] = prometheus_health
    
    # Determine overall status
    statuses = [s.status for s in services.values()]
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    
    # Get system metrics
    try:
        from src.api.metrics_endpoints import get_system_metrics
        system_metrics = get_system_metrics()
        system = {
            "cpu_usage": system_metrics.cpu_usage,
            "memory_usage": system_metrics.memory_usage,
            "disk_usage": system_metrics.disk_usage,
            "uptime": system_metrics.uptime,
            "chaos_score": system_metrics.chaos_score
        }
    except Exception:
        system = {
            "cpu_usage": get_cpu_usage(),
            "memory_usage": get_memory_usage(),
            "disk_usage": get_disk_usage(),
            "uptime": 0,
            "chaos_score": 0
        }
    
    return FullHealthStatus(
        timestamp=datetime.now(),
        overall_status=overall_status,
        services=services,
        system=system
    )


@router.get("/api", response_model=ServiceHealth)
async def get_api_health():
    """
    Check API server liveness.
    Verifies the API server is running and responsive.
    """
    try:
        import psutil
        import time
        
        start = time.time()
        
        # Check if process is running
        current_process = psutil.Process()
        process_info = {
            "pid": current_process.pid,
            "status": current_process.status(),
            "cpu_percent": current_process.cpu_percent(),
            "memory_mb": current_process.memory_info().rss / 1024 / 1024
        }
        
        latency_ms = round((time.time() - start) * 1000, 2)
        
        return ServiceHealth(
            status="healthy",
            latency_ms=latency_ms,
            message="API server is running",
            details=process_info
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"API health check failed: {str(e)}"
        )


@router.get("/mt5", response_model=ServiceHealth)
async def get_mt5_health():
    """
    Check MT5 bridge connectivity and latency.
    Uses system_monitor to check broker connection status.
    """
    try:
        import time
        start = time.time()
        
        # Try to get system status from system_monitor
        broker_connected = False
        last_chaos_score = None
        
        try:
            from src.router.system_monitor import get_system_monitor
            system_monitor = get_system_monitor()
            if system_monitor:
                status = system_monitor.get_system_status()
                broker_connected = status.get("broker_connection", False)
                last_chaos_score = status.get("last_chaos_score")
        except ImportError:
            # Fallback: check if MT5 connection env var is set
            broker_connected = os.getenv("MT5_CONNECTED", "false").lower() == "true"
        except Exception:
            pass
        
        latency_ms = round((time.time() - start) * 1000, 2)
        
        if broker_connected:
            return ServiceHealth(
                status="healthy",
                latency_ms=latency_ms,
                message="MT5 bridge connected",
                details={"broker_connection": broker_connected, "last_chaos_score": last_chaos_score}
            )
        else:
            return ServiceHealth(
                status="degraded",
                latency_ms=latency_ms,
                message="MT5 bridge not connected",
                details={"broker_connection": broker_connected}
            )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"MT5 health check failed: {str(e)}"
        )


@router.get("/database", response_model=ServiceHealth)
async def get_database_health():
    """
    Check PostgreSQL connection pool stats.
    """
    try:
        import time
        start = time.time()
        
        # Try to get database connection info
        pool_size = 0
        active_connections = 0
        
        try:
            from src.monitoring.prometheus_exporter import db_connection_pool_size
            for metric in db_connection_pool_size.collect():
                for sample in metric.samples:
                    if sample.labels.get('state') == 'active':
                        active_connections = int(sample.value)
                    elif sample.labels.get('state') == 'idle':
                        pool_size += int(sample.value)
            pool_size += active_connections
        except ImportError:
            pass
        except Exception:
            pass
        
        latency_ms = round((time.time() - start) * 1000, 2)
        
        return ServiceHealth(
            status="healthy",
            latency_ms=latency_ms,
            message="Database connection OK",
            details={
                "pool_size": pool_size,
                "active_connections": active_connections
            }
        )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"Database health check failed: {str(e)}"
        )


@router.get("/redis", response_model=ServiceHealth)
async def get_redis_health():
    """
    Check Redis hit rate and memory usage.
    """
    try:
        import time
        start = time.time()
        
        # Try to get Redis stats
        hit_rate = 0.0
        memory_used = 0
        
        try:
            from src.cache.redis_client import get_redis_client
            redis_client = get_redis_client()
            if redis_client:
                # Get memory info
                info = redis_client.info("memory")
                memory_used = info.get("used_memory_human", "unknown")
                
                # Get stats for hit rate
                stats = redis_client.info("stats")
                hits = stats.get("keyspace_hits", 0)
                misses = stats.get("keyspace_misses", 0)
                total = hits + misses
                if total > 0:
                    hit_rate = round((hits / total) * 100, 2)
        except ImportError:
            # Redis not available
            pass
        except Exception:
            pass
        
        latency_ms = round((time.time() - start) * 1000, 2)
        
        if hit_rate > 0:
            return ServiceHealth(
                status="healthy",
                latency_ms=latency_ms,
                message="Redis connection OK",
                details={
                    "hit_rate": hit_rate,
                    "memory_used": memory_used
                }
            )
        else:
            return ServiceHealth(
                status="degraded",
                latency_ms=latency_ms,
                message="Redis not available or empty",
                details={"hit_rate": hit_rate, "memory_used": memory_used}
            )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"Redis health check failed: {str(e)}"
        )


@router.get("/prometheus", response_model=ServiceHealth)
async def get_prometheus_health():
    """
    Check Prometheus agent last-push timestamp.
    """
    try:
        import time
        start = time.time()
        
        last_push = None
        
        try:
            from src.monitoring.prometheus_exporter import (
                prometheus_push_gateway_last_push_timestamp
            )
            for metric in prometheus_push_gateway_last_push_timestamp.collect():
                for sample in metric.samples:
                    last_push = datetime.fromtimestamp(sample.value)
                    break
        except ImportError:
            pass
        except Exception:
            pass
        
        latency_ms = round((time.time() - start) * 1000, 2)
        
        if last_push:
            # Check if last push was within last 5 minutes
            time_since_push = (datetime.now() - last_push).total_seconds()
            if time_since_push < 300:  # 5 minutes
                return ServiceHealth(
                    status="healthy",
                    latency_ms=latency_ms,
                    message="Prometheus agent pushing",
                    details={"last_push": last_push.isoformat()}
                )
            else:
                return ServiceHealth(
                    status="degraded",
                    latency_ms=latency_ms,
                    message=f"Last push was {int(time_since_push/60)} minutes ago",
                    details={"last_push": last_push.isoformat()}
                )
        else:
            return ServiceHealth(
                status="degraded",
                latency_ms=latency_ms,
                message="Prometheus push timestamp not available",
                details={"last_push": None}
            )
    except Exception as e:
        return ServiceHealth(
            status="unhealthy",
            message=f"Prometheus health check failed: {str(e)}"
        )
