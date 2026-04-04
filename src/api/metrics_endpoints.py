"""
Metrics API Endpoints for QuantMindX

Provides real-time metrics via HTTP and WebSocket for the monitoring dashboard.
Aggregates data from Prometheus metrics, system monitoring, and trading operations.
"""

import asyncio
import json
import logging
import platform
import psutil
from datetime import datetime
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

# Import AlertManager for centralized alert management
try:
    from src.router.alert_manager import get_alert_manager, AlertManager
except ImportError:
    get_alert_manager = None
    AlertManager = None

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


# ============== Models ==============

class AlertData(BaseModel):
    id: str
    severity: str  # info, warning, critical
    message: str
    timestamp: datetime
    acknowledged: bool = False
    source: str


class SystemMetrics(BaseModel):
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_in: float
    network_out: float
    uptime: float
    chaos_score: float


class TradingMetrics(BaseModel):
    tick_latency_ms: float
    active_bots: int
    active_positions: int
    daily_pnl: float
    total_trades: int
    win_rate: float


class DatabaseMetrics(BaseModel):
    query_latency_ms: float
    connection_pool_size: int
    active_connections: int
    query_count: int


class TickStreamMetrics(BaseModel):
    ticks_per_second: float
    buffer_size: int
    processing_time_ms: float
    symbols_active: int


class MetricsResponse(BaseModel):
    system: SystemMetrics
    trading: TradingMetrics
    database: DatabaseMetrics
    tick_stream: TickStreamMetrics
    alerts: List[AlertData]
    timestamp: datetime


# ============== Global State ==============

# In-memory alert storage (in production, use database)
_active_alerts: Dict[str, AlertData] = {}

# Track start time for uptime
_start_time = datetime.now()


# ============== Helper Functions ==============

def get_system_metrics() -> SystemMetrics:
    """Collect system-level metrics."""
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()

        uptime = (datetime.now() - _start_time).total_seconds()

        # Get chaos score from monitoring if available
        chaos_score = 0.0
        try:
            from src.monitoring.prometheus_exporter import chaos_score as chaos_gauge
            # Read the current value from the gauge
            for metric in chaos_gauge.collect():
                for sample in metric.samples:
                    chaos_score = sample.value
                    break
        except Exception:
            pass

        return SystemMetrics(
            cpu_usage=round(cpu, 2),
            memory_usage=round(memory.percent, 2),
            disk_usage=round(disk.percent, 2),
            network_in=round(network.bytes_recv / 1024 / 1024, 2),  # MB
            network_out=round(network.bytes_sent / 1024 / 1024, 2),  # MB
            uptime=round(uptime, 2),
            chaos_score=round(chaos_score, 4)
        )
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return SystemMetrics(
            cpu_usage=0, memory_usage=0, disk_usage=0,
            network_in=0, network_out=0, uptime=0, chaos_score=0
        )


def get_trading_metrics() -> TradingMetrics:
    """Collect trading-related metrics."""
    try:
        # Default values
        tick_latency = 0.0
        active_bots = 0
        active_positions = 0
        daily_pnl = 0.0
        total_trades = 0
        win_rate = 0.0

        # Try to get from Prometheus metrics
        try:
            from src.monitoring.prometheus_exporter import (
                mt5_latency_seconds, active_eas, trade_profit_loss
            )

            # Get average tick latency
            for metric in mt5_latency_seconds.collect():
                for sample in metric.samples:
                    if 'operation' in sample.labels and sample.labels['operation'] == 'trade':
                        tick_latency = sample.value * 1000  # Convert to ms
                        break

            # Get active EA counts
            for metric in active_eas.collect():
                for sample in metric.samples:
                    if sample.labels.get('status') == 'primal':
                        active_bots += int(sample.value)

            # Get trade stats
            trade_count = 0
            profit_sum = 0.0
            for metric in trade_profit_loss.collect():
                for sample in metric.samples:
                    trade_count += sample.value
                    # This is a histogram, so we need to handle it differently
            total_trades = int(trade_count)

        except ImportError:
            pass

        # Try to get position count from database
        try:
            from src.database.models import get_session, Position
            with get_session() as session:
                active_positions = session.query(Position).filter(
                    Position.is_open == True
                ).count()
        except Exception:
            pass

        return TradingMetrics(
            tick_latency_ms=round(tick_latency, 2),
            active_bots=active_bots,
            active_positions=active_positions,
            daily_pnl=round(daily_pnl, 2),
            total_trades=total_trades,
            win_rate=round(win_rate, 4)
        )
    except Exception as e:
        logger.error(f"Failed to get trading metrics: {e}")
        return TradingMetrics(
            tick_latency_ms=0, active_bots=0, active_positions=0,
            daily_pnl=0, total_trades=0, win_rate=0
        )


def get_database_metrics() -> DatabaseMetrics:
    """Collect database-related metrics."""
    try:
        query_latency = 0.0
        pool_size = 0
        active_connections = 0
        query_count = 0

        # Try to get from Prometheus metrics
        try:
            from src.monitoring.prometheus_exporter import (
                db_query_duration_seconds, db_connection_pool_size
            )

            for metric in db_query_duration_seconds.collect():
                for sample in metric.samples:
                    # Get average latency
                    query_latency = sample.value * 1000  # Convert to ms
                    break

            for metric in db_connection_pool_size.collect():
                for sample in metric.samples:
                    if sample.labels.get('state') == 'active':
                        active_connections = int(sample.value)
                    elif sample.labels.get('state') == 'idle':
                        pool_size += int(sample.value)
            pool_size += active_connections

        except ImportError:
            pass

        return DatabaseMetrics(
            query_latency_ms=round(query_latency, 2),
            connection_pool_size=pool_size,
            active_connections=active_connections,
            query_count=query_count
        )
    except Exception as e:
        logger.error(f"Failed to get database metrics: {e}")
        return DatabaseMetrics(
            query_latency_ms=0, connection_pool_size=0,
            active_connections=0, query_count=0
        )


def get_tick_stream_metrics() -> TickStreamMetrics:
    """Collect tick streaming metrics."""
    try:
        ticks_per_second = 0.0
        buffer_size = 0
        processing_time = 0.0
        symbols_active = 0

        # Try to get from tick stream handler
        try:
            from src.api.tick_stream_handler import tick_stream_manager
            if tick_stream_manager:
                stats = tick_stream_manager.get_stats()
                ticks_per_second = stats.get('ticks_per_second', 0)
                buffer_size = stats.get('buffer_size', 0)
                processing_time = stats.get('processing_time_ms', 0)
                symbols_active = stats.get('symbols_active', 0)
        except ImportError:
            pass

        return TickStreamMetrics(
            ticks_per_second=round(ticks_per_second, 2),
            buffer_size=buffer_size,
            processing_time_ms=round(processing_time, 2),
            symbols_active=symbols_active
        )
    except Exception as e:
        logger.error(f"Failed to get tick stream metrics: {e}")
        return TickStreamMetrics(
            ticks_per_second=0, buffer_size=0,
            processing_time_ms=0, symbols_active=0
        )


def get_active_alerts() -> List[AlertData]:
    """Get all active (unacknowledged) alerts."""
    return [alert for alert in _active_alerts.values() if not alert.acknowledged]


def get_recent_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent alerts from AlertManager.
    
    Integrates with AlertManager for centralized alert management,
    combining active alerts with recent history.
    
    Transforms AlertManager alerts into AlertData schema:
    - Sets id from source and triggered_at
    - Maps level to severity string
    - Maps triggered_at to timestamp (datetime)
    - Includes source, copies message
    - Sets acknowledged default to False
    - Drops/ignores unsupported fields (tier, threshold_pct, metadata)
    """
    # Try to get alerts from AlertManager
    if get_alert_manager is not None:
        try:
            alert_manager = get_alert_manager()
            # Get active alerts
            status = alert_manager.get_status()
            active = status.get('active_alerts', [])
            # Get recent history
            history = alert_manager.get_history(limit=limit)
            
            # Combine and deduplicate
            all_alerts = active + history
            seen_ids = set()
            unique_alerts = []
            
            for alert in all_alerts:
                # Transform AlertManager alert to AlertData schema
                # Map level (GREEN/YELLOW/ORANGE/RED/BLACK) to severity (info/warning/critical)
                level = alert.get('level', 'GREEN')
                severity_map = {
                    'GREEN': 'info',
                    'YELLOW': 'warning',
                    'ORANGE': 'warning',
                    'RED': 'critical',
                    'BLACK': 'critical'
                }
                severity = severity_map.get(level, 'info')
                
                # Parse triggered_at to datetime
                triggered_at_str = alert.get('triggered_at')
                if isinstance(triggered_at_str, str):
                    try:
                        # Parse ISO format string to datetime
                        timestamp = datetime.fromisoformat(triggered_at_str.replace('Z', '+00:00'))
                    except ValueError:
                        # Fallback to current time if parsing fails
                        timestamp = datetime.now()
                elif isinstance(triggered_at_str, datetime):
                    timestamp = triggered_at_str
                else:
                    timestamp = datetime.now()
                
                # Generate id from source and triggered_at
                source = alert.get('source', 'unknown')
                alert_id = f"{source}_{alert.get('triggered_at')}"
                
                if alert_id not in seen_ids:
                    seen_ids.add(alert_id)
                    # Create normalized alert dict matching AlertData schema
                    normalized_alert = {
                        'id': alert_id,
                        'severity': severity,
                        'message': alert.get('message', ''),
                        'timestamp': timestamp,
                        'acknowledged': False,
                        'source': source
                    }
                    unique_alerts.append(normalized_alert)
            
            return unique_alerts[:limit]
        except Exception as e:
            logger.warning(f"Failed to get alerts from AlertManager: {e}")
    
    # Fallback to internal alerts
    return [alert.dict() for alert in _active_alerts.values()]


def create_alert(
    severity: str,
    message: str,
    source: str,
    alert_id: Optional[str] = None
) -> AlertData:
    """Create a new alert."""
    import uuid
    alert = AlertData(
        id=alert_id or str(uuid.uuid4()),
        severity=severity,
        message=message,
        timestamp=datetime.now(),
        acknowledged=False,
        source=source
    )
    _active_alerts[alert.id] = alert
    return alert


# ============== HTTP Endpoints ==============

@router.get("", response_model=MetricsResponse)
async def get_all_metrics():
    """Get all metrics in a single response."""
    return MetricsResponse(
        system=get_system_metrics(),
        trading=get_trading_metrics(),
        database=get_database_metrics(),
        tick_stream=get_tick_stream_metrics(),
        alerts=get_recent_alerts(limit=50),  # Use AlertManager
        timestamp=datetime.now()
    )


@router.get("/system", response_model=SystemMetrics)
async def get_system():
    """Get system metrics only."""
    return get_system_metrics()


@router.get("/trading", response_model=TradingMetrics)
async def get_trading():
    """Get trading metrics only."""
    return get_trading_metrics()


@router.get("/database", response_model=DatabaseMetrics)
async def get_database():
    """Get database metrics only."""
    return get_database_metrics()


@router.get("/tick-stream", response_model=TickStreamMetrics)
async def get_tick_stream():
    """Get tick stream metrics only."""
    return get_tick_stream_metrics()


@router.get("/alerts", response_model=PaginatedResponse[AlertData])
async def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    include_acknowledged: bool = Query(False, description="Include acknowledged alerts"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
):
    """Get alerts with optional filtering and pagination."""
    alerts = list(_active_alerts.values())

    if severity:
        alerts = [a for a in alerts if a.severity == severity]

    if not include_acknowledged:
        alerts = [a for a in alerts if not a.acknowledged]

    sorted_alerts = sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    total = len(sorted_alerts)
    paginated_alerts = sorted_alerts[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated_alerts,
        total=total,
        limit=limit,
        offset=offset
    )


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    if alert_id not in _active_alerts:
        return JSONResponse(
            status_code=404,
            content={"error": "Alert not found"}
        )

    _active_alerts[alert_id].acknowledged = True

    # Broadcast alert acknowledgment to all WebSocket clients
    await ws_manager.broadcast({
        "type": "alert_acknowledged",
        "alert_id": alert_id,
        "alert": _active_alerts[alert_id].dict()
    })

    return {"status": "acknowledged", "alert_id": alert_id}


@router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert."""
    if alert_id not in _active_alerts:
        return JSONResponse(
            status_code=404,
            content={"error": "Alert not found"}
        )

    deleted_alert = _active_alerts[alert_id].dict()
    del _active_alerts[alert_id]

    # Broadcast alert deletion to all WebSocket clients
    await ws_manager.broadcast({
        "type": "alert_deleted",
        "alert_id": alert_id,
        "alert": deleted_alert
    })

    return {"status": "deleted", "alert_id": alert_id}


@router.post("/alerts")
async def create_new_alert(
    severity: str,
    message: str,
    source: str = "manual"
):
    """Manually create an alert (for testing)."""
    alert = create_alert(severity, message, source)
    return alert


# ============== WebSocket Endpoint ==============

class MetricsWebSocketManager:
    """Manages WebSocket connections for metrics streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._broadcast_task = None

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Metrics WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Metrics WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def start_broadcasting(self, interval: float = 1.0):
        """Start periodic metrics broadcasting."""
        while True:
            try:
                # Get all metrics
                system_metrics = get_system_metrics().dict()
                trading_metrics = get_trading_metrics().dict()
                db_metrics = get_database_metrics().dict()
                tick_metrics = get_tick_stream_metrics().dict()
                
                # Get alerts from AlertManager
                alerts = get_recent_alerts(limit=50)
                
                metrics = {
                    "type": "metrics",
                    "payload": {
                        "system": system_metrics,
                        "trading": trading_metrics,
                        "database": db_metrics,
                        "tick_stream": tick_metrics,
                        "alerts": alerts  # Now includes AlertManager alerts
                    },
                    "timestamp": datetime.now().isoformat()
                }
                await self.broadcast(metrics)
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {e}")

            await asyncio.sleep(interval)


# Global WebSocket manager
ws_manager = MetricsWebSocketManager()


@router.websocket("/ws/metrics")
async def metrics_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics streaming."""
    await ws_manager.connect(websocket)

    try:
        # Start broadcasting if not already running
        if ws_manager._broadcast_task is None:
            ws_manager._broadcast_task = asyncio.create_task(
                ws_manager.start_broadcasting(interval=1.0)
            )

        while True:
            # Wait for messages from client (for ping/pong and commands)
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif message.get("type") == "acknowledge_alert":
                    alert_id = message.get("alert_id")
                    if alert_id and alert_id in _active_alerts:
                        _active_alerts[alert_id].acknowledged = True
                        await websocket.send_text(json.dumps({
                            "type": "alert_acknowledged",
                            "alert_id": alert_id
                        }))

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)
