"""
WebSocket endpoint for UI real-time updates.

Provides real-time updates for:
- Backtest progress
- Live trading status
- Bot performance
- System logs
- Trade journal entries
"""

from typing import List, Set, Dict, Any, Optional, TYPE_CHECKING
import asyncio
import json
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from fastapi import WebSocket, WebSocketDisconnect

try:
    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI not available for WebSocket support")
    FASTAPI_AVAILABLE = False
    # Create stub types for runtime when FastAPI is not available
    WebSocket = type('WebSocket', (), {})  # type: ignore
    WebSocketDisconnect = type('WebSocketDisconnect', (Exception,), {})  # type: ignore


class ConnectionManager:
    """Manages WebSocket connections with topic pooling (Phase 4.3)."""

    def __init__(self, heartbeat_interval: int = 30):
        self.active_connections: Set[WebSocket] = set()  # type: ignore
        self.subscriptions: Dict[str, Set[WebSocket]] = {}  # type: ignore[valid-type]
        self._topic_cache: Dict[str, List[WebSocket]] = {}  # Cache for faster lookup  # type: ignore[valid-type]
        self._heartbeat_interval = heartbeat_interval  # seconds
        self._heartbeat_task: Optional[asyncio.Task] = None
    
    async def connect(self, websocket: WebSocket):  # type: ignore
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):  # type: ignore
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from all subscriptions and invalidate caches
        for topic_subs in self.subscriptions.values():
            topic_subs.discard(websocket)
        self._topic_cache.clear()  # Invalidate all caches
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def subscribe(self, websocket: WebSocket, topic: str):  # type: ignore
        """Subscribe connection to specific topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)
        
        # Invalidate cache for this topic
        self._topic_cache.pop(topic, None)
        logger.debug(f"WebSocket subscribed to: {topic}")
    
    async def broadcast(self, message: Dict[str, Any], topic: Optional[str] = None):
        """
        Broadcast message with topic connection pooling.
        
        Args:
            message: Message dict to send
            topic: Optional topic filter
        """
        message_json = json.dumps(message)
        
        # Use cached topic connections if available
        if topic and topic in self._topic_cache:
            targets = self._topic_cache[topic]
        elif topic and topic in self.subscriptions:
            targets = list(self.subscriptions[topic])
            self._topic_cache[topic] = targets  # Cache for next time
        else:
            targets = list(self.active_connections)
        
        # Send to all targets
        disconnected = []
        for connection in targets:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected
        for conn in disconnected:
            self.disconnect(conn)

    async def _send_heartbeat(self):
        """Send periodic ping to all active connections."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            disconnected = []
            for websocket in list(self.active_connections):
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": asyncio.get_event_loop().time()
                    })
                except Exception as e:
                    logger.warning(f"Heartbeat failed for client: {e}")
                    disconnected.append(websocket)

            for ws in disconnected:
                self.disconnect(ws)

    def start_heartbeat(self):
        """Start the heartbeat task."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._send_heartbeat())
            logger.info("WebSocket heartbeat started")

    def stop_heartbeat(self):
        """Stop the heartbeat task."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            logger.info("WebSocket heartbeat stopped")


# Global connection manager
manager = ConnectionManager(heartbeat_interval=30)


# Chart streaming state
_chart_subscriptions: Dict[str, set] = {}


def create_websocket_endpoints(app):
    """
    Add WebSocket endpoints to FastAPI app.

    Usage:
        from src.api.websocket_endpoints import create_websocket_endpoints
        app = FastAPI()
        create_websocket_endpoints(app)
    """
    global manager

    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping WebSocket endpoints")
        return

    # Start heartbeat
    manager.start_heartbeat()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):  # type: ignore
        """
        Main WebSocket endpoint for UI updates.
        
        Message format:
            Client -> Server: {"action": "subscribe", "topic": "backtest"}
            Server -> Client: {"type": "backtest_progress", "data": {...}}
        """
        await manager.connect(websocket)
        try:
            while True:
                # Receive messages from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle subscription requests
                if message.get("action") == "subscribe":
                    topic = message.get("topic")
                    if topic:
                        await manager.subscribe(websocket, topic)
                        await websocket.send_json({
                            "type": "subscription_confirmed",
                            "topic": topic
                        })
                
                # Handle ping/pong for connection keep-alive
                elif message.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})

                # Handle server-initiated ping responses (client responding to our ping)
                elif message.get("type") == "pong":
                    # Client responded to our server-initiated ping - connection is alive
                    logger.debug(f"Received pong from client")
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)
    
    @app.websocket("/ws/chart/{symbol}/{timeframe}")
    async def chart_websocket_endpoint(websocket: WebSocket, symbol: str, timeframe: str):  # type: ignore
        """
        WebSocket endpoint for streaming live chart data.
        
        Subscribes to tick data for the specified symbol/timeframe and streams:
        - OHLCV bar updates
        - Regime changes
        - Trade events
        
        Message format:
            Server -> Client: {
                "type": "ohlcv" | "regime" | "trade",
                "data": {...}
            }
        """
        await websocket.accept()
        subscription_key = f"{symbol}:{timeframe}".upper()
        
        # Initialize subscription tracking
        if subscription_key not in _chart_subscriptions:
            _chart_subscriptions[subscription_key] = set()
        _chart_subscriptions[subscription_key].add(websocket)
        
        logger.info(f"Chart WebSocket connected: {symbol} {timeframe}")
        
        try:
            # Send initial confirmation
            await websocket.send_json({
                "type": "subscribed",
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper()
            })
            
            # Send initial historical data (last 100 candles)
            initial_data = await _get_historical_bars(symbol.upper(), timeframe.upper(), 100)
            if initial_data:
                await websocket.send_json({
                    "type": "historical_bars",
                    "data": initial_data
                })
            
            # Keep connection alive and forward tick data
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle ping/pong
                if message.get("action") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
        except WebSocketDisconnect:
            logger.info(f"Chart WebSocket disconnected: {symbol} {timeframe}")
        except Exception as e:
            logger.error(f"Chart WebSocket error: {e}")
        finally:
            if subscription_key in _chart_subscriptions:
                _chart_subscriptions[subscription_key].discard(websocket)
                if not _chart_subscriptions[subscription_key]:
                    del _chart_subscriptions[subscription_key]


async def _get_historical_bars(symbol: str, timeframe: str, count: int) -> List[Dict]:
    """Get historical OHLCV bars for initial data."""
    try:
        from src.data.historical_data import get_cached_bars
        bars = get_cached_bars(symbol, timeframe, count)
        return bars
    except Exception as e:
        logger.warning(f"Failed to get historical bars: {e}")
        return []


async def broadcast_chart_update(symbol: str, timeframe: str, update_type: str, data: Dict):
    """
    Broadcast chart update to all subscribed clients.
    
    Args:
        symbol: Trading symbol
        timeframe: Chart timeframe
        update_type: Type of update (ohlcv, regime, trade)
        data: Update data
    """
    subscription_key = f"{symbol}:{timeframe}".upper()
    
    if subscription_key in _chart_subscriptions:
        message = {
            "type": update_type,
            "data": data
        }
        # Send to all subscribers
        disconnected = set()
        for websocket in _chart_subscriptions[subscription_key]:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            _chart_subscriptions[subscription_key].discard(ws)


async def broadcast_backtest_progress(backtest_id: str, progress: float, status: str):
    """
    Broadcast backtest progress to UI.
    
    Args:
        backtest_id: Backtest ID
        progress: Progress percentage (0-100)
        status: Status message
    """
    await manager.broadcast({
        "type": "backtest_progress",
        "data": {
            "backtest_id": backtest_id,
            "progress": progress,
            "status": status
        }
    }, topic="backtest")


async def broadcast_trading_update(bot_id: str, status: str, pnl: Optional[float] = None):
    """
    Broadcast live trading update to UI.
    
    Args:
        bot_id: Bot identifier
        status: Bot status
        pnl: Current PnL
    """
    await manager.broadcast({
        "type": "trading_update",
        "data": {
            "bot_id": bot_id,
            "status": status,
            "pnl": pnl
        }
    }, topic="trading")


async def broadcast_log_entry(level: str, message: str, source: Optional[str] = None):
    """
    Broadcast log entry to UI.
    
    Args:
        level: Log level (INFO, WARNING, ERROR)
        message: Log message
        source: Source of log (bot_id, system, etc.)
    """
    await manager.broadcast({
        "type": "log_entry",
        "data": {
            "level": level,
            "message": message,
            "source": source
        }
    }, topic="logs")


async def broadcast_paper_trading_update(agent_id: str, status: str, 
                                         symbol: Optional[str] = None,
                                         timeframe: Optional[str] = None,
                                         uptime_seconds: Optional[int] = None,
                                         bot_id: Optional[str] = None,
                                         target_account: Optional[str] = None):
    """
    Broadcast paper trading update to UI.
    
    Args:
        agent_id: Agent identifier
        status: Agent status (running, stopped, error, promoted, etc.)
        symbol: Trading symbol (e.g., EURUSD)
        timeframe: Timeframe (e.g., H1, M5)
        uptime_seconds: Agent uptime in seconds
        bot_id: Bot ID if promoted to live trading
        target_account: Target account for promotion
    """
    data = {
        "agent_id": agent_id,
        "status": status,
        "symbol": symbol,
        "timeframe": timeframe,
        "uptime_seconds": uptime_seconds
    }
    
    # Add promotion data if available
    if bot_id:
        data["bot_id"] = bot_id
    if target_account:
        data["target_account"] = target_account
    
    await manager.broadcast({
        "type": "paper_trading_update",
        "data": data
    }, topic="paper-trading")


async def broadcast_paper_trading_performance(
    agent_id: str,
    metrics: Dict[str, Any]
):
    """
    Broadcast paper trading performance metrics update to UI.
    
    Triggered periodically (e.g., every 5 minutes) to update
    performance metrics in the UI.
    
    Args:
        agent_id: Agent identifier
        metrics: Performance metrics dict containing:
            - total_trades: int
            - winning_trades: int
            - losing_trades: int
            - win_rate: float (0-1)
            - total_pnl: float
            - sharpe_ratio: float or None
            - max_drawdown: float
            - profit_factor: float
            - validation_status: str (pending, validating, validated)
            - days_validated: int
            - meets_criteria: bool
    """
    await manager.broadcast({
        "type": "paper_trading_performance_update",
        "data": {
            "agent_id": agent_id,
            **metrics
        }
    }, topic="paper-trading")


async def broadcast_paper_trading_promotion(
    agent_id: str,
    bot_id: str,
    target_account: str,
    performance_summary: Dict[str, Any]
):
    """
    Broadcast paper trading promotion event to UI.
    
    Triggered when an agent is promoted to live trading.
    
    Args:
        agent_id: Original paper trading agent ID
        bot_id: New live trading bot ID
        target_account: Target account for live trading
        performance_summary: Final performance metrics before promotion
    """
    await manager.broadcast({
        "type": "paper_trading_promotion",
        "data": {
            "agent_id": agent_id,
            "bot_id": bot_id,
            "target_account": target_account,
            "performance_summary": performance_summary,
            "promoted_at": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        }
    }, topic="paper-trading")


async def broadcast_system_status(data: Dict[str, Any]):
    """
    Broadcast system status to UI.
    
    Args:
        data: System status dict with keys:
            - active_bots: int
            - pnl_today: float
            - regime: str
            - kelly: float
    """
    await manager.broadcast({
        "type": "system_status",
        "data": data
    }, topic="trading")


async def broadcast_bot_update(data: Dict[str, Any]):
    """
    Broadcast bot list update to UI.

    Args:
        data: Bot update dict with key:
            - bots: List[Dict] - list of bot dicts with id, name, state, symbol
    """
    await manager.broadcast({
        "type": "bot_update",
        "data": data
    }, topic="trading")


async def broadcast_fee_update(data: Dict[str, Any]):
    """
    Broadcast fee monitoring update to UI.

    Args:
        data: Fee update dict with keys:
            - daily_fees: float - Total fees paid today
            - daily_fee_burn_pct: float - Fee burn percentage
            - kill_switch_active: bool - Whether fee kill switch is active
            - fee_breakdown: List[Dict] - Per-bot fee breakdown
    """
    await manager.broadcast({
        "type": "fee_update",
        "data": data
    }, topic="trading")


async def broadcast_regime_update(data: Dict[str, Any]):
    """
    Broadcast multi-timeframe regime update to UI.

    Args:
        data: Regime update dict with keys:
            - dominant_regime: str - Dominant regime across timeframes
            - timeframe_regimes: Dict[str, Dict] - Per-timeframe regime data
                - regime: str - Regime name
                - quality: float - Regime quality (0-1)
            - consensus_strength: float - Percentage of timeframes agreeing (0-100)
    """
    await manager.broadcast({
        "type": "regime_update",
        "data": data
    }, topic="trading")


# =============================================================================
# HMM WebSocket Events
# =============================================================================

async def broadcast_hmm_prediction(
    symbol: str,
    timeframe: str,
    ising_regime: str,
    ising_confidence: float,
    hmm_regime: str,
    hmm_state: int,
    hmm_confidence: float,
    agreement: bool,
    decision_source: str
):
    """
    Broadcast HMM prediction event to UI.

    Triggered on every tick when shadow mode or hybrid mode is active.

    Args:
        symbol: Trading symbol
        timeframe: Timeframe
        ising_regime: Ising model regime prediction
        ising_confidence: Ising model confidence (0-1)
        hmm_regime: HMM regime prediction
        hmm_state: HMM state ID (0-3)
        hmm_confidence: HMM confidence (0-1)
        agreement: Whether models agree
        decision_source: Which model was used for decision
    """
    await manager.broadcast({
        "type": "hmm_prediction",
        "data": {
            "symbol": symbol,
            "timeframe": timeframe,
            "ising_regime": ising_regime,
            "ising_confidence": ising_confidence,
            "hmm_regime": hmm_regime,
            "hmm_state": hmm_state,
            "hmm_confidence": hmm_confidence,
            "agreement": agreement,
            "decision_source": decision_source,
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        }
    }, topic="hmm")


async def broadcast_hmm_sync_progress(
    status: str,
    progress: float,
    message: str,
    error: Optional[str] = None
):
    """
    Broadcast HMM model sync progress to UI.

    Triggered during model synchronization from Contabo to Cloudzy.

    Args:
        status: Sync status (connecting, downloading, verifying, deploying, complete, failed)
        progress: Progress percentage (0-100)
        message: Status message
        error: Error message if failed
    """
    await manager.broadcast({
        "type": "hmm_sync_progress",
        "data": {
            "status": status,
            "progress": progress,
            "message": message,
            "error": error,
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        }
    }, topic="hmm")


async def broadcast_hmm_mode_change(
    previous_mode: Optional[str],
    new_mode: str,
    hmm_weight: float
):
    """
    Broadcast HMM deployment mode change to UI.

    Triggered when deployment mode transitions.

    Args:
        previous_mode: Previous mode (None for initial state)
        new_mode: New deployment mode
        hmm_weight: Current HMM weight for hybrid decisions
    """
    await manager.broadcast({
        "type": "hmm_mode_change",
        "data": {
            "previous_mode": previous_mode,
            "new_mode": new_mode,
            "hmm_weight": hmm_weight,
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        }
    }, topic="hmm")


async def broadcast_hmm_training_status(
    job_id: str,
    status: str,
    progress: float,
    message: str,
    metrics: Optional[Dict[str, Any]] = None
):
    """
    Broadcast HMM training job status to UI.

    Triggered during training jobs on Contabo.

    Args:
        job_id: Training job ID
        status: Job status (pending, running, complete, failed)
        progress: Progress percentage (0-100)
        message: Status message
        metrics: Training metrics (log_likelihood, samples, etc.)
    """
    await manager.broadcast({
        "type": "hmm_training_status",
        "data": {
            "job_id": job_id,
            "status": status,
            "progress": progress,
            "message": message,
            "metrics": metrics,
            "timestamp": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
        }
    }, topic="hmm")


def get_manager():
    """Get the global connection manager for use by other modules."""
    return manager


async def broadcast_lifecycle_event(
    event_type: str,
    bot_id: str,
    from_tag: str,
    to_tag: str,
    reason: str,
    timestamp: Optional[str] = None
):
    """
    Broadcast lifecycle event to connected clients.
    
    Args:
        event_type: Type of event (promotion, quarantine, kill)
        bot_id: Bot identifier
        from_tag: Source tag
        to_tag: Destination tag
        reason: Reason for transition
        timestamp: Event timestamp (defaults to now)
    """
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()
    
    await manager.broadcast({
        "type": "lifecycle_event",
        "event": {
            "event_type": event_type,
            "bot_id": bot_id,
            "from_tag": from_tag,
            "to_tag": to_tag,
            "reason": reason,
            "timestamp": timestamp
        },
        "timestamp": timestamp
    }, topic="lifecycle")


async def broadcast_market_opportunity(
    scan_type: str,
    symbol: str,
    confidence: float,
    setup: str,
    recommended_bots: list,
    metadata: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
):
    """
    Broadcast market opportunity alert to connected clients.
    
    Args:
        scan_type: Type of scan (SESSION_BREAKOUT, VOLATILITY_SPIKE, etc.)
        symbol: Trading symbol
        confidence: Confidence score 0-1
        setup: Specific setup detected
        recommended_bots: List of recommended bot IDs
        metadata: Additional details
        timestamp: Event timestamp (defaults to now)
    """
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

    await manager.broadcast({
        "type": "market_opportunity",
        "alert": {
            "scan_type": scan_type,
            "symbol": symbol,
            "confidence": confidence,
            "setup": setup,
            "recommended_bots": recommended_bots,
            "metadata": metadata or {},
            "timestamp": timestamp
        },
        "timestamp": timestamp
    }, topic="market_scanner")


async def broadcast_approval_gate(
    gate_id: str,
    workflow_id: str,
    action: str,  # "created", "approved", "rejected"
    from_stage: str,
    to_stage: str,
    gate_type: str = "stage_transition",
    approver: Optional[str] = None,
    notes: Optional[str] = None,
    timestamp: Optional[str] = None
):
    """
    Broadcast approval gate events to connected WebSocket clients.

    Args:
        gate_id: ID of the approval gate
        workflow_id: ID of the workflow
        action: Action type - "created", "approved", or "rejected"
        from_stage: Current workflow stage
        to_stage: Next workflow stage
        gate_type: Type of approval gate
        approver: User who approved/rejected (for approved/rejected actions)
        notes: Notes from approver (for approved/rejected actions)
        timestamp: ISO timestamp (optional, will be generated if not provided)
    """
    if timestamp is None:
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

    await manager.broadcast({
        "type": "approval_gate",
        "action": action,
        "gate": {
            "gate_id": gate_id,
            "workflow_id": workflow_id,
            "from_stage": from_stage,
            "to_stage": to_stage,
            "gate_type": gate_type,
            "approver": approver,
            "notes": notes,
            "timestamp": timestamp
        },
        "timestamp": timestamp
    }, topic="approvals")


__all__ = [
    'create_websocket_endpoints',
    'broadcast_backtest_progress',
    'broadcast_trading_update',
    'broadcast_log_entry',
    'broadcast_paper_trading_update',
    'broadcast_paper_trading_performance',
    'broadcast_paper_trading_promotion',
    'broadcast_system_status',
    'broadcast_bot_update',
    'broadcast_fee_update',
    'broadcast_regime_update',
    'broadcast_hmm_prediction',
    'broadcast_hmm_sync_progress',
    'broadcast_hmm_mode_change',
    'broadcast_hmm_training_status',
    'broadcast_chart_update',
    'broadcast_lifecycle_event',
    'broadcast_market_opportunity',
    'broadcast_approval_gate',
    'manager'
]
