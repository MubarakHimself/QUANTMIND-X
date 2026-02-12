"""
WebSocket endpoint for UI real-time updates.

Provides real-time updates for:
- Backtest progress
- Live trading status
- Bot performance
- System logs
- Trade journal entries
"""

from typing import Set, Dict, Any, Optional, TYPE_CHECKING
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
    """Manages WebSocket connections for UI updates."""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()  # type: ignore
        self.subscriptions: Dict[str, Set[WebSocket]] = {}  # type: ignore
    
    async def connect(self, websocket: WebSocket):  # type: ignore
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):  # type: ignore
        """Remove WebSocket connection."""
        self.active_connections.discard(websocket)
        # Remove from all subscriptions
        for topic_subs in self.subscriptions.values():
            topic_subs.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def subscribe(self, websocket: WebSocket, topic: str):  # type: ignore
        """Subscribe connection to specific topic."""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(websocket)
        logger.debug(f"WebSocket subscribed to: {topic}")
    
    async def broadcast(self, message: Dict[str, Any], topic: Optional[str] = None):
        """
        Broadcast message to all connections or specific topic subscribers.
        
        Args:
            message: Message dict to send
            topic: Optional topic filter (e.g., "backtest", "trading", "logs")
        """
        message_json = json.dumps(message)
        
        # Select target connections
        if topic and topic in self.subscriptions:
            targets = self.subscriptions[topic]
        else:
            targets = self.active_connections
        
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


# Global connection manager
manager = ConnectionManager()


def create_websocket_endpoints(app):
    """
    Add WebSocket endpoints to FastAPI app.
    
    Usage:
        from src.api.websocket_endpoints import create_websocket_endpoints
        app = FastAPI()
        create_websocket_endpoints(app)
    """
    if not FASTAPI_AVAILABLE:
        logger.warning("FastAPI not available, skipping WebSocket endpoints")
        return
    
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
                    
        except WebSocketDisconnect:
            manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            manager.disconnect(websocket)


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


def get_manager():
    """Get the global connection manager for use by other modules."""
    return manager


__all__ = [
    'create_websocket_endpoints',
    'broadcast_backtest_progress',
    'broadcast_trading_update',
    'broadcast_log_entry',
    'manager'
]
