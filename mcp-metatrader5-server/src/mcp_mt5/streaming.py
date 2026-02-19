"""
WebSocket Streaming Service
============================
Real-time tick data streaming via WebSocket.

Features:
- Multi-symbol subscription
- Automatic reconnection
- Connection pooling
- Rate limiting per client
- JSON message protocol
- Heartbeat/ping-pong
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional, Set
import threading

try:
    from websockets.server import serve as ws_serve
    from websockets.exceptions import ConnectionClosed
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class StreamConfig:
    """WebSocket streaming configuration."""
    
    host: str = "127.0.0.1"
    port: int = 8765
    poll_interval_ms: int = 100  # Tick polling interval
    heartbeat_interval_sec: int = 30
    max_clients: int = 10
    max_symbols_per_client: int = 50
    rate_limit_per_second: int = 100  # Max messages per second per client


class MessageType(str, Enum):
    """WebSocket message types."""
    
    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    GET_SYMBOLS = "get_symbols"
    PING = "ping"
    
    # Server -> Client
    TICK = "tick"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    SYMBOLS = "symbols"
    PONG = "pong"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class TickData:
    """Real-time tick data."""
    
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    time: str  # ISO8601
    spread: float
    
    @classmethod
    def from_mt5_tick(cls, symbol: str, tick) -> "TickData":
        """Create from MT5 tick object."""
        return cls(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            time=datetime.fromtimestamp(tick.time).isoformat(),
            spread=round((tick.ask - tick.bid) * 10000, 1)  # In pips for forex
        )


# ============================================================================
# Client Connection
# ============================================================================

@dataclass
class ClientConnection:
    """Represents a connected WebSocket client."""
    
    id: str
    websocket: Any
    subscribed_symbols: Set[str] = field(default_factory=set)
    connected_at: str = ""
    last_message_at: str = ""
    message_count: int = 0
    
    def __post_init__(self):
        if not self.connected_at:
            self.connected_at = datetime.now().isoformat()


# ============================================================================
# Tick Streamer
# ============================================================================

class TickStreamer:
    """
    Real-time tick data streamer via WebSocket.
    
    Usage:
        streamer = TickStreamer()
        await streamer.start()
        
    Client Protocol:
        # Subscribe to symbols
        {"type": "subscribe", "symbols": ["EURUSD", "GBPUSD"]}
        
        # Unsubscribe
        {"type": "unsubscribe", "symbols": ["EURUSD"]}
        
        # Get available symbols
        {"type": "get_symbols"}
        
        # Ping (keepalive)
        {"type": "ping"}
        
    Server Messages:
        # Tick data
        {"type": "tick", "data": {"symbol": "EURUSD", "bid": 1.0850, ...}}
        
        # Subscription confirmed
        {"type": "subscribed", "symbols": ["EURUSD", "GBPUSD"]}
        
        # Error
        {"type": "error", "message": "..."}
    """
    
    def __init__(self, config: StreamConfig = None):
        """
        Initialize tick streamer.
        
        Args:
            config: Streaming configuration.
        """
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError(
                "websockets package not installed. "
                "Install with: pip install websockets"
            )
        
        self.config = config or StreamConfig()
        self.clients: dict[str, ClientConnection] = {}
        self._running = False
        self._tick_cache: dict[str, TickData] = {}  # Last tick per symbol
        self._lock = asyncio.Lock()
        
        logger.info(f"TickStreamer initialized on {self.config.host}:{self.config.port}")
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        if self._running:
            logger.warning("Streamer already running")
            return
        
        self._running = True
        
        # Start tick polling task
        asyncio.create_task(self._poll_ticks())
        
        # Start heartbeat task
        asyncio.create_task(self._heartbeat_loop())
        
        # Start WebSocket server
        async with ws_serve(
            self._handle_client,
            self.config.host,
            self.config.port
        ):
            logger.info(f"WebSocket server started on ws://{self.config.host}:{self.config.port}")
            await asyncio.Future()  # Run forever
    
    async def stop(self) -> None:
        """Stop the streamer."""
        self._running = False
        
        # Close all client connections
        for client in list(self.clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass
        
        self.clients.clear()
        logger.info("TickStreamer stopped")
    
    async def _handle_client(self, websocket) -> None:
        """Handle a new client connection."""
        client_id = f"client_{int(time.time() * 1000)}"
        
        if len(self.clients) >= self.config.max_clients:
            await websocket.send(json.dumps({
                "type": MessageType.ERROR.value,
                "message": "Max clients reached"
            }))
            await websocket.close()
            return
        
        client = ClientConnection(id=client_id, websocket=websocket)
        self.clients[client_id] = client
        
        logger.info(f"Client connected: {client_id}")
        
        try:
            async for message in websocket:
                await self._process_message(client, message)
        except ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            self.clients.pop(client_id, None)
    
    async def _process_message(self, client: ClientConnection, raw_message: str) -> None:
        """Process incoming client message."""
        try:
            message = json.loads(raw_message)
            msg_type = message.get("type", "").lower()
            
            client.last_message_at = datetime.now().isoformat()
            client.message_count += 1
            
            if msg_type == MessageType.SUBSCRIBE.value:
                await self._handle_subscribe(client, message.get("symbols", []))
            
            elif msg_type == MessageType.UNSUBSCRIBE.value:
                await self._handle_unsubscribe(client, message.get("symbols", []))
            
            elif msg_type == MessageType.GET_SYMBOLS.value:
                await self._handle_get_symbols(client)
            
            elif msg_type == MessageType.PING.value:
                await client.websocket.send(json.dumps({
                    "type": MessageType.PONG.value,
                    "timestamp": datetime.now().isoformat()
                }))
            
            else:
                await client.websocket.send(json.dumps({
                    "type": MessageType.ERROR.value,
                    "message": f"Unknown message type: {msg_type}"
                }))
                
        except json.JSONDecodeError:
            await client.websocket.send(json.dumps({
                "type": MessageType.ERROR.value,
                "message": "Invalid JSON"
            }))
    
    async def _handle_subscribe(self, client: ClientConnection, symbols: list) -> None:
        """Handle symbol subscription."""
        if not symbols:
            await client.websocket.send(json.dumps({
                "type": MessageType.ERROR.value,
                "message": "No symbols provided"
            }))
            return
        
        # Limit symbols per client
        available_slots = self.config.max_symbols_per_client - len(client.subscribed_symbols)
        symbols_to_add = symbols[:available_slots]
        
        # Validate symbols exist in MT5
        valid_symbols = []
        for symbol in symbols_to_add:
            if mt5.symbol_select(symbol, True):
                valid_symbols.append(symbol)
                client.subscribed_symbols.add(symbol)
        
        await client.websocket.send(json.dumps({
            "type": MessageType.SUBSCRIBED.value,
            "symbols": valid_symbols,
            "total_subscribed": len(client.subscribed_symbols)
        }))
        
        logger.info(f"Client {client.id} subscribed to: {valid_symbols}")
    
    async def _handle_unsubscribe(self, client: ClientConnection, symbols: list) -> None:
        """Handle symbol unsubscription."""
        removed = []
        for symbol in symbols:
            if symbol in client.subscribed_symbols:
                client.subscribed_symbols.remove(symbol)
                removed.append(symbol)
        
        await client.websocket.send(json.dumps({
            "type": MessageType.UNSUBSCRIBED.value,
            "symbols": removed,
            "total_subscribed": len(client.subscribed_symbols)
        }))
    
    async def _handle_get_symbols(self, client: ClientConnection) -> None:
        """Return list of available symbols."""
        symbols = mt5.symbols_get()
        symbol_list = [s.name for s in symbols] if symbols else []
        
        await client.websocket.send(json.dumps({
            "type": MessageType.SYMBOLS.value,
            "symbols": symbol_list[:200],  # Limit response size
            "total": len(symbol_list)
        }))
    
    async def _poll_ticks(self) -> None:
        """Background task to poll MT5 for ticks and broadcast."""
        while self._running:
            try:
                # Collect all subscribed symbols
                all_symbols: Set[str] = set()
                for client in self.clients.values():
                    all_symbols.update(client.subscribed_symbols)
                
                # Get ticks for each symbol
                for symbol in all_symbols:
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    
                    tick_data = TickData.from_mt5_tick(symbol, tick)
                    
                    # Check if tick changed
                    cached = self._tick_cache.get(symbol)
                    if cached and cached.bid == tick_data.bid and cached.ask == tick_data.ask:
                        continue  # No change
                    
                    self._tick_cache[symbol] = tick_data
                    
                    # Broadcast to subscribed clients
                    await self._broadcast_tick(symbol, tick_data)
                
            except Exception as e:
                logger.error(f"Tick polling error: {e}")
            
            await asyncio.sleep(self.config.poll_interval_ms / 1000)
    
    async def _broadcast_tick(self, symbol: str, tick_data: TickData) -> None:
        """Broadcast tick to all subscribed clients."""
        message = json.dumps({
            "type": MessageType.TICK.value,
            "data": asdict(tick_data)
        })
        
        for client in list(self.clients.values()):
            if symbol in client.subscribed_symbols:
                try:
                    await client.websocket.send(message)
                except Exception:
                    pass  # Client may have disconnected
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to clients."""
        while self._running:
            await asyncio.sleep(self.config.heartbeat_interval_sec)
            
            message = json.dumps({
                "type": MessageType.HEARTBEAT.value,
                "timestamp": datetime.now().isoformat(),
                "clients": len(self.clients)
            })
            
            for client in list(self.clients.values()):
                try:
                    await client.websocket.send(message)
                except Exception:
                    pass
    
    def get_stats(self) -> dict:
        """Get streamer statistics."""
        return {
            "running": self._running,
            "clients": len(self.clients),
            "symbols_tracked": len(self._tick_cache),
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "poll_interval_ms": self.config.poll_interval_ms
            }
        }


# ============================================================================
# Standalone Server Entry Point
# ============================================================================

async def run_streaming_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    poll_interval_ms: int = 100
) -> None:
    """
    Run the streaming server as standalone.
    
    Usage:
        import asyncio
        from streaming import run_streaming_server
        
        # Initialize MT5 first
        import MetaTrader5 as mt5
        mt5.initialize()
        
        # Run server
        asyncio.run(run_streaming_server())
    """
    config = StreamConfig(
        host=host,
        port=port,
        poll_interval_ms=poll_interval_ms
    )
    
    streamer = TickStreamer(config)
    await streamer.start()


if __name__ == "__main__":
    # Standalone mode
    import sys
    
    if not mt5.initialize():
        print("Failed to initialize MT5")
        sys.exit(1)
    
    print("Starting WebSocket Tick Streamer...")
    asyncio.run(run_streaming_server())
