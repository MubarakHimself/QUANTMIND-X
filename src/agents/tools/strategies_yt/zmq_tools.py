"""
ZMQ Communication Tools

Tools for ZMQ-based communication with the QuantMindX Strategy Router.
Provides low-level socket communication for EA integration and monitoring.

Implements the ZMQ REP/REQ pattern for sub-5ms latency trade events.
Message types: TRADE_OPEN, TRADE_CLOSE, TRADE_MODIFY, HEARTBEAT, RISK_UPDATE

Based on:
- src/router/socket_server.py - Python ZMQ server implementation
- src/mql5/Include/QuantMind/Utils/Sockets.mqh - MQL5 socket client
"""

import asyncio
import json
import logging
import zmq
import zmq.asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import time

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class MessageType(Enum):
    """Socket message types for event-driven communication."""
    TRADE_OPEN = "trade_open"
    TRADE_CLOSE = "trade_close"
    TRADE_MODIFY = "trade_modify"
    HEARTBEAT = "heartbeat"
    RISK_UPDATE = "risk_update"
    REGISTRATION = "registration"
    CIRCUIT_BREAKER = "circuit_breaker"
    REGIME_CHANGE = "regime_change"


class ConnectionStatus(Enum):
    """ZMQ connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ZMQMessage:
    """Base ZMQ message structure."""
    type: MessageType
    ea_name: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Convert to JSON string."""
        msg_dict = {
            "type": self.type.value,
            "ea_name": self.ea_name,
            "timestamp": self.timestamp,
            **self.data
        }
        return json.dumps(msg_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "ZMQMessage":
        """Create from JSON string."""
        data = json.loads(json_str)
        msg_type = MessageType(data.pop("type", "heartbeat"))
        ea_name = data.pop("ea_name", "")
        timestamp = data.pop("timestamp", time.time())
        return cls(type=msg_type, ea_name=ea_name, timestamp=timestamp, data=data)


@dataclass
class TradeOpenMessage(ZMQMessage):
    """Message for trade open events."""
    symbol: str = ""
    volume: float = 0.0
    magic: int = 0
    order_type: str = "market"  # market, limit, stop
    stop_loss: float = 0.0
    take_profit: float = 0.0
    current_balance: float = 0.0

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = MessageType.TRADE_OPEN
        else:
            self.type = MessageType.TRADE_OPEN

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "trade_open",
            "ea_name": self.ea_name,
            "symbol": self.symbol,
            "volume": self.volume,
            "magic": self.magic,
            "order_type": self.order_type,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "current_balance": self.current_balance,
            "timestamp": self.timestamp
        }


@dataclass
class TradeCloseMessage(ZMQMessage):
    """Message for trade close events."""
    symbol: str = ""
    ticket: int = 0
    profit: float = 0.0
    close_reason: str = ""  # sl, tp, manual, signal

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = MessageType.TRADE_CLOSE
        else:
            self.type = MessageType.TRADE_CLOSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "trade_close",
            "ea_name": self.ea_name,
            "symbol": self.symbol,
            "ticket": self.ticket,
            "profit": self.profit,
            "close_reason": self.close_reason,
            "timestamp": self.timestamp
        }


@dataclass
class HeartbeatMessage(ZMQMessage):
    """Message for heartbeat events."""
    symbol: str = ""
    magic: int = 0
    current_positions: int = 0

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = MessageType.HEARTBEAT
        else:
            self.type = MessageType.HEARTBEAT

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "heartbeat",
            "ea_name": self.ea_name,
            "symbol": self.symbol,
            "magic": self.magic,
            "current_positions": self.current_positions,
            "timestamp": self.timestamp
        }


@dataclass
class ZMQResponse:
    """Response from ZMQ server."""
    status: str  # success, rejected, error
    approved: bool = False
    risk_multiplier: float = 1.0
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ZMQResponse":
        """Create from response dictionary."""
        return cls(
            status=data.get("status", "error"),
            approved=data.get("approved", False),
            risk_multiplier=data.get("risk_multiplier", 1.0),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {})
        )


# =============================================================================
# ZMQ CLIENT CLASS
# =============================================================================

class ZMQClient:
    """
    Async ZMQ client for communicating with Strategy Router.

    Provides REQ/REP pattern communication with timeout and retry logic.
    """

    def __init__(
        self,
        server_address: str = "localhost",
        server_port: int = 5555,
        timeout_ms: int = 5000,
        max_retries: int = 3
    ):
        """
        Initialize ZMQ client.

        Args:
            server_address: ZMQ server hostname or IP
            server_port: ZMQ server port
            timeout_ms: Request timeout in milliseconds
            max_retries: Maximum retry attempts
        """
        self.server_address = server_address
        self.server_port = server_port
        self.endpoint = f"tcp://{server_address}:{server_port}"
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries

        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.status = ConnectionStatus.DISCONNECTED
        self.message_count = 0
        self.total_latency = 0.0

    async def connect(self) -> bool:
        """
        Connect to ZMQ server.

        Returns:
            True if connected successfully
        """
        try:
            self.status = ConnectionStatus.CONNECTING

            # Create context and socket
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

            # Connect to server
            await self.socket.connect(self.endpoint)

            self.status = ConnectionStatus.CONNECTED
            logger.info(f"ZMQ client connected to {self.endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ZMQ server: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> None:
        """Disconnect from ZMQ server."""
        try:
            if self.socket:
                self.socket.close()
            if self.context:
                self.context.term()

            self.status = ConnectionStatus.DISCONNECTED
            logger.info("ZMQ client disconnected")

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    async def send_message(self, message: ZMQMessage) -> ZMQResponse:
        """
        Send message to ZMQ server and await response.

        Args:
            message: ZMQMessage to send

        Returns:
            ZMQResponse from server
        """
        if self.status != ConnectionStatus.CONNECTED:
            if not await self.connect():
                return ZMQResponse(status="error", message="Not connected to server")

        try:
            start_time = time.time()

            # Send message
            await self.socket.send_string(message.to_json())

            # Receive response
            response_str = await self.socket.recv_string()
            response = ZMQResponse.from_dict(json.loads(response_str))

            # Update metrics
            latency_ms = (time.time() - start_time) * 1000
            self.message_count += 1
            self.total_latency += latency_ms

            # Log if latency exceeds target
            if latency_ms > 5.0:
                logger.warning(f"High ZMQ latency: {latency_ms:.2f}ms (target: <5ms)")

            return response

        except zmq.Again:
            logger.error("ZMQ request timeout")
            return ZMQResponse(status="error", message="Request timeout")
        except Exception as e:
            logger.error(f"ZMQ send error: {e}")
            return ZMQResponse(status="error", message=str(e))

    async def send_and_retry(self, message: ZMQMessage) -> ZMQResponse:
        """
        Send message with retry logic.

        Args:
            message: ZMQMessage to send

        Returns:
            ZMQResponse from server
        """
        for attempt in range(self.max_retries):
            response = await self.send_message(message)

            if response.status == "success" or response.status == "rejected":
                return response

            logger.warning(f"Retry {attempt + 1}/{self.max_retries} for message: {message.type.value}")

            # Reconnect before retry
            await self.disconnect()
            await asyncio.sleep(0.1 * (attempt + 1))

        return ZMQResponse(status="error", message="Max retries exceeded")

    def get_average_latency(self) -> float:
        """Get average message latency in milliseconds."""
        if self.message_count == 0:
            return 0.0
        return self.total_latency / self.message_count


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

async def send_to_router(
    message: ZMQMessage,
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Send message to Strategy Router.

    Args:
        message: ZMQMessage to send
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary containing:
        - success: Send status
        - response: ZMQResponse
        - latency_ms: Round-trip latency
    """
    logger.info(f"Sending to router: {message.type.value} from {message.ea_name}")

    try:
        client = ZMQClient(server_address, server_port)

        async with client:
            response = await client.send_and_retry(message)

            return {
                "success": response.status == "success",
                "response": asdict(response),
                "latency_ms": client.get_average_latency(),
                "message_type": message.type.value,
                "ea_name": message.ea_name
            }

    except Exception as e:
        logger.error(f"Failed to send to router: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "message_type": message.type.value,
            "ea_name": message.ea_name
        }


async def receive_from_router(
    ea_name: str,
    timeout_ms: int = 5000,
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Receive messages from router (subscription-based).

    Note: This uses a SUB socket pattern for receiving router broadcasts.

    Args:
        ea_name: EA identifier for filtering
        timeout_ms: Receive timeout
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary with received messages
    """
    logger.info(f"Receiving from router for: {ea_name}")

    try:
        context = zmq.asyncio.Context()
        socket = context.socket(zmq.SUB)

        # Subscribe to all messages (filter in application)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")

        socket.connect(f"tcp://{server_address}:{server_port + 1}")  # PUB/SUB port
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

        messages = []

        # Receive messages with timeout
        start_time = time.time()
        while (time.time() - start_time) * 1000 < timeout_ms:
            try:
                msg_str = await socket.recv_string()
                msg_data = json.loads(msg_str)

                # Filter by EA name if specified
                if ea_name and msg_data.get("ea_name") != ea_name:
                    continue

                messages.append(msg_data)

            except zmq.Again:
                break

        socket.close()
        context.term()

        return {
            "success": True,
            "messages": messages,
            "count": len(messages),
            "ea_name": ea_name
        }

    except Exception as e:
        logger.error(f"Failed to receive from router: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ea_name": ea_name,
            "messages": []
        }


async def register_ea(
    ea_name: str,
    strategy_id: str,
    magic: int,
    symbols: List[str],
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Register EA with Strategy Router.

    Args:
        ea_name: EA identifier
        strategy_id: Strategy identifier
        magic: Magic number for EA
        symbols: Trading symbols
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary with registration result
    """
    logger.info(f"Registering EA: {ea_name} (magic: {magic})")

    try:
        message = ZMQMessage(
            type=MessageType.REGISTRATION,
            ea_name=ea_name,
            data={
                "strategy_id": strategy_id,
                "magic": magic,
                "symbols": symbols,
                "register_at": datetime.now(timezone.utc).isoformat()
            }
        )

        result = await send_to_router(message, server_address, server_port)

        if result["success"]:
            result["registration_id"] = f"{ea_name}_{magic}_{int(time.time())}"

        return result

    except Exception as e:
        logger.error(f"Failed to register EA: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ea_name": ea_name
        }


async def send_heartbeat(
    ea_name: str,
    symbol: str,
    magic: int,
    current_positions: int = 0,
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Send heartbeat to Strategy Router.

    Args:
        ea_name: EA identifier
        symbol: Current symbol
        magic: Magic number
        current_positions: Number of open positions
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary with heartbeat result and risk multiplier
    """
    logger.debug(f"Sending heartbeat: {ea_name}")

    try:
        message = HeartbeatMessage(
            ea_name=ea_name,
            symbol=symbol,
            magic=magic,
            current_positions=current_positions
        )

        result = await send_to_router(message, server_address, server_port)

        # Extract risk multiplier from response
        if result.get("success") and result.get("response"):
            result["risk_multiplier"] = result["response"].get("risk_multiplier", 1.0)

        return result

    except Exception as e:
        logger.error(f"Failed to send heartbeat: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ea_name": ea_name
        }


async def send_trade_open(
    ea_name: str,
    symbol: str,
    volume: float,
    magic: int,
    stop_loss: float = 0.0,
    take_profit: float = 0.0,
    current_balance: float = 0.0,
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Send trade open event to Strategy Router.

    Args:
        ea_name: EA identifier
        symbol: Trading symbol
        volume: Position size in lots
        magic: Magic number
        stop_loss: Stop loss price
        take_profit: Take profit price
        current_balance: Current account balance
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary with approval status and risk multiplier
    """
    logger.info(f"Sending trade open: {ea_name} {symbol} {volume} lots")

    try:
        message = TradeOpenMessage(
            ea_name=ea_name,
            symbol=symbol,
            volume=volume,
            magic=magic,
            stop_loss=stop_loss,
            take_profit=take_profit,
            current_balance=current_balance
        )

        result = await send_to_router(message, server_address, server_port)

        # Extract approval status from response
        if result.get("success") and result.get("response"):
            result["approved"] = result["response"].get("approved", False)
            result["risk_multiplier"] = result["response"].get("risk_multiplier", 1.0)

        return result

    except Exception as e:
        logger.error(f"Failed to send trade open: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ea_name": ea_name,
            "approved": False
        }


async def send_trade_close(
    ea_name: str,
    symbol: str,
    ticket: int,
    profit: float,
    close_reason: str = "manual",
    server_address: str = "localhost",
    server_port: int = 5555
) -> Dict[str, Any]:
    """
    Send trade close event to Strategy Router.

    Args:
        ea_name: EA identifier
        symbol: Trading symbol
        ticket: Order ticket number
        profit: Trade profit/loss
        close_reason: Reason for close (sl, tp, manual, signal)
        server_address: Router hostname/IP
        server_port: Router port

    Returns:
        Dictionary with close result
    """
    logger.info(f"Sending trade close: {ea_name} {symbol} ticket #{ticket}")

    try:
        message = TradeCloseMessage(
            ea_name=ea_name,
            symbol=symbol,
            ticket=ticket,
            profit=profit,
            close_reason=close_reason
        )

        return await send_to_router(message, server_address, server_port)

    except Exception as e:
        logger.error(f"Failed to send trade close: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "ea_name": ea_name
        }


# =============================================================================
# TOOL REGISTRY
# =============================================================================

ZMQ_TOOLS = {
    "send_to_router": {
        "function": send_to_router,
        "description": "Send message to Strategy Router",
        "parameters": {
            "message": {"type": "object", "required": True},
            "server_address": {"type": "string", "required": False, "default": "localhost"},
            "server_port": {"type": "integer", "required": False, "default": 5555}
        }
    },
    "receive_from_router": {
        "function": receive_from_router,
        "description": "Receive messages from router (subscription-based)",
        "parameters": {
            "ea_name": {"type": "string", "required": True},
            "timeout_ms": {"type": "integer", "required": False, "default": 5000},
            "server_address": {"type": "string", "required": False, "default": "localhost"},
            "server_port": {"type": "integer", "required": False, "default": 5556}
        }
    },
    "register_ea": {
        "function": register_ea,
        "description": "Register EA with Strategy Router",
        "parameters": {
            "ea_name": {"type": "string", "required": True},
            "strategy_id": {"type": "string", "required": True},
            "magic": {"type": "integer", "required": True},
            "symbols": {"type": "array", "required": True},
            "server_address": {"type": "string", "required": False, "default": "localhost"},
            "server_port": {"type": "integer", "required": False, "default": 5555}
        }
    },
    "send_heartbeat": {
        "function": send_heartbeat,
        "description": "Send heartbeat to Strategy Router",
        "parameters": {
            "ea_name": {"type": "string", "required": True},
            "symbol": {"type": "string", "required": True},
            "magic": {"type": "integer", "required": True},
            "current_positions": {"type": "integer", "required": False, "default": 0},
            "server_address": {"type": "string", "required": False, "default": "localhost"},
            "server_port": {"type": "integer", "required": False, "default": 5555}
        }
    }
}


def get_zmq_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a ZMQ tool by name."""
    return ZMQ_TOOLS.get(name)


def list_zmq_tools() -> List[str]:
    """List all available ZMQ tools."""
    return list(ZMQ_TOOLS.keys())
