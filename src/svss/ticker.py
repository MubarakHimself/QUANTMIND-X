"""
MT5 ZMQ Tick Subscription Handler

Subscribes to the MT5 ZMQ tick stream and converts ticks to TickData format.
"""

import logging
import zmq
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class TickData:
    """
    Normalized tick data from MT5 ZMQ stream.

    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD')
        bid: Bid price
        ask: Ask price
        last: Last (deal) price
        volume: Tick volume
        timestamp: Tick timestamp (UTC)
        raw_data: Original tick data if needed
    """

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime
    raw_data: Optional[dict] = None

    @property
    def typical_price(self) -> float:
        """
        Calculate typical price: (High + Low + Close) / 3.

        Note: For MT5 ZMQ tick data, we don't have separate high/low values.
        We use bid as low estimate and ask as high estimate.
        This is an approximation - true high/low would require OHLC data.
        """
        # Use bid as low, ask as high, last as close
        high = self.ask
        low = self.bid
        close = self.last
        return (high + low + close) / 3

    @property
    def spread(self) -> float:
        """Calculate spread in pips."""
        return (self.ask - self.bid) * 10000  # Convert to pips for forex


class SVSSTicker:
    """
    MT5 ZMQ tick subscription handler for SVSS.

    Connects to the MT5 ZMQ tick stream and converts ticks to normalized TickData.
    """

    def __init__(
        self,
        zmq_endpoint: str = "tcp://localhost:5555",
        subscription_filter: str = "TICK",
    ):
        """
        Initialize SVSSTicker.

        Args:
            zmq_endpoint: ZMQ endpoint for MT5 tick stream
            subscription_filter: ZMQ subscription filter (default: 'TICK')
        """
        self._zmq_endpoint = zmq_endpoint
        self._subscription_filter = subscription_filter
        self._context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._connected = False
        self._tick_callback: Optional[Callable[[TickData], None]] = None

    def connect(self) -> bool:
        """
        Establish connection to MT5 ZMQ tick stream.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self._zmq_endpoint)
            self._socket.setsockopt_string(zmq.SUBSCRIBE, self._subscription_filter)
            self._connected = True

            logger.info(
                f"SVSS Ticker connected to {self._zmq_endpoint}",
                extra={"endpoint": self._zmq_endpoint},
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect to MT5 ZMQ stream: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close ZMQ connection."""
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()
        self._connected = False
        logger.info("SVSS Ticker disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if ticker is connected."""
        return self._connected

    def set_tick_callback(self, callback: Callable[[TickData], None]) -> None:
        """
        Set callback to be called on each tick.

        Args:
            callback: Function that accepts TickData
        """
        self._tick_callback = callback

    def poll(self) -> Optional[TickData]:
        """
        Poll for a single tick.

        Non-blocking poll - returns None if no tick available.

        Returns:
            TickData if available, None otherwise.
        """
        if not self._connected or not self._socket:
            return None

        try:
            message = self._socket.recv_string(flags=zmq.NOBLOCK)
            tick_data = self._parse_tick(message)

            if tick_data and self._tick_callback:
                self._tick_callback(tick_data)

            return tick_data

        except zmq.Again:
            # No message available
            return None
        except Exception as e:
            logger.error(f"Error polling tick: {e}")
            return None

    def _parse_tick(self, message: str) -> Optional[TickData]:
        """
        Parse tick message from ZMQ stream.

        Expected format: "TICK {\"symbol\": \"EURUSD\", \"bid\": 1.08542, ...}"

        Args:
            message: Raw ZMQ message string

        Returns:
            TickData if parsing successful, None otherwise.
        """
        try:
            # Remove the subscription filter prefix if present
            if message.startswith(self._subscription_filter):
                message = message[len(self._subscription_filter) :].strip()

            data = json.loads(message)

            # Parse timestamp - MT5 sends as milliseconds since epoch
            ts_ms = data.get("timestamp", 0)
            if ts_ms:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            return TickData(
                symbol=data.get("symbol", "").upper(),
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                last=float(data.get("last", data.get("bid", 0))),
                volume=float(data.get("volume", 0)),
                timestamp=timestamp,
                raw_data=data,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse tick message: {e}")
            return None

    def reconnect(self) -> bool:
        """
        Attempt to reconnect after disconnection.

        Returns:
            True if reconnection successful, False otherwise.
        """
        logger.info("Attempting to reconnect SVSS Ticker...")
        self.disconnect()
        return self.connect()
