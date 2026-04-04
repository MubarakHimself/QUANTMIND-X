"""
SVSS Engine Module

asyncio service that connects to MT5 ZMQ tick feed and drives SVSS calculations.
Runs as a background service on Cloudzy alongside the live tick feed.

The engine:
1. Connects to MT5 ZMQ tick stream
2. On each tick, calls SVSSService.update_*() methods
3. Publishes combined readings to Redis at svss:{symbol}:readings (TTL 60s)
4. Detects session boundaries and resets accumulators

Usage:
    # Run as standalone service
    asyncio.run(SVSSEngine.run(symbols=["EURUSD", "GBPUSD"]))

    # Or use as a service
    engine = SVSSEngine(symbols=["EURUSD"])
    await engine.start()
"""

import asyncio
import json
import logging
import os
import signal
import sys
import zmq.asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

import zmq

logger = logging.getLogger(__name__)


def _resolve_redis_url(redis_url: Optional[str] = None) -> str:
    """Resolve Redis URL from explicit argument or deployment environment."""
    return redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

# ZMQ tick subscription filter
TICK_SUBSCRIPTION = "TICK"

# Session boundaries (GMT hours)
SESSION_BOUNDARIES = [8, 12, 13, 16, 21]

# RVOL warm storage path
WARM_STORAGE_PATH = "data/svss/warm_storage.db"


@dataclass
class TickMessage:
    """Normalized tick data from ZMQ stream."""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: datetime

    @property
    def spread_pips(self) -> float:
        """Calculate spread in pips."""
        return (self.ask - self.bid) * 10000

    @property
    def typical_price(self) -> float:
        """Calculate typical price: (High + Low + Close) / 3."""
        return (self.ask + self.bid + self.last) / 3.0


class SVSSEngine:
    """
    asyncio SVSS engine that consumes MT5 ZMQ ticks and drives SVSS calculations.

    Runs as a background service on Cloudzy. Publishes combined readings to Redis
    at svss:{symbol}:readings with 60s TTL.

    Features:
    - Multi-symbol support (runs engine per symbol)
    - Session boundary detection (resets accumulators on new session)
    - Graceful shutdown on SIGTERM/SIGINT
    - Auto-reconnect on ZMQ disconnection
    """

    def __init__(
        self,
        symbols: List[str],
        zmq_endpoint: str = "tcp://localhost:5555",
        redis_url: Optional[str] = None,
        warm_storage_path: str = WARM_STORAGE_PATH,
        session_boundaries: List[int] = None,
        poll_timeout_ms: int = 100,
    ):
        """
        Initialize SVSS Engine.

        Args:
            symbols: List of trading symbols to process
            zmq_endpoint: MT5 ZMQ tick stream endpoint
            redis_url: Redis connection URL
            warm_storage_path: Path to warm storage DuckDB for RVOL rolling averages
            session_boundaries: GMT hours that trigger session resets
            poll_timeout_ms: ZMQ poll timeout in milliseconds
        """
        self._symbols = [s.upper() for s in symbols]
        self._zmq_endpoint = zmq_endpoint
        self._redis_url = _resolve_redis_url(redis_url)
        self._warm_storage_path = warm_storage_path
        self._session_boundaries = session_boundaries or SESSION_BOUNDARIES
        self._poll_timeout_ms = poll_timeout_ms

        # ZMQ async context and socket
        self._zmq_context: Optional[zmq.asyncio.Context] = None
        self._zmq_socket: Optional[zmq.asyncio.Socket] = None
        self._connected = False

        # asyncio state
        self._running = False
        self._tasks: List[asyncio.Task] = []

        # Per-symbol state
        self._services: Dict[str, "SVSSService"] = {}
        self._session_managers: Dict[str, "SessionState"] = {}

        # Graceful shutdown event
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """
        Start the SVSS engine.

        Creates per-symbol SVSS services, connects to ZMQ and Redis,
        and starts the tick processing loop.
        """
        logger.info(f"Starting SVSS Engine for symbols: {self._symbols}")

        # Import here to avoid circular imports
        from src.risk.svss_service import SVSSService

        # Initialize services for each symbol
        for symbol in self._symbols:
            service = SVSSService(
                symbol=symbol,
                redis_url=self._redis_url,
            )
            if not service.connect():
                logger.error(f"Failed to connect service for {symbol}")
                continue
            self._services[symbol] = service

            # Initialize session state
            self._session_managers[symbol] = SessionState(
                last_known_hour=None,
                current_session_id=None,
            )

            # Load RVOL rolling average profile from warm storage
            await self._load_rvol_profile(symbol)

        if not self._services:
            logger.error("No symbols could be initialized")
            return

        # Connect to ZMQ
        await self._connect_zmq()

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        self._running = True

        try:
            await self._process_loop()
        except asyncio.CancelledError:
            logger.info("SVSS Engine cancelled")
        finally:
            await self._cleanup()

    async def _connect_zmq(self) -> None:
        """Connect to MT5 ZMQ tick stream."""
        self._zmq_context = zmq.asyncio.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.SUB)
        self._zmq_socket.connect(self._zmq_endpoint)
        self._zmq_socket.setsockopt_string(zmq.SUBSCRIBE, TICK_SUBSCRIPTION)
        self._connected = True
        logger.info(f"Connected to MT5 ZMQ at {self._zmq_endpoint}")

    async def _cleanup(self) -> None:
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up SVSS Engine")

        self._running = False

        # Disconnect ZMQ
        if self._zmq_socket:
            self._zmq_socket.close()
            self._zmq_socket = None
        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        # Disconnect services
        for service in self._services.values():
            service.disconnect()
        self._services.clear()

        self._connected = False
        logger.info("SVSS Engine cleanup complete")

    def _handle_shutdown(self) -> None:
        """Handle shutdown signals."""
        logger.info("Shutdown signal received")
        self._shutdown_event.set()
        self._running = False

    async def _load_rvol_profile(self, symbol: str) -> None:
        """
        Load RVOL rolling average profile from warm storage.

        This is called once per symbol on startup to load historical
        volume data for RVOL calculation.

        Args:
            symbol: Trading symbol
        """
        # Check if duckdb is available
        try:
            import duckdb

            service = self._services.get(symbol)
            if not service:
                return

            if not self._warm_storage_path:
                logger.warning(f"No warm storage path configured for {symbol}")
                return

            conn = duckdb.connect(self._warm_storage_path, read_only=True)

            # Load rolling average profile
            query = """
            WITH session_volumes AS (
                SELECT
                    session_id,
                    bucket_minute,
                    avg_volume,
                    ROW_NUMBER() OVER (PARTITION BY bucket_minute ORDER BY updated_at DESC) as rn
                FROM volume_profile_history
                WHERE symbol = ?
            ),
            recent_sessions AS (
                SELECT DISTINCT session_id
                FROM volume_profile_history
                WHERE symbol = ?
                ORDER BY session_id DESC
                LIMIT 20
            )
            SELECT
                sv.bucket_minute,
                AVG(sv.avg_volume) as rolling_avg_volume
            FROM session_volumes sv
            JOIN recent_sessions rs ON sv.session_id = rs.session_id
            WHERE sv.rn = 1
            GROUP BY sv.bucket_minute
            ORDER BY sv.bucket_minute;
            """

            result = conn.execute(query, [symbol, symbol])
            rows = result.fetchall()

            profile = {int(row[0]): float(row[1]) for row in rows}
            service.set_rvol_rolling_avg_profile(profile)

            conn.close()
            logger.info(f"Loaded RVOL profile for {symbol}: {len(profile)} buckets")

        except ImportError:
            logger.warning("DuckDB not available, RVOL will use default profile")
        except Exception as e:
            logger.error(f"Failed to load RVOL profile for {symbol}: {e}")

    async def _process_loop(self) -> None:
        """
        Main tick processing loop.

        Polls ZMQ for ticks, processes them, and publishes to Redis.
        """
        logger.info("Starting tick processing loop")

        poll_task = asyncio.create_task(self._zmq_poller())

        # Also create a periodic publish task (every 1 second)
        publish_task = asyncio.create_task(self._periodic_publisher())

        await asyncio.gather(poll_task, publish_task)

    async def _zmq_poller(self) -> None:
        """
        Poll ZMQ for ticks and process them.

        Runs continuously until shutdown.
        """
        if not self._zmq_socket:
            return

        while self._running:
            try:
                # Poll for message with timeout
                msg = await self._zmq_socket.recv_string(flags=zmq.NOBLOCK)
                tick = self._parse_tick(msg)

                if tick:
                    await self._process_tick(tick)

            except zmq.Again:
                # No message available, sleep briefly
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(f"Error polling ZMQ: {e}")
                if self._running:
                    await asyncio.sleep(1)  # Back off on error

    async def _periodic_publisher(self) -> None:
        """
        Periodically publish readings to Redis.

        Runs every 1 second to ensure consumers always have fresh data.
        """
        while self._running:
            await asyncio.sleep(1.0)

            for symbol, service in self._services.items():
                if not self._connected:
                    continue

                try:
                    # Check if session is active
                    readings = service.get_current_readings()
                    if readings is None:
                        continue

                    # Publish to Redis
                    service.publish_to_redis()

                except Exception as e:
                    logger.error(f"Failed to publish readings for {symbol}: {e}")

    def _parse_tick(self, message: str) -> Optional[TickMessage]:
        """
        Parse tick message from ZMQ stream.

        Expected format: "TICK {\"symbol\": \"EURUSD\", \"bid\": 1.08542, ...}"

        Args:
            message: Raw ZMQ message string

        Returns:
            TickMessage if parsing successful, None otherwise.
        """
        try:
            # Remove subscription filter prefix
            if message.startswith(TICK_SUBSCRIPTION):
                message = message[len(TICK_SUBSCRIPTION):].strip()

            data = json.loads(message)

            # Parse timestamp (milliseconds since epoch)
            ts_ms = data.get("timestamp", 0)
            if ts_ms:
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            else:
                timestamp = datetime.now(timezone.utc)

            return TickMessage(
                symbol=data.get("symbol", "").upper(),
                bid=float(data.get("bid", 0)),
                ask=float(data.get("ask", 0)),
                last=float(data.get("last", data.get("bid", 0))),
                volume=float(data.get("volume", 0)),
                timestamp=timestamp,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse tick: {e}")
            return None

    async def _process_tick(self, tick: TickMessage) -> None:
        """
        Process a single tick through all SVSS calculations.

        Args:
            tick: Parsed tick data
        """
        # Ignore ticks for symbols we're not tracking
        if tick.symbol not in self._services:
            return

        service = self._services[tick.symbol]
        session_state = self._session_managers[tick.symbol]

        # Check for session boundary
        new_session = self._check_session_boundary(
            symbol=tick.symbol,
            timestamp=tick.timestamp,
            session_state=session_state,
        )

        if new_session:
            # Generate session ID
            session_id = f"{new_session}_{tick.timestamp.strftime('%Y%m%d_%H%M')}"
            service.reset_session(session_id, tick.timestamp)
            session_state.current_session_id = session_id

            # Reload RVOL profile for new session
            await self._load_rvol_profile(tick.symbol)

            logger.info(f"New session started for {tick.symbol}: {session_id}")

        # Update all indicators
        service.update_vwap(
            price=tick.last,
            volume=tick.volume,
            timestamp=tick.timestamp,
        )

        service.update_volume_profile(
            price=tick.last,
            volume=tick.volume,
        )

        service.update_mfi(
            bid=tick.bid,
            ask=tick.ask,
            last=tick.last,
            volume=tick.volume,
            timestamp=tick.timestamp,
        )

        # RVOL update (most accurate on bar close, but we update on each tick)
        service.update_rvol(
            volume=tick.volume,
            timestamp=tick.timestamp,
        )

    def _check_session_boundary(
        self,
        symbol: str,
        timestamp: datetime,
        session_state: "SessionState",
    ) -> Optional[str]:
        """
        Check if timestamp crosses a session boundary.

        Args:
            symbol: Trading symbol
            timestamp: Current tick timestamp
            session_state: Current session state

        Returns:
            Session type string if new session detected, None otherwise.
        """
        current_hour = timestamp.hour
        current_minute = timestamp.minute

        # Check if we've crossed a boundary
        if session_state.last_known_hour is not None:
            for boundary in self._session_boundaries:
                if (current_hour >= boundary and
                    session_state.last_known_hour < boundary and
                    current_minute < 5):  # Within first 5 minutes
                    session_state.last_known_hour = current_hour
                    return self._session_type_from_hour(boundary)

        session_state.last_known_hour = current_hour

        # Check if we need to initialize session
        if session_state.current_session_id is None:
            # Find the applicable session type for current hour
            applicable_boundary = 0
            for boundary in sorted(self._session_boundaries, reverse=True):
                if current_hour >= boundary:
                    applicable_boundary = boundary
                    break
            return self._session_type_from_hour(applicable_boundary)

        return None

    @staticmethod
    def _session_type_from_hour(hour: int) -> str:
        """Get session type from GMT hour."""
        session_types = {
            8: "london_open",
            12: "ny_am",
            13: "lunch",
            16: "ny_close",
            21: "late_close",
        }
        return session_types.get(hour, "unknown")

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._running

    @property
    def connected_symbols(self) -> List[str]:
        """Get list of symbols currently being processed."""
        return list(self._services.keys())


@dataclass
class SessionState:
    """Internal state for session tracking."""

    last_known_hour: Optional[int]
    current_session_id: Optional[str]


async def run(
    symbols: List[str],
    zmq_endpoint: str = "tcp://localhost:5555",
    redis_url: Optional[str] = None,
) -> None:
    """
    Run SVSS Engine as a standalone service.

    Args:
        symbols: List of trading symbols to process
        zmq_endpoint: MT5 ZMQ tick stream endpoint
        redis_url: Redis connection URL
    """
    engine = SVSSEngine(
        symbols=symbols,
        zmq_endpoint=zmq_endpoint,
        redis_url=redis_url,
    )

    await engine.start()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Example: run for EURUSD and GBPUSD
    symbols = ["EURUSD", "GBPUSD"]
    zmq_endpoint = "tcp://localhost:5555"
    redis_url = _resolve_redis_url()

    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
    if len(sys.argv) > 2:
        zmq_endpoint = sys.argv[2]
    if len(sys.argv) > 3:
        redis_url = sys.argv[3]

    asyncio.run(run(symbols=symbols, zmq_endpoint=zmq_endpoint, redis_url=redis_url))
