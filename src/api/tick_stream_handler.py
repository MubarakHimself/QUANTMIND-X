"""
Live Tick Data Streaming Handler

Streams real-time tick data from MT5 to UI via WebSocket broadcasting.
Supports both ZMQ-based real-time streaming (<5ms latency) and polling fallback.
"""

import asyncio
import os
import time
import logging
import yaml
from typing import Dict, Set, Optional, Literal
from datetime import datetime, timezone, timedelta
import json

try:
    import zmq
    import zmq.asyncio
    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
from src.api.websocket_endpoints import manager
from src.api.websocket_endpoints import (
    broadcast_position_update,
    broadcast_pnl_update,
    broadcast_bridge_status,
    get_trading_broadcaster
)
from src.database.db_manager import DBManager
from src.database.models import TickCache, SymbolSubscription
from src.monitoring.prometheus_exporter import (
    tick_stream_rate, tick_stream_latency_ms, tick_stream_drops,
    symbol_subscriptions, circuit_breaker_state, circuit_breaker_errors
)

# Import adaptive throttler for resource-aware streaming (optional - guard import)
try:
    from src.data.adaptive_throttler import AdaptiveThrottler, DegradationLevel
    ADAPTIVE_THROTTLER_AVAILABLE = True
except ImportError:
    AdaptiveThrottler = None
    DegradationLevel = None
    ADAPTIVE_THROTTLER_AVAILABLE = False

logger = logging.getLogger(__name__)

# HOT tier retention: 1 hour
TICK_RETENTION_HOURS = 1

# Demo tick configuration from environment
USE_DEMO_TICKS = os.environ.get('USE_DEMO_TICKS', 'false').lower() == 'true'


def get_tick_source() -> Literal['demo', 'live']:
    """
    Determine the tick data source based on configuration.

    When USE_DEMO_TICKS=true, tick data is sourced from a separate demo connection
    to prevent broker manipulation of live data.

    Returns:
        'demo' if demo ticks are enabled, 'live' otherwise
    """
    return 'demo' if USE_DEMO_TICKS else 'live'


def load_streaming_config() -> Dict:
    """Load streaming configuration from YAML file."""
    try:
        with open('config/streaming.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning("streaming.yaml not found, using defaults")
        return {
            'zmq_enabled': False,
            'zmq_port': 5555,
            'polling_interval_ms': 100,
            'enable_fallback': True,
            'max_tick_age_ms': 100,
            'validate_sequence': True
        }


class TickStreamHandler:
    def __init__(
        self,
        mt5_adapter: MT5SocketAdapter,
        db_manager: Optional[DBManager] = None,
        subscription_manager=None,
        circuit_breaker=None,
        resource_monitor=None,
        adaptive_throttler=None
    ):
        self.mt5_adapter = mt5_adapter
        self.db = db_manager or DBManager()
        self.subscription_manager = subscription_manager
        self.circuit_breaker = circuit_breaker
        self.resource_monitor = resource_monitor
        self.adaptive_throttler = adaptive_throttler

        self.subscribed_symbols: Set[str] = set()
        self.is_streaming = False
        self.streaming_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self.throttle_interval = 0.1
        self.last_update_time: Dict[str, float] = {}

        # Load configuration
        self._config = load_streaming_config()

        # Sequence tracking for tick validation
        self._last_sequence: Dict[str, int] = {}

        # ZMQ support
        self._zmq_enabled = self._config.get('zmq_enabled', False) and ZMQ_AVAILABLE
        self._zmq_socket = None
        self._zmq_context = None
        self._using_zmq = False

        # Tick validation settings
        self._max_tick_age_ms = self._config.get('max_tick_age_ms', 100)
        self._validate_sequence = self._config.get('validate_sequence', True)

        # Fallback settings
        self._enable_fallback = self._config.get('enable_fallback', True)
        self._zmq_port = self._config.get('zmq_port', 5555)

        # Latency tracking
        self._tick_latencies: list = []

        # Tick rate tracking for metrics
        self._tick_count = 0
        self._tick_count_start = time.time()

        # Track bot subscriptions for SymbolSubscription table (legacy, for backwards compat)
        # Format: {symbol: {bot_id: priority}}
        self._bot_subscriptions: Dict[str, Dict[str, int]] = {}

    def set_subscription_manager(self, manager):
        """Set the subscription manager reference."""
        self.subscription_manager = manager

    def set_circuit_breaker(self, circuit_breaker):
        """Set the circuit breaker reference."""
        self.circuit_breaker = circuit_breaker

    def set_resource_monitor(self, monitor):
        """Set the resource monitor reference."""
        self.resource_monitor = monitor

    def set_adaptive_throttler(self, throttler):
        """Set the adaptive throttler reference for resource-aware streaming."""
        self.adaptive_throttler = throttler
        # Wire up the throttler to call back when polling interval changes
        if throttler:
            throttler.set_tick_handler(self)
            throttler.set_actuation_callbacks(
                on_adjust_polling=self._on_throttler_adjust_polling
            )

    async def _on_throttler_adjust_polling(self, interval: float):
        """Callback from throttler when polling interval should change."""
        self.throttle_interval = interval
        logger.info(f"Throttler adjusted polling interval to {interval}s")

    async def _drop_symbol_subscription(self, symbol: str):
        """Drop a symbol subscription due to throttling/resource limits."""
        # Find bots subscribed to this symbol and remove them
        # Prioritize removing PAPER > DEMO > LIVE
        if symbol in self._bot_subscriptions:
            # Sort by priority (higher number = lower priority)
            bots_by_priority = sorted(
                self._bot_subscriptions[symbol].items(),
                key=lambda x: x[1],  # Sort by priority value
                reverse=True  # Higher priority value (PAPER=3) first
            )

            # Remove the lowest priority bot's subscription
            if bots_by_priority:
                bot_id, priority = bots_by_priority[0]
                await self.unsubscribe(symbol, bot_id)
                logger.debug(f"Dropped {symbol} subscription for bot {bot_id} (priority: {priority})")

    async def subscribe(
        self,
        symbol: str,
        bot_id: str = "default",
        priority: int = 3,
        timeframe: str = "M1"
    ):
        """
        Subscribe to tick data for a symbol.

        Args:
            symbol: Trading symbol to subscribe to
            bot_id: Bot identifier for subscription tracking (default: "default")
            priority: Priority level (1=LIVE, 2=DEMO, 3=PAPER)
            timeframe: Timeframe for the subscription (default: M1)
        """
        self.subscribed_symbols.add(symbol)
        # Initialize sequence tracking for new symbol
        if symbol not in self._last_sequence:
            self._last_sequence[symbol] = -1

        # Track bot subscription in memory (legacy)
        if symbol not in self._bot_subscriptions:
            self._bot_subscriptions[symbol] = {}
        self._bot_subscriptions[symbol][bot_id] = priority

        # Use subscription manager if available
        if self.subscription_manager:
            from src.data.symbol_subscription_manager import BotPriority
            try:
                bot_priority = BotPriority(priority)
            except ValueError:
                bot_priority = BotPriority.PAPER
            await self.subscription_manager.subscribe(
                bot_id=bot_id,
                symbol=symbol,
                timeframe=timeframe,
                priority=bot_priority
            )
        else:
            # Fallback to direct DB update
            await self._update_symbol_subscription(symbol, bot_id, priority, timeframe, active=True)

        # Update Prometheus metrics
        priority_name = {1: "LIVE", 2: "DEMO", 3: "PAPER"}.get(priority, "PAPER")
        symbol_subscriptions.labels(symbol=symbol, priority=priority_name).inc()

        if not self.is_streaming and self.subscribed_symbols:
            await self.start_streaming()
        logger.info(f"Subscribed to tick data for {symbol} (bot: {bot_id}, priority: {priority}, tf: {timeframe})")

    async def unsubscribe(self, symbol: str, bot_id: str = "default", timeframe: str = "M1"):
        """
        Unsubscribe from tick data for a symbol.

        Args:
            symbol: Trading symbol to unsubscribe from
            bot_id: Bot identifier to unsubscribe
            timeframe: Timeframe for the subscription (default: M1)
        """
        # Get the stored priority before removing from _bot_subscriptions
        # Fall back to PAPER (3) if not found
        stored_priority = 3  # Default to PAPER
        if symbol in self._bot_subscriptions:
            stored_priority = self._bot_subscriptions[symbol].get(bot_id, 3)
            # Only remove this bot's subscription, keep the symbol if other bots remain
            self._bot_subscriptions[symbol].pop(bot_id, None)
            if not self._bot_subscriptions[symbol]:
                # No more bots subscribed to this symbol, clean up
                del self._bot_subscriptions[symbol]
                self.subscribed_symbols.discard(symbol)
                # Clean up sequence tracking only when no bots remain
                self._last_sequence.pop(symbol, None)

        # Use subscription manager if available
        if self.subscription_manager:
            await self.subscription_manager.unsubscribe(
                bot_id=bot_id,
                symbol=symbol,
                timeframe=timeframe
            )
        else:
            # Fallback to direct DB update
            await self._update_symbol_subscription(symbol, bot_id, 0, timeframe, active=False)

        # Update Prometheus metrics with the stored priority
        priority_name = {1: "LIVE", 2: "DEMO", 3: "PAPER"}.get(stored_priority, "PAPER")
        symbol_subscriptions.labels(symbol=symbol, priority=priority_name).dec()

        # Only stop streaming when no symbols have any bot subscriptions
        if not self._bot_subscriptions and self.is_streaming:
            await self.stop_streaming()
        logger.info(f"Unsubscribed from tick data for {symbol} (bot: {bot_id}, tf: {timeframe})")

    async def start_streaming(self):
        if self.is_streaming:
            return

        # Try to initialize ZMQ if enabled
        if self._zmq_enabled:
            await self._init_zmq()

        self.is_streaming = True
        self.streaming_task = asyncio.create_task(self._stream_loop())

        # Start the cleanup task for stale ticks
        self._cleanup_task = asyncio.create_task(self._cleanup_stale_ticks())

    async def _init_zmq(self):
        """Initialize ZMQ subscriber socket."""
        try:
            self._zmq_context = zmq.asyncio.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.SUB)
            self._zmq_socket.connect(f"tcp://localhost:{self._zmq_port}")
            self._zmq_socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._using_zmq = True
            logger.info(f"ZMQ subscriber connected to tcp://localhost:{self._zmq_port}")
        except Exception as e:
            logger.warning(f"Failed to initialize ZMQ: {e}. Falling back to polling.")
            self._using_zmq = False
            if self._zmq_socket:
                try:
                    self._zmq_socket.close()
                except:
                    pass
                self._zmq_socket = None

    async def stop_streaming(self):
        if not self.is_streaming:
            return
        self.is_streaming = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clean up ZMQ
        if self._zmq_socket:
            try:
                self._zmq_socket.close()
            except:
                pass
            self._zmq_socket = None
        if self._zmq_context:
            try:
                self._zmq_context.term()
            except:
                pass
            self._zmq_context = None

    async def _stream_loop(self):
        """Main streaming loop - uses ZMQ if available, otherwise polling."""
        # Check circuit breaker state before deciding mode
        if self.circuit_breaker:
            from src.data.tick_stream_circuit_breaker import CircuitState
            if self.circuit_breaker.is_open():
                # Circuit is open, force polling mode
                logger.warning("Circuit breaker open, using polling mode")
                await self._polling_stream_loop()
                return

        # Check adaptive throttler state for resource-aware streaming
        if self.adaptive_throttler:
            await self._apply_throttler_state()

        if self._using_zmq and self._zmq_socket:
            await self._zmq_stream_loop()
        else:
            await self._polling_stream_loop()

    async def _apply_throttler_state(self):
        """Apply adaptive throttler state to streaming behavior."""
        if not self.adaptive_throttler:
            return

        # Update throttler with current resource metrics
        if self.resource_monitor:
            await self.adaptive_throttler.update()

        # Get current degradation level
        level = self.adaptive_throttler.get_current_level()

        # Get allowed symbols based on current level
        allowed_symbols = self.adaptive_throttler.get_allowed_symbols()

        # Adjust throttle interval based on level
        self.throttle_interval = self.adaptive_throttler.get_polling_interval()

        # If in critical/emergency level, drop subscriptions for non-allowed symbols
        # Prioritize LIVE > DEMO > PAPER
        if level >= 1:  # WARNING or higher
            current_symbols = set(self.subscribed_symbols)
            symbols_to_remove = current_symbols - allowed_symbols

            if symbols_to_remove:
                logger.warning(
                    f"Throttler level {level.name}: Dropping {len(symbols_to_remove)} "
                    f"lower-priority symbols to conserve resources"
                )
                # Remove lower-priority subscriptions
                for symbol in symbols_to_remove:
                    await self._drop_symbol_subscription(symbol)

        # Force polling mode in critical/emergency levels
        if ADAPTIVE_THROTTLER_AVAILABLE and DegradationLevel is not None:
            if level >= DegradationLevel.CRITICAL:
                if self._using_zmq:
                    logger.warning("Forcing polling mode due to high resource usage")
                    self._using_zmq = False

    async def _zmq_stream_loop(self):
        """ZMQ-based streaming loop for <5ms latency."""
        while self.is_streaming:
            # Check circuit breaker state
            if self.circuit_breaker:
                from src.data.tick_stream_circuit_breaker import CircuitState
                state = self.circuit_breaker.get_state()
                if state == CircuitState.OPEN:
                    logger.info("Circuit breaker open, switching to polling")
                    await self._polling_stream_loop()
                    break
                # In half-open, probe ZMQ but be ready to fall back

            try:
                # Non-blocking receive with timeout
                message = await asyncio.wait_for(
                    self._zmq_socket.recv_string(),
                    timeout=0.1
                )

                if message:
                    await self._process_zmq_message(message)

            except asyncio.TimeoutError:
                # No message received, continue
                pass
            except Exception as e:
                logger.error(f"ZMQ receive error: {e}")

                # Record error with circuit breaker
                if self.circuit_breaker:
                    await self.circuit_breaker.record_error(e)

                # Try to reconnect if using ZMQ failed
                if self._enable_fallback:
                    logger.info("Switching to polling fallback")
                    self._using_zmq = False
                    # Notify circuit breaker of ZMQ disconnect
                    if self.circuit_breaker:
                        self.circuit_breaker.notify_zmq_disconnected()
                    await self._polling_stream_loop()
                    break
                else:
                    await asyncio.sleep(1)

    async def _process_zmq_message(self, message: str):
        """Process incoming ZMQ message."""
        try:
            data = json.loads(message)
            msg_type = data.get('type', '')
            
            if msg_type == 'tick':
                await self._handle_zmq_tick(data)
            elif msg_type == 'heartbeat':
                logger.debug(f"Received heartbeat from MT5")
            elif msg_type == 'disconnect':
                logger.warning("MT5 streamer disconnected")
                
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in ZMQ message: {e}")
        except Exception as e:
            logger.error(f"Error processing ZMQ message: {e}")

    async def _handle_zmq_tick(self, data: Dict):
        """Handle incoming tick from ZMQ."""
        symbol = data.get('symbol')
        if not symbol or symbol not in self.subscribed_symbols:
            return

        method = "zmq"

        # Track latency
        tick_time_ms = data.get('time_msc', 0)
        if tick_time_ms:
            latency_ms = (time.time() * 1000) - tick_time_ms
            self._tick_latencies.append(latency_ms)
            # Keep last 1000 latencies
            if len(self._tick_latencies) > 1000:
                self._tick_latencies = self._tick_latencies[-1000:]

            # Record latency to Prometheus
            tick_stream_latency_ms.labels(symbol=symbol).observe(latency_ms)

            # Report latency to circuit breaker
            if self.circuit_breaker:
                await self.circuit_breaker.record_latency(latency_ms)

        # Track tick rate for Prometheus
        self._tick_count += 1
        current_time = time.time()
        elapsed = current_time - self._tick_count_start
        if elapsed > 0:
            rate = self._tick_count / elapsed
            tick_stream_rate.labels(symbol=symbol, method=method).set(rate)

        # Record tick with resource monitor
        if self.resource_monitor:
            self.resource_monitor.record_tick(symbol)

        # Validate tick
        is_valid, error_msg = self._validate_tick(symbol, data)
        if not is_valid:
            logger.debug(f"Tick validation failed for {symbol}: {error_msg}")
            # Record drop with reason
            reason = "validation_error"
            if "old" in error_msg.lower():
                reason = "stale"
            elif "order" in error_msg.lower():
                reason = "out_of_order"
            tick_stream_drops.labels(symbol=symbol, reason=reason).inc()

            # Record error with circuit breaker
            if self.circuit_breaker:
                await self.circuit_breaker.record_error(Exception(error_msg))
            return

        # Record success with circuit breaker
        if self.circuit_breaker:
            await self.circuit_breaker.record_success()

        # Build message
        bid = data.get('bid', 0.0)
        ask = data.get('ask', 0.0)
        spread = ask - bid
        sequence = data.get('sequence', 0)

        # Convert timestamp
        if tick_time_ms:
            tick_timestamp = datetime.fromtimestamp(tick_time_ms / 1000.0, tz=timezone.utc)
            timestamp = tick_timestamp.isoformat(timespec='milliseconds') + 'Z'
        else:
            tick_timestamp = datetime.now(timezone.utc)
            timestamp = tick_timestamp.isoformat(timespec='milliseconds') + 'Z'

        # Persist validated tick to TickCache (HOT tier)
        await self._persist_tick_to_cache(
            symbol=symbol,
            bid=bid,
            ask=ask,
            tick_timestamp=tick_timestamp,
            sequence=sequence
        )

        message = {
            "type": "tick_data",
            "data": {
                "symbol": symbol,
                "bid": round(bid, 5),
                "ask": round(ask, 5),
                "spread": round(spread, 5),
                "timestamp": timestamp,
                "time_msc": tick_time_ms,
                "source": method
            }
        }

        await manager.broadcast(message, topic="tick_data")
        self.last_update_time[symbol] = time.time()

    async def _polling_stream_loop(self):
        """Polling-based streaming loop (fallback)."""
        while self.is_streaming:
            # Check circuit breaker - in half-open, probe for recovery
            if self.circuit_breaker:
                from src.data.tick_stream_circuit_breaker import CircuitState
                state = self.circuit_breaker.get_state()
                if state == CircuitState.HALF_OPEN:
                    # Probe mode - try one tick to test recovery
                    pass
                await self.circuit_breaker.check_auto_reset()

            # Check adaptive throttler for resource-aware polling
            if self.adaptive_throttler:
                await self._apply_throttler_state()

            # Get symbols to process - honor throttler limits
            symbols_to_process = self.subscribed_symbols
            if self.adaptive_throttler:
                allowed = self.adaptive_throttler.get_allowed_symbols()
                symbols_to_process = self.subscribed_symbols & allowed

            tasks = [self._fetch_and_broadcast_tick(symbol) for symbol in list(symbols_to_process)]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.throttle_interval)

    def _validate_tick(self, symbol: str, tick_data: Dict) -> tuple[bool, str]:
        """
        Validate tick data for freshness and sequence.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        current_time_ms = time.time() * 1000
        
        # Extract timestamp from tick data
        tick_time_ms = tick_data.get('time_msc', 0)
        if tick_time_ms == 0:
            # Fallback to current time if time_msc not available
            tick_time_ms = current_time_ms
        
        # Check staleness - reject ticks older than 100ms
        tick_age_ms = current_time_ms - tick_time_ms
        if tick_age_ms > self._max_tick_age_ms:
            return False, f"Tick too old: {tick_age_ms:.0f}ms"
        
        # Validate sequence number to detect out-of-order ticks
        if self._validate_sequence:
            sequence = tick_data.get('sequence', 0)
            last_sequence = self._last_sequence.get(symbol, -1)
            
            # Allow sequence 0 or sequences > last (normal flow)
            # Reject if sequence < last (out of order)
            if sequence > 0 and sequence <= last_sequence:
                return False, f"Out-of-order tick: seq {sequence} <= last {last_sequence}"
            
            # Update last sequence
            self._last_sequence[symbol] = sequence
        
        return True, ""

    async def _fetch_and_broadcast_tick(self, symbol: str):
        method = "polling"
        now = time.time()
        if symbol in self.last_update_time and (now - self.last_update_time[symbol]) < self.throttle_interval:
            return

        try:
            order_book = await self.mt5_adapter.get_order_book(symbol)
            bid = order_book['bids'][0][0] if order_book['bids'] else 0.0
            ask = order_book['asks'][0][0] if order_book['asks'] else 0.0
            spread = ask - bid

            # Get actual MT5 tick timestamp with millisecond precision
            # Use time_msc for precise timestamp (microseconds since epoch)
            tick_timestamp_ms = order_book.get('time_msc', 0)
            sequence = order_book.get('sequence', 0)
            if tick_timestamp_ms == 0:
                # Fallback to current time if not available
                tick_timestamp = datetime.now(timezone.utc)
            else:
                # Convert milliseconds to datetime
                tick_timestamp = datetime.fromtimestamp(
                    tick_timestamp_ms / 1000.0,
                    tz=timezone.utc
                )
                # Track latency
                latency_ms = (time.time() * 1000) - tick_timestamp_ms
                self._tick_latencies.append(latency_ms)
                if len(self._tick_latencies) > 1000:
                    self._tick_latencies = self._tick_latencies[-1000:]

                # Record latency to Prometheus
                tick_stream_latency_ms.labels(symbol=symbol).observe(latency_ms)

                # Report latency to circuit breaker
                if self.circuit_breaker:
                    await self.circuit_breaker.record_latency(latency_ms)

            # Track tick rate for Prometheus
            self._tick_count += 1
            current_time = time.time()
            elapsed = current_time - self._tick_count_start
            if elapsed > 0:
                rate = self._tick_count / elapsed
                tick_stream_rate.labels(symbol=symbol, method=method).set(rate)

            # Record tick with resource monitor
            if self.resource_monitor:
                self.resource_monitor.record_tick(symbol)

            # Prepare tick data for validation
            tick_data = {
                'symbol': symbol,
                'time_msc': tick_timestamp_ms,
                'sequence': sequence,
                'bid': bid,
                'ask': ask
            }

            # Validate tick
            is_valid, error_msg = self._validate_tick(symbol, tick_data)
            if not is_valid:
                logger.debug(f"Tick validation failed for {symbol}: {error_msg}")
                # Record drop with reason
                reason = "validation_error"
                if "old" in error_msg.lower():
                    reason = "stale"
                elif "order" in error_msg.lower():
                    reason = "out_of_order"
                tick_stream_drops.labels(symbol=symbol, reason=reason).inc()

                # Record error with circuit breaker
                if self.circuit_breaker:
                    await self.circuit_breaker.record_error(Exception(error_msg))
                return

            # Record success with circuit breaker
            if self.circuit_breaker:
                await self.circuit_breaker.record_success()

            # Persist validated tick to TickCache (HOT tier)
            await self._persist_tick_to_cache(
                symbol=symbol,
                bid=bid,
                ask=ask,
                tick_timestamp=tick_timestamp,
                sequence=sequence
            )

            # Format timestamp with millisecond precision
            timestamp = tick_timestamp.isoformat(timespec='milliseconds') + 'Z'

            message = {
                "type": "tick_data",
                "data": {
                    "symbol": symbol,
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "spread": round(spread, 5),
                    "timestamp": timestamp,
                    "time_msc": tick_timestamp_ms,
                    "source": method
                }
            }
            await manager.broadcast(message, topic="tick_data")
            self.last_update_time[symbol] = now
        except Exception as e:
            logger.error(f"Failed to fetch/broadcast tick for {symbol}: {e}")
            # Record error with circuit breaker
            if self.circuit_breaker:
                await self.circuit_breaker.record_error(e)
            # Record drop
            tick_stream_drops.labels(symbol=symbol, reason="fetch_error").inc()

    # =============================================================================
    # Position & P&L Broadcasting Integration (Story 3-1)
    # =============================================================================

    async def broadcast_position_change(
        self,
        ticket: int,
        symbol: str,
        volume: float,
        open_price: float,
        current_price: float,
        profit: float,
        action: str = "open",
        **kwargs
    ):
        """
        Broadcast a position update event.

        This method integrates the trading data broadcast with the tick stream.
        Call this when positions are opened, modified, or closed.

        Args:
            ticket: MT5 position ticket
            symbol: Trading symbol
            volume: Position volume in lots
            open_price: Open price
            current_price: Current market price
            profit: Current profit
            action: One of 'open', 'modify', 'close'
            **kwargs: Additional position data
        """
        position_data = {
            "ticket": ticket,
            "symbol": symbol,
            "volume": volume,
            "open_price": open_price,
            "current_price": current_price,
            "profit": profit,
            "action": action,
            **kwargs
        }

        # Broadcast the position update
        await broadcast_position_update(position_data)

        # Also update the local cache for state replay
        broadcaster = get_trading_broadcaster()
        await broadcaster.cache_position(position_data)

    async def broadcast_pnl_change(
        self,
        daily_pnl: float,
        open_positions: int,
        open_pnl: float = 0.0,
        closed_pnl: float = 0.0,
        equity: float = 0.0,
        balance: float = 0.0,
        **kwargs
    ):
        """
        Broadcast a P&L update event.

        Call this when P&L calculations are updated.

        Args:
            daily_pnl: Total daily P&L
            open_positions: Number of open positions
            open_pnl: P&L from open positions
            closed_pnl: P&L from closed positions
            equity: Current equity
            balance: Account balance
            **kwargs: Additional P&L data
        """
        pnl_data = {
            "daily_pnl": daily_pnl,
            "open_positions": open_positions,
            "open_pnl": open_pnl,
            "closed_pnl": closed_pnl,
            "equity": equity,
            "balance": balance,
            **kwargs
        }

        # Broadcast the P&L update
        await broadcast_pnl_update(pnl_data)

        # Also update the local cache
        broadcaster = get_trading_broadcaster()
        await broadcaster.cache_pnl(pnl_data)

    async def report_bridge_status(self, connected: bool, latency_ms: float = None):
        """
        Report MT5 bridge connection status.

        Call this to broadcast bridge status changes.

        Args:
            connected: Whether the MT5 bridge is connected
            latency_ms: Current latency in milliseconds
        """
        message = "Connected to MT5" if connected else "ZMQ connection lost"
        await broadcast_bridge_status(
            connected=connected,
            latency_ms=latency_ms,
            message=message
        )

    def get_stats(self) -> Dict:
        """Get streaming statistics."""
        avg_latency = sum(self._tick_latencies) / len(self._tick_latencies) if self._tick_latencies else 0
        return {
            'zmq_enabled': self._zmq_enabled,
            'using_zmq': self._using_zmq,
            'subscribed_symbols': list(self.subscribed_symbols),
            'avg_latency_ms': round(avg_latency, 2),
            'tick_count': len(self._tick_latencies)
        }

    async def _update_symbol_subscription(
        self,
        symbol: str,
        bot_id: str,
        priority: int,
        timeframe: str = "M1",
        active: bool = True
    ):
        """
        Update SymbolSubscription table when bots subscribe/unsubscribe.

        Args:
            symbol: Trading symbol
            bot_id: Bot identifier
            priority: Priority level (1=LIVE, 2=DEMO, 3=PAPER, 0=unsubscribed)
            timeframe: Timeframe for the subscription
            active: Whether subscription is active
        """
        try:
            with self.db.get_session() as session:
                if active:
                    # Upsert subscription
                    existing = session.query(SymbolSubscription).filter_by(
                        symbol=symbol,
                        timeframe=timeframe,
                        bot_id=bot_id
                    ).first()

                    if existing:
                        existing.priority = priority
                        existing.subscribed_at = datetime.now(timezone.utc)
                    else:
                        subscription = SymbolSubscription(
                            symbol=symbol,
                            timeframe=timeframe,
                            bot_id=bot_id,
                            priority=priority
                        )
                        session.add(subscription)
                else:
                    # Remove subscription
                    session.query(SymbolSubscription).filter_by(
                        symbol=symbol,
                        timeframe=timeframe,
                        bot_id=bot_id
                    ).delete()

                session.commit()
        except Exception as e:
            logger.error(f"Failed to update subscription for {symbol}/{timeframe}/{bot_id}: {e}")

    async def _persist_tick_to_cache(self, symbol: str, bid: float, ask: float,
                                      tick_timestamp: datetime, sequence: int = 0):
        """
        Persist validated tick to TickCache table for HOT tier storage.

        Args:
            symbol: Trading symbol
            bid: Bid price
            ask: Ask price
            tick_timestamp: Tick timestamp
            sequence: Sequence number for ordering
        """
        try:
            with self.db.get_session() as session:
                tick = TickCache(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=tick_timestamp,
                    sequence=sequence
                )
                session.add(tick)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to persist tick for {symbol}: {e}")

    async def _cleanup_stale_ticks(self):
        """
        Background task to clean up ticks older than the retention period (1 hour).
        Runs every 5 minutes.
        """
        while self.is_streaming:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=TICK_RETENTION_HOURS)

                with self.db.get_session() as session:
                    deleted = session.query(TickCache).filter(
                        TickCache.timestamp < cutoff_time
                    ).delete(synchronize_session=False)
                    session.commit()

                    if deleted > 0:
                        logger.info(f"Cleaned up {deleted} stale ticks older than {TICK_RETENTION_HOURS} hour(s)")

            except Exception as e:
                logger.error(f"Failed to cleanup stale ticks: {e}")

            # Run cleanup every 5 minutes
            await asyncio.sleep(300)

_tick_handler_instance: Optional[TickStreamHandler] = None
_tick_handler_adapter: Optional[MT5SocketAdapter] = None
_tick_handler_db: Optional[DBManager] = None
_tick_handler_subscription_manager = None
_tick_handler_circuit_breaker = None
_tick_handler_resource_monitor = None


def get_tick_handler(
    mt5_adapter: Optional[MT5SocketAdapter] = None,
    db_manager: Optional[DBManager] = None,
    subscription_manager=None,
    circuit_breaker=None,
    resource_monitor=None
) -> TickStreamHandler:
    """
    Get or create the singleton TickStreamHandler instance.

    Args:
        mt5_adapter: MT5 socket adapter for tick data
        db_manager: Database manager for HOT tier persistence (optional, creates default if None)
        subscription_manager: Symbol subscription manager for coordinating subscriptions
        circuit_breaker: Tick stream circuit breaker for fault tolerance
        resource_monitor: Resource monitor for tracking system health

    Returns:
        TickStreamHandler instance
    """
    global _tick_handler_instance, _tick_handler_adapter, _tick_handler_db
    global _tick_handler_subscription_manager, _tick_handler_circuit_breaker, _tick_handler_resource_monitor

    if _tick_handler_instance is None:
        if mt5_adapter is None:
            raise ValueError("MT5 adapter required for first initialization")
        _tick_handler_adapter = mt5_adapter
        _tick_handler_db = db_manager
        _tick_handler_subscription_manager = subscription_manager
        _tick_handler_circuit_breaker = circuit_breaker
        _tick_handler_resource_monitor = resource_monitor
        _tick_handler_instance = TickStreamHandler(
            mt5_adapter,
            db_manager,
            subscription_manager,
            circuit_breaker,
            resource_monitor
        )
    elif mt5_adapter is not None and _tick_handler_adapter != mt5_adapter:
        logger.warning("MT5 adapter mismatch, using existing handler")

    # Allow setting additional components on existing instance
    if subscription_manager is not None and _tick_handler_instance:
        _tick_handler_instance.set_subscription_manager(subscription_manager)
    if circuit_breaker is not None and _tick_handler_instance:
        _tick_handler_instance.set_circuit_breaker(circuit_breaker)
    if resource_monitor is not None and _tick_handler_instance:
        _tick_handler_instance.set_resource_monitor(resource_monitor)

    return _tick_handler_instance


# -----------------------------------------------------------------------------
# Position Query Functions (Story 3-6)
# -----------------------------------------------------------------------------


def get_position_by_ticket(ticket: int) -> Optional[Dict]:
    """
    Get position details by MT5 ticket number.

    This function queries the current position from the MT5 adapter.

    Args:
        ticket: MT5 position ticket number

    Returns:
        Position dict with keys: ticket, bot_id, symbol, direction, lot, price, profit
        Returns None if position not found
    """
    try:
        from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
        adapter = MT5SocketAdapter.get_instance()
        if adapter:
            position = adapter.get_position(ticket)
            if position:
                return {
                    "ticket": position.get("ticket", ticket),
                    "bot_id": position.get("bot_id", ""),
                    "symbol": position.get("symbol", ""),
                    "direction": position.get("type", "buy"),
                    "lot": position.get("volume", 0.0),
                    "price": position.get("price", 0.0),
                    "profit": position.get("profit", 0.0)
                }
    except Exception as e:
        logger.warning(f"Failed to get position {ticket}: {e}")

    # Return None if position not found or adapter unavailable
    return None


def get_open_positions(bot_id: Optional[str] = None) -> list:
    """
    Get all open positions, optionally filtered by bot_id.

    This function queries open positions from the MT5 adapter.

    Args:
        bot_id: Optional bot ID to filter positions

    Returns:
        List of position dicts with keys: ticket, bot_id, symbol, direction, lot, price, profit
    """
    try:
        from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
        adapter = MT5SocketAdapter.get_instance()
        if adapter:
            positions = adapter.get_open_positions(bot_id=bot_id)
            return [
                {
                    "ticket": pos.get("ticket", 0),
                    "bot_id": pos.get("bot_id", ""),
                    "symbol": pos.get("symbol", ""),
                    "direction": pos.get("type", "buy"),
                    "lot": pos.get("volume", 0.0),
                    "price": pos.get("price", 0.0),
                    "profit": pos.get("profit", 0.0)
                }
                for pos in positions
            ]
    except Exception as e:
        logger.warning(f"Failed to get open positions: {e}")

    # Return empty list if positions not found or adapter unavailable
    return []
