"""
Position Monitor Engine (Layer 2) - Cloudzy Local Service

Async engine that:
1. Subscribes to Sentinel Redis channel for regime updates
2. Runs check_and_modify_positions() on every regime change
3. Runs a 10-second ticker for break-even checks
4. Logs all modifications with reason

Spec: Addendum Section 3, Layer 2
"""

import asyncio
import logging
import signal
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

import redis
import redis.asyncio as aioredis

from .position_monitor import PositionMonitorService, RegimeState, ModificationResult

logger = logging.getLogger(__name__)


# Redis channel for Sentinel regime pub/sub
SENTINEL_REGIME_CHANNEL = "sentinel:regime:updates"


class PositionMonitorEngine:
    """
    Async engine for Position Monitor Service (Layer 2).

    Runs on Cloudzy and coordinates:
    1. Redis pub/sub subscription for Sentinel regime updates
    2. 10-second ticker for break-even checks
    3. Position modification orchestration

    Attributes:
        position_monitor: PositionMonitorService instance
        redis_client: Async Redis client for pub/sub
        sync_redis: Synchronous Redis client for regime cache reads
        is_running: Engine running state
    """

    def __init__(
        self,
        position_monitor: PositionMonitorService,
        redis_url: str = "redis://localhost:6379",
        ticker_interval: float = 10.0
    ):
        """
        Initialize Position Monitor Engine.

        Args:
            position_monitor: PositionMonitorService instance to orchestrate
            redis_url: Redis connection URL
            ticker_interval: Break-even check interval in seconds (default: 10s)
        """
        self._monitor = position_monitor
        self._redis_url = redis_url
        self._ticker_interval = ticker_interval

        # Redis clients (async for pub/sub, sync for cache reads)
        self._async_redis: Optional[aioredis.Redis] = None
        self._sync_redis: Optional[redis.Redis] = None

        # Control flags
        self._is_running = False
        self._shutdown_event = asyncio.Event()

        # Price cache for ticker (symbol -> price)
        self._prices: Dict[str, float] = {}

        # Modification log
        self._modification_log: List[ModificationResult] = []

        # Callbacks for regime updates (optional)
        self._regime_callbacks: List[Callable[[RegimeState], None]] = []

        logger.info(
            f"PositionMonitorEngine initialized: ticker_interval={ticker_interval}s "
            f"on Cloudzy (Layer 2)"
        )

    # =========================================================================
    # Engine Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """
        Start the Position Monitor Engine.

        Initializes Redis connections and starts:
        1. Regime subscriber task
        2. Break-even ticker task
        """
        if self._is_running:
            logger.warning("Engine already running")
            return

        logger.info("Starting Position Monitor Engine...")

        # Initialize Redis clients
        try:
            self._async_redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            self._sync_redis = redis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True
            )

            # Test connection
            await self._async_redis.ping()
            logger.info(f"Connected to Redis: {self._redis_url}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Load existing positions from MT5
        positions_loaded = self._monitor.load_open_positions_from_mt5()
        logger.info(f"Loaded {positions_loaded} positions from MT5")

        # Start tasks
        self._is_running = True
        self._shutdown_event.clear()

        # Create async tasks
        self._subscriber_task = asyncio.create_task(self._regime_subscriber())
        self._ticker_task = asyncio.create_task(self._break_even_ticker())

        logger.info(
            "Position Monitor Engine started on Cloudzy. "
            "Subscribing to regime updates and running break-even ticker."
        )

    async def stop(self) -> None:
        """
        Stop the Position Monitor Engine gracefully.

        Cancels tasks and closes Redis connections.
        """
        if not self._is_running:
            return

        logger.info("Stopping Position Monitor Engine...")
        self._is_running = False
        self._shutdown_event.set()

        # Cancel tasks
        if hasattr(self, '_subscriber_task') and self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass

        if hasattr(self, '_ticker_task') and self._ticker_task:
            self._ticker_task.cancel()
            try:
                await self._ticker_task
            except asyncio.CancelledError:
                pass

        # Close Redis connections
        if self._async_redis:
            await self._async_redis.close()
        if self._sync_redis:
            self._sync_redis.close()

        logger.info("Position Monitor Engine stopped")

    # =========================================================================
    # Regime Subscriber (Redis Pub/Sub)
    # =========================================================================

    async def _regime_subscriber(self) -> None:
        """
        Subscribe to Sentinel regime updates via Redis pub/sub.

        When a regime change is published, fetches current regime and
        triggers check_and_modify_positions().
        """
        if not self._async_redis:
            logger.error("Async Redis not initialized")
            return

        try:
            pubsub = self._async_redis.pubsub()
            await pubsub.subscribe(SENTINEL_REGIME_CHANNEL)

            logger.info(f"Subscribed to Redis channel: {SENTINEL_REGIME_CHANNEL}")

            async for message in pubsub.listen():
                if not self._is_running:
                    break

                if message['type'] != 'message':
                    continue

                try:
                    # Parse regime update message
                    # Expected format: {"symbol": "EURUSD", "regimes": {"M5": "...", "H1": "...", "H4": "..."}}
                    import json
                    data = json.loads(message['data'])

                    symbol = data.get('symbol', self._monitor._symbol)
                    regimes = data.get('regimes', {})

                    regime_state = RegimeState(
                        m5_regime=regimes.get('M5'),
                        h1_regime=regimes.get('H1'),
                        h4_regime=regimes.get('H4'),
                        chaos_signal=regimes.get('CHAOS', False),
                        timestamp=datetime.now(timezone.utc)
                    )

                    logger.info(
                        f"Regime update received: {symbol} "
                        f"M5={regime_state.m5_regime}, "
                        f"H1={regime_state.h1_regime}, "
                        f"H4={regime_state.h4_regime}, "
                        f"CHAOS={regime_state.chaos_signal}"
                    )

                    # Invoke callbacks
                    for callback in self._regime_callbacks:
                        try:
                            callback(regime_state)
                        except Exception as e:
                            logger.error(f"Regime callback error: {e}")

                    # Trigger position checks
                    await self._on_regime_update(regime_state)

                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid regime message format: {e}")
                except Exception as e:
                    logger.error(f"Error processing regime update: {e}")

        except asyncio.CancelledError:
            logger.info("Regime subscriber cancelled")
            raise
        except Exception as e:
            logger.error(f"Regime subscriber error: {e}")
            if self._is_running:
                # Attempt reconnect after delay
                await asyncio.sleep(5)
                if self._is_running:
                    self._subscriber_task = asyncio.create_task(self._regime_subscriber())

    async def _on_regime_update(self, regime_state: RegimeState) -> None:
        """
        Handle regime update by checking positions.

        Args:
            regime_state: New regime state from Sentinel
        """
        if not self._prices:
            logger.debug("No prices available, skipping regime check")
            return

        # Run position check
        results = self._monitor.check_and_modify_positions(
            prices=self._prices,
            regime_state=regime_state
        )

        # Log results
        for result in results:
            self._log_modification(result)

    # =========================================================================
    # Break-Even Ticker
    # =========================================================================

    async def _break_even_ticker(self) -> None:
        """
        10-second ticker for break-even checks.

        On each tick:
        1. Updates prices from Redis cache
        2. Runs check_and_modify_positions() with current prices
        """
        logger.info(f"Break-even ticker started (interval: {self._ticker_interval}s)")

        while self._is_running:
            try:
                await asyncio.sleep(self._ticker_interval)

                if not self._is_running:
                    break

                # Update prices from Redis cache
                await self._update_prices_from_cache()

                if not self._prices:
                    continue

                # Run position check (regime_state=None uses cached regime)
                results = self._monitor.check_and_modify_positions(prices=self._prices)

                # Log results
                for result in results:
                    self._log_modification(result)

            except asyncio.CancelledError:
                logger.info("Break-even ticker cancelled")
                raise
            except Exception as e:
                logger.error(f"Break-even ticker error: {e}")

    async def _update_prices_from_cache(self) -> None:
        """
        Update current prices from Redis cache.

        Reads from price cache keys: price:current:{symbol}
        """
        if not self._sync_redis:
            return

        try:
            symbols = set(pos.symbol for pos in self._monitor.open_positions.values())

            for symbol in symbols:
                price_key = f"price:current:{symbol}"
                price = self._sync_redis.get(price_key)

                if price:
                    self._prices[symbol] = float(price)
                else:
                    # Try alternative key format
                    price_key_alt = f"ticker:{symbol}:last"
                    price_alt = self._sync_redis.get(price_key_alt)
                    if price_alt:
                        self._prices[symbol] = float(price_alt)

        except Exception as e:
            logger.debug(f"Error updating prices from cache: {e}")

    # =========================================================================
    # Modification Logging
    # =========================================================================

    def _log_modification(self, result: ModificationResult) -> None:
        """
        Log position modification with full details.

        Args:
            result: ModificationResult from position check
        """
        if result.success and result.new_sl is not None:
            logger.info(
                f"POSITION MODIFIED: ticket={result.ticket}, "
                f"action={result.action}, "
                f"SL={result.old_sl:.5f} -> {result.new_sl:.5f}, "
                f"reason={result.reason}"
            )
        elif result.success:
            logger.debug(
                f"Position check: ticket={result.ticket}, "
                f"action={result.action}, {result.reason}"
            )
        else:
            logger.warning(
                f"POSITION MODIFY FAILED: ticket={result.ticket}, "
                f"action={result.action}, error={result.error}"
            )

        # Store in log
        self._modification_log.append(result)

        # Trim log if too large
        if len(self._modification_log) > 10000:
            self._modification_log = self._modification_log[-5000:]

    def get_modification_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent modification log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of modification result dicts
        """
        recent = self._modification_log[-limit:]
        return [
            {
                'ticket': r.ticket,
                'action': r.action,
                'success': r.success,
                'old_sl': r.old_sl,
                'new_sl': r.new_sl,
                'error': r.error,
                'reason': r.reason
            }
            for r in recent
        ]

    # =========================================================================
    # External Controls
    # =========================================================================

    def register_regime_callback(self, callback: Callable[[RegimeState], None]) -> None:
        """
        Register a callback to be called on regime updates.

        Args:
            callback: Function that takes RegimeState
        """
        self._regime_callbacks.append(callback)

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a symbol.

        Called externally when new prices are available.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        self._prices[symbol] = price

    async def trigger_check(self) -> List[ModificationResult]:
        """
        Manually trigger a position check.

        Returns:
            List of ModificationResults
        """
        if not self._prices:
            await self._update_prices_from_cache()

        results = self._monitor.check_and_modify_positions(prices=self._prices)
        for result in results:
            self._log_modification(result)

        return results

    @property
    def is_running(self) -> bool:
        """Check if engine is running."""
        return self._is_running


# =============================================================================
# Standalone Engine Runner
# =============================================================================

async def run_position_monitor_engine(
    redis_url: str = "redis://localhost:6379",
    symbol: str = "EURUSD",
    ticker_interval: float = 10.0
) -> None:
    """
    Run Position Monitor Engine as standalone service.

    Args:
        redis_url: Redis connection URL
        symbol: Symbol to monitor
        ticker_interval: Break-even check interval in seconds
    """
    from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter

    # Initialize MT5 adapter
    try:
        mt5_adapter = MT5SocketAdapter.get_instance()
    except Exception as e:
        logger.warning(f"MT5 adapter not available: {e}")
        mt5_adapter = None

    # Initialize Redis
    try:
        sync_redis = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
        sync_redis.ping()
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
        sync_redis = None

    # Create services
    position_monitor = PositionMonitorService(
        mt5_adapter=mt5_adapter,
        redis_client=sync_redis,
        symbol=symbol
    )

    engine = PositionMonitorEngine(
        position_monitor=position_monitor,
        redis_url=redis_url,
        ticker_interval=ticker_interval
    )

    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.stop()))

    # Run engine
    await engine.start()

    try:
        await asyncio.Event().wait()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        await engine.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(run_position_monitor_engine())
