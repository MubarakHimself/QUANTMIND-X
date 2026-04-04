"""
Redis Pub/Sub Publisher Module

Publishes SVSS indicator values to Redis channels.
Uses Redis pipeline for atomic multi-channel publish.

Channel format: svss:{symbol}:{indicator}
Message format: {"symbol": str, "value": float, "timestamp": str, "session_id": str}

Also integrates with SVSSCacheManager for cache stability:
- Jittered TTL (30s + 0-10s random) to prevent thundering herd
- Probabilistic early refresh coordination
- Stale-while-revalidate for graceful degradation
"""

import json
import logging
import random
from datetime import datetime, timezone
from typing import Optional, List, Dict

import redis

from svss.indicators.base import IndicatorResult
from svss.cache.cache_manager import SVSSCacheManager

logger = logging.getLogger(__name__)


class SVSSPublisher:
    """
    Redis pub/sub publisher for SVSS indicators.

    Publishes all 4 indicators (VWAP, RVOL, Volume Profile, MFI) to their
    respective Redis channels atomically using pipeline.

    Also provides cache management for consumers via SVSSCacheManager instances.
    """

    CHANNEL_PATTERN = "svss:{symbol}:{indicator}"

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize SVSS Publisher.

        Args:
            redis_url: Redis connection URL
        """
        self._redis_url = redis_url
        self._redis_client: Optional[redis.Redis] = None
        self._connected = False
        self._cache_managers: Dict[str, SVSSCacheManager] = {}

    def connect(self) -> bool:
        """
        Establish connection to Redis.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            self._connected = True
            logger.info(f"SVSS Publisher connected to {self._redis_url}")
            return True

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis_client:
            self._redis_client.close()
        self._connected = False
        logger.info("SVSS Publisher disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if publisher is connected."""
        return self._connected

    def _get_channel_name(self, symbol: str, indicator: str) -> str:
        """
        Get Redis channel name for indicator.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            indicator: Indicator name (e.g., 'vwap', 'rvvol')

        Returns:
            Channel name (e.g., 'svss:EURUSD:vwap')
        """
        return self.CHANNEL_PATTERN.format(symbol=symbol.lower(), indicator=indicator.lower())

    def publish(self, result: IndicatorResult) -> bool:
        """
        Publish a single indicator result to Redis.

        Args:
            result: IndicatorResult to publish

        Returns:
            True if published successfully, False otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("Cannot publish - not connected to Redis")
            return False

        try:
            channel = self._get_channel_name(
                result.metadata.get("symbol", ""),
                result.name
            )
            message = json.dumps(result.to_dict())
            self._redis_client.publish(channel, message)
            return True

        except Exception as e:
            logger.error(f"Failed to publish {result.name}: {e}")
            return False

    def publish_all(
        self,
        results: List[IndicatorResult],
        symbol: str,
        session_id: str,
        timestamp: Optional[datetime] = None
    ) -> bool:
        """
        Publish all indicator results atomically using Redis pipeline.

        Args:
            results: List of IndicatorResult from all indicators
            symbol: Trading symbol
            session_id: Current session identifier
            timestamp: Timestamp for the publish (uses current time if None)

        Returns:
            True if all published successfully, False otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("Cannot publish - not connected to Redis")
            return False

        if not results:
            logger.warning("No results to publish")
            return False

        try:
            pipe = self._redis_client.pipeline()
            ts = timestamp or datetime.now(timezone.utc)

            for result in results:
                channel = self._get_channel_name(
                    symbol,
                    result.name
                )

                # Update timestamp if provided
                if timestamp:
                    result.timestamp = timestamp

                message = json.dumps(result.to_dict())
                pipe.publish(channel, message)

            pipe.execute()

            logger.debug(
                f"Published {len(results)} indicators for {symbol} "
                f"session {session_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish indicators: {e}")
            return False

    def get_cache_manager(self, symbol: str, indicator: str) -> Optional[SVSSCacheManager]:
        """
        Get or create a cache manager for a specific indicator.

        Cache managers are cached per symbol+indicator to maintain consistent
        tick frequency tracking.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            indicator: Indicator name (e.g., 'vwap', 'rvvol')

        Returns:
            SVSSCacheManager instance if connected, None otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("Cannot get cache manager - not connected to Redis")
            return None

        cache_key = f"{symbol.lower()}:{indicator.lower()}"

        if cache_key not in self._cache_managers:
            self._cache_managers[cache_key] = SVSSCacheManager(
                redis_client=self._redis_client,
                symbol=symbol,
                indicator=indicator,
            )

        return self._cache_managers[cache_key]

    def cache_indicator(
        self,
        symbol: str,
        indicator: str,
        value: float,
        session_id: str,
    ) -> bool:
        """
        Cache an indicator value with jittered TTL.

        Args:
            symbol: Trading symbol
            indicator: Indicator name
            value: Indicator value to cache
            session_id: Current session identifier

        Returns:
            True if cached successfully, False otherwise.
        """
        manager = self.get_cache_manager(symbol, indicator)
        if manager is None:
            return False

        return manager.set_with_jitter(value, session_id)

    def get_cached_indicator(self, symbol: str, indicator: str) -> Optional[object]:
        """
        Get cached indicator value.

        Args:
            symbol: Trading symbol
            indicator: Indicator name

        Returns:
            CacheEntry if available, None otherwise.
        """
        manager = self.get_cache_manager(symbol, indicator)
        if manager is None:
            return None

        return manager.get()

    def record_tick_for_frequency(self, symbol: str, indicator: str) -> None:
        """
        Record a tick for high-load frequency tracking.

        Args:
            symbol: Trading symbol
            indicator: Indicator name
        """
        manager = self.get_cache_manager(symbol, indicator)
        if manager is not None:
            manager.record_tick()

    def publish_and_cache(
        self,
        results: List[IndicatorResult],
        symbol: str,
        session_id: str,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """
        Publish all indicator results AND cache them with jittered TTL.

        Combines publish_all and cache_indicator into a single operation
        for efficiency.

        Args:
            results: List of IndicatorResult from all indicators
            symbol: Trading symbol
            session_id: Current session identifier
            timestamp: Timestamp for the publish (uses current time if None)

        Returns:
            True if all published and cached successfully, False otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("Cannot publish and cache - not connected to Redis")
            return False

        if not results:
            logger.warning("No results to publish and cache")
            return False

        ts = timestamp or datetime.now(timezone.utc)
        success = True

        try:
            pipe = self._redis_client.pipeline()

            for result in results:
                # Publish to channel
                channel = self._get_channel_name(symbol, result.name)
                if timestamp:
                    result.timestamp = timestamp
                message = json.dumps(result.to_dict())
                pipe.publish(channel, message)

                # Cache with jittered TTL
                cache_key = f"svss:cache:{symbol.lower()}:{result.name.lower()}"
                # Compute jittered TTL inline to avoid creating cache manager per indicator
                ttl = 30 + int(random.uniform(0, 10))
                cache_data = {
                    "value": result.value,
                    "timestamp": result.timestamp.isoformat(),
                    "session_id": session_id,
                }
                pipe.setex(cache_key, ttl, json.dumps(cache_data))

            pipe.execute()

            logger.debug(
                f"Published and cached {len(results)} indicators for {symbol} "
                f"session {session_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to publish and cache indicators: {e}")
            return False
