"""
SVSS Consumer Module

Consumes SVSS (Shared Volume Session Service) indicator data from Redis.
Supports both:
1. RVOL-only consumption via SVSSRVOLConsumer (legacy pub/sub pattern)
2. Full readings consumption via SVSSConsumer (new cache key pattern)

Full SVSS readings are published to Redis at key `svss:{symbol}:readings`
with TTL 60s. The SVSSConsumer reads this cache key and provides access
to VWAP, Volume Profile, MFI, and RVOL.

RVOL usage in Governor pipeline:
- RVOL > 1.0: Volume confirming entry (increase position size)
- RVOL < 1.0: Volume thin (decrease position size)
- RVOL < 0.5: Hard block (no entry)

Cache key pattern: svss:{symbol}:readings
Example: svss:EURUSD:readings
"""

import json
import logging
import os
import time
from typing import Optional, Dict
import redis

logger = logging.getLogger(__name__)


def _resolve_redis_url(redis_url: Optional[str] = None) -> str:
    """Resolve Redis URL from explicit argument or deployment environment."""
    return redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis channel pattern for RVOL data (legacy pub/sub)
RVOL_CHANNEL_PATTERN = "svss:{symbol}:rvvol"

# Redis key pattern for full SVSS readings (new cache pattern)
READINGS_KEY_PATTERN = "svss:{symbol}:readings"
READINGS_TTL = 60  # seconds

# Default RVOL value when cache is missing (no multiplier, no block)
DEFAULT_RVOL = 1.0

# RVOL clamping bounds
RVOL_MIN = 0.5  # Below this = hard block
RVOL_MAX = 1.5  # Above this = cap at 1.5x


class RVOLCache:
    """
    Cache for RVOL values with TTL support.

    RVOL values are published by SVSS at a regular frequency.
    This cache stores the latest RVOL value with timestamp for TTL validation.
    """

    def __init__(self, ttl: float = 5.0):
        """
        Initialize RVOL cache.

        Args:
            ttl: Time-to-live in seconds for cached RVOL values.
                 Default 5.0 seconds matches typical SVSS publication frequency.
        """
        self._cache: Dict[str, tuple[float, float]] = {}  # symbol -> (rvol, timestamp)
        self._ttl = ttl
        self._cache_miss_count = 0
        self._cache_hit_count = 0

    def set(self, symbol: str, rvol: float) -> None:
        """Store RVOL value for symbol with current timestamp."""
        self._cache[symbol.upper()] = (rvol, time.time())

    def get(self, symbol: str) -> Optional[float]:
        """
        Get cached RVOL value for symbol.

        Returns:
            RVOL value if cached and not expired, None otherwise.
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self._cache:
            self._cache_miss_count += 1
            return None

        rvol, timestamp = self._cache[symbol_upper]
        if time.time() - timestamp > self._ttl:
            # Cache expired
            del self._cache[symbol_upper]
            self._cache_miss_count += 1
            return None

        self._cache_hit_count += 1
        return rvol

    def get_with_default(self, symbol: str) -> tuple[float, bool]:
        """
        Get RVOL value with default fallback.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            Tuple of (rvol_value, was_cache_miss)
            - rvol_value: Cached RVOL or DEFAULT_RVOL (1.0) if cache miss
            - was_cache_miss: True if cache was missing/expired, False if cache hit
        """
        rvol = self.get(symbol)
        if rvol is None:
            return DEFAULT_RVOL, True
        return rvol, False

    @property
    def cache_miss_rate(self) -> float:
        """Get cache miss rate for monitoring."""
        total = self._cache_miss_count + self._cache_hit_count
        if total == 0:
            return 0.0
        return self._cache_miss_count / total

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            'hits': self._cache_hit_count,
            'misses': self._cache_miss_count,
            'cached_symbols': len(self._cache)
        }


class SVSSRVOLConsumer:
    """
    Redis subscriber for SVSS RVOL data.

    Connects to Redis and subscribes to the SVSS RVOL channel pattern
    to receive real-time RVOL updates for trading symbols.

    Usage:
        consumer = SVSSRVOLConsumer()
        consumer.connect()
        # or use class methods for cached access
        rvol = get_rvol("EURUSD")  # Returns 1.0 if unavailable
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        channel_pattern: str = RVOL_CHANNEL_PATTERN,
        cache_ttl: float = 5.0
    ):
        """
        Initialize SVSS RVOL Consumer.

        Args:
            redis_url: Redis connection URL
            channel_pattern: Redis channel pattern for RVOL data
            cache_ttl: Cache TTL in seconds
        """
        self._redis_url = _resolve_redis_url(redis_url)
        self._channel_pattern = channel_pattern
        self._cache = RVOLCache(ttl=cache_ttl)
        self._redis_client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._connected = False

    def connect(self) -> bool:
        """
        Establish Redis connection and subscribe to RVOL channels.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()  # Verify connection
            self._pubsub = self._redis_client.pubsub()

            # Subscribe to RVOL channel pattern
            self._pubsub.psubscribe(self._channel_pattern.replace("{symbol}", "*"))
            self._connected = True

            logger.info(
                "SVSS RVOL Consumer connected",
                extra={'redis_url': self._redis_url, 'pattern': self._channel_pattern}
            )
            return True

        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close Redis connection."""
        if self._pubsub:
            self._pubsub.close()
        if self._redis_client:
            self._redis_client.close()
        self._connected = False
        logger.info("SVSS RVOL Consumer disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if consumer is connected to Redis."""
        return self._connected

    def _get_channel_symbol(self, channel: str) -> str:
        """Extract symbol from Redis channel name."""
        # Channel format: svss:{symbol}:rvvol
        parts = channel.split(':')
        if len(parts) >= 2:
            return parts[1]
        return channel

    def poll(self) -> None:
        """
        Poll for RVOL updates. Call this in a loop to receive updates.

        Non-blocking poll - processes any pending messages.
        """
        if not self._connected or not self._pubsub:
            return

        try:
            message = self._pubsub.get_message(ignore_subscribe_messages=True)
            if message and message['type'] == 'pmessage':
                channel = message['channel']
                symbol = self._get_channel_symbol(channel)
                try:
                    rvol = float(message['data'])
                    self._cache.set(symbol, rvol)
                    logger.debug(f"RVOL update: {symbol} = {rvol}")
                except (ValueError, TypeError):
                    logger.warning(f"Invalid RVOL data on {channel}: {message['data']}")
        except redis.ConnectionError:
            logger.error("Redis connection lost during poll")
            self._connected = False

    def get_rvol(self, symbol: str) -> float:
        """
        Get RVOL value for symbol.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            RVOL value from cache, or DEFAULT_RVOL (1.0) if unavailable.
            Logs RVOLCacheMissing warning on cache miss.
        """
        rvol, was_miss = self._cache.get_with_default(symbol)
        if was_miss:
            logger.warning(
                "RVOLCacheMissing",
                extra={'symbol': symbol, 'default_rvol': DEFAULT_RVOL}
            )
        return rvol

    def get_rvol_detailed(self, symbol: str) -> tuple[float, bool]:
        """
        Get RVOL value with cache status.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            Tuple of (rvol_value, was_cache_miss)
        """
        return self._cache.get_with_default(symbol)

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return self._cache.get_stats()

    @property
    def cache(self) -> RVOLCache:
        """Access the underlying cache for direct operations."""
        return self._cache


# Global consumer instance for cross-module access
_global_consumer: Optional[SVSSRVOLConsumer] = None


def get_global_consumer() -> Optional[SVSSRVOLConsumer]:
    """Get the global RVOL consumer instance."""
    return _global_consumer


def init_global_consumer(
    redis_url: Optional[str] = None,
    cache_ttl: float = 5.0
) -> SVSSRVOLConsumer:
    """
    Initialize the global RVOL consumer.

    Args:
        redis_url: Redis connection URL
        cache_ttl: Cache TTL in seconds

    Returns:
        Initialized global consumer instance.
    """
    global _global_consumer
    _global_consumer = SVSSRVOLConsumer(redis_url=redis_url, cache_ttl=cache_ttl)
    return _global_consumer


def get_rvol(symbol: str) -> float:
    """
    Convenience function to get RVOL value for symbol.

    Uses the global consumer if initialized, otherwise returns DEFAULT_RVOL.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')

    Returns:
        RVOL value or DEFAULT_RVOL (1.0) if consumer not initialized.
    """
    if _global_consumer is None:
        logger.debug("Global RVOL consumer not initialized, returning default")
        return DEFAULT_RVOL
    return _global_consumer.get_rvol(symbol)


def clamp_rvol(rvol: float) -> float:
    """
    Clamp RVOL value to valid range [RVOL_MIN, RVOL_MAX].

    Args:
        rvol: Raw RVOL value from SVSS

    Returns:
        Clamped RVOL value:
        - rvol < 0.5: Returns 0.0 (hard block indicator)
        - 0.5 <= rvol <= 1.5: Returns clamped value
        - rvol > 1.5: Returns 1.5 (cap)
    """
    if rvol < RVOL_MIN:
        return 0.0  # Hard block
    return min(rvol, RVOL_MAX)


def should_block_entry(rvol: float) -> bool:
    """
    Determine if entry should be blocked based on RVOL.

    Args:
        rvol: RVOL value

    Returns:
        True if entry should be blocked (RVOL < 0.5), False otherwise.
    """
    return rvol < RVOL_MIN


def apply_rvol_multiplier(base_kelly: float, rvol: float) -> float:
    """
    Apply RVOL multiplier to base Kelly fraction.

    Formula: final_size = Kelly_base × clamp(RVOL, 0.5, 1.5)

    Args:
        base_kelly: Base Kelly fraction
        rvol: RVOL value

    Returns:
        Modified Kelly fraction, or 0.0 if entry should be blocked.
    """
    clamped_rvol = clamp_rvol(rvol)
    if clamped_rvol == 0.0:
        return 0.0  # Hard block
    return base_kelly * clamped_rvol


# =============================================================================
# SVSS Consumer - Reads full SVSS readings from Redis cache
# =============================================================================


class SVSSConsumer:
    """
    Consumer for full SVSS indicator readings from Redis cache.

    Reads combined VWAP, Volume Profile, MFI, and RVOL from the
    `svss:{symbol}:readings` Redis cache key (TTL 60s).

    This is the primary consumer for bots - they read all SVSS indicators
    from Redis instead of calculating them independently.

    Usage:
        consumer = SVSSConsumer(redis_url="redis://localhost:6379")
        consumer.connect()

        # Read current readings for a symbol
        readings = consumer.get_readings("EURUSD")
        if readings:
            print(f"VWAP: {readings.vwap.value}")
            print(f"POC: {readings.volume_profile.poc}")
            print(f"MFI: {readings.mfi.value}")
            print(f"RVOL: {readings.rvol.value}")
            print(f"MFI overbought: {readings.mfi.is_overbought}")
            print(f"RVOL block: {readings.rvol.value < 0.5}")

        # Direct convenience methods
        rvol = consumer.get_rvol("EURUSD")  # Uses clamp internally
        mfi = consumer.get_mfi("EURUSD")  # Returns 50.0 if unavailable
        vwap = consumer.get_vwap("EURUSD")  # Returns 0.0 if unavailable
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        cache_ttl: float = 5.0,
    ):
        """
        Initialize SVSS Consumer.

        Args:
            redis_url: Redis connection URL
            cache_ttl: Local cache TTL in seconds for readings
        """
        self._redis_url = _resolve_redis_url(redis_url)
        self._cache_ttl = cache_ttl
        self._redis_client: Optional[redis.Redis] = None
        self._connected = False

        # Local cache: symbol -> (readings_dict, timestamp)
        self._readings_cache: Dict[str, tuple[dict, float]] = {}

        # Statistics
        self._cache_hit_count = 0
        self._cache_miss_count = 0

    def connect(self) -> bool:
        """
        Connect to Redis.

        Returns:
            True if connected successfully, False otherwise.
        """
        try:
            self._redis_client = redis.from_url(self._redis_url)
            self._redis_client.ping()
            self._connected = True
            logger.info(f"SVSS Consumer connected to {self._redis_url}")
            return True
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
        self._connected = False
        logger.info("SVSS Consumer disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    def _get_readings_key(self, symbol: str) -> str:
        """Get Redis key for symbol readings."""
        return READINGS_KEY_PATTERN.format(symbol=symbol.upper())

    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if local cache is valid for symbol."""
        if symbol.upper() not in self._readings_cache:
            return False
        _, timestamp = self._readings_cache[symbol.upper()]
        return (time.time() - timestamp) < self._cache_ttl

    def get_readings(self, symbol: str) -> Optional[dict]:
        """
        Get full SVSS readings for a symbol from Redis.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')

        Returns:
            Dict with vwap, volume_profile, mfi, rvol keys if available,
            None otherwise.
        """
        if not self._connected or not self._redis_client:
            logger.warning("SVSS Consumer not connected to Redis")
            return None

        # Check local cache first
        if self._is_cache_valid(symbol):
            self._cache_hit_count += 1
            return self._readings_cache[symbol.upper()][0]

        # Fetch from Redis
        try:
            key = self._get_readings_key(symbol)
            data = self._redis_client.get(key)

            if data is None:
                self._cache_miss_count += 1
                logger.debug(f"SVSS readings cache miss for {symbol}")
                return None

            readings = json.loads(data)

            # Cache locally
            self._readings_cache[symbol.upper()] = (readings, time.time())
            self._cache_hit_count += 1

            return readings

        except (json.JSONDecodeError, redis.RedisError) as e:
            logger.error(f"Error fetching SVSS readings for {symbol}: {e}")
            self._cache_miss_count += 1
            return None

    def get_rvol(self, symbol: str) -> float:
        """
        Get RVOL value for symbol.

        Applies clamping:
        - RVOL < 0.5: Returns 0.0 (hard block)
        - RVOL > 1.5: Returns 1.5 (cap)

        Args:
            symbol: Trading symbol

        Returns:
            RVOL value, clamped to valid range.
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return DEFAULT_RVOL

        rvol = readings.get("rvol", {}).get("value", DEFAULT_RVOL)
        return clamp_rvol(rvol)

    def get_rvol_raw(self, symbol: str) -> tuple[float, bool]:
        """
        Get RVOL value with cache status.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (rvol_value, was_cache_miss)
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return DEFAULT_RVOL, True

        rvol = readings.get("rvol", {}).get("value", DEFAULT_RVOL)
        return rvol, False

    def get_mfi(self, symbol: str) -> float:
        """
        Get MFI value for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            MFI value (0-100), or 50.0 if unavailable.
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return 50.0  # Neutral MFI

        return readings.get("mfi", {}).get("value", 50.0)

    def get_mfi_status(self, symbol: str) -> tuple[float, bool, bool]:
        """
        Get MFI value with overbought/oversold status.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (mfi_value, is_overbought, is_oversold)
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return 50.0, False, False

        mfi_data = readings.get("mfi", {})
        return (
            mfi_data.get("value", 50.0),
            mfi_data.get("is_overbought", False),
            mfi_data.get("is_oversold", False),
        )

    def get_vwap(self, symbol: str) -> float:
        """
        Get VWAP value for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            VWAP value, or 0.0 if unavailable.
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return 0.0

        return readings.get("vwap", {}).get("value", 0.0)

    def get_volume_profile(self, symbol: str) -> Optional[dict]:
        """
        Get Volume Profile data for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Dict with poc, vah, val, profile, total_volume,
            or None if unavailable.
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return None

        return readings.get("volume_profile")

    def get_session_id(self, symbol: str) -> Optional[str]:
        """
        Get current session ID for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Session ID string, or None if no active session.
        """
        readings = self.get_readings(symbol)
        if readings is None:
            return None

        return readings.get("session_id")

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """
        Invalidate local cache.

        Args:
            symbol: Specific symbol to invalidate, or None for all symbols.
        """
        if symbol:
            self._readings_cache.pop(symbol.upper(), None)
        else:
            self._readings_cache.clear()

    @property
    def cache_miss_rate(self) -> float:
        """Get cache miss rate for monitoring."""
        total = self._cache_miss_count + self._cache_hit_count
        if total == 0:
            return 0.0
        return self._cache_miss_count / total

    def get_stats(self) -> Dict:
        """Get consumer statistics."""
        return {
            "hits": self._cache_hit_count,
            "misses": self._cache_miss_count,
            "cached_symbols": len(self._readings_cache),
            "cache_miss_rate": self.cache_miss_rate,
            "connected": self._connected,
        }


# Global SVSS consumer instance
_global_svss_consumer: Optional[SVSSConsumer] = None


def get_global_svss_consumer() -> Optional[SVSSConsumer]:
    """Get the global SVSS consumer instance."""
    return _global_svss_consumer


def init_global_svss_consumer(
    redis_url: Optional[str] = None,
    cache_ttl: float = 5.0,
) -> SVSSConsumer:
    """
    Initialize the global SVSS consumer.

    Args:
        redis_url: Redis connection URL
        cache_ttl: Local cache TTL in seconds

    Returns:
        Initialized global consumer instance.
    """
    global _global_svss_consumer
    _global_svss_consumer = SVSSConsumer(redis_url=redis_url, cache_ttl=cache_ttl)
    return _global_svss_consumer


def get_svss_readings(symbol: str, redis_url: Optional[str] = None) -> Optional[dict]:
    """
    Convenience function to get full SVSS readings for a symbol.

    Creates a temporary connection, fetches readings, and closes.

    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        redis_url: Redis connection URL

    Returns:
        Full readings dict if available, None otherwise.
    """
    try:
        client = redis.from_url(_resolve_redis_url(redis_url))
        key = READINGS_KEY_PATTERN.format(symbol=symbol.upper())
        data = client.get(key)
        client.close()

        if data is None:
            return None

        return json.loads(data)
    except Exception as e:
        logger.error(f"Failed to get SVSS readings: {e}")
        return None


def get_svss_rvol(symbol: str, redis_url: Optional[str] = None) -> float:
    """
    Convenience function to get RVOL value for a symbol.

    Args:
        symbol: Trading symbol
        redis_url: Redis connection URL

    Returns:
        RVOL value (clamped to valid range).
    """
    readings = get_svss_readings(symbol, redis_url)
    if readings is None:
        return DEFAULT_RVOL

    rvol = readings.get("rvol", {}).get("value", DEFAULT_RVOL)
    return clamp_rvol(rvol)


def get_svss_mfi(symbol: str, redis_url: Optional[str] = None) -> tuple[float, bool, bool]:
    """
    Convenience function to get MFI value and status for a symbol.

    Args:
        symbol: Trading symbol
        redis_url: Redis connection URL

    Returns:
        Tuple of (mfi_value, is_overbought, is_oversold)
    """
    readings = get_svss_readings(symbol, redis_url)
    if readings is None:
        return 50.0, False, False

    mfi_data = readings.get("mfi", {})
    return (
        mfi_data.get("value", 50.0),
        mfi_data.get("is_overbought", False),
        mfi_data.get("is_oversold", False),
    )
