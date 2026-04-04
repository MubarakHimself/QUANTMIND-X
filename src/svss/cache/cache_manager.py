"""
SVSS Cache Manager

Provides cache stability for SVSS indicators using:
1. Jittered TTL (30s + 0-10s random) to prevent thundering herd
2. Probabilistic early refresh to coordinate cache updates
3. Stale-while-revalidate pattern for graceful degradation

Cache Key Format:
- svss:cache:{symbol}:{indicator} -> cached indicator value

Jittered TTL Formula:
- Normal mode: TTL = 30 + random(0, 10) seconds
- High load mode (tick_freq > 10 ticks/sec): TTL = 30 + random(0, 20) seconds

Probabilistic Early Refresh Formula:
- p_refresh = 1 - (TTL_remaining / 8)
- When TTL_remaining < 8 seconds
- First consumer to acquire lock does the refresh
- Others use stale value (max 60 seconds stale)
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, List

import redis

from svss.cache.lock_manager import DistributedLock

logger = logging.getLogger(__name__)

# Cache key patterns
CACHE_KEY_PATTERN = "svss:cache:{symbol}:{indicator}"

# TTL configuration
BASE_TTL = 30  # seconds
NORMAL_JITTER_MAX = 10  # seconds
HIGH_LOAD_JITTER_MAX = 20  # seconds
HIGH_LOAD_TICK_THRESHOLD = 10  # ticks per second

# Early refresh configuration
EARLY_REFRESH_THRESHOLD = 8  # seconds
STALE_MAX_DURATION = 60  # seconds


@dataclass
class CacheEntry:
    """
    Cached indicator value with metadata.

    Attributes:
        value: Cached indicator value
        timestamp: When the value was cached (UTC)
        session_id: Session identifier when cached
        ttl_remaining: Estimated seconds until cache expires
        is_stale: Whether the value exceeds normal TTL (in stale-while-revalidate)
    """

    value: float
    timestamp: datetime
    session_id: str
    ttl_remaining: int
    is_stale: bool = False


class SVSSCacheManager:
    """
    Manages SVSS indicator cache with jittered TTL and probabilistic early refresh.

    Prevents thundering herd by:
    1. Jittering TTL so cache entries don't all expire at once
    2. Using probabilistic early refresh so consumers don't all refresh simultaneously
    3. First-refresher-wins lock pattern for coordination
    4. Stale-while-revalidate for graceful degradation
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        symbol: str,
        indicator: str,
    ):
        """
        Initialize SVSS Cache Manager.

        Args:
            redis_client: Redis client instance
            symbol: Trading symbol (e.g., 'EURUSD')
            indicator: Indicator name (e.g., 'vwap', 'rvvol')
        """
        self._redis_client = redis_client
        self._symbol = symbol.lower()
        self._indicator = indicator.lower()
        self._cache_key = CACHE_KEY_PATTERN.format(
            symbol=self._symbol,
            indicator=self._indicator
        )
        self._lock = DistributedLock(
            redis_client=redis_client,
            symbol=symbol,
            indicator=indicator,
        )

        # Tick frequency tracking for high-load detection
        self._tick_times: List[float] = []
        self._tick_window = 1.0  # 1 second window for frequency calculation

    @property
    def symbol(self) -> str:
        """Get symbol."""
        return self._symbol

    @property
    def indicator(self) -> str:
        """Get indicator name."""
        return self._indicator

    @property
    def cache_key(self) -> str:
        """Get cache Redis key."""
        return self._cache_key

    def _compute_jittered_ttl(self) -> int:
        """
        Compute jittered TTL based on current load.

        Returns:
            TTL in seconds: 30 + random(0, 10) normally,
            or 30 + random(0, 20) under high load.
        """
        tick_freq = self._get_tick_frequency()
        if tick_freq > HIGH_LOAD_TICK_THRESHOLD:
            jitter_max = HIGH_LOAD_JITTER_MAX
            logger.debug(
                f"High load detected ({tick_freq:.1f} ticks/sec), "
                f"using extended jitter range"
            )
        else:
            jitter_max = NORMAL_JITTER_MAX

        jitter = random.uniform(0, jitter_max)
        ttl = BASE_TTL + jitter
        return int(ttl)

    def _get_tick_frequency(self) -> float:
        """
        Calculate current tick frequency.

        Returns:
            Ticks per second over the tracking window.
        """
        now = time.monotonic()

        # Remove old tick times outside the window
        self._tick_times = [
            t for t in self._tick_times
            if now - t <= self._tick_window
        ]

        if not self._tick_times:
            return 0.0

        # Calculate frequency
        if len(self._tick_times) >= 2:
            time_span = self._tick_times[-1] - self._tick_times[0]
            if time_span > 0:
                return len(self._tick_times) / time_span

        return 0.0

    def record_tick(self) -> None:
        """Record a tick for frequency tracking."""
        self._tick_times.append(time.monotonic())

    def set_with_jitter(self, value: float, session_id: str) -> bool:
        """
        Set cache value with jittered TTL.

        Args:
            value: Indicator value to cache
            session_id: Current session identifier

        Returns:
            True if set successfully, False otherwise.
        """
        try:
            ttl = self._compute_jittered_ttl()
            now = datetime.now(timezone.utc)

            cache_data = {
                "value": value,
                "timestamp": now.isoformat(),
                "session_id": session_id,
            }

            self._redis_client.setex(
                self._cache_key,
                ttl,
                json.dumps(cache_data)
            )

            logger.debug(
                f"Cache set for {self._cache_key} with TTL {ttl}s "
                f"(session={session_id})"
            )
            return True

        except redis.RedisError as e:
            logger.error(f"Failed to set cache: {e}")
            return False

    def get(self) -> Optional[CacheEntry]:
        """
        Get cached value if available.

        Returns:
            CacheEntry if cache hit, None if cache miss.
        """
        try:
            data = self._redis_client.get(self._cache_key)
            if data is None:
                logger.debug(f"Cache miss for {self._cache_key}")
                return None

            cache_data = json.loads(data)
            ttl = self._redis_client.ttl(self._cache_key)
            # Preserve actual TTL (negative means expired, how many seconds ago)
            # But don't return None for expired keys - they're still usable as stale
            ttl = ttl if ttl is not None else 0

            entry = CacheEntry(
                value=cache_data["value"],
                timestamp=datetime.fromisoformat(cache_data["timestamp"]),
                session_id=cache_data["session_id"],
                ttl_remaining=ttl,
                is_stale=False,
            )

            logger.debug(
                f"Cache hit for {self._cache_key}: "
                f"value={entry.value}, TTL={ttl}s"
            )
            return entry

        except redis.RedisError as e:
            logger.error(f"Failed to get cache: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse cache data: {e}")
            return None

    def should_early_refresh(self, entry: CacheEntry) -> bool:
        """
        Determine if probabilistic early refresh should be triggered.

        Uses formula: p_refresh = 1 - (TTL_remaining / 8)

        When TTL_remaining < 8 seconds:
        - Generate random r in [0, 1]
        - If r < p_refresh: trigger early refresh

        Args:
            entry: Current cache entry

        Returns:
            True if early refresh should be attempted.
        """
        if entry.ttl_remaining >= EARLY_REFRESH_THRESHOLD:
            return False

        # Calculate refresh probability
        p_refresh = 1.0 - (entry.ttl_remaining / EARLY_REFRESH_THRESHOLD)

        # Generate random threshold
        r = random.random()
        should_refresh = r < p_refresh

        logger.debug(
            f"Early refresh check: TTL_remaining={entry.ttl_remaining:.1f}s, "
            f"p_refresh={p_refresh:.3f}, r={r:.3f}, "
            f"should_refresh={should_refresh}"
        )

        return should_refresh

    def get_with_stale(self, max_stale_age: int = STALE_MAX_DURATION) -> Optional[CacheEntry]:
        """
        Get cached value, allowing stale values up to max_stale_age.

        This is the stale-while-revalidate pattern:
        1. If cache is fresh, return it
        2. If cache is stale but within max_stale_age, return it with is_stale=True
        3. If cache is older than max_stale_age, return None

        Args:
            max_stale_age: Maximum age in seconds for stale values

        Returns:
            CacheEntry if available (fresh or stale within limit), None otherwise.
        """
        entry = self.get()

        if entry is None:
            return None

        # Fresh cache
        if entry.ttl_remaining > 0:
            return entry

        # Cache expired but might be usable as stale
        # Calculate how long since expiration
        stale_age = abs(entry.ttl_remaining)

        if stale_age <= max_stale_age:
            # Mark as stale but still usable
            stale_entry = CacheEntry(
                value=entry.value,
                timestamp=entry.timestamp,
                session_id=entry.session_id,
                ttl_remaining=entry.ttl_remaining,
                is_stale=True,
            )
            logger.debug(
                f"Returning stale value for {self._cache_key}: "
                f"stale_age={stale_age:.1f}s (max={max_stale_age}s)"
            )
            return stale_entry

        # Too stale
        logger.debug(
            f"Cache value too stale for {self._cache_key}: "
            f"stale_age={stale_age:.1f}s > max={max_stale_age}s"
        )
        return None

    def try_refresh(
        self,
        refresh_fn,
        session_id: str,
        max_stale_age: int = STALE_MAX_DURATION,
    ) -> Optional[CacheEntry]:
        """
        Attempt to refresh cache using stale-while-revalidate pattern.

        Algorithm:
        1. Get current cache entry
        2. If no entry or TTL >= 8s, return current entry
        3. If TTL < 8s, calculate p_refresh and potentially attempt refresh
        4. If attempting refresh, try to acquire lock
        5. If lock acquired, do refresh and return new value
        6. If lock not acquired, return stale value (if within max_stale_age)

        Args:
            refresh_fn: Function to call to compute new value (no args)
            session_id: Current session identifier
            max_stale_age: Maximum stale age in seconds

        Returns:
            CacheEntry (possibly stale) if available, None otherwise.
        """
        # Get current entry
        entry = self.get()

        # No cache at all
        if entry is None:
            logger.debug(f"No cache entry for {self._cache_key}, attempting refresh")
            return self._do_refresh(refresh_fn, session_id)

        # TTL still has plenty of time
        if entry.ttl_remaining >= EARLY_REFRESH_THRESHOLD:
            return entry

        # TTL is low, check if we should early refresh
        if not self.should_early_refresh(entry):
            # Not refreshing yet, use current entry (possibly stale)
            return self.get_with_stale(max_stale_age)

        # Try to acquire lock for refresh
        if self._lock.acquire(timeout=0.1):
            try:
                # Double-check cache hasn't been refreshed by another consumer
                current = self.get()
                if current and current.ttl_remaining >= EARLY_REFRESH_THRESHOLD:
                    logger.debug(
                        f"Cache refreshed by another consumer, using current value"
                    )
                    return current

                # We hold the lock, do the refresh
                return self._do_refresh(refresh_fn, session_id)

            finally:
                self._lock.release()
        else:
            # Lock not acquired, another consumer is refreshing
            logger.debug(
                f"Lock not acquired for {self._cache_key}, using stale value"
            )
            return self.get_with_stale(max_stale_age)

    def _do_refresh(
        self,
        refresh_fn,
        session_id: str,
    ) -> Optional[CacheEntry]:
        """
        Perform cache refresh.

        Args:
            refresh_fn: Function to compute new value
            session_id: Current session identifier

        Returns:
            New CacheEntry if refresh successful, None otherwise.
        """
        try:
            new_value = refresh_fn()
            if new_value is None:
                logger.warning(f"Refresh function returned None for {self._cache_key}")
                return None

            self.set_with_jitter(new_value, session_id)
            return self.get()

        except Exception as e:
            logger.error(f"Failed to refresh cache for {self._cache_key}: {e}")
            return None

    def invalidate(self) -> bool:
        """
        Invalidate (delete) the cache entry.

        Returns:
            True if deleted, False otherwise.
        """
        try:
            result = self._redis_client.delete(self._cache_key)
            logger.debug(f"Cache invalidated for {self._cache_key}")
            return result > 0

        except redis.RedisError as e:
            logger.error(f"Failed to invalidate cache: {e}")
            return False

    def get_ttl_remaining(self) -> int:
        """
        Get remaining TTL for cache entry.

        Returns:
            TTL in seconds, -1 if no TTL (key doesn't exist), -2 if no expiry.
        """
        try:
            ttl = self._redis_client.ttl(self._cache_key)
            return max(0, ttl) if ttl and ttl > 0 else 0

        except redis.RedisError as e:
            logger.error(f"Failed to get TTL: {e}")
            return 0

    def get_lock_info(self) -> Dict[str, any]:
        """
        Get information about the distributed lock.

        Returns:
            Dict with lock holder and status.
        """
        holder = self._lock.get_holder()
        return {
            "lock_key": self._lock.lock_key,
            "holder": holder,
            "our_consumer_id": self._lock.consumer_id,
            "is_held_by_us": self._lock.is_held(),
        }
