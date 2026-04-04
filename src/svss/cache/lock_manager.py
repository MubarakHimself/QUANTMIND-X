"""
Redis Distributed Lock Manager

Provides distributed locking for SVSS cache coordination using Redis.
Ensures only one consumer refreshes the cache while others use stale values.

Lock Pattern:
- Lock key: svss:lock:{symbol}:{indicator}
- Lock value: consumer_id (unique per consumer)
- Lock TTL: 5 seconds (auto-release if holder crashes)
- Acquire: SETNX with NX and EX (atomic)
- Release: DEL if value matches (Lua script for atomicity)
"""

import logging
import time
import uuid
from typing import Optional

import redis

logger = logging.getLogger(__name__)

# Lua script for atomic lock release
# Only releases if the lock value matches the consumer_id
RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class LockAcquisitionError(Exception):
    """Raised when lock acquisition fails unexpectedly."""
    pass


class DistributedLock:
    """
    Redis-based distributed lock for coordinating cache refresh.

    Implements the first-refresher-wins pattern where:
    - The first consumer to acquire the lock does the refresh
    - Other consumers use the stale value
    - Lock auto-releases after TTL to handle holder crashes
    """

    LOCK_KEY_PATTERN = "svss:lock:{symbol}:{indicator}"
    DEFAULT_LOCK_TTL = 5  # seconds

    def __init__(
        self,
        redis_client: redis.Redis,
        symbol: str,
        indicator: str,
        lock_ttl: int = DEFAULT_LOCK_TTL,
    ):
        """
        Initialize distributed lock.

        Args:
            redis_client: Redis client instance
            symbol: Trading symbol (e.g., 'EURUSD')
            indicator: Indicator name (e.g., 'vwap', 'rvvol')
            lock_ttl: Lock TTL in seconds (auto-release if holder crashes)
        """
        self._redis_client = redis_client
        self._symbol = symbol.lower()
        self._indicator = indicator.lower()
        self._lock_ttl = lock_ttl
        self._consumer_id = str(uuid.uuid4())
        self._lock_key = self.LOCK_KEY_PATTERN.format(
            symbol=self._symbol,
            indicator=self._indicator
        )
        self._release_script = self._redis_client.register_script(RELEASE_LOCK_SCRIPT)

    @property
    def consumer_id(self) -> str:
        """Get unique consumer identifier."""
        return self._consumer_id

    @property
    def lock_key(self) -> str:
        """Get Redis lock key."""
        return self._lock_key

    def acquire(self, timeout: float = 0.0) -> bool:
        """
        Attempt to acquire the distributed lock.

        Uses SETNX with NX and EX for atomic acquisition.
        Only one consumer can acquire the lock; others get False.

        Args:
            timeout: Maximum time to wait for lock (0 = non-blocking)

        Returns:
            True if lock acquired, False otherwise.
        """
        if timeout <= 0:
            # Non-blocking acquisition
            return self._try_acquire()

        # Blocking acquisition with timeout
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            if self._try_acquire():
                return True
            # Small sleep to avoid busy-waiting
            time.sleep(0.01)

        return False

    def _try_acquire(self) -> bool:
        """
        Try to acquire lock in a single attempt.

        Returns:
            True if acquired, False otherwise.
        """
        try:
            # SET key value NX EX ttl
            # Returns True if key was set (lock acquired), None if key exists (lock not acquired)
            result = self._redis_client.set(
                self._lock_key,
                self._consumer_id,
                nx=True,
                ex=self._lock_ttl
            )
            acquired = result is True
            if acquired:
                logger.debug(
                    f"Lock acquired for {self._lock_key} by {self._consumer_id}"
                )
            return acquired

        except redis.RedisError as e:
            logger.error(f"Redis error during lock acquisition: {e}")
            return False

    def release(self) -> bool:
        """
        Release the distributed lock.

        Uses Lua script for atomic check-and-delete.
        Only releases if the lock value matches our consumer_id.

        Returns:
            True if lock released, False if lock was not held or already released.
        """
        try:
            result = self._release_script(
                keys=[self._lock_key],
                args=[self._consumer_id]
            )
            released = result == 1
            if released:
                logger.debug(
                    f"Lock released for {self._lock_key} by {self._consumer_id}"
                )
            return released

        except redis.RedisError as e:
            logger.error(f"Redis error during lock release: {e}")
            return False

    def extend(self, additional_ttl: int) -> bool:
        """
        Extend the lock TTL if we still hold it.

        Useful for long-running refresh operations.

        Args:
            additional_ttl: Additional seconds to add to lock TTL

        Returns:
            True if lock extended, False if lock not held.
        """
        try:
            # Check if we still hold the lock
            current_value = self._redis_client.get(self._lock_key)
            if current_value is None:
                logger.warning(
                    f"Cannot extend lock {self._lock_key}: lock not held"
                )
                return False

            if current_value.decode() != self._consumer_id:
                logger.warning(
                    f"Cannot extend lock {self._lock_key}: held by different consumer"
                )
                return False

            # Extend TTL
            new_ttl = self._redis_client.expire(
                self._lock_key,
                self._lock_ttl + additional_ttl
            )
            return new_ttl

        except redis.RedisError as e:
            logger.error(f"Redis error during lock extension: {e}")
            return False

    def is_held(self) -> bool:
        """
        Check if this consumer still holds the lock.

        Returns:
            True if lock is held by this consumer, False otherwise.
        """
        try:
            current_value = self._redis_client.get(self._lock_key)
            if current_value is None:
                return False
            return current_value.decode() == self._consumer_id

        except redis.RedisError as e:
            logger.error(f"Redis error checking lock status: {e}")
            return False

    def get_holder(self) -> Optional[str]:
        """
        Get the consumer ID of the current lock holder.

        Returns:
            Consumer ID if lock is held, None if lock is free.
        """
        try:
            current_value = self._redis_client.get(self._lock_key)
            if current_value is None:
                return None
            return current_value.decode()

        except redis.RedisError as e:
            logger.error(f"Redis error getting lock holder: {e}")
            return None
