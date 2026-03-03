"""
Caching utilities for MT5 client.
"""

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CachedValue:
    """Cached value with TTL."""

    def __init__(self, value: Any, timestamp: float, ttl: float = 10.0):
        self.value = value
        self.timestamp = timestamp
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if cached value has expired."""
        return time.time() - self.timestamp > self.ttl


class MT5Cache:
    """Cache manager for MT5 client data."""

    def __init__(self, ttl: float = 10.0):
        """
        Initialize cache with TTL.

        Args:
            ttl: Time-to-live in seconds (default: 10.0)
        """
        self._ttl = ttl
        self._cache: Dict[str, CachedValue] = {}

    @property
    def ttl(self) -> float:
        """Get cache TTL."""
        return self._ttl

    @ttl.setter
    def ttl(self, value: float) -> None:
        """Set cache TTL."""
        self._ttl = value

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        cached = self._cache.get(key)
        if cached and not cached.is_expired():
            logger.debug(f"Cache hit for {key}")
            return cached.value
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = CachedValue(
            value=value,
            timestamp=time.time(),
            ttl=self._ttl
        )
        logger.debug(f"Cached {key} with TTL {self._ttl}s")

    def clear(self, pattern: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match keys (None clears all)
        """
        if pattern is None:
            self._cache.clear()
            logger.debug("Cleared all cache")
        else:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
            logger.debug(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")

    def size(self) -> int:
        """Get number of cached items."""
        return len(self._cache)
