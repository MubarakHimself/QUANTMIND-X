import json
import logging
import os
from typing import Any, Optional, Union, Callable, Awaitable
import asyncio
from datetime import timedelta

import redis.asyncio as redis
from redis.asyncio.client import Redis

logger = logging.getLogger(__name__)

class GlobalCache:
    """
    Standardized Redis caching layer for QuantMindX.
    
    Provides async methods for caching JSON-serializable data with TTL support.
    """
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        db: Optional[int] = None,
        password: Optional[str] = None,
        prefix: str = "quantmind:",
    ):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = port or int(os.getenv("REDIS_PORT", "6379"))
        self.db = db if db is not None else int(os.getenv("REDIS_DB", "0"))
        self.password = password or os.getenv("REDIS_PASSWORD")
        self.prefix = prefix
        self._redis: Optional[Redis] = None

    async def connect(self):
        """Establish connection to Redis."""
        if self._redis is None:
            try:
                self._redis = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True
                )
                await self._redis.ping()
                logger.info(f"Connected to Redis at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._redis = None
                raise

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    def _get_key(self, key: str) -> str:
        """Apply prefix to key."""
        return f"{self.prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis:
            await self.connect()
        
        try:
            raw_value = await self._redis.get(self._get_key(key))
            if raw_value:
                return json.loads(raw_value)
            return None
        except Exception as e:
            logger.warning(f"Cache GET failed for {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[Union[int, timedelta]] = 3600):
        """Set value in cache with TTL."""
        if not self._redis:
            await self.connect()
        
        try:
            serialized = json.dumps(value)
            await self._redis.set(self._get_key(key), serialized, ex=ttl)
        except Exception as e:
            logger.warning(f"Cache SET failed for {key}: {e}")

    async def delete(self, key: str):
        """Delete key from cache."""
        if not self._redis:
            await self.connect()
        
        try:
            await self._redis.delete(self._get_key(key))
        except Exception as e:
            logger.warning(f"Cache DELETE failed for {key}: {e}")

    async def get_or_set(
        self, 
        key: str, 
        func: Callable[[], Awaitable[Any]], 
        ttl: Optional[Union[int, timedelta]] = 3600
    ) -> Any:
        """
        Implementation of the cache-aside pattern.
        Returns cached value if present, otherwise calls func() and caches result.
        """
        cached_value = await self.get(key)
        if cached_value is not None:
            return cached_value
        
        value = await func()
        if value is not None:
            await self.set(key, value, ttl=ttl)
        
        return value

# Global instance
_cache: Optional[GlobalCache] = None

def get_cache() -> GlobalCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = GlobalCache()
    return _cache
