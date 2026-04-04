"""
SQS Cache with Jittered TTL and Probabilistic Early Refresh

Redis cache manager implementing:
- Jittered TTL: 30s base + 0-10s random to prevent cache stampede
- Probabilistic early refresh: when TTL < 8s, p = 1 - (TTL_remaining / 8)
- Stale-while-revalidate: consumers may use stale value for up to 60s max

Story: 4-7-spread-quality-score-sqs-system
GG-3: This pattern is shared with Epic 15 Story 15.2 (SVSS Cache Stability)
"""

import asyncio
import logging
import os
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JitteredTTLConfig:
    """Configuration for jittered TTL behavior."""
    BASE_TTL_SECONDS = 30
    JITTER_MAX_SECONDS = 10
    EARLY_REFRESH_THRESHOLD_SECONDS = 8
    MAX_STALE_SECONDS = 60


class SQSCacheEntry(BaseModel):
    """Cached SQS value entry."""
    sqs: float
    threshold: float
    calculated_at_utc: datetime
    expires_at_utc: datetime
    is_stale: bool = False


class SQSHistoryBucket(BaseModel):
    """Historical spread bucket for aggregation."""
    avg_spread: float
    sample_count: int
    updated_at_utc: datetime


class SQSRedisCache:
    """
    Redis cache manager for SQS values and historical spread data.

    Implements:
    - Jittered TTL (30s + 0-10s random) to prevent cache stampede
    - Probabilistic early refresh when TTL < 8s
    - Stale-while-revalidate up to 60s max
    """

    # Redis key prefixes
    PREFIX_CURRENT = "sqs:current:"
    PREFIX_HISTORY = "sqs:history:"
    PREFIX_WEEKEND_GUARD = "sqs:weekend_guard:"
    PREFIX_NEWS_OVERRIDE = "sqs:news_override:"
    PREFIX_MONDAY_WARMUP = "sqs:monday_warmup:"

    def __init__(self, redis_client=None):
        """
        Initialize SQS cache.

        Args:
            redis_client: Optional Redis client instance. If None, uses mock for testing.
        """
        self._redis = redis_client
        self._config = JitteredTTLConfig()
        logger.info("SQS Redis Cache initialized")

    @property
    def is_available(self) -> bool:
        """Check if Redis is available."""
        return self._redis is not None

    async def get_current_sqs(self, symbol: str) -> Optional[SQSCacheEntry]:
        """
        Get current SQS value for symbol from cache.

        Returns cached value even if stale (stale-while-revalidate).
        """
        if not self.is_available:
            return None

        try:
            import json
            key = f"{self.PREFIX_CURRENT}{symbol}"
            data = await self._redis.get(key)

            if data is None:
                return None

            parsed = json.loads(data)
            entry = SQSCacheEntry(**parsed)

            # Check if stale but within max stale time
            now = datetime.now(timezone.utc)
            if now > entry.expires_at_utc:
                age_seconds = (now - entry.expires_at_utc).total_seconds()
                if age_seconds <= self._config.MAX_STALE_SECONDS:
                    entry.is_stale = True
                    logger.debug(f"SQS cache: Returning stale value for {symbol}, age={age_seconds:.1f}s")
                else:
                    # Beyond max stale - treat as cache miss
                    logger.debug(f"SQS cache: Stale value for {symbol} beyond max age")
                    return None

            return entry

        except Exception as e:
            logger.warning(f"SQS cache: Error getting current SQS for {symbol}: {e}")
            return None

    async def set_current_sqs(
        self,
        symbol: str,
        sqs: float,
        threshold: float
    ) -> None:
        """
        Store current SQS value with jittered TTL.

        TTL = base (30s) + random jitter (0-10s) = 30-40s total
        """
        if not self.is_available:
            return

        try:
            import json
            now = datetime.now(timezone.utc)
            jitter = random.uniform(0, self._config.JITTER_MAX_SECONDS)
            ttl_seconds = self._config.BASE_TTL_SECONDS + jitter
            expires_at = now.replace(microsecond=0) + timedelta(seconds=ttl_seconds)

            entry = SQSCacheEntry(
                sqs=sqs,
                threshold=threshold,
                calculated_at_utc=now,
                expires_at_utc=expires_at
            )

            key = f"{self.PREFIX_CURRENT}{symbol}"
            await self._redis.setex(
                key,
                int(ttl_seconds),
                entry.model_dump_json()
            )

            logger.debug(f"SQS cache: Stored {symbol} sqs={sqs:.4f} ttl={ttl_seconds:.1f}s")

        except Exception as e:
            logger.warning(f"SQS cache: Error setting current SQS for {symbol}: {e}")

    async def should_early_refresh(self, symbol: str) -> bool:
        """
        Determine if probabilistic early refresh should trigger.

        When TTL remaining < 8s:
            p_refresh = 1 - (TTL_remaining / 8)

        This means:
        - TTL = 8s remaining -> 0% chance refresh
        - TTL = 4s remaining -> 50% chance refresh
        - TTL = 0s remaining -> 100% chance refresh
        """
        if not self.is_available:
            return True  # Always refresh if cache unavailable

        try:
            key = f"{self.PREFIX_CURRENT}{symbol}"
            ttl = await self._redis.ttl(key)

            if ttl < 0:
                # Key doesn't exist or has no TTL
                return True

            if ttl >= self._config.EARLY_REFRESH_THRESHOLD_SECONDS:
                # TTL still high enough - no early refresh
                return False

            # Compute refresh probability
            p_refresh = 1 - (ttl / self._config.EARLY_REFRESH_THRESHOLD_SECONDS)
            should_refresh = random.random() < p_refresh

            logger.debug(
                f"SQS cache: Early refresh check for {symbol}: "
                f"ttl={ttl}s p_refresh={p_refresh:.2f} result={should_refresh}"
            )

            return should_refresh

        except Exception as e:
            logger.warning(f"SQS cache: Error checking early refresh for {symbol}: {e}")
            return True

    async def get_historical_bucket(
        self,
        symbol: str,
        bucket_key: str
    ) -> Optional[SQSHistoryBucket]:
        """
        Get historical spread bucket for symbol and bucket key.

        Bucket key format: "{dow}:{hour}:{minute_bucket}"
        """
        if not self.is_available:
            return None

        try:
            import json
            key = f"{self.PREFIX_HISTORY}{symbol}:{bucket_key}"
            data = await self._redis.get(key)

            if data is None:
                return None

            return SQSHistoryBucket(**json.loads(data))

        except Exception as e:
            logger.warning(f"SQS cache: Error getting history bucket for {symbol}:{bucket_key}: {e}")
            return None

    async def update_historical_bucket(
        self,
        symbol: str,
        bucket_key: str,
        new_spread: float
    ) -> None:
        """
        Update historical spread bucket with new observation.

        Uses exponential moving average to update bucket:
        new_avg = (1 - alpha) * old_avg + alpha * new_spread
        where alpha = 1 / sample_count (gives more weight to new observations as we accumulate samples)
        """
        if not self.is_available:
            return

        try:
            import json
            key = f"{self.PREFIX_HISTORY}{symbol}:{bucket_key}"
            data = await self._redis.get(key)

            if data is None:
                # New bucket
                bucket = SQSHistoryBucket(
                    avg_spread=new_spread,
                    sample_count=1,
                    updated_at_utc=datetime.now(timezone.utc)
                )
            else:
                # Update existing bucket
                bucket = SQSHistoryBucket(**json.loads(data))
                alpha = 1.0 / (bucket.sample_count + 1)
                bucket.avg_spread = (1 - alpha) * bucket.avg_spread + alpha * new_spread
                bucket.sample_count += 1
                bucket.updated_at_utc = datetime.now(timezone.utc)

            # Store with 24-hour TTL for historical buckets
            await self._redis.setex(key, 86400, bucket.model_dump_json())

        except Exception as e:
            logger.warning(f"SQS cache: Error updating history bucket for {symbol}:{bucket_key}: {e}")

    async def get_weekend_guard_state(self, symbol: str) -> bool:
        """Get weekend guard state for symbol. Returns True if guard is active."""
        if not self.is_available:
            return False

        try:
            key = f"{self.PREFIX_WEEKEND_GUARD}{symbol}"
            state = await self._redis.get(key)
            return state == b"1" or state == "1"

        except Exception as e:
            logger.warning(f"SQS cache: Error getting weekend guard for {symbol}: {e}")
            return False

    async def set_weekend_guard_state(self, symbol: str, active: bool) -> None:
        """Set weekend guard state for symbol."""
        if not self.is_available:
            return

        try:
            key = f"{self.PREFIX_WEEKEND_GUARD}{symbol}"
            # Weekend guard lasts until Sunday 21:00 GMT
            await self._redis.setex(key, 172800, "1" if active else "0")  # 48 hour TTL max

        except Exception as e:
            logger.warning(f"SQS cache: Error setting weekend guard for {symbol}: {e}")

    async def get_news_override(self, symbol: str) -> Optional[float]:
        """
        Get news threshold override for symbol.

        Returns threshold modifier (e.g., +0.10) if news blackout active, None otherwise.
        """
        if not self.is_available:
            return None

        try:
            import json
            key = f"{self.PREFIX_NEWS_OVERRIDE}{symbol}"
            data = await self._redis.get(key)

            if data is None:
                return None

            return float(json.loads(data))

        except Exception as e:
            logger.warning(f"SQS cache: Error getting news override for {symbol}: {e}")
            return None

    async def set_news_override(self, symbol: str, threshold_modifier: float, ttl_seconds: int = 3600) -> None:
        """Set news threshold override for symbol."""
        if not self.is_available:
            return

        try:
            import json
            key = f"{self.PREFIX_NEWS_OVERRIDE}{symbol}"
            await self._redis.setex(key, ttl_seconds, json.dumps(threshold_modifier))

        except Exception as e:
            logger.warning(f"SQS cache: Error setting news override for {symbol}: {e}")

    async def get_monday_warmup_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get Monday warm-up state for symbol.

        Returns dict with:
        - enabled: bool
        - started_at: float (Unix timestamp)
        - current_threshold: float
        Or None if not in warmup period.
        """
        if not self.is_available:
            return None

        try:
            import json
            key = f"{self.PREFIX_MONDAY_WARMUP}{symbol}"
            data = await self._redis.get(key)

            if data is None:
                return None

            return json.loads(data)

        except Exception as e:
            logger.warning(f"SQS cache: Error getting Monday warmup for {symbol}: {e}")
            return None

    async def set_monday_warmup_state(
        self,
        symbol: str,
        enabled: bool,
        started_at: float,
        current_threshold: float,
        ttl_seconds: int = 3600
    ) -> None:
        """Set Monday warm-up state for symbol."""
        if not self.is_available:
            return

        try:
            import json
            key = f"{self.PREFIX_MONDAY_WARMUP}{symbol}"
            state = {
                "enabled": enabled,
                "started_at": started_at,
                "current_threshold": current_threshold
            }
            await self._redis.setex(key, ttl_seconds, json.dumps(state))

        except Exception as e:
            logger.warning(f"SQS cache: Error setting Monday warmup for {symbol}: {e}")

    async def clear_stale_entries(self, symbol: str) -> int:
        """
        Clear all stale entries for symbol (used during weekend guard activation).

        Returns count of cleared entries.
        """
        if not self.is_available:
            return 0

        try:
            pattern = f"{self.PREFIX_HISTORY}{symbol}:*"
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)
                logger.info(f"SQS cache: Cleared {len(keys)} history entries for {symbol} (weekend guard)")
                return len(keys)

            return 0

        except Exception as e:
            logger.warning(f"SQS cache: Error clearing stale entries for {symbol}: {e}")
            return 0


def create_sqs_cache(redis_client=None) -> SQSRedisCache:
    """
    Factory function to create SQS cache.

    Attempts to connect to Redis if no client provided.
    """
    if redis_client is not None:
        return SQSRedisCache(redis_client=redis_client)

    # Try to connect to Redis
    try:
        import redis.asyncio as aioredis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        client = aioredis.from_url(redis_url)
        return SQSRedisCache(redis_client=client)
    except Exception as e:
        logger.warning(f"Could not connect to Redis: {e}. SQS cache will use mock mode.")
        return SQSRedisCache(redis_client=None)
