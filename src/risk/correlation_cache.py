"""
Correlation Matrix Redis Cache Manager

Manages caching of correlation matrices in Redis with jittered TTL and
probabilistic early refresh to prevent cache stampede.

Redis key pattern: risk:correlation:{timeframe} where timeframe = 'M5' or 'H1'
"""

import json
import logging
import random
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrixData:
    """Data structure for cached correlation matrix."""
    matrix: list
    computed_at: float
    sample_count: int

    def to_json(self) -> str:
        return json.dumps({
            'matrix': self.matrix,
            'computed_at': self.computed_at,
            'sample_count': self.sample_count
        })

    @classmethod
    def from_json(cls, data: str) -> 'CorrelationMatrixData':
        parsed = json.loads(data)
        return cls(
            matrix=parsed['matrix'],
            computed_at=parsed['computed_at'],
            sample_count=parsed['sample_count']
        )


class JitteredTTLConfig:
    """Configuration for jittered TTL pattern (GG-3)."""
    BASE_TTL: int = 30  # seconds
    JITTER_MAX: int = 10  # additional seconds
    EARLY_REFRESH_THRESHOLD: int = 8  # seconds remaining before TTL expiry


class CorrelationCache:
    """
    Redis cache manager for correlation matrices.

    Implements:
    - Jittered TTL (30s base + 0-10s random) to prevent cache stampede
    - Probabilistic early refresh when TTL < 8s
    - 5-minute bar close trigger for matrix recalculation
    """

    def __init__(self, redis_client=None):
        """
        Initialize CorrelationCache.

        Args:
            redis_client: Optional Redis client instance. If None, uses local cache.
        """
        self._redis = redis_client
        self._local_cache: Dict[str, CorrelationMatrixData] = {}
        self._local_cache_timestamps: Dict[str, float] = {}
        self._ttl_config = JitteredTTLConfig()

    def _get_ttl_seconds(self) -> int:
        """Generate jittered TTL value: 30s base + 0-10s random."""
        return self._ttl_config.BASE_TTL + random.randint(0, self._ttl_config.JITTER_MAX)

    def _should_early_refresh(self, ttl_remaining: float) -> bool:
        """
        Determine if probabilistic early refresh should trigger.

        When TTL remaining < 8s, p_refresh = 1 - (TTL_remaining / 8)
        This means closer to expiry = higher probability of refresh.
        """
        if ttl_remaining >= self._ttl_config.EARLY_REFRESH_THRESHOLD:
            return False

        # p_refresh = 1 - (TTL_remaining / 8)
        p_refresh = 1.0 - (ttl_remaining / self._ttl_config.EARLY_REFRESH_THRESHOLD)
        return random.random() < p_refresh

    def _get_redis_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for a Redis key in seconds."""
        if self._redis is None:
            return None
        try:
            ttl = self._redis.ttl(key)
            return ttl if ttl > 0 else None
        except Exception:
            return None

    def get_correlation_matrix(self, timeframe: str) -> Optional[CorrelationMatrixData]:
        """
        Get cached correlation matrix for timeframe.

        Args:
            timeframe: 'M5' or 'H1'

        Returns:
            CorrelationMatrixData if cached, None otherwise
        """
        cache_key = f"risk:correlation:{timeframe}"

        # Try Redis first
        if self._redis is not None:
            try:
                data = self._redis.get(cache_key)
                if data:
                    return CorrelationMatrixData.from_json(data)
            except Exception as e:
                logger.warning(f"Redis get failed for {cache_key}: {e}")

        # Fall back to local cache
        if cache_key in self._local_cache:
            cached_data = self._local_cache[cache_key]
            cache_age = time.time() - self._local_cache_timestamps[cache_key]
            jittered_ttl = self._ttl_config.BASE_TTL + self._ttl_config.JITTER_MAX

            # Check if local cache is still valid
            if cache_age < jittered_ttl:
                # Check for early refresh
                ttl_remaining = jittered_ttl - cache_age
                if self._should_early_refresh(ttl_remaining):
                    logger.info(f"Early refresh triggered for {cache_key}, TTL remaining: {ttl_remaining:.2f}s")
                    return None  # Signal caller to refresh
                return cached_data
            else:
                # Cache expired
                del self._local_cache[cache_key]
                del self._local_cache_timestamps[cache_key]

        return None

    def set_correlation_matrix(
        self,
        timeframe: str,
        matrix: list,
        sample_count: int = 0
    ) -> bool:
        """
        Cache correlation matrix for timeframe with jittered TTL.

        Args:
            timeframe: 'M5' or 'H1'
            matrix: 2D correlation matrix as nested list
            sample_count: Number of samples used to compute matrix

        Returns:
            True if cached successfully, False otherwise
        """
        cache_key = f"risk:correlation:{timeframe}"
        ttl_seconds = self._get_ttl_seconds()
        computed_at = time.time()

        cache_data = CorrelationMatrixData(
            matrix=matrix,
            computed_at=computed_at,
            sample_count=sample_count
        )

        # Store in Redis if available
        if self._redis is not None:
            try:
                self._redis.setex(
                    cache_key,
                    ttl_seconds,
                    cache_data.to_json()
                )
                logger.debug(f"Cached correlation matrix for {cache_key}, TTL: {ttl_seconds}s")
                return True
            except Exception as e:
                logger.warning(f"Redis set failed for {cache_key}: {e}")

        # Fall back to local cache
        self._local_cache[cache_key] = cache_data
        self._local_cache_timestamps[cache_key] = computed_at
        logger.debug(f"Local cached correlation matrix for {cache_key}, TTL: {ttl_seconds}s")
        return True

    def invalidate(self, timeframe: Optional[str] = None) -> None:
        """
        Invalidate cached correlation matrix.

        Args:
            timeframe: Specific timeframe to invalidate, or None for all
        """
        if timeframe:
            cache_key = f"risk:correlation:{timeframe}"
            if self._redis is not None:
                try:
                    self._redis.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Redis delete failed for {cache_key}: {e}")
            if cache_key in self._local_cache:
                del self._local_cache[cache_key]
                del self._local_cache_timestamps[cache_key]
        else:
            # Invalidate all timeframes
            for tf in ['M5', 'H1']:
                self.invalidate(tf)

    def get_ttl_remaining(self, timeframe: str) -> Optional[int]:
        """Get remaining TTL for a cached timeframe in seconds."""
        cache_key = f"risk:correlation:{timeframe}"

        if self._redis is not None:
            return self._get_redis_ttl(cache_key)

        # Local cache TTL calculation
        if cache_key in self._local_cache_timestamps:
            cache_age = time.time() - self._local_cache_timestamps[cache_key]
            jittered_ttl = self._ttl_config.BASE_TTL + self._ttl_config.JITTER_MAX
            remaining = jittered_ttl - cache_age
            return max(0, int(remaining))
        return None

    def should_recalculate(self, timeframe: str, force: bool = False) -> bool:
        """
        Determine if correlation matrix should be recalculated.

        Args:
            timeframe: 'M5' or 'H1'
            force: Force recalculation regardless of cache state

        Returns:
            True if recalculation needed, False if cached value can be used
        """
        if force:
            return True

        cached = self.get_correlation_matrix(timeframe)
        if cached is None:
            return True

        # Check if early refresh was signaled
        ttl_remaining = self.get_ttl_remaining(timeframe)
        if ttl_remaining is not None and ttl_remaining < self._ttl_config.EARLY_REFRESH_THRESHOLD:
            if self._should_early_refresh(ttl_remaining):
                return True

        return False
