"""
QuantMindLib V1 — DPR Redis Publisher

Phase 10 Packet 10A: DPRRedisPublisher
Bridges DPR scores from DPRBridge to Redis for Governor consumption.
Fixes G-18 gap: DPR scores computed but not persisted to Redis.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from src.library.core.types.enums import BotTier
from src.library.core.bridges.sentinel_dpr_bridges import DPRScore, DPRBridge

logger = logging.getLogger(__name__)

# Redis key constants
_DPR_SCORES_KEY = "dpr:scores"
_DPR_BOT_KEY_PREFIX = "dpr:bot:"
_DPR_TTL_SECONDS = 300  # 5 minutes default TTL


class DPRRedisPublisher:
    """
    Publishes DPR scores to Redis for Governor consumption.

    Redis key pattern:
    - `dpr:scores` (sorted set, score as float, bot_id as member)
    - `dpr:bot:{bot_id}` (hash with all DPRScore fields)

    This publisher bridges DPRBridge -> Redis -> Governor.
    """

    def __init__(self, redis_client: Optional[Any] = None) -> None:
        """
        Initialize with optional Redis client. Uses real client if available.

        Args:
            redis_client: Optional Redis client instance. If None, creates
                a real redis.Redis client connected to localhost:6379.
        """
        self._client = redis_client
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(host="localhost", port=6379, decode_responses=True)
            except Exception as e:
                logger.warning(f"Failed to create Redis client: {e}. DPRRedisPublisher will operate in degraded mode.")
                self._client = None

    def publish_scores(self, dpr_bridge: DPRBridge) -> int:
        """
        Publish all DPR scores from DPRBridge to Redis.

        Steps:
        1. Get all scores from dpr_bridge.scores
        2. For each score:
           - ZADD to dpr:scores with score as float
           - HSET dpr:bot:{bot_id} with all DPRScore fields
        3. Set TTL on the sorted set and individual keys
        4. Return count of published scores

        Args:
            dpr_bridge: DPRBridge instance containing computed scores.

        Returns:
            Number of scores published to Redis.
        """
        if self._client is None:
            return 0

        scores = dpr_bridge.scores
        if not scores:
            return 0

        try:
            pipe = self._client.pipeline(transaction=True)
            now_ms = int(time.time() * 1000)

            for bot_id, score in scores.items():
                # ZADD: add to sorted set with score as the member score
                pipe.zadd(_DPR_SCORES_KEY, {bot_id: score.dpr_score})
                # HSET: store all DPRScore fields as a hash
                pipe.hset(
                    f"{_DPR_BOT_KEY_PREFIX}{bot_id}",
                    mapping={
                        "bot_id": str(score.bot_id),
                        "dpr_score": str(score.dpr_score),
                        "sharpe_today": str(score.sharpe_today),
                        "win_rate_today": str(score.win_rate_today),
                        "daily_pnl": str(score.daily_pnl),
                        "rank": str(score.rank),
                        "tier": str(score.tier),
                        "computed_at_ms": str(score.computed_at_ms),
                    },
                )
                # Set TTL on individual bot key
                pipe.expire(f"{_DPR_BOT_KEY_PREFIX}{bot_id}", _DPR_TTL_SECONDS)

            # Set TTL on sorted set
            pipe.expire(_DPR_SCORES_KEY, _DPR_TTL_SECONDS)
            pipe.execute()

            logger.info(f"Published {len(scores)} DPR scores to Redis")
            return len(scores)

        except Exception as e:
            logger.error(f"Failed to publish DPR scores to Redis: {e}")
            return 0

    def publish_single(
        self,
        bot_id: str,
        dpr_score: DPRScore,
        ttl_seconds: int = 300,
    ) -> bool:
        """
        Publish a single bot's DPR score to Redis.

        Uses pipeline for atomicity.

        Args:
            bot_id: Bot identifier.
            dpr_score: DPRScore object for the bot.
            ttl_seconds: TTL in seconds for Redis keys (default 300).

        Returns:
            True on success, False on failure.
        """
        if self._client is None:
            return False

        try:
            pipe = self._client.pipeline(transaction=True)
            pipe.zadd(_DPR_SCORES_KEY, {bot_id: dpr_score.dpr_score})
            pipe.hset(
                f"{_DPR_BOT_KEY_PREFIX}{bot_id}",
                mapping={
                    "bot_id": str(dpr_score.bot_id),
                    "dpr_score": str(dpr_score.dpr_score),
                    "sharpe_today": str(dpr_score.sharpe_today),
                    "win_rate_today": str(dpr_score.win_rate_today),
                    "daily_pnl": str(dpr_score.daily_pnl),
                    "rank": str(dpr_score.rank),
                    "tier": str(dpr_score.tier),
                    "computed_at_ms": str(dpr_score.computed_at_ms),
                },
            )
            pipe.expire(f"{_DPR_BOT_KEY_PREFIX}{bot_id}", ttl_seconds)
            pipe.expire(_DPR_SCORES_KEY, ttl_seconds)
            pipe.execute()
            logger.debug(f"Published DPR score for bot {bot_id} to Redis")
            return True

        except Exception as e:
            logger.error(f"Failed to publish DPR score for bot {bot_id}: {e}")
            return False

    def get_cached_scores(self, limit: int = 10) -> List[DPRScore]:
        """
        Retrieve DPR scores from Redis (for Governor startup/recovery).

        Args:
            limit: Maximum number of scores to retrieve (default 10).

        Returns:
            List of DPRScore objects from the sorted set.
            Deserializes hash fields back to DPRScore.
        """
        if self._client is None:
            return []

        try:
            # Get top N bots from sorted set (highest scores first)
            bot_ids = self._client.zrevrange(_DPR_SCORES_KEY, 0, limit - 1)
            if not bot_ids:
                return []

            results: List[DPRScore] = []
            for bot_id in bot_ids:
                score = self.get_cached_score(bot_id)
                if score is not None:
                    results.append(score)

            return results

        except Exception as e:
            logger.error(f"Failed to retrieve cached DPR scores: {e}")
            return []

    def get_cached_score(self, bot_id: str) -> Optional[DPRScore]:
        """
        Get a single bot's DPR score from Redis.

        Args:
            bot_id: Bot identifier.

        Returns:
            DPRScore object if found, None otherwise.
        """
        if self._client is None:
            return None

        try:
            key = f"{_DPR_BOT_KEY_PREFIX}{bot_id}"
            data = self._client.hgetall(key)
            if not data:
                return None

            return DPRScore(
                bot_id=data["bot_id"],
                dpr_score=float(data["dpr_score"]),
                sharpe_today=float(data["sharpe_today"]),
                win_rate_today=float(data["win_rate_today"]),
                daily_pnl=float(data["daily_pnl"]),
                rank=int(data["rank"]),
                tier=data["tier"],
                computed_at_ms=int(data["computed_at_ms"]),
            )

        except Exception as e:
            logger.error(f"Failed to retrieve cached DPR score for bot {bot_id}: {e}")
            return None

    def clear_scores(self) -> int:
        """
        Clear all DPR scores from Redis.

        Returns:
            Count of keys deleted.
        """
        if self._client is None:
            return 0

        try:
            # Delete sorted set
            self._client.delete(_DPR_SCORES_KEY)

            # Find and delete all dpr:bot:* keys
            pattern = f"{_DPR_BOT_KEY_PREFIX}*"
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += self._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared DPR scores from Redis: {deleted} keys deleted")
            return deleted

        except Exception as e:
            logger.error(f"Failed to clear DPR scores from Redis: {e}")
            return 0

    def is_healthy(self) -> bool:
        """
        Ping Redis to verify connectivity.

        Returns:
            True if Redis is reachable, False otherwise.
        """
        if self._client is None:
            return False

        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False


def dpr_tier_to_bot_tier(dpr_tier: str) -> BotTier:
    """
    Map DPR tier string to BotTier enum.

    DPR tier names are derived from score thresholds:
    - ELITE: score >= 0.85
    - PERFORMING: score >= 0.70 (maps to BotTier.STANDARD)
    - STANDARD: score >= 0.50
    - AT_RISK: score >= 0.30
    - CIRCUIT_BROKEN: score < 0.30

    Args:
        dpr_tier: DPR tier string from DPRScore.tier field.

    Returns:
        Corresponding BotTier enum value.
    """
    mapping: Dict[str, BotTier] = {
        "ELITE": BotTier.ELITE,
        "PERFORMING": BotTier.STANDARD,  # No PERFORMING in BotTier, map to STANDARD
        "STANDARD": BotTier.STANDARD,
        "AT_RISK": BotTier.AT_RISK,
        "CIRCUIT_BROKEN": BotTier.CIRCUIT_BROKEN,
    }
    return mapping.get(dpr_tier, BotTier.STANDARD)


__all__ = ["DPRRedisPublisher", "dpr_tier_to_bot_tier"]
