"""Tests for QuantMindLib V1 — DPRRedisPublisher."""

import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.library.core.bridges.dpr_redis_bridge import DPRRedisPublisher, _DPR_SCORES_KEY, _DPR_BOT_KEY_PREFIX
from src.library.core.bridges.sentinel_dpr_bridges import DPRScore, DPRBridge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dpr_score(
    bot_id: str = "bot_001",
    dpr_score: float = 0.85,
    sharpe: float = 1.2,
    win_rate: float = 0.65,
    daily_pnl: float = 150.0,
    rank: int = 1,
    tier: str = "ELITE",
) -> DPRScore:
    return DPRScore(
        bot_id=bot_id,
        dpr_score=dpr_score,
        sharpe_today=sharpe,
        win_rate_today=win_rate,
        daily_pnl=daily_pnl,
        rank=rank,
        tier=tier,
        computed_at_ms=int(time.time() * 1000),
    )


def _make_mock_pipeline() -> MagicMock:
    """Create a mock Redis pipeline that supports chaining."""
    pipe = MagicMock()
    pipe.zadd = MagicMock(return_value=pipe)
    pipe.hset = MagicMock(return_value=pipe)
    pipe.expire = MagicMock(return_value=pipe)
    pipe.execute = MagicMock(return_value=[])
    return pipe


def _make_mock_redis_client() -> MagicMock:
    """Create a fully-mocked Redis client."""
    client = MagicMock()
    client.pipeline = MagicMock(return_value=_make_mock_pipeline())
    client.zrevrange = MagicMock(return_value=[])
    client.hgetall = MagicMock(return_value={})
    client.delete = MagicMock(return_value=0)
    client.scan = MagicMock(return_value=(0, []))
    client.ping = MagicMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# TestPublisherInit
# ---------------------------------------------------------------------------

class TestPublisherInit:
    def test_default_init_degraded_mode(self):
        """DPRRedisPublisher falls back to no client when Redis connection fails."""
        # Patch redis.Redis at the global level so the import inside __init__ gets
        # the mock that raises, leaving _client as None after the try/except.
        with patch("redis.Redis", side_effect=Exception("Connection refused")):
            publisher = DPRRedisPublisher()
            assert publisher._client is None

    def test_custom_client(self):
        """DPRRedisPublisher accepts a custom Redis client."""
        mock_client = _make_mock_redis_client()
        publisher = DPRRedisPublisher(redis_client=mock_client)
        assert publisher._client is mock_client


# ---------------------------------------------------------------------------
# TestPublishScores
# ---------------------------------------------------------------------------

class TestPublishScores:
    def test_publishes_all_scores_to_sorted_set(self):
        """publish_scores calls ZADD for each bot in the sorted set."""
        mock_client = _make_mock_redis_client()
        mock_pipe = mock_client.pipeline.return_value
        publisher = DPRRedisPublisher(redis_client=mock_client)

        bridge = DPRBridge(scores={
            "bot_a": _make_dpr_score(bot_id="bot_a", dpr_score=0.9),
            "bot_b": _make_dpr_score(bot_id="bot_b", dpr_score=0.7),
        })

        count = publisher.publish_scores(bridge)

        assert count == 2
        zadd_calls = mock_pipe.zadd.call_args_list
        assert len(zadd_calls) == 2
        # Verify each ZADD was called with the sorted set key and bot->score mapping
        for call in zadd_calls:
            args, kwargs = call
            assert args[0] == _DPR_SCORES_KEY
            assert isinstance(args[1], dict)
            assert len(args[1]) == 1  # one bot per call

    def test_publishes_hash_per_bot(self):
        """publish_scores calls HSET with all DPRScore fields for each bot."""
        mock_client = _make_mock_redis_client()
        mock_pipe = mock_client.pipeline.return_value
        publisher = DPRRedisPublisher(redis_client=mock_client)

        score = _make_dpr_score(bot_id="bot_x", dpr_score=0.8, sharpe=1.5, tier="ELITE")
        bridge = DPRBridge(scores={"bot_x": score})

        publisher.publish_scores(bridge)

        # Should have one HSET call for the bot hash
        hset_calls = [c for c in mock_pipe.hset.call_args_list]
        assert len(hset_calls) == 1
        # Verify the hash mapping contains all required fields
        _, kwargs = hset_calls[0]
        mapping = kwargs["mapping"]
        assert mapping["bot_id"] == "bot_x"
        assert mapping["tier"] == "ELITE"
        assert "sharpe_today" in mapping
        assert "win_rate_today" in mapping
        assert "daily_pnl" in mapping
        assert "rank" in mapping
        assert "computed_at_ms" in mapping

    def test_returns_count(self):
        """publish_scores returns the number of scores published."""
        mock_client = _make_mock_redis_client()
        publisher = DPRRedisPublisher(redis_client=mock_client)

        bridge = DPRBridge(scores={
            "bot_1": _make_dpr_score(bot_id="bot_1"),
            "bot_2": _make_dpr_score(bot_id="bot_2"),
            "bot_3": _make_dpr_score(bot_id="bot_3"),
        })

        count = publisher.publish_scores(bridge)
        assert count == 3

    def test_returns_zero_when_no_scores(self):
        """publish_scores returns 0 when bridge has no scores."""
        mock_client = _make_mock_redis_client()
        publisher = DPRRedisPublisher(redis_client=mock_client)

        bridge = DPRBridge(scores={})
        count = publisher.publish_scores(bridge)
        assert count == 0

    def test_returns_zero_when_client_is_none(self):
        """publish_scores returns 0 in degraded mode (no client)."""
        publisher = DPRRedisPublisher()
        bridge = DPRBridge(scores={"bot_1": _make_dpr_score()})
        count = publisher.publish_scores(bridge)
        assert count == 0


# ---------------------------------------------------------------------------
# TestPublishSingle
# ---------------------------------------------------------------------------

class TestPublishSingle:
    def test_single_score_publish(self):
        """publish_single publishes a single bot score via pipeline."""
        mock_client = _make_mock_redis_client()
        mock_pipe = mock_client.pipeline.return_value
        publisher = DPRRedisPublisher(redis_client=mock_client)

        score = _make_dpr_score(bot_id="single_bot", dpr_score=0.75)

        result = publisher.publish_single("single_bot", score)

        assert result is True
        mock_client.pipeline.assert_called_once_with(transaction=True)
        mock_pipe.zadd.assert_called()
        mock_pipe.hset.assert_called()
        mock_pipe.expire.assert_called()
        mock_pipe.execute.assert_called_once()

    def test_ttl_set(self):
        """publish_single sets TTL on both sorted set and bot hash."""
        mock_client = _make_mock_redis_client()
        mock_pipe = mock_client.pipeline.return_value
        publisher = DPRRedisPublisher(redis_client=mock_client)

        score = _make_dpr_score(bot_id="ttl_bot")
        publisher.publish_single("ttl_bot", score, ttl_seconds=600)

        expire_calls = mock_pipe.expire.call_args_list
        assert len(expire_calls) == 2  # One for sorted set, one for bot hash
        # expire(name, time_sec) — time_sec is positional arg index 1
        ttl_values = {c[0][1] for c in expire_calls}
        assert 600 in ttl_values

    def test_returns_false_when_client_is_none(self):
        """publish_single returns False in degraded mode."""
        publisher = DPRRedisPublisher()
        score = _make_dpr_score()
        result = publisher.publish_single("any_bot", score)
        assert result is False


# ---------------------------------------------------------------------------
# TestGetCachedScores
# ---------------------------------------------------------------------------

class TestGetCachedScores:
    def test_retrieves_scores_from_sorted_set(self):
        """get_cached_scores retrieves top N bots from sorted set and deserializes."""
        mock_client = _make_mock_redis_client()
        publisher = DPRRedisPublisher(redis_client=mock_client)

        mock_client.zrevrange.return_value = ["bot_a", "bot_b"]
        mock_client.hgetall.side_effect = [
            {
                "bot_id": "bot_a",
                "dpr_score": "0.9",
                "sharpe_today": "1.2",
                "win_rate_today": "0.65",
                "daily_pnl": "150.0",
                "rank": "1",
                "tier": "ELITE",
                "computed_at_ms": "1234567890",
            },
            {
                "bot_id": "bot_b",
                "dpr_score": "0.7",
                "sharpe_today": "0.8",
                "win_rate_today": "0.55",
                "daily_pnl": "75.0",
                "rank": "2",
                "tier": "PERFORMING",
                "computed_at_ms": "1234567891",
            },
        ]

        results = publisher.get_cached_scores(limit=5)

        assert len(results) == 2
        assert results[0].bot_id == "bot_a"
        assert results[0].dpr_score == 0.9
        assert results[0].tier == "ELITE"
        assert results[1].bot_id == "bot_b"
        assert results[1].dpr_score == 0.7
        mock_client.zrevrange.assert_called_once_with(_DPR_SCORES_KEY, 0, 4)

    def test_empty_when_no_scores(self):
        """get_cached_scores returns empty list when no scores in Redis."""
        mock_client = _make_mock_redis_client()
        mock_client.zrevrange.return_value = []
        publisher = DPRRedisPublisher(redis_client=mock_client)

        results = publisher.get_cached_scores()
        assert results == []

    def test_returns_empty_when_client_is_none(self):
        """get_cached_scores returns [] in degraded mode."""
        publisher = DPRRedisPublisher()
        results = publisher.get_cached_scores()
        assert results == []


# ---------------------------------------------------------------------------
# TestGetCachedScore
# ---------------------------------------------------------------------------

class TestGetCachedScore:
    def test_retrieves_single_bot_score(self):
        """get_cached_score retrieves and deserializes a single bot's DPRScore."""
        mock_client = _make_mock_redis_client()
        publisher = DPRRedisPublisher(redis_client=mock_client)

        mock_client.hgetall.return_value = {
            "bot_id": "cached_bot",
            "dpr_score": "0.82",
            "sharpe_today": "1.1",
            "win_rate_today": "0.6",
            "daily_pnl": "200.0",
            "rank": "2",
            "tier": "PERFORMING",
            "computed_at_ms": "9876543210",
        }

        result = publisher.get_cached_score("cached_bot")

        assert result is not None
        assert result.bot_id == "cached_bot"
        assert result.dpr_score == 0.82
        assert result.sharpe_today == 1.1
        assert result.win_rate_today == 0.6
        assert result.daily_pnl == 200.0
        assert result.rank == 2
        assert result.tier == "PERFORMING"
        assert result.computed_at_ms == 9876543210
        mock_client.hgetall.assert_called_once_with(f"{_DPR_BOT_KEY_PREFIX}cached_bot")

    def test_returns_none_for_unknown_bot(self):
        """get_cached_score returns None when bot is not found in Redis."""
        mock_client = _make_mock_redis_client()
        mock_client.hgetall.return_value = {}
        publisher = DPRRedisPublisher(redis_client=mock_client)

        result = publisher.get_cached_score("unknown_bot")

        assert result is None

    def test_returns_none_when_client_is_none(self):
        """get_cached_score returns None in degraded mode."""
        publisher = DPRRedisPublisher()
        result = publisher.get_cached_score("any_bot")
        assert result is None


# ---------------------------------------------------------------------------
# TestClearScores
# ---------------------------------------------------------------------------

class TestClearScores:
    def test_clears_all_dpr_keys(self):
        """clear_scores deletes the sorted set and all dpr:bot:* keys."""
        mock_client = _make_mock_redis_client()
        mock_client.scan.return_value = (0, ["dpr:bot:bot1", "dpr:bot:bot2", "dpr:bot:bot3"])
        mock_client.delete.return_value = 3
        publisher = DPRRedisPublisher(redis_client=mock_client)

        count = publisher.clear_scores()

        mock_client.delete.assert_any_call(_DPR_SCORES_KEY)
        mock_client.scan.assert_called()
        assert count == 3

    def test_returns_count(self):
        """clear_scores returns the count of deleted keys."""
        mock_client = _make_mock_redis_client()
        mock_client.scan.return_value = (0, ["dpr:bot:a", "dpr:bot:b"])
        mock_client.delete.return_value = 2
        publisher = DPRRedisPublisher(redis_client=mock_client)

        count = publisher.clear_scores()

        assert count == 2

    def test_returns_zero_when_client_is_none(self):
        """clear_scores returns 0 in degraded mode."""
        publisher = DPRRedisPublisher()
        count = publisher.clear_scores()
        assert count == 0


# ---------------------------------------------------------------------------
# TestHealthCheck
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_is_healthy_returns_true_on_ping_success(self):
        """is_healthy returns True when Redis ping succeeds."""
        mock_client = _make_mock_redis_client()
        mock_client.ping.return_value = True
        publisher = DPRRedisPublisher(redis_client=mock_client)

        assert publisher.is_healthy() is True
        mock_client.ping.assert_called_once()

    def test_is_healthy_returns_false_on_ping_failure(self):
        """is_healthy returns False when Redis ping fails."""
        mock_client = _make_mock_redis_client()
        mock_client.ping.side_effect = Exception("Connection refused")
        publisher = DPRRedisPublisher(redis_client=mock_client)

        assert publisher.is_healthy() is False

    def test_is_healthy_returns_false_when_client_is_none(self):
        """is_healthy returns False in degraded mode."""
        publisher = DPRRedisPublisher()
        assert publisher.is_healthy() is False
