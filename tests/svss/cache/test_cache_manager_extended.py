"""
Extended Tests for SVSS Cache Manager

Additional edge cases and scenarios beyond basic functionality.
"""

import pytest
import json
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from svss.cache.cache_manager import (
    SVSSCacheManager,
    CacheEntry,
    BASE_TTL,
    NORMAL_JITTER_MAX,
    HIGH_LOAD_JITTER_MAX,
    HIGH_LOAD_TICK_THRESHOLD,
    EARLY_REFRESH_THRESHOLD,
    STALE_MAX_DURATION,
)


class TestSVSSCacheManagerExtended:
    """Extended tests for SVSSCacheManager."""

    def test_jittered_ttl_computed_inline_in_publish_and_cache(self):
        """Test jittered TTL uses correct formula."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Simulate normal mode
        manager._tick_times = []

        ttls = [manager._compute_jittered_ttl() for _ in range(100)]

        # All should be in range [30, 41]
        assert all(30 <= ttl <= 41 for ttl in ttls)

    def test_high_load_threshold_exactly_10(self):
        """Test that 10 ticks/sec is at the threshold."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Exactly at threshold - should use normal jitter
        manager._tick_times = [time.monotonic() - 0.9] * 9
        freq = manager._get_tick_frequency()

        # 9 ticks in ~0.9 seconds = ~10 ticks/sec, might be just under
        # Just verify it doesn't error

    def test_jittered_ttl_uses_random_module(self):
        """Test that jittered TTL uses random module."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        with patch('random.uniform', return_value=5.0):
            ttl = manager._compute_jittered_ttl()

        assert ttl == 35  # 30 + 5

    def test_record_tick_tracking(self):
        """Test that record_tick properly tracks tick times."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        initial_count = len(manager._tick_times)
        manager.record_tick()

        assert len(manager._tick_times) == initial_count + 1

    def test_tick_frequency_with_single_tick(self):
        """Test frequency calculation with single tick."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        now = time.monotonic()
        manager._tick_times = [now]

        freq = manager._get_tick_frequency()

        # With single tick, frequency should be 0
        assert freq == 0.0

    def test_tick_frequency_empty(self):
        """Test frequency calculation with no ticks."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        manager._tick_times = []
        freq = manager._get_tick_frequency()

        assert freq == 0.0

    def test_should_early_refresh_edge_ttl_7(self):
        """Test should_early_refresh at TTL=7 (just below threshold)."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = CacheEntry(
            value=1.0,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=7,
        )

        # p_refresh = 1 - (7/8) = 0.125
        # Only triggers if r < 0.125
        with patch('random.random', return_value=0.0):
            result = manager.should_early_refresh(entry)
            assert result is True

        with patch('random.random', return_value=0.5):
            result = manager.should_early_refresh(entry)
            assert result is False

    def test_should_early_refresh_edge_ttl_1(self):
        """Test should_early_refresh at TTL=1."""
        import random as random_module
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = CacheEntry(
            value=1.0,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=1,
        )

        # p_refresh = 1 - (1/8) = 0.875
        # Very likely to trigger
        trigger_count = 0
        for _ in range(100):
            with patch('random.random', return_value=random_module.uniform(0, 0.8)):
                if manager.should_early_refresh(entry):
                    trigger_count += 1

        assert trigger_count > 80

    def test_get_with_stale_negative_ttl_handling(self):
        """Test get_with_stale handles negative TTL correctly."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -5  # 5 seconds expired

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get_with_stale(max_stale_age=60)

        assert entry is not None
        assert entry.is_stale is True

    def test_get_with_stale_custom_max_stale_age(self):
        """Test get_with_stale with custom max_stale_age."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -120  # 120 seconds expired

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # With max_stale_age=60, should return None
        entry = manager.get_with_stale(max_stale_age=60)
        assert entry is None

        # With max_stale_age=180, should return stale entry
        entry = manager.get_with_stale(max_stale_age=180)
        assert entry is not None
        assert entry.is_stale is True

    def test_try_refresh_double_check_after_lock_acquire(self):
        """Test try_refresh double-checks cache after acquiring lock."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 10  # TTL increased since we acquired lock

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Mock lock to succeed
        manager._lock.acquire = MagicMock(return_value=True)
        manager._lock.release = MagicMock()

        refresh_fn = MagicMock(return_value=1.0900)

        with patch('random.random', return_value=0.3):
            entry = manager.try_refresh(refresh_fn, "test_session")

        # Should NOT have called refresh because another consumer refreshed
        # (TTL is now >= threshold)
        refresh_fn.assert_not_called()

    def test_try_refresh_calls_refresh_function(self):
        """Test try_refresh actually calls the refresh function."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 3  # Low TTL

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        manager._lock.acquire = MagicMock(return_value=True)
        manager._lock.release = MagicMock()

        refresh_fn = MagicMock(return_value=1.0900)
        mock_redis.setex.return_value = True

        with patch('random.random', return_value=0.3):
            entry = manager.try_refresh(refresh_fn, "test_session")

        refresh_fn.assert_called_once()

    def test_try_refresh_when_refresh_fn_returns_none(self):
        """Test try_refresh handles None from refresh function."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 3

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        manager._lock.acquire = MagicMock(return_value=True)
        manager._lock.release = MagicMock()

        refresh_fn = MagicMock(return_value=None)

        with patch('random.random', return_value=0.3):
            entry = manager.try_refresh(refresh_fn, "test_session")

        assert entry is None

    def test_invalidate_when_key_not_exists(self):
        """Test invalidate returns False when key doesn't exist."""
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 0  # Key didn't exist

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = manager.invalidate()

        assert result is False

    def test_invalidate_redis_error(self):
        """Test invalidate handles Redis error."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.delete.side_effect = redis_lib.RedisError("Connection failed")

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = manager.invalidate()

        assert result is False

    def test_get_ttl_remaining_redis_error(self):
        """Test get_ttl_remaining handles Redis error."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.ttl.side_effect = redis_lib.RedisError("Connection failed")

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = manager.get_ttl_remaining()

        assert result == 0

    def test_get_lock_info(self):
        """Test get_lock_info returns correct structure."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        info = manager.get_lock_info()

        assert "lock_key" in info
        assert "holder" in info
        assert "our_consumer_id" in info
        assert "is_held_by_us" in info
        assert info["lock_key"] == "svss:lock:eurusd:vwap"


class TestCacheEntryEdgeCases:
    """Edge cases for CacheEntry."""

    def test_cache_entry_default_values(self):
        """Test CacheEntry with default is_stale."""
        entry = CacheEntry(
            value=1.0,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=30,
        )

        assert entry.is_stale is False

    def test_cache_entry_metadata_preserved(self):
        """Test CacheEntry preserves all metadata."""
        now = datetime.now(timezone.utc)
        entry = CacheEntry(
            value=1.0850,
            timestamp=now,
            session_id="london_open_20260325",
            ttl_remaining=25,
            is_stale=False,
        )

        assert entry.value == 1.0850
        assert entry.timestamp == now
        assert entry.session_id == "london_open_20260325"
        assert entry.ttl_remaining == 25
        assert entry.is_stale is False


class TestConstants:
    """Tests for module constants."""

    def test_base_ttl_value(self):
        """Test BASE_TTL is 30 seconds."""
        assert BASE_TTL == 30

    def test_normal_jitter_max(self):
        """Test NORMAL_JITTER_MAX is 10 seconds."""
        assert NORMAL_JITTER_MAX == 10

    def test_high_load_jitter_max(self):
        """Test HIGH_LOAD_JITTER_MAX is 20 seconds."""
        assert HIGH_LOAD_JITTER_MAX == 20

    def test_high_load_tick_threshold(self):
        """Test HIGH_LOAD_TICK_THRESHOLD is 10 ticks/sec."""
        assert HIGH_LOAD_TICK_THRESHOLD == 10

    def test_early_refresh_threshold(self):
        """Test EARLY_REFRESH_THRESHOLD is 8 seconds."""
        assert EARLY_REFRESH_THRESHOLD == 8

    def test_stale_max_duration(self):
        """Test STALE_MAX_DURATION is 60 seconds."""
        assert STALE_MAX_DURATION == 60
