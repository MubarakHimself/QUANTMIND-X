"""
Unit Tests for SVSS Cache Manager

Tests jittered TTL, probabilistic early refresh, and stale-while-revalidate.
"""

import pytest
import json
import time
import random
from unittest.mock import MagicMock, patch, call
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


class TestSVSSCacheManager:
    """Tests for SVSSCacheManager."""

    def test_initialization(self):
        """Test cache manager initialization."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        assert manager.symbol == "eurusd"
        assert manager.indicator == "vwap"
        assert manager.cache_key == "svss:cache:eurusd:vwap"

    def test_symbol_normalized_to_lowercase(self):
        """Test symbol is normalized to lowercase."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="GBPUSD",
            indicator="RVOL",
        )

        assert manager.symbol == "gbpusd"
        assert manager.indicator == "rvol"

    def test_jittered_ttl_normal_mode(self):
        """Test jittered TTL in normal mode (30 + 0-10s)."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Sample TTL values
        ttls = [manager._compute_jittered_ttl() for _ in range(100)]

        # All should be in range [30, 40]
        assert all(30 <= ttl <= 41 for ttl in ttls)

        # Average should be close to 35 (30 + 5)
        avg_ttl = sum(ttls) / len(ttls)
        assert 33 <= avg_ttl <= 37

    def test_jittered_ttl_high_load_mode(self):
        """Test jittered TTL in high load mode (30 + 0-20s)."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Simulate high load by recording many ticks
        for _ in range(20):
            manager.record_tick()
            time.sleep(0.01)

        # Now record tick frequency should be high
        manager._tick_times = [time.monotonic() - 0.01] * 20  # 20 ticks in 10ms = 2000 ticks/sec

        ttls = [manager._compute_jittered_ttl() for _ in range(100)]

        # All should be in range [30, 50] with high load jitter
        # (might include some normal mode samples)
        assert all(30 <= ttl <= 51 for ttl in ttls)

    def test_tick_frequency_calculation(self):
        """Test tick frequency calculation."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # No ticks recorded
        assert manager._get_tick_frequency() == 0.0

        # Record ticks
        now = time.monotonic()
        manager._tick_times = [now - 0.5, now - 0.4, now - 0.3, now - 0.2, now - 0.1]

        freq = manager._get_tick_frequency()
        assert freq > 0

    def test_set_with_jitter_stores_correct_data(self):
        """Test set_with_jitter stores correct data in Redis."""
        mock_redis = MagicMock()
        mock_redis.setex.return_value = True

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = manager.set_with_jitter(value=1.0850, session_id="test_session")

        assert result is True
        mock_redis.setex.assert_called_once()

        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "svss:cache:eurusd:vwap"
        assert isinstance(call_args[0][1], int)  # TTL
        assert 30 <= call_args[0][1] <= 41

        # Verify stored data
        stored_data = json.loads(call_args[0][2])
        assert stored_data["value"] == 1.0850
        assert stored_data["session_id"] == "test_session"
        assert "timestamp" in stored_data

    def test_get_returns_cache_entry(self):
        """Test get returns CacheEntry on cache hit."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test_session",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 25

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get()

        assert entry is not None
        assert entry.value == 1.0850
        assert entry.session_id == "test_session"
        assert entry.ttl_remaining == 25
        assert entry.is_stale is False

    def test_get_returns_none_on_cache_miss(self):
        """Test get returns None on cache miss."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get()

        assert entry is None

    def test_should_early_refresh_ttl_above_threshold(self):
        """Test should_early_refresh returns False when TTL >= 8s."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = CacheEntry(
            value=1.0850,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=10,  # Above threshold
            is_stale=False,
        )

        # Mock random to control outcome
        with patch('random.random', return_value=0.5):
            result = manager.should_early_refresh(entry)

        assert result is False

    def test_should_early_refresh_probability_at_zero_ttl(self):
        """Test should_early_refresh always True when TTL = 0."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = CacheEntry(
            value=1.0850,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=0,
            is_stale=False,
        )

        # At TTL=0, p_refresh = 1 - (0/8) = 1, so always triggers
        for _ in range(10):
            result = manager.should_early_refresh(entry)
            assert result is True

    def test_should_early_refresh_probability_at_ttl_1(self):
        """Test should_early_refresh p=0.875 when TTL = 1."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = CacheEntry(
            value=1.0850,
            timestamp=datetime.now(timezone.utc),
            session_id="test",
            ttl_remaining=1,
            is_stale=False,
        )

        # p_refresh = 1 - (1/8) = 0.875
        # r < 0.875 should trigger, r >= 0.875 should not
        trigger_count = 0
        for _ in range(1000):
            # Force random to return values that should trigger
            with patch('random.random', return_value=random.uniform(0, 0.8)):
                if manager.should_early_refresh(entry):
                    trigger_count += 1

        # Most should trigger
        assert trigger_count > 800

    def test_get_with_stale_returns_fresh_value(self):
        """Test get_with_stale returns fresh value when TTL > 0."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 20  # Fresh

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get_with_stale()

        assert entry is not None
        assert entry.is_stale is False

    def test_get_with_stale_returns_stale_within_limit(self):
        """Test get_with_stale returns stale value within limit."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -30  # Expired 30 seconds ago

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get_with_stale(max_stale_age=60)

        assert entry is not None
        assert entry.is_stale is True

    def test_get_with_stale_returns_none_when_too_stale(self):
        """Test get_with_stale returns None when stale exceeds limit."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -90  # Expired 90 seconds ago

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        entry = manager.get_with_stale(max_stale_age=60)

        assert entry is None

    def test_try_refresh_fresh_cache_not_refreshed(self):
        """Test try_refresh doesn't refresh when cache is fresh."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 25  # Fresh

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        refresh_fn = MagicMock(return_value=1.0900)

        entry = manager.try_refresh(refresh_fn, "test_session")

        assert entry is not None
        assert entry.value == 1.0850
        refresh_fn.assert_not_called()

    def test_try_refresh_lock_acquisition_failure(self):
        """Test try_refresh uses stale value when lock not acquired."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -5  # Stale but within limit

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Mock lock to fail acquisition
        manager._lock.acquire = MagicMock(return_value=False)

        refresh_fn = MagicMock(return_value=1.0900)

        entry = manager.try_refresh(refresh_fn, "test_session")

        assert entry is not None
        assert entry.is_stale is True
        refresh_fn.assert_not_called()

    def test_try_refresh_success(self):
        """Test try_refresh successfully refreshes cache."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 3  # Low TTL, will trigger refresh

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Mock lock to succeed
        manager._lock.acquire = MagicMock(return_value=True)
        manager._lock.release = MagicMock()

        refresh_fn = MagicMock(return_value=1.0900)

        # Force early refresh to trigger (p=0.625 at TTL=3, so r must be < 0.625)
        with patch('random.random', return_value=0.3):
            entry = manager.try_refresh(refresh_fn, "test_session")

        assert entry is not None
        refresh_fn.assert_called_once()
        manager._lock.release.assert_called_once()

    def test_invalidate(self):
        """Test invalidate deletes cache entry."""
        mock_redis = MagicMock()
        mock_redis.delete.return_value = 1

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = manager.invalidate()

        assert result is True
        mock_redis.delete.assert_called_once_with("svss:cache:eurusd:vwap")

    def test_get_ttl_remaining(self):
        """Test get_ttl_remaining returns correct value."""
        mock_redis = MagicMock()
        mock_redis.ttl.return_value = 25

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        ttl = manager.get_ttl_remaining()

        assert ttl == 25

    def test_get_ttl_remaining_handles_negative(self):
        """Test get_ttl_remaining returns 0 for negative values."""
        mock_redis = MagicMock()
        mock_redis.ttl.return_value = -1

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        ttl = manager.get_ttl_remaining()

        assert ttl == 0

    def test_record_tick(self):
        """Test record_tick adds tick time."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        initial_len = len(manager._tick_times)
        manager.record_tick()

        assert len(manager._tick_times) == initial_len + 1


class TestJitteredTTLDistribution:
    """Tests for jittered TTL distribution properties."""

    def test_ttl_distribution_average(self):
        """Test average TTL approaches 35s over many samples."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        ttls = [manager._compute_jittered_ttl() for _ in range(1000)]
        avg_ttl = sum(ttls) / len(ttls)

        # Average should be close to 35 (base 30 + 5 average jitter)
        # Due to int() truncation, average is slightly lower
        assert 34.0 <= avg_ttl <= 35.5

    def test_ttl_distribution_variance(self):
        """Test TTL has appropriate variance."""
        mock_redis = MagicMock()
        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        ttls = [manager._compute_jittered_ttl() for _ in range(100)]

        # Min and max should be within expected range
        assert min(ttls) >= 30
        assert max(ttls) <= 41

        # Should have good variance (not all the same)
        # With integer TTLs in range [30, 40], we expect ~10-11 unique values
        unique_values = len(set(ttls))
        assert unique_values >= 8, f"Expected at least 8 unique TTL values, got {unique_values}"

        # Average should be close to 35
        assert 33 <= sum(ttls) / len(ttls) <= 37


class TestProbabilisticEarlyRefresh:
    """Tests for probabilistic early refresh formula."""

    def test_probability_formula_ttl_0(self):
        """Test p_refresh = 1 when TTL = 0."""
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
            ttl_remaining=0,
        )

        # p_refresh = 1 - (0/8) = 1
        # Should always trigger
        count = sum(1 for _ in range(100) if manager.should_early_refresh(entry))
        assert count == 100

    def test_probability_formula_ttl_8(self):
        """Test p_refresh = 0 when TTL = 8."""
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
            ttl_remaining=8,
        )

        # p_refresh = 1 - (8/8) = 0
        # Should never trigger
        count = sum(1 for _ in range(100) if manager.should_early_refresh(entry))
        assert count == 0

    def test_probability_formula_ttl_4(self):
        """Test p_refresh = 0.5 when TTL = 4."""
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
            ttl_remaining=4,
        )

        # p_refresh = 1 - (4/8) = 0.5
        # When r = 0.49 (< p_refresh), should trigger
        # When r = 0.51 (> p_refresh), should not trigger
        with patch('random.random', return_value=0.49):
            result = manager.should_early_refresh(entry)
            assert result is True, "r=0.49 < p_refresh=0.5 should trigger"

        with patch('random.random', return_value=0.51):
            result = manager.should_early_refresh(entry)
            assert result is False, "r=0.51 > p_refresh=0.5 should not trigger"

        # Also verify boundary cases
        with patch('random.random', return_value=0.0):
            result = manager.should_early_refresh(entry)
            assert result is True, "r=0.0 < p_refresh should always trigger"

        with patch('random.random', return_value=0.4999):
            result = manager.should_early_refresh(entry)
            assert result is True, "r=0.4999 < p_refresh should trigger"


class TestStaleWhileRevalidate:
    """Tests for stale-while-revalidate 60s maximum."""

    def test_stale_max_60_seconds(self):
        """Test stale values are returned up to 60s maximum."""
        mock_redis = MagicMock()
        now = datetime.now(timezone.utc)

        manager = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Test at 30 seconds stale
        cache_data = {"value": 1.0, "timestamp": now.isoformat(), "session_id": "test"}
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = -30

        entry = manager.get_with_stale(max_stale_age=60)
        assert entry is not None
        assert entry.is_stale is True

        # Test at 59 seconds stale
        mock_redis.ttl.return_value = -59
        entry = manager.get_with_stale(max_stale_age=60)
        assert entry is not None

        # Test at 61 seconds stale (should return None)
        mock_redis.ttl.return_value = -61
        entry = manager.get_with_stale(max_stale_age=60)
        assert entry is None
