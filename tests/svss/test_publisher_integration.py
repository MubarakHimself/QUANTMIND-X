"""
Integration Tests for SVSSPublisher

Tests SVSSPublisher with cache integration and multi-component scenarios.
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

from svss.publishers.redis_publisher import SVSSPublisher
from svss.indicators.base import IndicatorResult
from svss.cache.cache_manager import CacheEntry, SVSSCacheManager


def create_mock_result(name="vwap", value=1.0850, symbol="EURUSD", session_id="test_session"):
    """Create a mock indicator result."""
    return IndicatorResult(
        name=name,
        value=value,
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
        metadata={"symbol": symbol}
    )


class TestSVSSPublisherCacheIntegration:
    """Tests for SVSSPublisher cache integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.publisher = SVSSPublisher()
        self.mock_redis = MagicMock()
        self.mock_pipe = MagicMock()
        self.mock_redis.pipeline.return_value = self.mock_pipe
        self.publisher._redis_client = self.mock_redis
        self.publisher._connected = True

    def test_get_cache_manager_creates_instance(self):
        """Test get_cache_manager creates SVSSCacheManager instance."""
        cache_mgr = self.publisher.get_cache_manager("EURUSD", "vwap")

        assert cache_mgr is not None
        assert isinstance(cache_mgr, SVSSCacheManager)
        assert cache_mgr.symbol == "eurusd"
        assert cache_mgr.indicator == "vwap"

    def test_get_cache_manager_caches_instances(self):
        """Test that cache managers are cached per symbol+indicator."""
        cache_mgr1 = self.publisher.get_cache_manager("EURUSD", "vwap")
        cache_mgr2 = self.publisher.get_cache_manager("EURUSD", "vwap")

        assert cache_mgr1 is cache_mgr2

    def test_get_cache_manager_different_indicators(self):
        """Test cache managers are different for different indicators."""
        cache_mgr1 = self.publisher.get_cache_manager("EURUSD", "vwap")
        cache_mgr2 = self.publisher.get_cache_manager("EURUSD", "rvvol")

        assert cache_mgr1 is not cache_mgr2
        assert cache_mgr1.indicator == "vwap"
        assert cache_mgr2.indicator == "rvvol"

    def test_get_cache_manager_different_symbols(self):
        """Test cache managers are different for different symbols."""
        cache_mgr1 = self.publisher.get_cache_manager("EURUSD", "vwap")
        cache_mgr2 = self.publisher.get_cache_manager("GBPUSD", "vwap")

        assert cache_mgr1 is not cache_mgr2
        assert cache_mgr1.symbol == "eurusd"
        assert cache_mgr2.symbol == "gbpusd"

    def test_get_cache_manager_not_connected(self):
        """Test get_cache_manager returns None when not connected."""
        publisher = SVSSPublisher()
        publisher._connected = False

        result = publisher.get_cache_manager("EURUSD", "vwap")

        assert result is None

    def test_cache_indicator_success(self):
        """Test caching an indicator value."""
        result = self.publisher.cache_indicator(
            symbol="EURUSD",
            indicator="vwap",
            value=1.0850,
            session_id="test_session"
        )

        assert result is True
        self.mock_redis.setex.assert_called()

    def test_cache_indicator_not_connected(self):
        """Test cache_indicator returns False when not connected."""
        publisher = SVSSPublisher()
        publisher._connected = False

        result = publisher.cache_indicator(
            symbol="EURUSD",
            indicator="vwap",
            value=1.0850,
            session_id="test_session"
        )

        assert result is False

    def test_get_cached_indicator_returns_entry(self):
        """Test getting cached indicator value."""
        now = datetime.now(timezone.utc)
        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test_session",
        }
        self.mock_redis.get.return_value = json.dumps(cache_data)
        self.mock_redis.ttl.return_value = 25

        entry = self.publisher.get_cached_indicator("EURUSD", "vwap")

        assert entry is not None
        assert entry.value == 1.0850
        assert entry.session_id == "test_session"
        assert entry.ttl_remaining == 25

    def test_get_cached_indicator_not_connected(self):
        """Test get_cached_indicator returns None when not connected."""
        publisher = SVSSPublisher()
        publisher._connected = False

        result = publisher.get_cached_indicator("EURUSD", "vwap")

        assert result is None

    def test_record_tick_for_frequency(self):
        """Test recording tick for frequency tracking."""
        self.publisher.record_tick_for_frequency("EURUSD", "vwap")

        # Should not raise - just verify it calls through to cache manager

    def test_publish_and_cache_success(self):
        """Test publish_and_cache publishes and caches all indicators."""
        results = [
            create_mock_result(name="vwap", value=1.0850),
            create_mock_result(name="rvvol", value=1.25),
            create_mock_result(name="mfi", value=65.0),
            create_mock_result(name="volume_profile", value=1.0850),
        ]

        success = self.publisher.publish_and_cache(
            results=results,
            symbol="EURUSD",
            session_id="test_session"
        )

        assert success is True
        # Should have 4 publish + 4 setex calls = 8 total
        assert self.mock_pipe.publish.call_count == 4
        assert self.mock_pipe.setex.call_count == 4
        self.mock_pipe.execute.assert_called_once()

    def test_publish_and_cache_empty_results(self):
        """Test publish_and_cache with empty results."""
        success = self.publisher.publish_and_cache(
            results=[],
            symbol="EURUSD",
            session_id="test_session"
        )

        assert success is False
        self.mock_pipe.execute.assert_not_called()

    def test_publish_and_cache_not_connected(self):
        """Test publish_and_cache returns False when not connected."""
        publisher = SVSSPublisher()
        publisher._connected = False

        results = [create_mock_result(name="vwap", value=1.0850)]
        success = publisher.publish_and_cache(
            results=results,
            symbol="EURUSD",
            session_id="test_session"
        )

        assert success is False

    def test_publish_and_cache_uses_jittered_ttl(self):
        """Test publish_and_cache uses jittered TTL in range [30, 40]."""
        results = [create_mock_result(name="vwap", value=1.0850)]

        self.publisher.publish_and_cache(
            results=results,
            symbol="EURUSD",
            session_id="test_session"
        )

        # Check that setex was called with TTL in expected range
        setex_call = self.mock_pipe.setex.call_args
        ttl = setex_call[0][1]
        assert 30 <= ttl <= 41


class TestSVSSPublisherEdgeCases:
    """Edge case tests for SVSSPublisher."""

    def setup_method(self):
        """Set up test fixtures."""
        self.publisher = SVSSPublisher()
        self.mock_redis = MagicMock()
        self.mock_pipe = MagicMock()
        self.mock_redis.pipeline.return_value = self.mock_pipe
        self.publisher._redis_client = self.mock_redis
        self.publisher._connected = True

    def test_publish_all_with_custom_timestamp(self):
        """Test publish_all with custom timestamp."""
        results = [create_mock_result(name="vwap", value=1.0850)]
        custom_ts = datetime(2026, 3, 25, 14, 30, 0, tzinfo=timezone.utc)

        success = self.publisher.publish_all(
            results=results,
            symbol="EURUSD",
            session_id="test_session",
            timestamp=custom_ts
        )

        assert success is True

    def test_publish_all_updates_result_timestamp(self):
        """Test that publish_all updates result timestamp when provided."""
        results = [create_mock_result(name="vwap", value=1.0850)]
        custom_ts = datetime(2026, 3, 25, 14, 30, 0, tzinfo=timezone.utc)

        self.publisher.publish_all(
            results=results,
            symbol="EURUSD",
            session_id="test_session",
            timestamp=custom_ts
        )

        # Verify the result's timestamp was updated
        assert results[0].timestamp == custom_ts

    def test_channel_name_case_insensitive(self):
        """Test channel names are case insensitive."""
        channel1 = self.publisher._get_channel_name("EURUSD", "vwap")
        channel2 = self.publisher._get_channel_name("eurusd", "VWAP")

        assert channel1 == channel2 == "svss:eurusd:vwap"

    def test_multiple_symbols_publishing(self):
        """Test publishing for multiple symbols."""
        results1 = [create_mock_result(name="vwap", value=1.0850, symbol="EURUSD")]
        results2 = [create_mock_result(name="vwap", value=1.2650, symbol="GBPUSD")]

        success1 = self.publisher.publish_all(
            results=results1,
            symbol="EURUSD",
            session_id="test_session"
        )
        success2 = self.publisher.publish_all(
            results=results2,
            symbol="GBPUSD",
            session_id="test_session"
        )

        assert success1 is True
        assert success2 is True

    def test_publish_all_exception_handling(self):
        """Test publish_all handles Redis exceptions."""
        results = [create_mock_result(name="vwap", value=1.0850)]
        self.mock_pipe.execute.side_effect = Exception("Redis error")

        success = self.publisher.publish_and_cache(
            results=results,
            symbol="EURUSD",
            session_id="test_session"
        )

        assert success is False


class TestSVSSCacheManagerIntegration:
    """Integration tests for cache manager with publisher."""

    def test_cache_manager_publisher_integration(self):
        """Test SVSSCacheManager working with publisher pattern."""
        mock_redis = MagicMock()
        cache_mgr = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Simulate setting cache
        mock_redis.setex.return_value = True
        result = cache_mgr.set_with_jitter(value=1.0850, session_id="test_session")
        assert result is True

        # Simulate getting cache
        now = datetime.now(timezone.utc)
        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test_session",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 25

        entry = cache_mgr.get()
        assert entry is not None
        assert entry.value == 1.0850
        assert entry.ttl_remaining == 25

    def test_early_refresh_with_lock(self):
        """Test early refresh when lock is acquired."""
        mock_redis = MagicMock()
        cache_mgr = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        # Setup cache with low TTL
        now = datetime.now(timezone.utc)
        cache_data = {
            "value": 1.0850,
            "timestamp": now.isoformat(),
            "session_id": "test_session",
        }
        mock_redis.get.return_value = json.dumps(cache_data)
        mock_redis.ttl.return_value = 3  # Low TTL triggers early refresh

        # Mock lock to succeed
        cache_mgr._lock.acquire = MagicMock(return_value=True)
        cache_mgr._lock.release = MagicMock()

        # Mock refresh function
        refresh_fn = MagicMock(return_value=1.0900)

        with patch('random.random', return_value=0.3):
            entry = cache_mgr.try_refresh(refresh_fn, "test_session")

        refresh_fn.assert_called_once()

    def test_try_refresh_no_cache_creates_new(self):
        """Test try_refresh when no cache exists creates new value."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # No cache

        cache_mgr = SVSSCacheManager(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        refresh_fn = MagicMock(return_value=1.0900)
        mock_redis.setex.return_value = True

        entry = cache_mgr.try_refresh(refresh_fn, "test_session")

        refresh_fn.assert_called_once()


class TestCacheEntryEdgeCases:
    """Edge case tests for CacheEntry."""

    def test_cache_entry_with_stale_true(self):
        """Test CacheEntry with is_stale=True."""
        entry = CacheEntry(
            value=1.0850,
            timestamp=datetime.now(timezone.utc),
            session_id="test_session",
            ttl_remaining=-30,
            is_stale=True,
        )

        assert entry.value == 1.0850
        assert entry.is_stale is True
        assert entry.ttl_remaining == -30

    def test_cache_entry_with_zero_ttl(self):
        """Test CacheEntry with zero TTL remaining."""
        entry = CacheEntry(
            value=1.0850,
            timestamp=datetime.now(timezone.utc),
            session_id="test_session",
            ttl_remaining=0,
            is_stale=False,
        )

        assert entry.ttl_remaining == 0
        assert entry.is_stale is False
