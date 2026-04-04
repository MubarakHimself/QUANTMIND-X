"""
Unit Tests for Distributed Lock Manager

Tests distributed locking behavior without requiring actual Redis connection.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from svss.cache.lock_manager import DistributedLock, LockAcquisitionError


class TestDistributedLock:
    """Tests for DistributedLock."""

    def test_initialization(self):
        """Test lock initialization."""
        mock_redis = MagicMock()
        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
            lock_ttl=5,
        )

        assert lock.lock_key == "svss:lock:eurusd:vwap"
        assert lock.consumer_id is not None
        assert len(lock.consumer_id) > 0

    def test_lock_key_format(self):
        """Test lock key follows correct format."""
        mock_redis = MagicMock()
        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="GBPUSD",
            indicator="rvvol",
        )

        assert lock.lock_key == "svss:lock:gbpusd:rvvol"

    def test_acquire_success(self):
        """Test successful lock acquisition."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = True  # Lock acquired

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = lock.acquire()

        assert result is True
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        assert call_args[0][0] == lock.lock_key
        assert call_args[0][1] == lock.consumer_id
        assert call_args[1]["nx"] is True
        assert call_args[1]["ex"] == 5

    def test_acquire_failure_already_held(self):
        """Test lock acquisition fails when already held."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = None  # Lock not acquired (key exists)

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = lock.acquire()

        assert result is False

    def test_acquire_blocks_then_times_out(self):
        """Test blocking acquisition with timeout."""
        mock_redis = MagicMock()
        # First attempt fails, second succeeds
        mock_redis.set.side_effect = [None, True]

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        start = time.monotonic()
        result = lock.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is True
        assert elapsed >= 0.01  # Should have waited at least briefly

    def test_acquire_non_blocking(self):
        """Test non-blocking acquisition returns immediately."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = None

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        start = time.monotonic()
        result = lock.acquire(timeout=0.0)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.05  # Should return almost immediately

    def test_release_success(self):
        """Test successful lock release."""
        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_script.return_value = 1  # Lock released
        mock_redis.register_script.return_value = mock_script

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._release_script = mock_script

        result = lock.release()

        assert result is True

    def test_release_not_held(self):
        """Test release fails when lock not held."""
        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_script.return_value = 0  # Lock not released (not held)
        mock_redis.register_script.return_value = mock_script

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._release_script = mock_script

        result = lock.release()

        assert result is False

    def test_is_held_by_us(self):
        """Test is_held returns True when we hold the lock."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"test-consumer-id"

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._consumer_id = "test-consumer-id"

        assert lock.is_held() is True

    def test_is_held_by_other(self):
        """Test is_held returns False when another consumer holds."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"other-consumer-id"

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._consumer_id = "test-consumer-id"

        assert lock.is_held() is False

    def test_is_held_not_locked(self):
        """Test is_held returns False when lock doesn't exist."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        assert lock.is_held() is False

    def test_get_holder_returns_id(self):
        """Test get_holder returns consumer ID."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"holder-id-123"

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        assert lock.get_holder() == "holder-id-123"

    def test_get_holder_returns_none_when_free(self):
        """Test get_holder returns None when lock is free."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        assert lock.get_holder() is None

    def test_extend_success(self):
        """Test successful lock extension."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"test-consumer-id"
        mock_redis.expire.return_value = True

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._consumer_id = "test-consumer-id"

        result = lock.extend(additional_ttl=10)

        assert result is True
        mock_redis.expire.assert_called_once()

    def test_extend_not_held(self):
        """Test extend fails when lock not held."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = lock.extend(additional_ttl=10)

        assert result is False

    def test_unique_consumer_ids(self):
        """Test that each lock instance has a unique consumer ID."""
        mock_redis = MagicMock()

        lock1 = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock2 = DistributedLock(mock_redis, "EURUSD", "vwap")

        assert lock1.consumer_id != lock2.consumer_id

    def test_redis_error_on_acquire(self):
        """Test graceful handling of Redis error during acquire."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.set.side_effect = redis_lib.RedisError("Connection failed")

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )

        result = lock.acquire()

        assert result is False

    def test_redis_error_on_release(self):
        """Test graceful handling of Redis error during release."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_script.side_effect = redis_lib.RedisError("Connection failed")
        mock_redis.register_script.return_value = mock_script

        lock = DistributedLock(
            redis_client=mock_redis,
            symbol="EURUSD",
            indicator="vwap",
        )
        lock._release_script = mock_script

        result = lock.release()

        assert result is False
