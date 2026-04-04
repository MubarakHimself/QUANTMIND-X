"""
Extended Tests for Distributed Lock Manager

Additional edge cases and scenarios beyond basic functionality.
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from svss.cache.lock_manager import DistributedLock, LockAcquisitionError


class TestDistributedLockExtended:
    """Extended tests for DistributedLock."""

    def test_acquire_multiple_locks_same_consumer(self):
        """Test that a consumer can acquire locks for different indicators."""
        mock_redis = MagicMock()

        lock1 = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock2 = DistributedLock(mock_redis, "EURUSD", "rvvol")

        assert lock1.lock_key == "svss:lock:eurusd:vwap"
        assert lock2.lock_key == "svss:lock:eurusd:rvvol"

    def test_lock_key_symbol_case_normalization(self):
        """Test lock keys are normalized to lowercase."""
        mock_redis = MagicMock()
        lock = DistributedLock(mock_redis, "EURUSD", "VWAP")

        assert lock.lock_key == "svss:lock:eurusd:vwap"

    def test_lock_key_indicator_case_normalization(self):
        """Test indicator in lock key is normalized to lowercase."""
        mock_redis = MagicMock()
        lock = DistributedLock(mock_redis, "EURUSD", "VWAP")

        assert lock.lock_key == "svss:lock:eurusd:vwap"

    def test_release_with_wrong_consumer_id(self):
        """Test release fails when lock held by different consumer."""
        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_script.return_value = 0  # Lock not released
        mock_redis.register_script.return_value = mock_script

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock._consumer_id = "consumer-1"

        # Simulate another consumer holding the lock
        with patch.object(lock._redis_client, 'get', return_value=b"consumer-2"):
            result = lock.release()

        # Should return False because our consumer_id doesn't match
        assert result is False

    def test_extend_with_wrong_consumer(self):
        """Test extend fails when held by different consumer."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"different-consumer"
        mock_redis.expire.return_value = True

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock._consumer_id = "our-consumer"

        result = lock.extend(additional_ttl=10)

        assert result is False

    def test_extend_success(self):
        """Test successful lock extension."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"our-consumer"
        mock_redis.expire.return_value = True

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock._consumer_id = "our-consumer"

        result = lock.extend(additional_ttl=10)

        assert result is True
        mock_redis.expire.assert_called_once()

    def test_extend_when_lock_not_held(self):
        """Test extend returns False when lock not held."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # Lock not held

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        result = lock.extend(additional_ttl=10)

        assert result is False

    def test_is_held_decodes_bytes(self):
        """Test is_held properly decodes bytes from Redis."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"test-consumer"

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock._consumer_id = "test-consumer"

        assert lock.is_held() is True

    def test_get_holder_decodes_bytes(self):
        """Test get_holder properly decodes bytes from Redis."""
        mock_redis = MagicMock()
        mock_redis.get.return_value = b"holder-123"

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        assert lock.get_holder() == "holder-123"

    def test_acquire_non_blocking_timeout_zero(self):
        """Test non-blocking acquire with timeout=0."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = None  # Lock not acquired

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        start = time.monotonic()
        result = lock.acquire(timeout=0)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed < 0.01  # Should be nearly instant

    def test_acquire_blocks_until_timeout(self):
        """Test that blocking acquire respects timeout."""
        mock_redis = MagicMock()
        mock_redis.set.return_value = None  # Always fail

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        start = time.monotonic()
        result = lock.acquire(timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.1  # Should have waited for timeout

    def test_acquire_succeeds_on_retry(self):
        """Test acquire succeeds on second attempt."""
        mock_redis = MagicMock()
        mock_redis.set.side_effect = [None, True]  # First fails, second succeeds

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        result = lock.acquire(timeout=0.1)

        assert result is True
        assert mock_redis.set.call_count == 2

    def test_consumer_id_is_uuid(self):
        """Test consumer ID is a valid UUID format."""
        mock_redis = MagicMock()
        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        # UUID format check
        parts = lock.consumer_id.split('-')
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12

    def test_redis_error_on_is_held(self):
        """Test graceful handling of Redis error in is_held."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.get.side_effect = redis_lib.RedisError("Connection failed")

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        result = lock.is_held()

        assert result is False

    def test_redis_error_on_get_holder(self):
        """Test graceful handling of Redis error in get_holder."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.get.side_effect = redis_lib.RedisError("Connection failed")

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        result = lock.get_holder()

        assert result is None

    def test_redis_error_on_extend(self):
        """Test graceful handling of Redis error in extend."""
        import redis as redis_lib

        mock_redis = MagicMock()
        mock_redis.get.return_value = b"our-consumer"
        mock_redis.expire.side_effect = redis_lib.RedisError("Connection failed")

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock._consumer_id = "our-consumer"

        result = lock.extend(additional_ttl=10)

        assert result is False


class TestDistributedLockConcurrency:
    """Concurrency-related tests for DistributedLock."""

    def test_concurrent_acquire_simulation(self):
        """Simulate concurrent lock acquisition by multiple consumers."""
        mock_redis = MagicMock()

        # First consumer acquires
        mock_redis.set.return_value = True
        lock1 = DistributedLock(mock_redis, "EURUSD", "vwap")
        result1 = lock1.acquire()

        # Second consumer fails (lock already held)
        mock_redis.set.return_value = None
        lock2 = DistributedLock(mock_redis, "EURUSD", "vwap")
        result2 = lock2.acquire()

        assert result1 is True
        assert result2 is False
        assert lock1.consumer_id != lock2.consumer_id

    def test_lock_released_by_different_consumer(self):
        """Test that releasing a lock held by another consumer fails."""
        mock_redis = MagicMock()
        mock_script = MagicMock()
        mock_script.return_value = 0  # Not released
        mock_redis.register_script.return_value = mock_script

        lock1 = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock1._consumer_id = "consumer-1"

        # Consumer 2 tries to release lock1's lock
        lock2 = DistributedLock(mock_redis, "EURUSD", "vwap")
        lock2._consumer_id = "consumer-2"

        # The release script will check if consumer_id matches
        # In real scenario, lock1's lock key would be held by consumer-1
        result = lock2.release()

        # This should fail because consumer-2's ID doesn't match what's in the lock
        # (which is consumer-1's ID, or if the lock was never set, the script returns 0)
        assert result is False


class TestLuaScriptAtomicity:
    """Tests for Lua script atomicity in lock release."""

    def test_release_lua_script_format(self):
        """Test the release Lua script has correct format."""
        from svss.cache.lock_manager import RELEASE_LOCK_SCRIPT

        # Verify script contains expected commands
        assert "redis.call(\"get\", KEYS[1])" in RELEASE_LOCK_SCRIPT
        assert "redis.call(\"del\", KEYS[1])" in RELEASE_LOCK_SCRIPT
        assert "ARGV[1]" in RELEASE_LOCK_SCRIPT  # consumer_id check

    def test_register_script_called(self):
        """Test that register_script is called during initialization."""
        mock_redis = MagicMock()

        lock = DistributedLock(mock_redis, "EURUSD", "vwap")

        mock_redis.register_script.assert_called_once()
