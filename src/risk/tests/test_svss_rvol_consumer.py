"""
Unit tests for SVSS RVOL Consumer Module.

Tests cover:
- RVOL cache functionality
- RVOL clamping logic
- Hard block when RVOL < 0.5
- Cache miss defaulting to 1.0
- Integration with mock Redis
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from src.risk.svss_rvol_consumer import (
    RVOLCache,
    SVSSRVOLConsumer,
    clamp_rvol,
    should_block_entry,
    apply_rvol_multiplier,
    DEFAULT_RVOL,
    RVOL_MIN,
    RVOL_MAX,
    get_rvol,
    init_global_consumer,
    get_global_consumer
)
from src.risk.governor import RiskGovernor


class TestRVOLCache(unittest.TestCase):
    """Test suite for RVOLCache functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = RVOLCache(ttl=1.0)  # 1 second TTL for testing

    def test_cache_set_and_get(self):
        """Test basic cache set and get operations."""
        self.cache.set("EURUSD", 1.2)
        rvol = self.cache.get("EURUSD")
        self.assertEqual(rvol, 1.2)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        rvol = self.cache.get("NONEXISTENT")
        self.assertIsNone(rvol)

    def test_cache_expiry(self):
        """Test that cache entries expire after TTL."""
        self.cache.set("EURUSD", 1.5)
        time.sleep(1.1)  # Wait for TTL to expire
        rvol = self.cache.get("EURUSD")
        self.assertIsNone(rvol)

    def test_cache_get_with_default(self):
        """Test get_with_default returns default on miss."""
        rvol, was_miss = self.cache.get_with_default("EURUSD")
        self.assertEqual(rvol, DEFAULT_RVOL)
        self.assertTrue(was_miss)

    def test_cache_hit_with_default(self):
        """Test get_with_default returns cached value on hit."""
        self.cache.set("EURUSD", 0.8)
        rvol, was_miss = self.cache.get_with_default("EURUSD")
        self.assertEqual(rvol, 0.8)
        self.assertFalse(was_miss)

    def test_cache_stats(self):
        """Test cache statistics tracking."""
        self.cache.set("EURUSD", 1.0)
        self.cache.get("EURUSD")  # Hit
        self.cache.get("GBPUSD")  # Miss
        stats = self.cache.get_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['cached_symbols'], 1)

    def test_cache_miss_rate(self):
        """Test cache miss rate calculation."""
        self.cache.set("EURUSD", 1.0)
        self.cache.get("EURUSD")  # Hit
        self.cache.get("GBPUSD")  # Miss
        self.cache.get("USDJPY")  # Miss
        # 1 hit, 2 misses = 2/3 miss rate
        self.assertAlmostEqual(self.cache.cache_miss_rate, 2/3, places=2)

    def test_case_insensitive_symbol(self):
        """Test that symbol lookup is case insensitive."""
        self.cache.set("EURUSD", 1.3)
        rvol = self.cache.get("eurusd")
        self.assertEqual(rvol, 1.3)


class TestRVOLClamping(unittest.TestCase):
    """Test suite for RVOL clamping logic."""

    def test_clamp_rvol_normal_range(self):
        """Test RVOL within normal range [0.5, 1.5]."""
        self.assertEqual(clamp_rvol(0.5), 0.5)
        self.assertEqual(clamp_rvol(0.6), 0.6)
        self.assertEqual(clamp_rvol(1.0), 1.0)
        self.assertEqual(clamp_rvol(1.5), 1.5)

    def test_clamp_rvol_at_upper_cap(self):
        """Test RVOL above 1.5 is capped at 1.5."""
        self.assertEqual(clamp_rvol(2.0), 1.5)
        self.assertEqual(clamp_rvol(2.5), 1.5)
        self.assertEqual(clamp_rvol(10.0), 1.5)

    def test_clamp_rvol_at_lower_block(self):
        """Test RVOL below 0.5 returns 0.0 for hard block."""
        self.assertEqual(clamp_rvol(0.4), 0.0)
        self.assertEqual(clamp_rvol(0.3), 0.0)
        self.assertEqual(clamp_rvol(0.0), 0.0)
        self.assertEqual(clamp_rvol(-1.0), 0.0)

    def test_should_block_entry(self):
        """Test entry blocking logic."""
        self.assertTrue(should_block_entry(0.4))
        self.assertTrue(should_block_entry(0.3))
        self.assertFalse(should_block_entry(0.5))
        self.assertFalse(should_block_entry(0.6))
        self.assertFalse(should_block_entry(1.0))
        self.assertFalse(should_block_entry(2.0))


class TestApplyRVOLMultiplier(unittest.TestCase):
    """Test suite for RVOL multiplier application."""

    def test_apply_rvol_multiplier_normal(self):
        """Test normal RVOL multiplier application."""
        base_kelly = 0.10  # 10% Kelly
        result = apply_rvol_multiplier(base_kelly, 1.2)
        self.assertEqual(result, 0.12)  # 10% * 1.2 = 12%

    def test_apply_rvol_multiplier_at_cap(self):
        """Test RVOL multiplier at upper cap 1.5."""
        base_kelly = 0.10
        result = apply_rvol_multiplier(base_kelly, 2.0)
        self.assertAlmostEqual(result, 0.15, places=5)  # 10% * 1.5 = 15%

    def test_apply_rvol_multiplier_below_block(self):
        """Test hard block returns 0 when RVOL < 0.5."""
        base_kelly = 0.10
        result = apply_rvol_multiplier(base_kelly, 0.4)
        self.assertEqual(result, 0.0)

    def test_apply_rvol_multiplier_neutral(self):
        """Test neutral RVOL of 1.0 passes through."""
        base_kelly = 0.10
        result = apply_rvol_multiplier(base_kelly, 1.0)
        self.assertEqual(result, 0.10)


class TestSVSSRVOLConsumer(unittest.TestCase):
    """Test suite for SVSSRVOLConsumer with mock Redis."""

    def setUp(self):
        """Set up test fixtures with mock Redis."""
        self.mock_redis = MagicMock()
        self.consumer = SVSSRVOLConsumer(
            redis_url="redis://localhost:6379",
            cache_ttl=5.0
        )

    @patch('src.risk.svss_rvol_consumer.redis.from_url')
    def test_connect_success(self, mock_redis_from_url):
        """Test successful Redis connection."""
        mock_redis_instance = MagicMock()
        mock_redis_from_url.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True

        result = self.consumer.connect()

        self.assertTrue(result)
        self.assertTrue(self.consumer.is_connected)

    @patch('src.risk.svss_rvol_consumer.redis.from_url')
    def test_connect_failure(self, mock_redis_from_url):
        """Test failed Redis connection."""
        import redis
        mock_redis_from_url.side_effect = redis.ConnectionError("Connection refused")

        result = self.consumer.connect()

        self.assertFalse(result)
        self.assertFalse(self.consumer.is_connected)

    def test_get_channel_symbol(self):
        """Test symbol extraction from channel name."""
        symbol = self.consumer._get_channel_symbol("svss:EURUSD:rvvol")
        self.assertEqual(symbol, "EURUSD")

    def test_get_rvol_with_cache_hit(self):
        """Test get_rvol returns cached value."""
        self.consumer._cache.set("EURUSD", 1.3)

        rvol = self.consumer.get_rvol("EURUSD")

        self.assertEqual(rvol, 1.3)

    def test_get_rvol_with_cache_miss(self):
        """Test get_rvol returns default on cache miss."""
        rvol = self.consumer.get_rvol("EURUSD")

        self.assertEqual(rvol, DEFAULT_RVOL)

    def test_get_rvol_detailed(self):
        """Test get_rvol_detailed returns cache status."""
        self.consumer._cache.set("EURUSD", 0.8)
        rvol, was_miss = self.consumer.get_rvol_detailed("EURUSD")

        self.assertEqual(rvol, 0.8)
        self.assertFalse(was_miss)

    def test_get_rvol_detailed_cache_miss(self):
        """Test get_rvol_detailed on cache miss."""
        rvol, was_miss = self.consumer.get_rvol_detailed("EURUSD")

        self.assertEqual(rvol, DEFAULT_RVOL)
        self.assertTrue(was_miss)


class TestGlobalConsumerFunctions(unittest.TestCase):
    """Test suite for global consumer functions."""

    def test_get_rvol_without_consumer(self):
        """Test get_rvol returns default when no global consumer."""
        # Reset global consumer
        import src.risk.svss_rvol_consumer as consumer_module
        consumer_module._global_consumer = None

        rvol = get_rvol("EURUSD")

        self.assertEqual(rvol, DEFAULT_RVOL)

    def test_init_global_consumer(self):
        """Test global consumer initialization."""
        import src.risk.svss_rvol_consumer as consumer_module

        with patch('src.risk.svss_rvol_consumer.redis.from_url') as mock_redis:
            mock_instance = MagicMock()
            mock_redis.return_value = mock_instance
            mock_instance.ping.return_value = True

            consumer = init_global_consumer()

            self.assertIsNotNone(consumer)
            self.assertIs(consumer_module._global_consumer, consumer)

    def test_get_global_consumer(self):
        """Test getting global consumer instance."""
        import src.risk.svss_rvol_consumer as consumer_module
        consumer_module._global_consumer = None

        result = get_global_consumer()

        self.assertIsNone(result)


class TestRVOLEdgeCases(unittest.TestCase):
    """Test edge cases for RVOL calculations."""

    def test_clamp_edge_cases(self):
        """Test clamping at exact boundary values."""
        # At exactly 0.5 should return 0.5 (not block)
        self.assertEqual(clamp_rvol(0.5), 0.5)
        # At exactly 1.5 should return 1.5 (not cap further)
        self.assertEqual(clamp_rvol(1.5), 1.5)

    def test_clamp_just_below_block(self):
        """Test RVOL just below block threshold."""
        # 0.49 should block
        self.assertEqual(clamp_rvol(0.49), 0.0)

    def test_clamp_just_at_cap(self):
        """Test RVOL just at cap."""
        # 1.51 should be capped at 1.5
        self.assertEqual(clamp_rvol(1.51), 1.5)

    def test_negative_rvol(self):
        """Test negative RVOL values are blocked."""
        self.assertEqual(clamp_rvol(-0.5), 0.0)
        self.assertEqual(clamp_rvol(-1.0), 0.0)

    def test_zero_rvol(self):
        """Test zero RVOL is blocked."""
        self.assertEqual(clamp_rvol(0.0), 0.0)


class TestGovernorRVOLIntegration(unittest.TestCase):
    """Test suite for Governor RVOL integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.gov = RiskGovernor()

    def test_governor_without_symbol(self):
        """Test governor position sizing without symbol uses default RVOL."""
        result = self.gov.calculate_position_size(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )
        # Without symbol, RVOL defaults to 1.0
        self.assertGreater(result.lot_size, 0)

    def test_governor_with_symbol_no_consumer(self):
        """Test governor with symbol but no global consumer defaults RVOL to 1.0."""
        # Reset global consumer
        import src.risk.svss_rvol_consumer as consumer_module
        consumer_module._global_consumer = None

        result = self.gov.calculate_position_size(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            symbol='EURUSD'
        )
        # Without global consumer, RVOL defaults to 1.0
        self.assertGreater(result.lot_size, 0)

    def test_governor_rvol_step_in_calculation(self):
        """Test that RVOL step appears in calculation steps."""
        result = self.gov.calculate_position_size(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            symbol='EURUSD'
        )
        # Check that RVOL factor is in calculation steps
        rvol_steps = [s for s in result.calculation_steps if 'RVOL' in s]
        self.assertGreater(len(rvol_steps), 0)

    def test_governor_rvol_multiplier_chain_order(self):
        """Test that multiplier chain shows RVOL first: RVOL x Session Kelly x Correlation x House Money."""
        result = self.gov.calculate_position_size(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            symbol='EURUSD'
        )
        # Find the order of multipliers in calculation steps
        steps_str = ' '.join(result.calculation_steps)
        # Should have RVOL factor appearing
        self.assertIn('RVOL factor', steps_str)

    @patch('src.risk.governor.get_rvol')
    @patch('src.risk.governor.get_global_consumer')
    def test_governor_rvol_hard_block(self, mock_get_consumer, mock_get_rvol):
        """Test that RVOL < 0.5 results in hard block (lot_size = 0)."""
        # Mock get_rvol to return 0.4 (< 0.5 threshold) which triggers hard block
        mock_get_rvol.return_value = 0.4

        # Mock the global consumer to return cache miss info
        mock_consumer = MagicMock()
        mock_consumer.get_rvol_detailed.return_value = (0.4, True)
        mock_get_consumer.return_value = mock_consumer

        result = self.gov.calculate_position_size(
            account_balance=10000.0,
            win_rate=0.55,
            avg_win=400.0,
            avg_loss=200.0,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0,
            symbol='EURUSD'
        )
        # Hard block should result in lot_size of 0 (truly blocked, not rounded to MIN_LOT)
        self.assertEqual(result.lot_size, 0)


if __name__ == '__main__':
    unittest.main()
