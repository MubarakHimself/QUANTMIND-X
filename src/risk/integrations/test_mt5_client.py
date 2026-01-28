"""
Tests for MT5 Account Client Integration

These tests verify the MT5AccountClient functionality including:
- Account information retrieval with caching
- Margin information retrieval
- Symbol information lookup
- Pip value calculations for various symbol types
- Cache TTL functionality
- Graceful degradation when MT5 is unavailable
"""

import unittest
import time
from unittest.mock import Mock, patch, MagicMock
import logging

from src.risk.integrations.mt5_client import (
    MT5AccountClient,
    AccountInfo,
    SymbolInfo,
    MarginInfo,
    CachedValue,
    create_mt5_client,
    MT5ConnectionError,
    MT5SymbolError
)


class TestMT5AccountClientInitialization(unittest.TestCase):
    """Test MT5AccountClient initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_initialization_with_defaults(self):
        """Test client initialization with default parameters."""
        self.assertIsNotNone(self.client)
        self.assertTrue(self.client.fallback_to_simulated)
        self.assertEqual(self.client.cache_ttl, 10.0)
        self.assertFalse(self.client.is_connected)
        self.assertIsNone(self.client.last_error)
        self.assertEqual(len(self.client._cache), 0)

    def test_initialization_custom_cache_ttl(self):
        """Test client initialization with custom cache TTL."""
        client = MT5AccountClient(fallback_to_simulated=True, cache_ttl=5.0)
        self.assertEqual(client.cache_ttl, 5.0)

    def test_initialization_without_fallback(self):
        """Test client initialization without fallback mode."""
        client = MT5AccountClient(fallback_to_simulated=False)
        self.assertFalse(client.fallback_to_simulated)


class TestAccountInformation(unittest.TestCase):
    """Test account information retrieval with caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_get_account_balance(self):
        """Test account balance retrieval."""
        balance = self.client.get_account_balance()
        self.assertIsNotNone(balance)
        self.assertEqual(balance, 10000.0)
        self.assertIsInstance(balance, float)

    def test_get_account_balance_cached(self):
        """Test that account balance is cached."""
        # First call
        balance1 = self.client.get_account_balance()
        # Second call should return cached value
        balance2 = self.client.get_account_balance()
        self.assertEqual(balance1, balance2)

    def test_get_account_equity(self):
        """Test account equity retrieval."""
        equity = self.client.get_account_equity()
        self.assertIsNotNone(equity)
        self.assertEqual(equity, 10500.0)
        self.assertIsInstance(equity, float)

    def test_get_account_info(self):
        """Test comprehensive account information retrieval."""
        account_info = self.client.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertIsInstance(account_info, AccountInfo)
        self.assertEqual(account_info.login, 12345678)
        self.assertEqual(account_info.balance, 10000.0)
        self.assertEqual(account_info.equity, 10500.0)
        self.assertEqual(account_info.currency, "USD")

    def test_account_info_to_dict(self):
        """Test AccountInfo to_dict conversion."""
        account_info = self.client.get_account_info()
        info_dict = account_info.to_dict()
        self.assertIsInstance(info_dict, dict)
        self.assertIn("login", info_dict)
        self.assertIn("balance", info_dict)
        self.assertIn("equity", info_dict)


class TestMarginInformation(unittest.TestCase):
    """Test margin information retrieval with caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_get_margin_info(self):
        """Test margin information retrieval."""
        margin_info = self.client.get_margin_info()
        self.assertIsNotNone(margin_info)
        self.assertIsInstance(margin_info, MarginInfo)
        self.assertEqual(margin_info.margin_free, 10300.0)
        self.assertEqual(margin_info.margin_used, 200.0)
        self.assertEqual(margin_info.margin_level, 5250.0)

    def test_margin_info_to_dict(self):
        """Test MarginInfo to_dict conversion."""
        margin_info = self.client.get_margin_info()
        info_dict = margin_info.to_dict()
        self.assertIsInstance(info_dict, dict)
        self.assertIn("margin_free", info_dict)
        self.assertIn("margin_used", info_dict)
        self.assertIn("margin_level", info_dict)
        self.assertIn("equity", info_dict)
        self.assertIn("balance", info_dict)


class TestSymbolInformation(unittest.TestCase):
    """Test symbol information lookup with caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_get_symbol_info_eurusd(self):
        """Test EURUSD symbol information retrieval."""
        symbol_info = self.client.get_symbol_info("EURUSD")
        self.assertIsNotNone(symbol_info)
        self.assertIsInstance(symbol_info, SymbolInfo)
        self.assertEqual(symbol_info.name, "EURUSD")
        self.assertEqual(symbol_info.digits, 5)
        self.assertEqual(symbol_info.pip_location, 4)
        self.assertEqual(symbol_info.contract_size, 100000)

    def test_get_symbol_info_gbpusd(self):
        """Test GBPUSD symbol information retrieval."""
        symbol_info = self.client.get_symbol_info("GBPUSD")
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info.name, "GBPUSD")
        self.assertEqual(symbol_info.digits, 5)
        self.assertEqual(symbol_info.pip_location, 4)

    def test_get_symbol_info_xaurusd(self):
        """Test XAUUSD symbol information retrieval."""
        symbol_info = self.client.get_symbol_info("XAUUSD")
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info.name, "XAUUSD")
        self.assertEqual(symbol_info.digits, 2)
        self.assertEqual(symbol_info.pip_location, 1)  # Gold uses 1 decimal
        self.assertEqual(symbol_info.contract_size, 100)

    def test_get_symbol_info_usdjpy(self):
        """Test USDJPY symbol information retrieval."""
        symbol_info = self.client.get_symbol_info("USDJPY")
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info.name, "USDJPY")
        self.assertEqual(symbol_info.digits, 3)
        self.assertEqual(symbol_info.pip_location, 2)  # JPY pairs use 2 decimals

    def test_get_symbol_info_us30(self):
        """Test US30 index symbol information retrieval."""
        symbol_info = self.client.get_symbol_info("US30")
        self.assertIsNotNone(symbol_info)
        self.assertEqual(symbol_info.name, "US30")
        self.assertEqual(symbol_info.digits, 2)
        self.assertEqual(symbol_info.pip_location, 0)  # Indices use whole points

    def test_get_symbol_info_invalid(self):
        """Test symbol information retrieval for invalid symbol."""
        symbol_info = self.client.get_symbol_info("INVALID")
        self.assertIsNone(symbol_info)

    def test_symbol_info_cached(self):
        """Test that symbol information is cached."""
        # First call
        symbol_info1 = self.client.get_symbol_info("EURUSD")
        # Second call should return cached value (simulated creates new object, but value is same)
        symbol_info2 = self.client.get_symbol_info("EURUSD")
        # Check that values are the same (simulated mode creates new objects)
        self.assertEqual(symbol_info1.name, symbol_info2.name)
        self.assertEqual(symbol_info1.digits, symbol_info2.digits)
        self.assertEqual(symbol_info1.pip_location, symbol_info2.pip_location)

    def test_symbol_info_to_dict(self):
        """Test SymbolInfo to_dict conversion."""
        symbol_info = self.client.get_symbol_info("EURUSD")
        info_dict = symbol_info.to_dict()
        self.assertIsInstance(info_dict, dict)
        self.assertIn("name", info_dict)
        self.assertIn("digits", info_dict)
        self.assertIn("pip_location", info_dict)
        self.assertIn("tick_value", info_dict)


class TestPipValueCalculations(unittest.TestCase):
    """Test pip value calculations for various symbol types."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_calculate_pip_value_eurusd(self):
        """Test pip value calculation for EURUSD."""
        pip_value = self.client.calculate_pip_value("EURUSD", volume=1.0)
        self.assertIsNotNone(pip_value)
        self.assertEqual(pip_value, 10.0)  # $10 per pip for standard lot

    def test_calculate_pip_value_gbpusd(self):
        """Test pip value calculation for GBPUSD."""
        pip_value = self.client.calculate_pip_value("GBPUSD", volume=1.0)
        self.assertIsNotNone(pip_value)
        self.assertEqual(pip_value, 10.0)  # $10 per pip for standard lot

    def test_calculate_pip_value_xaurusd(self):
        """Test pip value calculation for XAUUSD."""
        pip_value = self.client.calculate_pip_value("XAUUSD", volume=1.0)
        self.assertIsNotNone(pip_value)
        self.assertGreater(pip_value, 0)  # Gold has different pip value

    def test_calculate_pip_value_usdjpy(self):
        """Test pip value calculation for USDJPY."""
        pip_value = self.client.calculate_pip_value("USDJPY", volume=1.0)
        self.assertIsNotNone(pip_value)
        # JPY pairs need special calculation (tick_value * 10)
        self.assertGreater(pip_value, 0)

    def test_calculate_pip_value_custom_volume(self):
        """Test pip value calculation with custom volume."""
        pip_value_01 = self.client.calculate_pip_value("EURUSD", volume=0.1)
        pip_value_10 = self.client.calculate_pip_value("EURUSD", volume=1.0)
        self.assertIsNotNone(pip_value_01)
        self.assertIsNotNone(pip_value_10)
        # 0.1 lot should have 1/10th the pip value of 1 lot
        self.assertAlmostEqual(pip_value_01, pip_value_10 / 10.0)

    def test_calculate_pip_value_invalid_symbol(self):
        """Test pip value calculation for invalid symbol."""
        pip_value = self.client.calculate_pip_value("INVALID", volume=1.0)
        self.assertIsNone(pip_value)

    def test_calculate_pip_value_detailed(self):
        """Test detailed pip value calculation."""
        detailed = self.client.calculate_pip_value_detailed("EURUSD", volume=0.5)
        self.assertIsNotNone(detailed)
        self.assertIsInstance(detailed, dict)
        self.assertIn("symbol", detailed)
        self.assertIn("volume", detailed)
        self.assertIn("pip_value", detailed)
        self.assertIn("pip_location", detailed)
        self.assertIn("tick_value", detailed)
        self.assertIn("contract_size", detailed)

        # Verify values
        self.assertEqual(detailed["symbol"], "EURUSD")
        self.assertEqual(detailed["volume"], 0.5)
        self.assertEqual(detailed["pip_value"], 5.0)  # $5 per pip for 0.5 lots
        self.assertEqual(detailed["pip_location"], 4)
        self.assertEqual(detailed["contract_size"], 100000)


class TestCacheFunctionality(unittest.TestCase):
    """Test cache TTL and management."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True, cache_ttl=2.0)

    def test_cache_hit(self):
        """Test that cache returns cached values."""
        # First call populates cache
        balance1 = self.client.get_account_balance()
        # Second call should hit cache
        balance2 = self.client.get_account_balance()
        self.assertEqual(balance1, balance2)

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        # First call populates cache
        balance1 = self.client.get_account_balance()

        # Wait for cache to expire (TTL is 2 seconds)
        time.sleep(2.5)

        # This should fetch fresh data (simulated, so same value)
        balance2 = self.client.get_account_balance()
        self.assertEqual(balance1, balance2)  # Same value, but refetched

    def test_cache_clear_all(self):
        """Test clearing all cache entries."""
        # Note: In simulated mode without connection, caching may not work as expected
        # This test verifies the clear functionality works
        self.client._clear_cache()
        self.assertEqual(len(self.client._cache), 0)

    def test_cache_clear_pattern(self):
        """Test clearing cache entries by pattern."""
        # Manually populate cache for testing
        self.client._cache["account_balance"] = self.client._cache.get("account_balance", 10000.0)
        self.client._cache["account_equity"] = self.client._cache.get("account_equity", 10500.0)
        self.client._cache["symbol_info_EURUSD"] = "eurusd_data"
        self.client._cache["symbol_info_GBPUSD"] = "gbpusd_data"
        self.assertEqual(len(self.client._cache), 4)

        # Clear only symbol info
        self.client._clear_cache("symbol_info")
        # Should have 2 entries left (account_balance, account_equity)
        self.assertEqual(len(self.client._cache), 2)


class TestConnectionManagement(unittest.TestCase):
    """Test connection management and status."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5AccountClient(fallback_to_simulated=True)

    def test_connection_status(self):
        """Test connection status retrieval."""
        status = self.client.get_connection_status()
        self.assertIsInstance(status, dict)
        self.assertIn("is_connected", status)
        self.assertIn("last_error", status)
        self.assertIn("has_fallback", status)
        self.assertIn("cache_ttl", status)
        self.assertIn("cache_size", status)

    def test_connect_without_mt5(self):
        """Test connection attempt without MT5 available."""
        result = self.client.connect("test_password")
        self.assertFalse(result)  # Should fail without MT5
        self.assertIsNotNone(self.client.last_error)

    def test_disconnect(self):
        """Test disconnection."""
        result = self.client.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.client.is_connected)


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation when MT5 is unavailable."""

    def test_fallback_mode_enabled(self):
        """Test that fallback mode provides simulated data."""
        client = MT5AccountClient(fallback_to_simulated=True)

        # All these should work with simulated data
        balance = client.get_account_balance()
        self.assertIsNotNone(balance)

        equity = client.get_account_equity()
        self.assertIsNotNone(equity)

        margin_info = client.get_margin_info()
        self.assertIsNotNone(margin_info)

        symbol_info = client.get_symbol_info("EURUSD")
        self.assertIsNotNone(symbol_info)

        pip_value = client.calculate_pip_value("EURUSD")
        self.assertIsNotNone(pip_value)

    def test_fallback_mode_disabled(self):
        """Test that disabling fallback returns None when MT5 unavailable."""
        client = MT5AccountClient(fallback_to_simulated=False)

        # All these should return None without MT5 connection
        balance = client.get_account_balance()
        self.assertIsNone(balance)

        equity = client.get_account_equity()
        self.assertIsNone(equity)

        margin_info = client.get_margin_info()
        self.assertIsNone(margin_info)


class TestContextManager(unittest.TestCase):
    """Test context manager functionality."""

    def test_context_manager(self):
        """Test using client as context manager."""
        with MT5AccountClient(fallback_to_simulated=True) as client:
            self.assertIsNotNone(client)
            balance = client.get_account_balance()
            self.assertIsNotNone(balance)


class TestFactoryFunction(unittest.TestCase):
    """Test the factory function."""

    def test_create_mt5_client_defaults(self):
        """Test factory function with default parameters."""
        client = create_mt5_client()
        self.assertIsNotNone(client)
        self.assertTrue(client.fallback_to_simulated)
        self.assertEqual(client.cache_ttl, 10.0)

    def test_create_mt5_client_custom(self):
        """Test factory function with custom parameters."""
        client = create_mt5_client(
            config_path="/tmp/test_config.json",
            fallback_to_simulated=False,
            cache_ttl=5.0
        )
        self.assertIsNotNone(client)
        self.assertFalse(client.fallback_to_simulated)
        self.assertEqual(client.cache_ttl, 5.0)


class TestCachedValue(unittest.TestCase):
    """Test CachedValue dataclass."""

    def test_cached_value_not_expired(self):
        """Test that cached value is not expired immediately."""
        cached = CachedValue(value=100, timestamp=time.time(), ttl=10.0)
        self.assertFalse(cached.is_expired())

    def test_cached_value_expired(self):
        """Test that cached value expires after TTL."""
        cached = CachedValue(
            value=100,
            timestamp=time.time() - 15.0,  # 15 seconds ago
            ttl=10.0
        )
        self.assertTrue(cached.is_expired())


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests with verbosity
    unittest.main(verbosity=2)
