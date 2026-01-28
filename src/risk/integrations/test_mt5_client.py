"""
Tests for MT5 Client Integration

These tests verify the MT5 integration wrapper functionality with both
real MT5 connection (when available) and simulated fallback mode.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import logging
from datetime import datetime

from src.risk.integrations.mt5_client import MT5Client, get_mt5_account_info, get_mt5_symbol_info, calculate_mt5_pip_value


class TestMT5Client(unittest.TestCase):
    """Test suite for MT5Client functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5Client(fallback_to_simulated=True)
        self.logger = logging.getLogger("test_mt5_client")

    def test_initialization(self):
        """Test MT5Client initialization."""
        self.assertIsNotNone(self.client)
        self.assertTrue(self.client.fallback_to_simulated)
        self.assertFalse(self.client.is_connected)
        self.assertIsNone(self.client.last_error)

    def test_connect_disconnect(self):
        """Test connection and disconnection functionality."""
        # Test with invalid password (should fail but fallback works)
        result = self.client.connect("invalid_password")
        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)
        self.assertIsNotNone(self.client.last_error)

        # Test disconnection
        result = self.client.disconnect()
        self.assertTrue(result)
        self.assertFalse(self.client.is_connected)

    def test_get_account_info(self):
        """Test account info retrieval."""
        account_info = self.client.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertIn("login", account_info)
        self.assertIn("balance", account_info)
        self.assertIn("equity", account_info)
        self.assertIn("currency", account_info)

    def test_get_symbol_info(self):
        """Test symbol info retrieval."""
        # Test with valid symbol
        symbol_info = self.client.get_symbol_info("EURUSD")
        self.assertIsNotNone(symbol_info)
        self.assertIn("name", symbol_info)
        self.assertIn("digits", symbol_info)
        self.assertIn("point", symbol_info)
        self.assertIn("tick_value", symbol_info)

        # Test with invalid symbol
        symbol_info = self.client.get_symbol_info("INVALID_SYMBOL")
        self.assertIsNone(symbol_info)

    def test_calculate_pip_value(self):
        """Test pip value calculation."""
        # Test with valid symbol
        pip_value = self.client.calculate_pip_value("EURUSD")
        self.assertIsNotNone(pip_value)
        self.assertGreater(pip_value, 0)

        # Test with invalid symbol
        pip_value = self.client.calculate_pip_value("INVALID_SYMBOL")
        self.assertIsNone(pip_value)

    def test_calculate_position_size(self):
        """Test position size calculation."""
        # Test with valid parameters
        position_size = self.client.calculate_position_size(
            symbol="EURUSD",
            risk_amount=50.0,
            stop_loss_pips=20.0,
            account_risk_percent=1.0
        )
        self.assertIsNotNone(position_size)
        self.assertGreater(position_size, 0)

        # Test with invalid symbol
        position_size = self.client.calculate_position_size(
            symbol="INVALID_SYMBOL",
            risk_amount=50.0,
            stop_loss_pips=20.0
        )
        self.assertIsNone(position_size)

        # Test with zero stop loss
        position_size = self.client.calculate_position_size(
            symbol="EURUSD",
            risk_amount=50.0,
            stop_loss_pips=0
        )
        self.assertIsNone(position_size)

    def test_switch_account(self):
        """Test account switching."""
        # Test with valid login (simulated)
        result = self.client.switch_account(12345678)
        self.assertTrue(result)

        # Test with invalid login
        result = self.client.switch_account(99999999)
        self.assertFalse(result)

    def test_connection_status(self):
        """Test connection status retrieval."""
        status = self.client.get_connection_status()
        self.assertIn("is_connected", status)
        self.assertIn("last_error", status)
        self.assertIn("has_fallback", status)

    def test_context_manager(self):
        """Test context manager functionality."""
        with MT5Client(fallback_to_simulated=True) as client:
            self.assertIsNotNone(client)
            self.assertFalse(client.is_connected)

    def test_direct_mcp_functions(self):
        """Test direct MCP function wrappers."""
        # Test get_mt5_account_info
        account_info = get_mt5_account_info()
        self.assertIsNotNone(account_info)
        self.assertIn("login", account_info)

        # Test get_mt5_symbol_info
        symbol_info = get_mt5_symbol_info("EURUSD")
        self.assertIsNotNone(symbol_info)
        self.assertIn("name", symbol_info)

        # Test calculate_mt5_pip_value
        pip_value = calculate_mt5_pip_value("EURUSD")
        self.assertIsNotNone(pip_value)
        self.assertGreater(pip_value, 0)


class TestMT5ClientWithoutFallback(unittest.TestCase):
    """Test MT5Client without fallback mode."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5Client(fallback_to_simulated=False)

    def test_no_fallback_connection(self):
        """Test that no fallback occurs when disabled."""
        # Connect with invalid password
        result = self.client.connect("invalid_password")
        self.assertFalse(result)
        self.assertFalse(self.client.is_connected)
        self.assertIsNotNone(self.client.last_error)

        # Try to get account info (should fail without fallback)
        account_info = self.client.get_account_info()
        self.assertIsNone(account_info)


class TestMT5ClientErrorHandling(unittest.TestCase):
    """Test MT5Client error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = MT5Client(fallback_to_simulated=True)

    def test_symbol_info_error_handling(self):
        """Test error handling in symbol info retrieval."""
        # Mock the get_symbol_info method to raise an exception
        with patch.object(self.client, 'get_symbol_info', side_effect=Exception("Test error")):
            symbol_info = self.client.get_symbol_info("EURUSD")
            self.assertIsNone(symbol_info)
            self.assertIsNotNone(self.client.last_error)

    def test_pip_value_error_handling(self):
        """Test error handling in pip value calculation."""
        # Mock the calculate_pip_value method to raise an exception
        with patch.object(self.client, 'calculate_pip_value', side_effect=Exception("Test error")):
            pip_value = self.client.calculate_pip_value("EURUSD")
            self.assertIsNone(pip_value)
            self.assertIsNotNone(self.client.last_error)


if __name__ == "__main__":
    # Configure logging for test output
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)