"""
MT5 Integration Client
====================

Wrapper for mcp-metatrader5-server AccountManager with fallback mode.
Provides a simplified interface for Enhanced Kelly Position Sizing calculations.

Features:
- Graceful degradation when MT5 is unavailable
- Fallback to simulated data for testing
- Comprehensive error handling
- Type hints and validation
- Connection state management
"""

import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
import time

from pydantic import BaseModel, Field, validator

from mcp_metatrader5 import AccountManager, get_account_manager
from mcp_metatrader5.models import AccountInfo, SymbolInfo

logger = logging.getLogger(__name__)


class MT5ConnectionError(Exception):
    """Custom exception for MT5 connection errors."""
    pass


class MT5Client:
    """
    MT5 Integration Client with fallback mode.

    This client provides a simplified interface to the mcp-metatrader5-server
    AccountManager with graceful degradation when MT5 is unavailable.

    Usage:
        client = MT5Client()
        if client.is_connected():
            account_info = client.get_account_info()
            symbol_info = client.get_symbol_info("EURUSD")
            pip_value = client.calculate_pip_value("EURUSD")
    """

    def __init__(self, config_path: str = None, fallback_to_simulated: bool = True):
        """
        Initialize MT5 client.

        Args:
            config_path: Path to MT5 configuration file
            fallback_to_simulated: Whether to use simulated data when MT5 is unavailable
        """
        self.config_path = config_path
        self.fallback_to_simulated = fallback_to_simulated
        self.account_manager = get_account_manager(config_path)
        self.is_connected = False
        self.last_error = None
        self.simulated_data = {}

        # Initialize simulated data
        self._initialize_simulated_data()

        logger.info("MT5Client initialized with fallback=%s", fallback_to_simulated)

    def _initialize_simulated_data(self):
        """Initialize simulated data for fallback mode."""
        self.simulated_data = {
            "account_info": {
                "login": 12345678,
                "server": "Demo-Server",
                "currency": "USD",
                "balance": 10000.0,
                "equity": 10500.0,
                "margin": 200.0,
                "margin_free": 10300.0,
                "profit": 500.0,
                "leverage": 100
            },
            "symbols": {
                "EURUSD": {
                    "name": "EURUSD",
                    "digits": 5,
                    "point": 0.00001,
                    "tick_value": 1.0,
                    "contract_size": 100000,
                    "volume_min": 0.01,
                    "volume_max": 1000.0,
                    "volume_step": 0.01
                },
                "GBPUSD": {
                    "name": "GBPUSD",
                    "digits": 5,
                    "point": 0.00001,
                    "tick_value": 1.0,
                    "contract_size": 100000,
                    "volume_min": 0.01,
                    "volume_max": 1000.0,
                    "volume_step": 0.01
                },
                "XAUUSD": {
                    "name": "XAUUSD",
                    "digits": 3,
                    "point": 0.001,
                    "tick_value": 0.1,
                    "contract_size": 100,
                    "volume_min": 0.01,
                    "volume_max": 100.0,
                    "volume_step": 0.01
                }
            }
        }

    def connect(self, master_password: str) -> bool:
        """
        Connect to MT5 using the account manager.

        Args:
            master_password: Master password for credential vault

        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if self.account_manager.unlock(master_password):
                self.is_connected = True
                self.last_error = None
                logger.info("MT5Client connected successfully")
                return True
            else:
                self.is_connected = False
                self.last_error = "Invalid master password or vault not unlocked"
                logger.error("Failed to unlock MT5 vault")
                return False
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            logger.error(f"Error connecting to MT5: {e}")
            return False

    def disconnect(self) -> bool:
        """
        Disconnect from MT5.

        Returns:
            True if disconnected successfully, False otherwise
        """
        try:
            if self.account_manager.disconnect():
                self.is_connected = False
                self.last_error = None
                logger.info("MT5Client disconnected successfully")
                return True
            else:
                self.is_connected = False
                self.last_error = "Failed to disconnect from MT5"
                logger.error("Failed to disconnect from MT5")
                return False
        except Exception as e:
            self.is_connected = False
            self.last_error = str(e)
            logger.error(f"Error disconnecting from MT5: {e}")
            return False

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """
        Get account information.

        Returns:
            Account information dictionary or None if failed
        """
        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning("MT5 not connected, using simulated account info")
                return self.simulated_data["account_info"]
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            account_info = self.account_manager.get_connection_status()
            if account_info.get("connected"):
                # Extract relevant account info
                return {
                    "login": account_info.get("current_login"),
                    "server": account_info.get("current_server"),
                    "currency": account_info.get("currency"),
                    "balance": account_info.get("balance"),
                    "equity": account_info.get("equity"),
                    "margin": account_info.get("margin"),
                    "margin_free": account_info.get("free_margin"),
                    "profit": account_info.get("profit"),
                    "leverage": account_info.get("leverage", 100)
                }
            else:
                self.last_error = "Not connected to any account"
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting account info: {e}")
            if self.fallback_to_simulated:
                logger.warning("Using simulated account info due to error")
                return self.simulated_data["account_info"]
            return None

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information.

        Args:
            symbol: Symbol name (e.g., "EURUSD", "XAUUSD")

        Returns:
            Symbol information dictionary or None if failed
        """
        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning(f"MT5 not connected, using simulated symbol info for {symbol}")
                return self.simulated_data["symbols"].get(symbol)
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            symbol_info = self.account_manager.get_symbol_info(symbol)
            if symbol_info:
                return {
                    "name": symbol_info.name,
                    "digits": symbol_info.digits,
                    "point": symbol_info.point,
                    "tick_value": symbol_info.tick_value,
                    "contract_size": symbol_info.contract_size,
                    "volume_min": symbol_info.volume_min,
                    "volume_max": symbol_info.volume_max,
                    "volume_step": symbol_info.volume_step
                }
            else:
                self.last_error = f"Symbol {symbol} not found"
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            if self.fallback_to_simulated:
                logger.warning(f"Using simulated symbol info for {symbol} due to error")
                return self.simulated_data["symbols"].get(symbol)
            return None

    def calculate_pip_value(self, symbol: str, volume: float = 1.0) -> Optional[float]:
        """
        Calculate pip value for a symbol.

        Args:
            symbol: Symbol name (e.g., "EURUSD", "XAUUSD")
            volume: Volume in lots (default: 1.0)

        Returns:
            Pip value in account currency or None if failed
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return None

        try:
            # Formula: pip_value = tick_value * volume
            pip_value = symbol_info["tick_value"] * volume
            return pip_value
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error calculating pip value for {symbol}: {e}")
            return None

    def calculate_position_size(
        self,
        symbol: str,
        risk_amount: float,
        stop_loss_pips: float,
        account_risk_percent: float = 1.0
    ) -> Optional[float]:
        """
        Calculate position size using Kelly position sizing.

        Args:
            symbol: Symbol name (e.g., "EURUSD", "XAUUSD")
            risk_amount: Risk amount in account currency
            stop_loss_pips: Stop loss in pips
            account_risk_percent: Account risk percentage (default: 1.0%)

        Returns:
            Position size in lots or None if failed
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return None

        try:
            # Get account info for available margin
            account_info = self.get_account_info()
            if not account_info:
                return None

            # Calculate available margin
            available_margin = account_info["margin_free"]

            # Calculate risk per trade (as percentage of available margin)
            risk_per_trade = (available_margin * account_risk_percent) / 100

            # Calculate position size: (risk_per_trade / (stop_loss_pips * pip_value))
            pip_value = self.calculate_pip_value(symbol, 1.0)
            if pip_value is None or stop_loss_pips <= 0:
                return None

            position_size = risk_per_trade / (stop_loss_pips * pip_value)

            # Apply volume constraints
            min_volume = symbol_info["volume_min"]
            max_volume = symbol_info["volume_max"]
            volume_step = symbol_info["volume_step"]

            # Round to nearest volume step
            position_size = round(position_size / volume_step) * volume_step

            # Apply min/max constraints
            position_size = max(min_volume, min(position_size, max_volume))

            return position_size
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error calculating position size for {symbol}: {e}")
            return None

    def switch_account(self, login: int) -> bool:
        """
        Switch to a different MT5 account.

        Args:
            login: Account number to switch to

        Returns:
            True if switched successfully, False otherwise
        """
        if not self.is_connected:
            self.last_error = "MT5 not connected"
            return False

        try:
            result = self.account_manager.switch_account(login)
            if result.get("success"):
                self.is_connected = True
                self.last_error = None
                logger.info(f"Switched to account {login}")
                return True
            else:
                self.last_error = result.get("error", "Unknown error switching account")
                logger.error(f"Failed to switch to account {login}: {self.last_error}")
                return False
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error switching to account {login}: {e}")
            return False

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Dictionary with connection status information
        """
        return {
            "is_connected": self.is_connected,
            "last_error": self.last_error,
            "has_fallback": self.fallback_to_simulated
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure disconnection."""
        self.disconnect()


# Convenience functions for direct MCP tool usage
def get_mt5_account_info() -> Optional[Dict[str, Any]]:
    """
    Get MT5 account info using MCP tools directly.

    Returns:
        Account information dictionary or None if failed
    """
    try:
        from mcp_metatrader5 import get_account_info
        return get_account_info().dict()
    except Exception as e:
        logger.error(f"Error getting MT5 account info: {e}")
        return None


def get_mt5_symbol_info(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get MT5 symbol info using MCP tools directly.

    Args:
        symbol: Symbol name

    Returns:
        Symbol information dictionary or None if failed
    """
    try:
        from mcp_metatrader5 import get_symbol_info
        return get_symbol_info(symbol).dict()
    except Exception as e:
        logger.error(f"Error getting symbol info for {symbol}: {e}")
        return None


def calculate_mt5_pip_value(symbol: str, volume: float = 1.0) -> Optional[float]:
    """
    Calculate pip value using MCP tools directly.

    Args:
        symbol: Symbol name
        volume: Volume in lots

    Returns:
        Pip value or None if failed
    """
    symbol_info = get_mt5_symbol_info(symbol)
    if not symbol_info:
        return None

    try:
        return symbol_info["tick_value"] * volume
    except Exception as e:
        logger.error(f"Error calculating pip value for {symbol}: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    # Test the MT5 client
    client = MT5Client(fallback_to_simulated=True)

    # Test account info
    account_info = client.get_account_info()
    print("Account Info:", account_info)

    # Test symbol info
    symbol_info = client.get_symbol_info("EURUSD")
    print("EURUSD Info:", symbol_info)

    # Test pip value calculation
    pip_value = client.calculate_pip_value("EURUSD")
    print("EURUSD Pip Value:", pip_value)

    # Test position size calculation
    position_size = client.calculate_position_size(
        symbol="EURUSD",
        risk_amount=50.0,
        stop_loss_pips=20.0,
        account_risk_percent=1.0
    )
    print("Position Size for EURUSD:", position_size)

    # Test with XAUUSD (different pip size)
    position_size_xau = client.calculate_position_size(
        symbol="XAUUSD",
        risk_amount=50.0,
        stop_loss_pips=50.0,
        account_risk_percent=1.0
    )
    print("Position Size for XAUUSD:", position_size_xau)