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
- 10-second TTL cache for account and symbol data
- Enhanced pip value calculations for various symbol types
- Separate demo and live connection support for tick data
"""

import os
import logging
from typing import Optional, Dict, Any, Literal

from .account import AccountManager, AccountInfo, MarginInfo, SIMULATED_ACCOUNT
from .symbols import SymbolInfo, TickData
from .exceptions import MT5ConnectionError, MT5SymbolError, MT5CacheError
from .cache import MT5Cache

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_simulated_fallback() -> bool:
    """Simulated MT5 fallback must be explicit; production defaults to off."""
    return _env_flag("MT5_ALLOW_SIMULATED_FALLBACK", default=False)


# Environment Configuration
USE_DEMO_TICKS = os.environ.get('USE_DEMO_TICKS', 'false').lower() == 'true'
MT5_DEMO_LOGIN = os.environ.get('MT5_DEMO_LOGIN', '')
MT5_DEMO_PASSWORD = os.environ.get('MT5_DEMO_PASSWORD', '')
MT5_DEMO_SERVER = os.environ.get('MT5_DEMO_SERVER', '')


# Simulated symbols data
SIMULATED_SYMBOLS = {
    "EURUSD": {
        "name": "EURUSD",
        "digits": 5,
        "point": 0.00001,
        "tick_value": 1.0,
        "tick_size": 0.00001,
        "contract_size": 100000,
        "currency_base": "EUR",
        "currency_profit": "USD",
        "currency_margin": "EUR",
        "volume_min": 0.01,
        "volume_max": 1000.0,
        "volume_step": 0.01,
        "pip_location": 4
    },
    "GBPUSD": {
        "name": "GBPUSD",
        "digits": 5,
        "point": 0.00001,
        "tick_value": 1.0,
        "tick_size": 0.00001,
        "contract_size": 100000,
        "currency_base": "GBP",
        "currency_profit": "USD",
        "currency_margin": "GBP",
        "volume_min": 0.01,
        "volume_max": 1000.0,
        "volume_step": 0.01,
        "pip_location": 4
    },
    "XAUUSD": {
        "name": "XAUUSD",
        "digits": 2,
        "point": 0.01,
        "tick_value": 1.0,
        "tick_size": 0.01,
        "contract_size": 100,
        "currency_base": "XAU",
        "currency_profit": "USD",
        "currency_margin": "USD",
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "pip_location": 1  # Gold uses 1 decimal for pips
    },
    "USDJPY": {
        "name": "USDJPY",
        "digits": 3,
        "point": 0.001,
        "tick_value": 9.0909,  # Approximate for USD account
        "tick_size": 0.001,
        "contract_size": 100000,
        "currency_base": "USD",
        "currency_profit": "JPY",
        "currency_margin": "USD",
        "volume_min": 0.01,
        "volume_max": 1000.0,
        "volume_step": 0.01,
        "pip_location": 2  # JPY pairs use 2 decimals
    },
    "US30": {
        "name": "US30",
        "digits": 2,
        "point": 0.01,
        "tick_value": 1.0,
        "tick_size": 1.0,
        "contract_size": 1,
        "currency_base": "USD",
        "currency_profit": "USD",
        "currency_margin": "USD",
        "volume_min": 1.0,
        "volume_max": 1000.0,
        "volume_step": 1.0,
        "pip_location": 0  # Indices often use whole points
    }
}


class MT5Client:
    """
    MT5 Client with caching and graceful degradation.

    This client provides a simplified interface to the mcp-metatrader5-server
    AccountManager with:
    - 10-second TTL cache for account and symbol data
    - Graceful degradation when MT5 is unavailable
    - Enhanced pip value calculations for various symbol types
    - Comprehensive error handling

    Usage:
        client = MT5Client()
        balance = client.get_account_balance()
        equity = client.get_account_equity()
        margin_info = client.get_margin_info()
        pip_value = client.calculate_pip_value("EURUSD")
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        fallback_to_simulated: Optional[bool] = None,
        cache_ttl: float = 10.0,
        use_demo_ticks: Optional[bool] = None
    ):
        """
        Initialize MT5 Client.

        Args:
            config_path: Path to MT5 configuration file
            fallback_to_simulated: Whether to use simulated data when MT5 is unavailable
            cache_ttl: Cache time-to-live in seconds (default: 10.0)
            use_demo_ticks: Override for USE_DEMO_TICKS env var. If True, uses demo
                           connection for tick data. If None, uses env var.
        """
        self.config_path = config_path
        self.fallback_to_simulated = (
            _default_simulated_fallback()
            if fallback_to_simulated is None
            else fallback_to_simulated
        )
        self.cache_ttl = cache_ttl
        self.account_manager = None
        self.is_connected = False
        self.last_error = None

        # Demo connection for tick data (prevents broker manipulation)
        self._use_demo_ticks = use_demo_ticks if use_demo_ticks is not None else USE_DEMO_TICKS
        self._demo_account_manager = None
        self._demo_connected = False

        # Initialize cache (both MT5Cache and dict for backward compatibility)
        self._mt5_cache = MT5Cache(cache_ttl)
        self._cache: Dict[str, Any] = {}  # Backward compatible dict access

        # Initialize account manager
        self._account_manager_internal = None
        try:
            from mcp_metatrader5 import get_account_manager
            self._account_manager_internal = get_account_manager(config_path)
            logger.info("MT5Client initialized with MT5 integration")
        except ImportError as e:
            logger.warning(f"Could not import mcp_metatrader5: {e}")
            self._account_manager_internal = None
            if not self.fallback_to_simulated:
                logger.error("MT5 integration not available and fallback disabled")
            else:
                logger.warning("Using simulated MT5 fallback; disable in production")

        # Create AccountManager
        self.account_manager = AccountManager(
            self._account_manager_internal,
            self.fallback_to_simulated
        )

        # Log demo tick configuration
        if self._use_demo_ticks:
            logger.info("Demo tick source enabled - tick data will use separate demo connection")

    # ============================================================================
    # Connection Management
    # ============================================================================

    def connect(self, master_password: str) -> bool:
        """
        Connect to MT5 using the account manager.

        Args:
            master_password: Master password for credential vault

        Returns:
            True if connected successfully, False otherwise
        """
        if not self._account_manager_internal:
            self.is_connected = False
            self.last_error = "MT5 account manager not available"
            return False

        try:
            if self._account_manager_internal.unlock(master_password):
                self.is_connected = True
                self.last_error = None
                self._cache.clear()
                self._mt5_cache.clear()
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
        if not self._account_manager_internal:
            self.is_connected = False
            return True

        try:
            if self._account_manager_internal.disconnect():
                self.is_connected = False
                self.last_error = None
                self._cache.clear()
                self._mt5_cache.clear()
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

    # ============================================================================
    # Account Information
    # ============================================================================

    def get_account_balance(self) -> Optional[float]:
        """Get account balance with caching."""
        cache_key = "account_balance"
        cached = self._mt5_cache.get(cache_key)
        if cached is not None:
            return cached

        balance = self.account_manager.get_account_balance()
        if balance is not None:
            self._mt5_cache.set(cache_key, balance)
        return balance

    def get_account_equity(self) -> Optional[float]:
        """Get account equity with caching."""
        cache_key = "account_equity"
        cached = self._mt5_cache.get(cache_key)
        if cached is not None:
            return cached

        equity = self.account_manager.get_account_equity()
        if equity is not None:
            self._mt5_cache.set(cache_key, equity)
        return equity

    def get_margin_info(self) -> Optional[MarginInfo]:
        """Get margin information with caching."""
        cache_key = "margin_info"
        cached = self._mt5_cache.get(cache_key)
        if cached is not None:
            return cached

        margin_info = self.account_manager.get_margin_info()
        if margin_info is not None:
            self._mt5_cache.set(cache_key, margin_info)
        return margin_info

    def get_account_info(self) -> Optional[AccountInfo]:
        """Get comprehensive account information with caching."""
        cache_key = "account_info"
        cached = self._mt5_cache.get(cache_key)
        if cached is not None:
            return cached

        account_info = self.account_manager.get_account_info()
        if account_info is not None:
            self._mt5_cache.set(cache_key, account_info)
        return account_info

    # ============================================================================
    # Symbol Information
    # ============================================================================

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """
        Get symbol information with caching.

        Args:
            symbol: Symbol name (e.g., "EURUSD", "XAUUSD")

        Returns:
            SymbolInfo object or None if failed
        """
        cache_key = f"symbol_info_{symbol}"
        cached = self._mt5_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning(f"MT5 not connected, using simulated symbol info for {symbol}")
                sim_data = SIMULATED_SYMBOLS.get(symbol)
                if sim_data:
                    return SymbolInfo(**sim_data)
            return None

        try:
            logger.warning(f"MT5 symbol_info() not fully implemented, using simulated data for {symbol}")
            sim_data = SIMULATED_SYMBOLS.get(symbol)
            if sim_data:
                symbol_info = SymbolInfo(**sim_data)
                self._mt5_cache.set(cache_key, symbol_info)
                return symbol_info
            return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            if self.fallback_to_simulated:
                sim_data = SIMULATED_SYMBOLS.get(symbol)
                if sim_data:
                    return SymbolInfo(**sim_data)
            return None

    # ============================================================================
    # Pip Value Calculations
    # ============================================================================

    def calculate_pip_value(
        self,
        symbol: str,
        volume: float = 1.0,
        account_currency: str = "USD"
    ) -> Optional[float]:
        """
        Calculate pip value for a symbol.

        The pip value calculation considers:
        - Symbol's tick value and tick size
        - Contract size
        - Currency conversion for cross pairs

        Args:
            symbol: Symbol name (e.g., "EURUSD", "XAUUSD")
            volume: Volume in lots (default: 1.0)
            account_currency: Account currency for conversion (default: "USD")

        Returns:
            Pip value in account currency or None if failed
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Cannot calculate pip value: symbol info not available for {symbol}")
            return None

        try:
            # Basic pip value formula: tick_value * volume
            pip_value = symbol_info.tick_value * volume

            # Special handling for different symbol types
            if symbol.startswith("USD") and symbol.endswith("JPY"):
                # USD/JPY pairs: pip value needs adjustment
                pip_value = pip_value * 10

            elif symbol == "XAUUSD":
                # Gold: 1 pip = $0.10 per lot (typically)
                pip_value = symbol_info.tick_value * volume

            elif symbol.endswith("USD") and not symbol.startswith("USD"):
                # XXX/USD pairs: pip value is typically $10 per standard lot
                pip_value = 10.0 * volume

            elif symbol.startswith("USD") and not symbol.endswith("USD"):
                # USD/XXX pairs: need currency conversion
                quote_currency = symbol[3:6]
                exchange_rate = 1.0

                try:
                    direct_tick = self.symbol_info_tick(f"{quote_currency}USD")
                    if direct_tick and direct_tick.ask:
                        exchange_rate = direct_tick.ask
                    else:
                        inverse_tick = self.symbol_info_tick(f"USD{quote_currency}")
                        if inverse_tick and inverse_tick.ask:
                            exchange_rate = 1.0 / inverse_tick.ask
                except Exception as e:
                    logger.warning(f"Could not get conversion rate for {symbol}: {e}")

                pip_value = (symbol_info.tick_value * volume) / exchange_rate if exchange_rate != 1.0 else symbol_info.tick_value * volume
                logger.debug(f"Converted pip value for {symbol} using rate {exchange_rate}: {pip_value}")

            logger.debug(f"Calculated pip value for {symbol}: {pip_value} {account_currency}")
            return pip_value

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error calculating pip value for {symbol}: {e}")
            return None

    def calculate_pip_value_detailed(
        self,
        symbol: str,
        volume: float = 1.0,
        account_currency: str = "USD"
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate pip value with detailed breakdown.

        Args:
            symbol: Symbol name
            volume: Volume in lots
            account_currency: Account currency

        Returns:
            Dictionary with detailed pip value breakdown or None if failed
        """
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return None

        try:
            pip_value = self.calculate_pip_value(symbol, volume, account_currency)
            if pip_value is None:
                return None

            return {
                "symbol": symbol,
                "volume": volume,
                "pip_value": pip_value,
                "currency": account_currency,
                "pip_location": symbol_info.pip_location,
                "tick_value": symbol_info.tick_value,
                "tick_size": symbol_info.tick_size,
                "contract_size": symbol_info.contract_size,
                "digits": symbol_info.digits
            }
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error calculating detailed pip value for {symbol}: {e}")
            return None

    def symbol_info_tick(self, symbol: str):
        """Get tick info for a symbol (placeholder for real implementation)."""
        # Placeholder - would need real MT5 integration
        return None

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status.

        Returns:
            Dictionary with connection status information
        """
        return {
            "is_connected": self.is_connected,
            "last_error": self.last_error,
            "has_fallback": self.fallback_to_simulated,
            "cache_ttl": self.cache_ttl,
            "cache_size": self._mt5_cache.size()
        }

    # ============================================================================
    # Backward Compatibility Methods
    # ============================================================================

    def _clear_cache(self, pattern: Optional[str] = None) -> None:
        """
        Clear cache entries (backward compatibility method).

        Args:
            pattern: Optional pattern to match keys (None clears all)
        """
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
        # Also clear the MT5Cache
        self._mt5_cache.clear(pattern)

    def switch_account(self, login: int) -> bool:
        """
        Switch to a different MT5 account.

        Args:
            login: Account number to switch to

        Returns:
            True if switched successfully, False otherwise
        """
        if not self.is_connected or not self._account_manager_internal:
            self.last_error = "MT5 not connected or account manager not available"
            return False

        try:
            result = self._account_manager_internal.switch_account(login)
            if result.get("success"):
                self._cache.clear()
                self._mt5_cache.clear()
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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure disconnection."""
        self.disconnect()


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mt5_client(
    config_path: Optional[str] = None,
    fallback_to_simulated: Optional[bool] = None,
    cache_ttl: float = 10.0
) -> MT5Client:
    """
    Factory function to create MT5Client.

    Args:
        config_path: Path to MT5 configuration file
        fallback_to_simulated: Whether to use simulated data when MT5 is unavailable
        cache_ttl: Cache time-to-live in seconds

    Returns:
        MT5Client instance
    """
    return MT5Client(
        config_path=config_path,
        fallback_to_simulated=fallback_to_simulated,
        cache_ttl=cache_ttl
    )


_MT5_CLIENT_SINGLETON: Optional[MT5Client] = None


def get_mt5_client(
    config_path: Optional[str] = None,
    fallback_to_simulated: Optional[bool] = None,
    cache_ttl: float = 10.0,
    reset: bool = False,
) -> MT5Client:
    """Legacy singleton accessor used by router/API compat call sites."""
    global _MT5_CLIENT_SINGLETON

    if reset or _MT5_CLIENT_SINGLETON is None:
        _MT5_CLIENT_SINGLETON = create_mt5_client(
            config_path=config_path,
            fallback_to_simulated=fallback_to_simulated,
            cache_ttl=cache_ttl,
        )

    return _MT5_CLIENT_SINGLETON
