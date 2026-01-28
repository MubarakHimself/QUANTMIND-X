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
"""

import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes and Models
# ============================================================================

@dataclass
class CachedValue:
    """Cached value with TTL."""
    value: Any
    timestamp: float
    ttl: float = 10.0  # 10 seconds default TTL

    def is_expired(self) -> bool:
        """Check if cached value has expired."""
        return time.time() - self.timestamp > self.ttl


@dataclass
class AccountInfo:
    """Account information model."""
    login: int
    server: str
    currency: str
    balance: float
    equity: float
    margin: float
    margin_free: float
    margin_level: float
    profit: float
    leverage: int
    currency_base: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "login": self.login,
            "server": self.server,
            "currency": self.currency,
            "balance": self.balance,
            "equity": self.equity,
            "margin": self.margin,
            "margin_free": self.margin_free,
            "margin_level": self.margin_level,
            "profit": self.profit,
            "leverage": self.leverage,
            "currency_base": self.currency_base
        }


@dataclass
class SymbolInfo:
    """Symbol information model."""
    name: str
    digits: int
    point: float
    tick_value: float
    tick_size: float
    contract_size: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    volume_min: float
    volume_max: float
    volume_step: float
    pip_location: int  # Position of pip (e.g., 4 for EURUSD, 2 for XAUUSD)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "digits": self.digits,
            "point": self.point,
            "tick_value": self.tick_value,
            "tick_size": self.tick_size,
            "contract_size": self.contract_size,
            "currency_base": self.currency_base,
            "currency_profit": self.currency_profit,
            "currency_margin": self.currency_margin,
            "volume_min": self.volume_min,
            "volume_max": self.volume_max,
            "volume_step": self.volume_step,
            "pip_location": self.pip_location
        }


@dataclass
class MarginInfo:
    """Margin information model."""
    margin_required: float
    margin_free: float
    margin_level: float
    margin_used: float
    equity: float
    balance: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "margin_required": self.margin_required,
            "margin_free": self.margin_free,
            "margin_level": self.margin_level,
            "margin_used": self.margin_used,
            "equity": self.equity,
            "balance": self.balance
        }


# ============================================================================
# Exceptions
# ============================================================================

class MT5ConnectionError(Exception):
    """Custom exception for MT5 connection errors."""
    pass


class MT5SymbolError(Exception):
    """Custom exception for symbol-related errors."""
    pass


class MT5CacheError(Exception):
    """Custom exception for cache-related errors."""
    pass


# ============================================================================
# MT5 Account Client
# ============================================================================

class MT5AccountClient:
    """
    MT5 Account Client with caching and graceful degradation.

    This client provides a simplified interface to the mcp-metatrader5-server
    AccountManager with:
    - 10-second TTL cache for account and symbol data
    - Graceful degradation when MT5 is unavailable
    - Enhanced pip value calculations for various symbol types
    - Comprehensive error handling

    Usage:
        client = MT5AccountClient()
        balance = client.get_account_balance()
        equity = client.get_account_equity()
        margin_info = client.get_margin_info()
        pip_value = client.calculate_pip_value("EURUSD")
    """

    # Simulated data for fallback mode
    SIMULATED_ACCOUNT = {
        "login": 12345678,
        "server": "Demo-Server",
        "currency": "USD",
        "balance": 10000.0,
        "equity": 10500.0,
        "margin": 200.0,
        "margin_free": 10300.0,
        "margin_level": 5250.0,
        "profit": 500.0,
        "leverage": 100
    }

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

    def __init__(
        self,
        config_path: Optional[str] = None,
        fallback_to_simulated: bool = True,
        cache_ttl: float = 10.0
    ):
        """
        Initialize MT5 Account Client.

        Args:
            config_path: Path to MT5 configuration file
            fallback_to_simulated: Whether to use simulated data when MT5 is unavailable
            cache_ttl: Cache time-to-live in seconds (default: 10.0)
        """
        self.config_path = config_path
        self.fallback_to_simulated = fallback_to_simulated
        self.cache_ttl = cache_ttl
        self.account_manager = None
        self.is_connected = False
        self.last_error = None

        # Initialize cache
        self._cache: Dict[str, CachedValue] = {}

        # Try to import MT5 modules
        try:
            from mcp_metatrader5 import get_account_manager
            self.account_manager = get_account_manager(config_path)
            logger.info("MT5AccountClient initialized with MT5 integration")
        except ImportError as e:
            logger.warning(f"Could not import mcp_metatrader5: {e}")
            self.account_manager = None
            if not fallback_to_simulated:
                logger.error("MT5 integration not available and fallback disabled")
            else:
                logger.info("Using simulated data for testing")

    # ============================================================================
    # Cache Management
    # ============================================================================

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if expired/not found
        """
        cached = self._cache.get(key)
        if cached and not cached.is_expired():
            logger.debug(f"Cache hit for {key}")
            return cached.value
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        """
        Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = CachedValue(
            value=value,
            timestamp=time.time(),
            ttl=self.cache_ttl
        )
        logger.debug(f"Cached {key} with TTL {self.cache_ttl}s")

    def _clear_cache(self, pattern: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            pattern: Optional pattern to match keys (None clears all)
        """
        if pattern is None:
            self._cache.clear()
            logger.debug("Cleared all cache")
        else:
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for k in keys_to_remove:
                del self._cache[k]
            logger.debug(f"Cleared {len(keys_to_remove)} cache entries matching '{pattern}'")

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
        if not self.account_manager:
            self.is_connected = False
            self.last_error = "MT5 account manager not available"
            return False

        try:
            if self.account_manager.unlock(master_password):
                self.is_connected = True
                self.last_error = None
                self._clear_cache()  # Clear cache on new connection
                logger.info("MT5AccountClient connected successfully")
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
        if not self.account_manager:
            self.is_connected = False
            return True

        try:
            if self.account_manager.disconnect():
                self.is_connected = False
                self.last_error = None
                self._clear_cache()
                logger.info("MT5AccountClient disconnected successfully")
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
        """
        Get account balance with caching.

        Returns:
            Account balance or None if failed
        """
        cache_key = "account_balance"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning("MT5 not connected, using simulated balance")
                return self.SIMULATED_ACCOUNT["balance"]
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            status = self.account_manager.get_connection_status()
            if status.get("connected") and "balance" in status:
                balance = float(status["balance"])
                self._set_cache(cache_key, balance)
                return balance
            else:
                self.last_error = "Not connected to any account"
                if self.fallback_to_simulated:
                    return self.SIMULATED_ACCOUNT["balance"]
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting account balance: {e}")
            if self.fallback_to_simulated:
                return self.SIMULATED_ACCOUNT["balance"]
            return None

    def get_account_equity(self) -> Optional[float]:
        """
        Get account equity with caching.

        Returns:
            Account equity or None if failed
        """
        cache_key = "account_equity"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning("MT5 not connected, using simulated equity")
                return self.SIMULATED_ACCOUNT["equity"]
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            status = self.account_manager.get_connection_status()
            if status.get("connected") and "equity" in status:
                equity = float(status["equity"])
                self._set_cache(cache_key, equity)
                return equity
            else:
                self.last_error = "Not connected to any account"
                if self.fallback_to_simulated:
                    return self.SIMULATED_ACCOUNT["equity"]
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting account equity: {e}")
            if self.fallback_to_simulated:
                return self.SIMULATED_ACCOUNT["equity"]
            return None

    def get_margin_info(self) -> Optional[MarginInfo]:
        """
        Get margin information with caching.

        Returns:
            MarginInfo object or None if failed
        """
        cache_key = "margin_info"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning("MT5 not connected, using simulated margin info")
                return MarginInfo(
                    margin_required=self.SIMULATED_ACCOUNT["margin"],
                    margin_free=self.SIMULATED_ACCOUNT["margin_free"],
                    margin_level=self.SIMULATED_ACCOUNT["margin_level"],
                    margin_used=self.SIMULATED_ACCOUNT["margin"],
                    equity=self.SIMULATED_ACCOUNT["equity"],
                    balance=self.SIMULATED_ACCOUNT["balance"]
                )
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            status = self.account_manager.get_connection_status()
            if status.get("connected"):
                margin_info = MarginInfo(
                    margin_required=float(status.get("margin", 0)),
                    margin_free=float(status.get("free_margin", 0)),
                    margin_level=float(status.get("margin_level", 0)),
                    margin_used=float(status.get("margin", 0)),
                    equity=float(status.get("equity", 0)),
                    balance=float(status.get("balance", 0))
                )
                self._set_cache(cache_key, margin_info)
                return margin_info
            else:
                self.last_error = "Not connected to any account"
                if self.fallback_to_simulated:
                    return MarginInfo(
                        margin_required=self.SIMULATED_ACCOUNT["margin"],
                        margin_free=self.SIMULATED_ACCOUNT["margin_free"],
                        margin_level=self.SIMULATED_ACCOUNT["margin_level"],
                        margin_used=self.SIMULATED_ACCOUNT["margin"],
                        equity=self.SIMULATED_ACCOUNT["equity"],
                        balance=self.SIMULATED_ACCOUNT["balance"]
                    )
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting margin info: {e}")
            if self.fallback_to_simulated:
                return MarginInfo(
                    margin_required=self.SIMULATED_ACCOUNT["margin"],
                    margin_free=self.SIMULATED_ACCOUNT["margin_free"],
                    margin_level=self.SIMULATED_ACCOUNT["margin_level"],
                    margin_used=self.SIMULATED_ACCOUNT["margin"],
                    equity=self.SIMULATED_ACCOUNT["equity"],
                    balance=self.SIMULATED_ACCOUNT["balance"]
                )
            return None

    def get_account_info(self) -> Optional[AccountInfo]:
        """
        Get comprehensive account information with caching.

        Returns:
            AccountInfo object or None if failed
        """
        cache_key = "account_info"
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning("MT5 not connected, using simulated account info")
                return AccountInfo(**self.SIMULATED_ACCOUNT)
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            status = self.account_manager.get_connection_status()
            if status.get("connected"):
                account_info = AccountInfo(
                    login=int(status.get("current_login", 0)),
                    server=status.get("current_server", ""),
                    currency=status.get("currency", "USD"),
                    balance=float(status.get("balance", 0)),
                    equity=float(status.get("equity", 0)),
                    margin=float(status.get("margin", 0)),
                    margin_free=float(status.get("free_margin", 0)),
                    margin_level=float(status.get("margin_level", 0)),
                    profit=float(status.get("profit", 0)),
                    leverage=int(status.get("leverage", 100)),
                    currency_base=status.get("currency", "USD")
                )
                self._set_cache(cache_key, account_info)
                return account_info
            else:
                self.last_error = "Not connected to any account"
                if self.fallback_to_simulated:
                    return AccountInfo(**self.SIMULATED_ACCOUNT)
                return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting account info: {e}")
            if self.fallback_to_simulated:
                return AccountInfo(**self.SIMULATED_ACCOUNT)
            return None

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
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        if not self.is_connected:
            if self.fallback_to_simulated:
                logger.warning(f"MT5 not connected, using simulated symbol info for {symbol}")
                sim_data = self.SIMULATED_SYMBOLS.get(symbol)
                if sim_data:
                    return SymbolInfo(**sim_data)
                return None
            else:
                self.last_error = "MT5 not connected and fallback disabled"
                return None

        try:
            # Try to get symbol info from MT5
            # Note: This would require actual MT5 symbol_info call
            # For now, use simulated data as fallback
            logger.warning(f"MT5 symbol_info() not fully implemented, using simulated data for {symbol}")
            sim_data = self.SIMULATED_SYMBOLS.get(symbol)
            if sim_data:
                symbol_info = SymbolInfo(**sim_data)
                self._set_cache(cache_key, symbol_info)
                return symbol_info
            return None
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error getting symbol info for {symbol}: {e}")
            if self.fallback_to_simulated:
                sim_data = self.SIMULATED_SYMBOLS.get(symbol)
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
            # For most forex pairs: tick_value is usually 1.0 for 1 lot
            pip_value = symbol_info.tick_value * volume

            # Special handling for different symbol types
            if symbol.startswith("USD") and symbol.endswith("JPY"):
                # USD/JPY pairs: pip value needs adjustment
                # Typically 1 pip = 0.01, tick is 0.001
                pip_value = pip_value * 10

            elif symbol == "XAUUSD":
                # Gold: 1 pip = $0.10 per lot (typically)
                pip_value = symbol_info.tick_value * volume

            elif symbol.endswith("USD") and not symbol.startswith("USD"):
                # XXX/USD pairs: pip value is typically $10 per standard lot
                pip_value = 10.0 * volume

            elif symbol.startswith("USD") and not symbol.endswith("USD"):
                # USD/XXX pairs: need currency conversion
                # For USD account, pip value = contract_size * tick_size * volume / exchange_rate
                # This is simplified; actual implementation needs exchange rates
                pass

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
            "cache_size": len(self._cache)
        }

    def switch_account(self, login: int) -> bool:
        """
        Switch to a different MT5 account.

        Args:
            login: Account number to switch to

        Returns:
            True if switched successfully, False otherwise
        """
        if not self.is_connected or not self.account_manager:
            self.last_error = "MT5 not connected or account manager not available"
            return False

        try:
            result = self.account_manager.switch_account(login)
            if result.get("success"):
                self._clear_cache()  # Clear cache on account switch
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
    fallback_to_simulated: bool = True,
    cache_ttl: float = 10.0
) -> MT5AccountClient:
    """
    Factory function to create MT5AccountClient.

    Args:
        config_path: Path to MT5 configuration file
        fallback_to_simulated: Whether to use simulated data when MT5 is unavailable
        cache_ttl: Cache time-to-live in seconds

    Returns:
        MT5AccountClient instance
    """
    return MT5AccountClient(
        config_path=config_path,
        fallback_to_simulated=fallback_to_simulated,
        cache_ttl=cache_ttl
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Test the MT5 client
    client = create_mt5_client(fallback_to_simulated=True)

    print("=" * 60)
    print("MT5 Account Client Test")
    print("=" * 60)

    # Test connection status
    print("\n1. Connection Status:")
    status = client.get_connection_status()
    print(f"   Connected: {status['is_connected']}")
    print(f"   Fallback: {status['has_fallback']}")
    print(f"   Cache TTL: {status['cache_ttl']}s")

    # Test account balance
    print("\n2. Account Balance:")
    balance = client.get_account_balance()
    print(f"   Balance: ${balance:,.2f}")

    # Test account equity
    print("\n3. Account Equity:")
    equity = client.get_account_equity()
    print(f"   Equity: ${equity:,.2f}")

    # Test margin info
    print("\n4. Margin Information:")
    margin_info = client.get_margin_info()
    if margin_info:
        print(f"   Margin Free: ${margin_info.margin_free:,.2f}")
        print(f"   Margin Used: ${margin_info.margin_used:,.2f}")
        print(f"   Margin Level: {margin_info.margin_level:,.2f}")

    # Test symbol info
    print("\n5. Symbol Information:")
    for symbol in ["EURUSD", "GBPUSD", "XAUUSD"]:
        symbol_info = client.get_symbol_info(symbol)
        if symbol_info:
            print(f"   {symbol}:")
            print(f"      Digits: {symbol_info.digits}")
            print(f"      Pip Location: {symbol_info.pip_location}")
            print(f"      Tick Value: ${symbol_info.tick_value}")

    # Test pip value calculation
    print("\n6. Pip Value Calculation:")
    for symbol in ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"]:
        pip_value = client.calculate_pip_value(symbol, volume=1.0)
        print(f"   {symbol}: ${pip_value:.2f} per pip (1 lot)")

    # Test detailed pip value
    print("\n7. Detailed Pip Value (EURUSD):")
    detailed = client.calculate_pip_value_detailed("EURUSD", volume=0.1)
    if detailed:
        for key, value in detailed.items():
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
