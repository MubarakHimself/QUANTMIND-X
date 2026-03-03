"""
MT5 Integration Client - Backward Compatibility Layer
====================================================

This module provides backward compatibility by re-exporting from the new
modular mt5/ package.

New code should use:
    from src.risk.integrations.mt5 import MT5Client, AccountManager, SymbolInfo

Legacy code can continue using:
    from src.risk.integrations.mt5_client import MT5AccountClient, AccountInfo, SymbolInfo
"""

# Re-export from new modular structure for backward compatibility
from src.risk.integrations.mt5 import (
    MT5Client,
    AccountManager,
    AccountInfo,
    SymbolInfo,
    MarginInfo,
    TickData,
    OrderManager,
    OrderInfo,
    MT5ConnectionError,
    MT5SymbolError,
    MT5CacheError,
    MT5Cache,
    create_mt5_client,
)

# Backward compatibility: MT5AccountClient is now MT5Client
MT5AccountClient = MT5Client

# Backward compatibility: CachedValue is now in cache module
from src.risk.integrations.mt5.cache import CachedValue

__all__ = [
    'MT5Client',
    'MT5AccountClient',  # Alias for backward compatibility
    'AccountManager',
    'AccountInfo',
    'SymbolInfo',
    'MarginInfo',
    'TickData',
    'OrderManager',
    'OrderInfo',
    'MT5ConnectionError',
    'MT5SymbolError',
    'MT5CacheError',
    'MT5Cache',
    'CachedValue',
    'create_mt5_client',
]
