"""MT5 Integration Module.

Modular structure for MT5 trading integration.
"""

from .client import MT5Client, create_mt5_client
from .account import AccountManager, AccountInfo, MarginInfo
from .symbols import SymbolInfo, TickData
from .orders import OrderManager, OrderInfo
from .exceptions import MT5ConnectionError, MT5SymbolError, MT5CacheError
from .cache import MT5Cache

__all__ = [
    'MT5Client',
    'create_mt5_client',
    'AccountManager',
    'AccountInfo',
    'MarginInfo',
    'SymbolInfo',
    'TickData',
    'OrderManager',
    'OrderInfo',
    'MT5ConnectionError',
    'MT5SymbolError',
    'MT5CacheError',
    'MT5Cache',
]
