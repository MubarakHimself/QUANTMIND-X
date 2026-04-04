"""MT5 Integration Module.

Modular structure for MT5 trading integration with Layer 1 SL/TP support (Story 14.1).
"""

from .client import MT5Client, create_mt5_client, get_mt5_client
from .account import AccountManager, AccountInfo, MarginInfo
from .symbols import SymbolInfo, TickData
from .orders import OrderManager, OrderInfo, EAInputParameters
from .exceptions import MT5ConnectionError, MT5SymbolError, MT5CacheError
from .cache import MT5Cache

__all__ = [
    'MT5Client',
    'create_mt5_client',
    'get_mt5_client',
    'AccountManager',
    'AccountInfo',
    'MarginInfo',
    'SymbolInfo',
    'TickData',
    'OrderManager',
    'OrderInfo',
    'EAInputParameters',
    'MT5ConnectionError',
    'MT5SymbolError',
    'MT5CacheError',
    'MT5Cache',
]
