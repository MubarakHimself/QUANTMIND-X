"""
V8 Unified Broker Registry

Provides factory pattern for managing multiple broker connections
(MT5, Binance, etc.) through a unified interface.

**Validates: Task Group 25**
"""

from .registry import BrokerRegistry
from .mock_mt5_adapter import MockMT5Adapter
from .mt5_socket_adapter import MT5SocketAdapter
from .binance_adapter import BinanceSpotAdapter, BinanceFuturesAdapter

__all__ = [
    'BrokerRegistry',
    'MockMT5Adapter',
    'MT5SocketAdapter',
    'BinanceSpotAdapter',
    'BinanceFuturesAdapter',
]
