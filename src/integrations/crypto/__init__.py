"""
V8 Crypto Module: Binance Integration

Provides unified broker interface for cryptocurrency trading with
WebSocket order book streaming and sub-10ms latency.
"""

from .broker_client import BrokerClient
from .binance_connector import BinanceConnector
from .stream_client import BinanceStreamClient

__all__ = [
    'BrokerClient',
    'BinanceConnector',
    'BinanceStreamClient',
]

