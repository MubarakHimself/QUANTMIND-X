"""
Live Tick Data Streaming Handler

Streams real-time tick data from MT5 to UI via WebSocket broadcasting.
"""

import asyncio
import time
import logging
from typing import Dict, Set, Optional
from datetime import datetime

from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
from src.api.websocket_endpoints import manager

logger = logging.getLogger(__name__)

class TickStreamHandler:
    def __init__(self, mt5_adapter: MT5SocketAdapter):
        self.mt5_adapter = mt5_adapter
        self.subscribed_symbols: Set[str] = set()
        self.is_streaming = False
        self.streaming_task: Optional[asyncio.Task] = None
        self.throttle_interval = 0.1
        self.last_update_time: Dict[str, float] = {}

    async def subscribe(self, symbol: str):
        self.subscribed_symbols.add(symbol)
        if not self.is_streaming and self.subscribed_symbols:
            await self.start_streaming()
        logger.info(f"Subscribed to tick data for {symbol}")

    async def unsubscribe(self, symbol: str):
        self.subscribed_symbols.discard(symbol)
        if not self.subscribed_symbols and self.is_streaming:
            await self.stop_streaming()
        logger.info(f"Unsubscribed from tick data for {symbol}")

    async def start_streaming(self):
        if self.is_streaming:
            return
        self.is_streaming = True
        self.streaming_task = asyncio.create_task(self._stream_loop())

    async def stop_streaming(self):
        if not self.is_streaming:
            return
        self.is_streaming = False
        if self.streaming_task:
            self.streaming_task.cancel()
            try:
                await self.streaming_task
            except asyncio.CancelledError:
                pass

    async def _stream_loop(self):
        while self.is_streaming:
            tasks = [self._fetch_and_broadcast_tick(symbol) for symbol in list(self.subscribed_symbols)]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(self.throttle_interval)

    async def _fetch_and_broadcast_tick(self, symbol: str):
        now = time.time()
        if symbol in self.last_update_time and (now - self.last_update_time[symbol]) < self.throttle_interval:
            return

        try:
            order_book = await self.mt5_adapter.get_order_book(symbol)
            bid = order_book['bids'][0][0] if order_book['bids'] else 0.0
            ask = order_book['asks'][0][0] if order_book['asks'] else 0.0
            spread = ask - bid

            timestamp = datetime.utcnow().isoformat() + 'Z'

            message = {
                "type": "tick_data",
                "data": {
                    "symbol": symbol,
                    "bid": round(bid, 5),
                    "ask": round(ask, 5),
                    "spread": round(spread, 5),
                    "timestamp": timestamp
                }
            }
            await manager.broadcast(message, topic="tick_data")
            self.last_update_time[symbol] = now
        except Exception as e:
            logger.error(f"Failed to fetch/broadcast tick for {symbol}: {e}")

_tick_handler_instance: Optional[TickStreamHandler] = None
_tick_handler_adapter: Optional[MT5SocketAdapter] = None

def get_tick_handler(mt5_adapter: Optional[MT5SocketAdapter] = None) -> TickStreamHandler:
    global _tick_handler_instance, _tick_handler_adapter
    if _tick_handler_instance is None:
        if mt5_adapter is None:
            raise ValueError("MT5 adapter required for first initialization")
        _tick_handler_adapter = mt5_adapter
        _tick_handler_instance = TickStreamHandler(mt5_adapter)
    elif mt5_adapter is not None and _tick_handler_adapter != mt5_adapter:
        logger.warning("MT5 adapter mismatch, using existing handler")
    return _tick_handler_instance
