"""
V8 Crypto Module: Binance WebSocket Stream Client

Provides real-time order book streaming with <10ms latency for
HFT scalping strategies.

**Validates: Requirement 18.3, 18.4, 18.5**
"""

import asyncio
import aiohttp
import json
import logging
from typing import Callable, Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceStreamClient:
    """
    WebSocket client for real-time order book streaming.
    
    Features:
    - Subscribe to multiple symbol streams
    - Depth5 order book updates (<10ms latency)
    - Automatic reconnection on disconnect
    - Callback-based update handling
    
    **Validates: Requirement 18.3, 18.4, 18.5**
    """
    
    def __init__(
        self,
        symbols: List[str],
        callback: Callable[[str, Dict[str, Any]], None],
        testnet: bool = False,
        futures: bool = False
    ):
        """
        Initialize WebSocket stream client.
        
        Args:
            symbols: List of symbols to stream (e.g., ["BTCUSDT", "ETHUSDT"])
            callback: Callback function for order book updates
            testnet: Use testnet environment
            futures: Use Futures streams
        """
        self.symbols = [s.lower() for s in symbols]
        self.callback = callback
        self.testnet = testnet
        self.futures = futures
        self.running = False
        self.ws = None
        
        # Set WebSocket URL
        if testnet:
            self.ws_url = "wss://testnet.binance.vision/ws"
        else:
            if futures:
                self.ws_url = "wss://fstream.binance.com/ws"
            else:
                self.ws_url = "wss://stream.binance.com:9443/ws"
        
        logger.info(f"BinanceStreamClient initialized: symbols={symbols}, testnet={testnet}, futures={futures}")
    
    async def start(self):
        """
        Start WebSocket connection and subscribe to streams.
        
        Connects to Binance WebSocket and subscribes to depth5 streams
        for all specified symbols.
        
        **Validates: Requirement 18.3, 18.4**
        """
        self.running = True
        
        # Build stream names (e.g., btcusdt@depth5)
        streams = [f"{symbol}@depth5" for symbol in self.symbols]
        stream_url = f"{self.ws_url}/{'/'.join(streams)}"
        
        logger.info(f"Connecting to Binance WebSocket: {streams}")
        
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(stream_url) as ws:
                        self.ws = ws
                        logger.info(f"✓ Connected to Binance WebSocket")
                        
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                await self.process_message(data)
                            
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error(f"WebSocket error: {ws.exception()}")
                                break
                            
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                logger.warning("WebSocket closed by server")
                                break
                        
                        # Connection closed, attempt reconnect
                        if self.running:
                            logger.warning("WebSocket disconnected, reconnecting in 5s...")
                            await asyncio.sleep(5)
                        
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                if self.running:
                    logger.info("Reconnecting in 5s...")
                    await asyncio.sleep(5)
    
    async def stop(self):
        """
        Stop WebSocket connection.
        
        Gracefully closes the WebSocket connection and stops the client.
        """
        self.running = False
        
        if self.ws and not self.ws.closed:
            await self.ws.close()
        
        logger.info("BinanceStreamClient stopped")
    
    async def process_message(self, data: Dict[str, Any]):
        """
        Process incoming order book update.
        
        Parses the WebSocket message and calls the callback with
        the symbol and order book data.
        
        Args:
            data: WebSocket message data
            
        **Validates: Requirement 18.5**
        """
        try:
            # Check if this is a depth update
            if 'stream' in data:
                stream_name = data['stream']
                symbol = stream_name.split('@')[0].upper()
                order_book_data = data['data']
                
                # Extract bids and asks
                bids = [[float(price), float(qty)] for price, qty in order_book_data.get('bids', [])]
                asks = [[float(price), float(qty)] for price, qty in order_book_data.get('asks', [])]
                
                # Build order book update
                order_book = {
                    'bids': bids,
                    'asks': asks,
                    'lastUpdateId': order_book_data.get('lastUpdateId')
                }
                
                # Call callback (async)
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(symbol, order_book)
                else:
                    self.callback(symbol, order_book)
                
                logger.debug(f"Order book update: {symbol} (bids: {len(bids)}, asks: {len(asks)})")
            
            elif 'e' in data and data['e'] == 'depthUpdate':
                # Single symbol stream format
                symbol = data['s']
                
                bids = [[float(price), float(qty)] for price, qty in data.get('b', [])]
                asks = [[float(price), float(qty)] for price, qty in data.get('a', [])]
                
                order_book = {
                    'bids': bids,
                    'asks': asks,
                    'lastUpdateId': data.get('u')
                }
                
                # Call callback
                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback(symbol, order_book)
                else:
                    self.callback(symbol, order_book)
                
                logger.debug(f"Order book update: {symbol} (bids: {len(bids)}, asks: {len(asks)})")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def is_connected(self) -> bool:
        """
        Check if WebSocket is connected.
        
        Returns:
            True if connected, False otherwise
        """
        return self.ws is not None and not self.ws.closed
    
    async def subscribe(self, symbols: List[str]):
        """
        Subscribe to additional symbols.
        
        Note: Binance doesn't support dynamic subscription on existing connection.
        Must reconnect with new symbol list.
        
        Args:
            symbols: List of symbols to add
        """
        logger.warning("Dynamic subscription not supported - must reconnect with new symbol list")
        self.symbols.extend([s.lower() for s in symbols])
    
    async def unsubscribe(self, symbols: List[str]):
        """
        Unsubscribe from symbols.
        
        Note: Binance doesn't support dynamic unsubscription on existing connection.
        Must reconnect with updated symbol list.
        
        Args:
            symbols: List of symbols to remove
        """
        logger.warning("Dynamic unsubscription not supported - must reconnect with updated symbol list")
        for symbol in symbols:
            if symbol.lower() in self.symbols:
                self.symbols.remove(symbol.lower())


class BinanceStreamManager:
    """
    Manager for multiple WebSocket stream clients.
    
    Handles multiple stream clients for different symbol groups
    and provides unified interface.
    """
    
    def __init__(self):
        """Initialize stream manager."""
        self.clients: Dict[str, BinanceStreamClient] = {}
        self.running = False
        
        logger.info("BinanceStreamManager initialized")
    
    async def add_stream(
        self,
        stream_id: str,
        symbols: List[str],
        callback: Callable[[str, Dict[str, Any]], None],
        testnet: bool = False,
        futures: bool = False
    ):
        """
        Add new stream client.
        
        Args:
            stream_id: Unique identifier for this stream
            symbols: List of symbols to stream
            callback: Callback function for updates
            testnet: Use testnet environment
            futures: Use Futures streams
        """
        if stream_id in self.clients:
            logger.warning(f"Stream {stream_id} already exists, stopping old stream")
            await self.remove_stream(stream_id)
        
        client = BinanceStreamClient(symbols, callback, testnet, futures)
        self.clients[stream_id] = client
        
        # Start client if manager is running
        if self.running:
            asyncio.create_task(client.start())
        
        logger.info(f"Stream added: {stream_id} with {len(symbols)} symbols")
    
    async def remove_stream(self, stream_id: str):
        """
        Remove stream client.
        
        Args:
            stream_id: Stream identifier to remove
        """
        if stream_id in self.clients:
            await self.clients[stream_id].stop()
            del self.clients[stream_id]
            logger.info(f"Stream removed: {stream_id}")
    
    async def start_all(self):
        """Start all stream clients."""
        self.running = True
        
        tasks = []
        for stream_id, client in self.clients.items():
            tasks.append(asyncio.create_task(client.start()))
            logger.info(f"Starting stream: {stream_id}")
        
        # Wait for all streams
        await asyncio.gather(*tasks)
    
    async def stop_all(self):
        """Stop all stream clients."""
        self.running = False
        
        for stream_id, client in self.clients.items():
            await client.stop()
            logger.info(f"Stopped stream: {stream_id}")
    
    def get_stream(self, stream_id: str) -> Optional[BinanceStreamClient]:
        """
        Get stream client by ID.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Stream client or None if not found
        """
        return self.clients.get(stream_id)
    
    def list_streams(self) -> List[str]:
        """
        List all active stream IDs.
        
        Returns:
            List of stream identifiers
        """
        return list(self.clients.keys())

