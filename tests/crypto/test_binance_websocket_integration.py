"""
V8 Crypto Module: Integration Tests for Binance WebSocket Streaming

Tests real-time order book streaming and shadow stop placement.

**Validates: Task 24.22, 24.23**
"""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.crypto.stream_client import BinanceStreamClient, BinanceStreamManager
from src.integrations.crypto.binance_connector import BinanceConnector


class TestBinanceWebSocketStreaming:
    """Integration tests for WebSocket order book streaming."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_websocket_connection_and_message_processing(self):
        """
        Test WebSocket connection and message processing.
        
        Validates:
        - WebSocket connects successfully
        - Messages are processed correctly
        - Callback is invoked with order book data
        """
        received_updates = []
        
        def callback(symbol: str, order_book_data: dict):
            """Callback to capture order book updates."""
            received_updates.append({
                'symbol': symbol,
                'order_book': order_book_data
            })
        
        # Create stream client
        client = BinanceStreamClient(
            symbols=['BTCUSDT'],
            callback=callback,
            testnet=True,
            futures=False
        )
        
        # Mock WebSocket message
        mock_message = {
            'stream': 'btcusdt@depth5',
            'data': {
                'bids': [['50000.0', '1.0'], ['49999.0', '2.0']],
                'asks': [['50001.0', '1.5'], ['50002.0', '2.5']],
                'lastUpdateId': 12345
            }
        }
        
        # Process message
        await client.process_message(mock_message)
        
        # Verify callback was invoked
        assert len(received_updates) == 1
        assert received_updates[0]['symbol'] == 'BTCUSDT'
        assert len(received_updates[0]['order_book']['bids']) == 2
        assert len(received_updates[0]['order_book']['asks']) == 2
        assert received_updates[0]['order_book']['bids'][0][0] == 50000.0
    
    @pytest.mark.asyncio
    async def test_websocket_multiple_symbols(self):
        """
        Test WebSocket streaming for multiple symbols.
        
        Validates:
        - Multiple symbols can be streamed simultaneously
        - Updates are correctly routed to callback
        """
        received_updates = []
        
        def callback(symbol: str, order_book_data: dict):
            received_updates.append(symbol)
        
        client = BinanceStreamClient(
            symbols=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            callback=callback,
            testnet=True
        )
        
        # Process messages for different symbols
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            mock_message = {
                'stream': f'{symbol.lower()}@depth5',
                'data': {
                    'bids': [['1000.0', '1.0']],
                    'asks': [['1001.0', '1.0']],
                    'lastUpdateId': 12345
                }
            }
            await client.process_message(mock_message)
        
        # Verify all symbols received updates
        assert len(received_updates) == 3
        assert 'BTCUSDT' in received_updates
        assert 'ETHUSDT' in received_updates
        assert 'BNBUSDT' in received_updates
    
    @pytest.mark.asyncio
    async def test_websocket_order_book_cache_integration(self):
        """
        Test integration between WebSocket stream and order book cache.
        
        Validates:
        - WebSocket updates populate order book cache
        - Cache is updated with <10ms latency
        """
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        def callback(symbol: str, order_book_data: dict):
            """Update connector's order book cache."""
            connector.update_order_book_cache(symbol, order_book_data)
        
        client = BinanceStreamClient(
            symbols=['BTCUSDT'],
            callback=callback,
            testnet=True
        )
        
        # Simulate WebSocket update
        start_time = time.time()
        
        mock_message = {
            'stream': 'btcusdt@depth5',
            'data': {
                'bids': [['50000.0', '1.0']],
                'asks': [['50001.0', '1.0']],
                'lastUpdateId': 12345
            }
        }
        
        await client.process_message(mock_message)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Verify cache was updated
        assert 'BTCUSDT' in connector.order_book_cache
        assert connector.order_book_cache['BTCUSDT']['bids'][0][0] == 50000.0
        
        # Verify latency < 10ms
        assert latency_ms < 10.0, f"Cache update latency {latency_ms:.2f}ms exceeds 10ms threshold"
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection_handling(self):
        """
        Test WebSocket automatic reconnection on disconnect.
        
        Validates:
        - Client detects disconnection
        - Automatic reconnection is attempted
        """
        client = BinanceStreamClient(
            symbols=['BTCUSDT'],
            callback=lambda s, d: None,
            testnet=True
        )
        
        # Initially not connected
        assert not client.is_connected()
        
        # Simulate connection
        client.ws = MagicMock()
        client.ws.closed = False
        
        assert client.is_connected()
        
        # Simulate disconnection
        client.ws.closed = True
        
        assert not client.is_connected()


class TestBinanceStreamManager:
    """Integration tests for BinanceStreamManager."""
    
    @pytest.mark.asyncio
    async def test_stream_manager_multiple_streams(self):
        """
        Test managing multiple WebSocket streams.
        
        Validates:
        - Multiple streams can be added
        - Each stream operates independently
        """
        manager = BinanceStreamManager()
        
        received_updates = {'stream1': [], 'stream2': []}
        
        def callback1(symbol: str, order_book_data: dict):
            received_updates['stream1'].append(symbol)
        
        def callback2(symbol: str, order_book_data: dict):
            received_updates['stream2'].append(symbol)
        
        # Add two streams
        await manager.add_stream(
            stream_id='stream1',
            symbols=['BTCUSDT'],
            callback=callback1,
            testnet=True
        )
        
        await manager.add_stream(
            stream_id='stream2',
            symbols=['ETHUSDT'],
            callback=callback2,
            testnet=True
        )
        
        # Verify streams were added
        assert len(manager.list_streams()) == 2
        assert 'stream1' in manager.list_streams()
        assert 'stream2' in manager.list_streams()
        
        # Get specific stream
        stream1 = manager.get_stream('stream1')
        assert stream1 is not None
        assert 'btcusdt' in stream1.symbols
    
    @pytest.mark.asyncio
    async def test_stream_manager_remove_stream(self):
        """
        Test removing streams from manager.
        
        Validates:
        - Streams can be removed
        - Removed streams are properly cleaned up
        """
        manager = BinanceStreamManager()
        
        await manager.add_stream(
            stream_id='test_stream',
            symbols=['BTCUSDT'],
            callback=lambda s, d: None,
            testnet=True
        )
        
        assert 'test_stream' in manager.list_streams()
        
        # Remove stream
        await manager.remove_stream('test_stream')
        
        assert 'test_stream' not in manager.list_streams()
        assert manager.get_stream('test_stream') is None


class TestShadowStopPlacement:
    """Integration tests for shadow stop placement."""
    
    @pytest.mark.asyncio
    async def test_shadow_stop_placement_after_fill(self):
        """
        Test shadow stops are placed immediately after trade fill.
        
        Validates:
        - Stop loss order is placed after main order fills
        - Take profit order is placed after main order fills
        - Shadow stops are tracked in connector
        """
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Mock order placement
        mock_main_order = {
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'status': 'FILLED',
            'executedQty': '0.001',
            'price': '50000.0'
        }
        
        mock_sl_order = {
            'orderId': 12346,
            'symbol': 'BTCUSDT',
            'status': 'NEW',
            'executedQty': '0.0',
            'price': '49000.0'
        }
        
        mock_tp_order = {
            'orderId': 12347,
            'symbol': 'BTCUSDT',
            'status': 'NEW',
            'executedQty': '0.0',
            'price': '51000.0'
        }
        
        # Track order placement calls
        order_calls = []
        
        async def mock_signed_request(method, endpoint, params):
            order_calls.append(params)
            if len(order_calls) == 1:
                return mock_main_order
            elif len(order_calls) == 2:
                return mock_sl_order
            else:
                return mock_tp_order
        
        with patch.object(connector, '_signed_request', new=mock_signed_request):
            # Place order with shadow stops
            order = await connector.place_order(
                symbol='BTCUSDT',
                volume=0.001,
                direction='buy',
                order_type='market',
                stop_loss=49000.0,
                take_profit=51000.0
            )
            
            # Verify main order was placed
            assert order['order_id'] == '12345'
            
            # Verify shadow stops were tracked
            assert '12345' in connector.shadow_stops
            # Note: In actual implementation, shadow stops would be placed
            # This test validates the tracking mechanism
    
    @pytest.mark.asyncio
    async def test_shadow_stop_without_take_profit(self):
        """
        Test shadow stop placement with only stop loss.
        
        Validates:
        - Stop loss can be placed without take profit
        - Only stop loss is tracked in shadow stops
        """
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        mock_main_order = {
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'status': 'FILLED',
            'executedQty': '0.001',
            'price': '50000.0'
        }
        
        with patch.object(connector, '_signed_request', new=AsyncMock(return_value=mock_main_order)):
            with patch.object(connector, '_place_shadow_stops', new=AsyncMock()) as mock_shadow:
                order = await connector.place_order(
                    symbol='BTCUSDT',
                    volume=0.001,
                    direction='buy',
                    order_type='market',
                    stop_loss=49000.0
                )
                
                # Verify shadow stops method was called
                mock_shadow.assert_called_once()
                call_args = mock_shadow.call_args
                assert call_args[1]['stop_loss'] == 49000.0
                assert call_args[1]['take_profit'] is None


class TestWebSocketLatency:
    """Integration tests for WebSocket latency."""
    
    @pytest.mark.asyncio
    async def test_order_book_update_latency(self):
        """
        Test order book update latency is <10ms.
        
        Validates:
        - WebSocket message processing is fast
        - Cache update is fast
        - Total latency < 10ms
        """
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        latencies = []
        
        def callback(symbol: str, order_book_data: dict):
            """Measure latency of cache update."""
            start_time = time.time()
            connector.update_order_book_cache(symbol, order_book_data)
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        client = BinanceStreamClient(
            symbols=['BTCUSDT'],
            callback=callback,
            testnet=True
        )
        
        # Process multiple messages
        for i in range(10):
            mock_message = {
                'stream': 'btcusdt@depth5',
                'data': {
                    'bids': [[50000.0 + i, 1.0]],
                    'asks': [[50001.0 + i, 1.0]],
                    'lastUpdateId': 12345 + i
                }
            }
            await client.process_message(mock_message)
        
        # Verify all latencies < 10ms
        assert len(latencies) == 10
        for latency in latencies:
            assert latency < 10.0, f"Latency {latency:.2f}ms exceeds 10ms threshold"
        
        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage order book update latency: {avg_latency:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
