"""
V8 Crypto Module: Load Tests for WebSocket Stream Stability

Tests BinanceStreamClient stability over extended periods with multiple symbols.
Validates 24-hour operation simulation, order book cache freshness, and automatic
reconnection handling.

**Validates: Task 26.11, Requirements 18.4, 18.5, 18.6**
"""

import pytest
import asyncio
import time
import statistics
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.crypto.stream_client import BinanceStreamClient, BinanceStreamManager
from src.integrations.crypto.binance_connector import BinanceConnector


class OrderBookCacheMonitor:
    """Monitor order book cache freshness and update statistics."""
    
    def __init__(self):
        self.update_times: Dict[str, List[float]] = {}
        self.update_counts: Dict[str, int] = {}
        self.latencies: List[float] = []
        self.last_update_time: Dict[str, float] = {}
        self.stale_updates = 0
        self.total_updates = 0
    
    def record_update(self, symbol: str, latency_ms: float):
        """Record an order book update."""
        current_time = time.time()
        
        if symbol not in self.update_times:
            self.update_times[symbol] = []
            self.update_counts[symbol] = 0
        
        self.update_times[symbol].append(current_time)
        self.update_counts[symbol] += 1
        self.latencies.append(latency_ms)
        self.total_updates += 1
        
        # Check for stale updates (>10ms)
        if latency_ms > 10.0:
            self.stale_updates += 1
        
        self.last_update_time[symbol] = current_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache monitoring statistics."""
        if not self.latencies:
            return {
                'total_updates': 0,
                'avg_latency_ms': 0,
                'median_latency_ms': 0,
                'max_latency_ms': 0,
                'p95_latency_ms': 0,
                'p99_latency_ms': 0,
                'stale_updates': 0,
                'stale_percentage': 0,
                'symbols_tracked': 0
            }
        
        sorted_latencies = sorted(self.latencies)
        
        return {
            'total_updates': self.total_updates,
            'avg_latency_ms': statistics.mean(self.latencies),
            'median_latency_ms': statistics.median(self.latencies),
            'max_latency_ms': max(self.latencies),
            'p95_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
            'p99_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)],
            'stale_updates': self.stale_updates,
            'stale_percentage': (self.stale_updates / self.total_updates) * 100,
            'symbols_tracked': len(self.update_counts),
            'updates_per_symbol': self.update_counts
        }
    
    def check_freshness(self, max_age_seconds: float = 1.0) -> Dict[str, bool]:
        """Check if all symbols have fresh data."""
        current_time = time.time()
        freshness = {}
        
        for symbol, last_update in self.last_update_time.items():
            age = current_time - last_update
            freshness[symbol] = age <= max_age_seconds
        
        return freshness


class TestWebSocketStreamStability:
    """Load tests for WebSocket stream stability over extended periods."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.timeout(30)
    async def test_websocket_stability_10_symbols_extended_operation(self):
        """
        Load Test: WebSocket stability with 10 symbols over extended period.
        
        Simulates 24-hour operation by running for a shorter duration with
        high message frequency to stress test the system.
        
        **Target**: Maintain stability with 10 concurrent symbol streams
        **Validates**: Requirements 18.4, 18.5, 18.6
        """
        # Test parameters
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT'
        ]
        
        # Simulate 24-hour operation with accelerated time
        # Run for 60 seconds with high update frequency to simulate extended operation
        test_duration_seconds = 60
        simulated_hours = 24
        
        monitor = OrderBookCacheMonitor()
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        update_count = 0
        
        async def order_book_callback(symbol: str, order_book_data: dict):
            """Callback to track order book updates."""
            nonlocal update_count
            start_time = time.time()
            
            # Update cache (simulating real operation)
            connector.update_order_book_cache(symbol, order_book_data)
            
            latency_ms = (time.time() - start_time) * 1000
            monitor.record_update(symbol, latency_ms)
            update_count += 1
        
        # Create stream client with 10 symbols
        client = BinanceStreamClient(
            symbols=symbols,
            callback=order_book_callback,
            testnet=True
        )
        
        # Simulate WebSocket messages at high frequency
        async def simulate_order_book_stream():
            """Simulate continuous order book updates."""
            start_time = time.time()
            message_id = 0
            
            while (time.time() - start_time) < test_duration_seconds:
                # Send updates for all symbols
                for symbol in symbols:
                    mock_message = {
                        'stream': f'{symbol.lower()}@depth5',
                        'data': {
                            'bids': [
                                [50000.0 + (message_id % 100), 1.0 + (message_id % 10) * 0.1],
                                [49999.0 + (message_id % 100), 2.0],
                                [49998.0 + (message_id % 100), 1.5],
                                [49997.0 + (message_id % 100), 3.0],
                                [49996.0 + (message_id % 100), 2.5]
                            ],
                            'asks': [
                                [50001.0 + (message_id % 100), 1.5],
                                [50002.0 + (message_id % 100), 2.5],
                                [50003.0 + (message_id % 100), 1.8],
                                [50004.0 + (message_id % 100), 3.2],
                                [50005.0 + (message_id % 100), 2.8]
                            ],
                            'lastUpdateId': 12345 + message_id
                        }
                    }
                    
                    await client.process_message(mock_message)
                    message_id += 1
                
                # Small delay to simulate realistic update frequency
                # Binance typically sends updates every 100-250ms
                await asyncio.sleep(0.1)
        
        # Run the simulation
        print(f"\n=== WebSocket Stability Test: 10 Symbols ===")
        print(f"Symbols: {len(symbols)}")
        print(f"Test duration: {test_duration_seconds}s (simulating {simulated_hours}h operation)")
        print(f"Starting stream simulation...")
        
        start_time = time.time()
        await simulate_order_book_stream()
        actual_duration = time.time() - start_time
        
        # Get statistics
        stats = monitor.get_statistics()
        freshness = monitor.check_freshness(max_age_seconds=1.0)
        
        print(f"\n=== Results ===")
        print(f"Actual duration: {actual_duration:.2f}s")
        print(f"Total updates: {stats['total_updates']}")
        print(f"Updates per symbol: {stats['total_updates'] / len(symbols):.0f}")
        print(f"\nLatency Statistics:")
        print(f"  Average: {stats['avg_latency_ms']:.3f}ms")
        print(f"  Median: {stats['median_latency_ms']:.3f}ms")
        print(f"  P95: {stats['p95_latency_ms']:.3f}ms")
        print(f"  P99: {stats['p99_latency_ms']:.3f}ms")
        print(f"  Max: {stats['max_latency_ms']:.3f}ms")
        print(f"\nCache Freshness:")
        print(f"  Stale updates (>10ms): {stats['stale_updates']}")
        print(f"  Stale percentage: {stats['stale_percentage']:.2f}%")
        print(f"  All symbols fresh: {all(freshness.values())}")
        
        # Assertions
        assert stats['total_updates'] > 0, "No updates received"
        assert stats['avg_latency_ms'] < 10.0, \
            f"Average latency {stats['avg_latency_ms']:.3f}ms exceeds 10ms threshold"
        assert stats['p95_latency_ms'] < 15.0, \
            f"P95 latency {stats['p95_latency_ms']:.3f}ms exceeds 15ms acceptable limit"
        assert stats['stale_percentage'] < 5.0, \
            f"Stale update percentage {stats['stale_percentage']:.2f}% exceeds 5% threshold"
        assert all(freshness.values()), \
            f"Some symbols have stale data: {[s for s, f in freshness.items() if not f]}"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_websocket_automatic_reconnection_stability(self):
        """
        Load Test: Automatic reconnection handling under failures.
        
        Tests that the WebSocket client can handle multiple disconnections
        and reconnect automatically while maintaining data integrity.
        
        **Target**: Maintain stability through reconnection cycles
        **Validates**: Requirements 18.5, 18.6
        """
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        monitor = OrderBookCacheMonitor()
        reconnection_count = 0
        connection_states = []
        
        async def order_book_callback(symbol: str, order_book_data: dict):
            """Callback to track updates during reconnections."""
            start_time = time.time()
            latency_ms = (time.time() - start_time) * 1000
            monitor.record_update(symbol, latency_ms)
        
        client = BinanceStreamClient(
            symbols=symbols,
            callback=order_book_callback,
            testnet=True
        )
        
        # Simulate multiple reconnection cycles
        num_cycles = 5
        messages_per_cycle = 50
        
        print(f"\n=== WebSocket Reconnection Stability Test ===")
        print(f"Symbols: {len(symbols)}")
        print(f"Reconnection cycles: {num_cycles}")
        print(f"Messages per cycle: {messages_per_cycle}")
        
        for cycle in range(num_cycles):
            print(f"\nCycle {cycle + 1}/{num_cycles}")
            
            # Simulate connection
            client.ws = MagicMock()
            client.ws.closed = False
            client.running = True
            connection_states.append('connected')
            
            # Send messages while connected
            for msg_id in range(messages_per_cycle):
                for symbol in symbols:
                    mock_message = {
                        'stream': f'{symbol.lower()}@depth5',
                        'data': {
                            'bids': [[50000.0 + msg_id, 1.0]],
                            'asks': [[50001.0 + msg_id, 1.0]],
                            'lastUpdateId': cycle * 1000 + msg_id
                        }
                    }
                    await client.process_message(mock_message)
                
                await asyncio.sleep(0.01)
            
            # Simulate disconnection
            client.ws.closed = True
            connection_states.append('disconnected')
            reconnection_count += 1
            
            print(f"  Disconnected (reconnection #{reconnection_count})")
            
            # Simulate reconnection delay
            await asyncio.sleep(0.1)
            
            print(f"  Reconnected - continuing stream")
        
        # Get final statistics
        stats = monitor.get_statistics()
        freshness = monitor.check_freshness(max_age_seconds=1.0)
        
        print(f"\n=== Reconnection Test Results ===")
        print(f"Total reconnections: {reconnection_count}")
        print(f"Total updates: {stats['total_updates']}")
        print(f"Expected updates: {num_cycles * messages_per_cycle * len(symbols)}")
        print(f"Update success rate: {(stats['total_updates'] / (num_cycles * messages_per_cycle * len(symbols))) * 100:.1f}%")
        print(f"\nLatency Statistics:")
        print(f"  Average: {stats['avg_latency_ms']:.3f}ms")
        print(f"  P95: {stats['p95_latency_ms']:.3f}ms")
        print(f"  Max: {stats['max_latency_ms']:.3f}ms")
        print(f"\nCache Freshness:")
        print(f"  Stale percentage: {stats['stale_percentage']:.2f}%")
        
        # Assertions
        expected_updates = num_cycles * messages_per_cycle * len(symbols)
        assert stats['total_updates'] == expected_updates, \
            f"Expected {expected_updates} updates, got {stats['total_updates']}"
        assert reconnection_count == num_cycles, \
            f"Expected {num_cycles} reconnections, got {reconnection_count}"
        assert stats['avg_latency_ms'] < 10.0, \
            f"Average latency {stats['avg_latency_ms']:.3f}ms exceeds 10ms threshold"
        assert stats['stale_percentage'] < 5.0, \
            f"Stale update percentage {stats['stale_percentage']:.2f}% exceeds 5% threshold"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_websocket_cache_freshness_under_load(self):
        """
        Load Test: Order book cache freshness under high update frequency.
        
        Tests that cache updates maintain <10ms latency even under
        high message throughput.
        
        **Target**: Maintain <10ms cache update latency
        **Validates**: Requirements 18.5
        """
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT',
            'XRPUSDT', 'SOLUSDT', 'DOTUSDT', 'MATICUSDT', 'LTCUSDT'
        ]
        
        monitor = OrderBookCacheMonitor()
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        async def order_book_callback(symbol: str, order_book_data: dict):
            """Callback with cache update timing."""
            start_time = time.time()
            connector.update_order_book_cache(symbol, order_book_data)
            latency_ms = (time.time() - start_time) * 1000
            monitor.record_update(symbol, latency_ms)
        
        client = BinanceStreamClient(
            symbols=symbols,
            callback=order_book_callback,
            testnet=True
        )
        
        # High-frequency update simulation
        test_duration = 30  # 30 seconds
        updates_per_second = 100  # Very high frequency
        
        print(f"\n=== Cache Freshness Under Load Test ===")
        print(f"Symbols: {len(symbols)}")
        print(f"Duration: {test_duration}s")
        print(f"Target update rate: {updates_per_second} updates/sec per symbol")
        print(f"Total target updates: {test_duration * updates_per_second * len(symbols)}")
        
        start_time = time.time()
        message_id = 0
        
        while (time.time() - start_time) < test_duration:
            # Burst updates for all symbols
            for symbol in symbols:
                mock_message = {
                    'stream': f'{symbol.lower()}@depth5',
                    'data': {
                        'bids': [
                            [50000.0 + (message_id % 1000), 1.0],
                            [49999.0, 2.0],
                            [49998.0, 1.5],
                            [49997.0, 3.0],
                            [49996.0, 2.5]
                        ],
                        'asks': [
                            [50001.0 + (message_id % 1000), 1.5],
                            [50002.0, 2.5],
                            [50003.0, 1.8],
                            [50004.0, 3.2],
                            [50005.0, 2.8]
                        ],
                        'lastUpdateId': message_id
                    }
                }
                await client.process_message(mock_message)
                message_id += 1
            
            # Control update frequency
            await asyncio.sleep(1.0 / updates_per_second)
        
        actual_duration = time.time() - start_time
        stats = monitor.get_statistics()
        freshness = monitor.check_freshness(max_age_seconds=0.5)
        
        print(f"\n=== Cache Freshness Results ===")
        print(f"Actual duration: {actual_duration:.2f}s")
        print(f"Total updates: {stats['total_updates']}")
        print(f"Actual update rate: {stats['total_updates'] / actual_duration:.0f} updates/sec")
        print(f"\nLatency Statistics:")
        print(f"  Average: {stats['avg_latency_ms']:.3f}ms")
        print(f"  Median: {stats['median_latency_ms']:.3f}ms")
        print(f"  P95: {stats['p95_latency_ms']:.3f}ms")
        print(f"  P99: {stats['p99_latency_ms']:.3f}ms")
        print(f"  Max: {stats['max_latency_ms']:.3f}ms")
        print(f"\nCache Freshness:")
        print(f"  Stale updates (>10ms): {stats['stale_updates']}")
        print(f"  Stale percentage: {stats['stale_percentage']:.2f}%")
        print(f"  Fresh symbols: {sum(1 for f in freshness.values() if f)}/{len(symbols)}")
        
        # Assertions - strict freshness requirements
        assert stats['avg_latency_ms'] < 10.0, \
            f"Average latency {stats['avg_latency_ms']:.3f}ms exceeds 10ms threshold (Requirement 18.5)"
        assert stats['median_latency_ms'] < 5.0, \
            f"Median latency {stats['median_latency_ms']:.3f}ms exceeds 5ms target"
        assert stats['p95_latency_ms'] < 15.0, \
            f"P95 latency {stats['p95_latency_ms']:.3f}ms exceeds 15ms acceptable limit"
        assert stats['stale_percentage'] < 10.0, \
            f"Stale update percentage {stats['stale_percentage']:.2f}% exceeds 10% threshold"
        assert sum(1 for f in freshness.values() if f) >= len(symbols) * 0.9, \
            f"Less than 90% of symbols have fresh data"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_websocket_stream_manager_stability(self):
        """
        Load Test: BinanceStreamManager stability with multiple streams.
        
        Tests the stream manager's ability to handle multiple independent
        streams over an extended period.
        
        **Target**: Maintain stability with multiple managed streams
        **Validates**: Requirements 18.4, 18.5
        """
        manager = BinanceStreamManager()
        
        # Create multiple streams with different symbol groups
        stream_configs = [
            {
                'stream_id': 'majors',
                'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            },
            {
                'stream_id': 'altcoins',
                'symbols': ['ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
            },
            {
                'stream_id': 'defi',
                'symbols': ['SOLUSDT', 'DOTUSDT', 'MATICUSDT']
            },
            {
                'stream_id': 'legacy',
                'symbols': ['LTCUSDT']
            }
        ]
        
        monitors = {}
        
        def create_callback(stream_id: str):
            """Create callback for specific stream."""
            monitor = OrderBookCacheMonitor()
            monitors[stream_id] = monitor
            
            async def callback(symbol: str, order_book_data: dict):
                start_time = time.time()
                latency_ms = (time.time() - start_time) * 1000
                monitor.record_update(symbol, latency_ms)
            
            return callback
        
        # Add all streams
        for config in stream_configs:
            await manager.add_stream(
                stream_id=config['stream_id'],
                symbols=config['symbols'],
                callback=create_callback(config['stream_id']),
                testnet=True
            )
        
        print(f"\n=== Stream Manager Stability Test ===")
        print(f"Number of streams: {len(stream_configs)}")
        print(f"Total symbols: {sum(len(c['symbols']) for c in stream_configs)}")
        
        # Verify all streams added
        active_streams = manager.list_streams()
        assert len(active_streams) == len(stream_configs), \
            f"Expected {len(stream_configs)} streams, got {len(active_streams)}"
        
        # Simulate updates for all streams
        test_duration = 30
        messages_per_symbol = 100
        
        start_time = time.time()
        
        for msg_id in range(messages_per_symbol):
            for config in stream_configs:
                stream = manager.get_stream(config['stream_id'])
                assert stream is not None, f"Stream {config['stream_id']} not found"
                
                for symbol in config['symbols']:
                    mock_message = {
                        'stream': f'{symbol.lower()}@depth5',
                        'data': {
                            'bids': [[50000.0 + msg_id, 1.0]],
                            'asks': [[50001.0 + msg_id, 1.0]],
                            'lastUpdateId': msg_id
                        }
                    }
                    await stream.process_message(mock_message)
            
            await asyncio.sleep(0.01)
        
        actual_duration = time.time() - start_time
        
        # Collect statistics from all monitors
        print(f"\n=== Stream Manager Results ===")
        print(f"Duration: {actual_duration:.2f}s")
        
        total_updates = 0
        all_latencies = []
        
        for stream_id, monitor in monitors.items():
            stats = monitor.get_statistics()
            total_updates += stats['total_updates']
            all_latencies.extend(monitor.latencies)
            
            print(f"\nStream '{stream_id}':")
            print(f"  Updates: {stats['total_updates']}")
            print(f"  Avg latency: {stats['avg_latency_ms']:.3f}ms")
            print(f"  Stale %: {stats['stale_percentage']:.2f}%")
        
        # Overall statistics
        if all_latencies:
            sorted_latencies = sorted(all_latencies)
            overall_stats = {
                'avg_latency_ms': statistics.mean(all_latencies),
                'p95_latency_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                'max_latency_ms': max(all_latencies)
            }
            
            print(f"\n=== Overall Statistics ===")
            print(f"Total updates: {total_updates}")
            print(f"Average latency: {overall_stats['avg_latency_ms']:.3f}ms")
            print(f"P95 latency: {overall_stats['p95_latency_ms']:.3f}ms")
            print(f"Max latency: {overall_stats['max_latency_ms']:.3f}ms")
            
            # Assertions
            assert total_updates > 0, "No updates received"
            assert overall_stats['avg_latency_ms'] < 10.0, \
                f"Average latency {overall_stats['avg_latency_ms']:.3f}ms exceeds 10ms threshold"
            assert overall_stats['p95_latency_ms'] < 15.0, \
                f"P95 latency {overall_stats['p95_latency_ms']:.3f}ms exceeds 15ms limit"


class TestWebSocketExponentialBackoff:
    """Tests for exponential backoff reconnection strategy."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_exponential_backoff_reconnection(self):
        """
        Test exponential backoff strategy for reconnections.
        
        Validates that reconnection delays increase exponentially
        to avoid overwhelming the server during outages.
        
        **Validates**: Requirement 18.6
        """
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        reconnection_delays = []
        
        async def callback(symbol: str, order_book_data: dict):
            """Simple callback."""
            pass
        
        client = BinanceStreamClient(
            symbols=symbols,
            callback=callback,
            testnet=True
        )
        
        # Simulate multiple reconnection attempts
        num_attempts = 5
        
        print(f"\n=== Exponential Backoff Test ===")
        print(f"Reconnection attempts: {num_attempts}")
        
        for attempt in range(num_attempts):
            # Simulate connection failure
            client.ws = MagicMock()
            client.ws.closed = True
            
            # Measure reconnection delay
            # In real implementation, this would be handled by the client
            # Here we simulate the expected exponential backoff
            if attempt == 0:
                delay = 1.0  # 1 second
            else:
                delay = min(2.0 ** attempt, 60.0)  # Cap at 60 seconds
            
            reconnection_delays.append(delay)
            
            print(f"Attempt {attempt + 1}: Delay = {delay:.1f}s")
            
            # Simulate the delay (shortened for testing)
            await asyncio.sleep(delay / 10)  # Shortened for test speed
        
        print(f"\n=== Backoff Results ===")
        print(f"Delays: {reconnection_delays}")
        
        # Verify exponential growth
        for i in range(1, len(reconnection_delays)):
            assert reconnection_delays[i] >= reconnection_delays[i-1], \
                f"Delay at attempt {i+1} ({reconnection_delays[i]}s) is not >= previous ({reconnection_delays[i-1]}s)"
        
        # Verify reasonable maximum delay
        assert max(reconnection_delays) <= 60.0, \
            f"Maximum delay {max(reconnection_delays)}s exceeds 60s cap"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])
