"""
Task 26.14: Order Book Cache Refresh Benchmark

Benchmarks the performance of WebSocket message processing and order book cache
updates to validate <10ms latency target for HFT crypto scalping.

Requirements: 18.5
"""

import pytest
import asyncio
import time
import statistics
import json
from typing import List, Dict, Any
from src.integrations.crypto.stream_client import BinanceStreamClient
from src.integrations.crypto.binance_connector import BinanceConnector


class TestOrderBookCacheBenchmark:
    """
    Benchmark tests for order book cache refresh performance.
    
    Target: <10ms for WebSocket message processing and cache update
    """
    
    @pytest.fixture
    def mock_order_book_data(self) -> Dict[str, Any]:
        """Generate realistic order book data for benchmarking."""
        return {
            'bids': [
                [50000.00, 1.5],
                [49999.50, 2.3],
                [49999.00, 0.8],
                [49998.50, 1.2],
                [49998.00, 3.1]
            ],
            'asks': [
                [50000.50, 1.2],
                [50001.00, 2.1],
                [50001.50, 0.9],
                [50002.00, 1.8],
                [50002.50, 2.5]
            ],
            'lastUpdateId': 123456789
        }
    
    def _calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for latency measurements."""
        return {
            'average': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'min': min(latencies),
            'max': max(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        }
    
    def _print_statistics(self, name: str, stats: Dict[str, float]):
        """Print formatted statistics."""
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        print(f"Average:  {stats['average']:.6f}ms")
        print(f"Median:   {stats['median']:.6f}ms")
        print(f"Std Dev:  {stats['stdev']:.6f}ms")
        print(f"Min:      {stats['min']:.6f}ms")
        print(f"Max:      {stats['max']:.6f}ms")
        print(f"P95:      {stats['p95']:.6f}ms")
        print(f"P99:      {stats['p99']:.6f}ms")
        print(f"{'='*70}\n")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_benchmark_websocket_message_processing(self, mock_order_book_data):
        """
        Benchmark: WebSocket message processing latency
        
        Measures the time to parse and process incoming WebSocket messages
        containing order book updates.
        
        Target: <10ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: WebSocket Message Processing")
        print("="*70)
        
        # Create stream client with callback
        latencies = []
        
        async def benchmark_callback(symbol: str, order_book: Dict[str, Any]):
            """Callback that measures processing time."""
            # Callback is called after message processing
            pass
        
        client = BinanceStreamClient(
            symbols=["BTCUSDT"],
            callback=benchmark_callback,
            testnet=True
        )
        
        # Test configuration
        num_iterations = 10000
        num_warmup = 100
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Message type: depth5 order book update")
        
        # Build WebSocket message format
        ws_message = {
            'stream': 'btcusdt@depth5',
            'data': mock_order_book_data
        }
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            await client.process_message(ws_message)
        
        # Benchmark phase
        print("Running benchmark...")
        
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            await client.process_message(ws_message)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("WebSocket Message Processing Performance", stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <10.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - stats['average']) / 10.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 10.0, \
            f"Average latency {stats['average']:.6f}ms exceeds 10ms target"
        assert stats['median'] < 10.0, \
            f"Median latency {stats['median']:.6f}ms exceeds 10ms target"
        assert stats['p95'] < 20.0, \
            f"P95 latency {stats['p95']:.6f}ms exceeds 20ms acceptable limit"
    
    @pytest.mark.asyncio
    async def test_benchmark_order_book_cache_update(self, mock_order_book_data):
        """
        Benchmark: Order book cache update time
        
        Measures the time to update the order book cache in BinanceConnector
        when receiving updates from WebSocket stream.
        
        Target: <10ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Order Book Cache Update")
        print("="*70)
        
        # Create connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Test configuration
        num_iterations = 10000
        num_warmup = 100
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        
        print(f"Test Configuration:")
        print(f"  - Iterations per symbol: {num_iterations:,}")
        print(f"  - Symbols: {len(symbols)}")
        print(f"  - Total updates: {num_iterations * len(symbols):,}")
        print(f"  - Warmup iterations: {num_warmup}")
        
        # Warmup phase
        print("\nWarming up...")
        for symbol in symbols:
            for _ in range(num_warmup):
                connector.update_order_book_cache(symbol, mock_order_book_data)
        
        # Benchmark phase
        print("Running benchmark...")
        all_latencies = []
        symbol_latencies = {symbol: [] for symbol in symbols}
        
        for symbol in symbols:
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                connector.update_order_book_cache(symbol, mock_order_book_data)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                all_latencies.append(latency_ms)
                symbol_latencies[symbol].append(latency_ms)
        
        # Calculate statistics
        overall_stats = self._calculate_statistics(all_latencies)
        
        # Print results
        self._print_statistics("Overall Cache Update Performance", overall_stats)
        
        # Per-symbol statistics
        print("Per-Symbol Performance:")
        print("-" * 70)
        for symbol in symbols:
            stats = self._calculate_statistics(symbol_latencies[symbol])
            print(f"{symbol}: avg={stats['average']:.6f}ms, median={stats['median']:.6f}ms, p95={stats['p95']:.6f}ms")
        print()
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <10.0ms average latency")
        print(f"  Actual: {overall_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if overall_stats['average'] < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - overall_stats['average']) / 10.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert overall_stats['average'] < 10.0, \
            f"Average cache update latency {overall_stats['average']:.6f}ms exceeds 10ms target"
        assert overall_stats['median'] < 10.0, \
            f"Median cache update latency {overall_stats['median']:.6f}ms exceeds 10ms target"
        assert overall_stats['p95'] < 20.0, \
            f"P95 cache update latency {overall_stats['p95']:.6f}ms exceeds 20ms acceptable limit"
    
    @pytest.mark.asyncio
    async def test_benchmark_end_to_end_cache_refresh(self, mock_order_book_data):
        """
        Benchmark: End-to-end cache refresh latency
        
        Measures the complete latency from WebSocket message arrival to
        cache update completion, simulating real-world HFT scenario.
        
        Target: <10ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: End-to-End Cache Refresh")
        print("="*70)
        
        # Create connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Create stream client with cache update callback
        latencies = []
        
        async def cache_update_callback(symbol: str, order_book: Dict[str, Any]):
            """Callback that updates cache."""
            connector.update_order_book_cache(symbol, order_book)
        
        client = BinanceStreamClient(
            symbols=["BTCUSDT", "ETHUSDT"],
            callback=cache_update_callback,
            testnet=True
        )
        
        # Test configuration
        num_iterations = 5000
        num_warmup = 50
        
        print(f"Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Symbols: BTCUSDT, ETHUSDT")
        print(f"  - Pipeline: WebSocket parse → Cache update")
        
        # Build WebSocket messages
        btc_message = {
            'stream': 'btcusdt@depth5',
            'data': mock_order_book_data
        }
        eth_message = {
            'stream': 'ethusdt@depth5',
            'data': mock_order_book_data
        }
        
        # Warmup phase
        print("\nWarming up...")
        for _ in range(num_warmup):
            await client.process_message(btc_message)
            await client.process_message(eth_message)
        
        # Benchmark phase
        print("Running benchmark...")
        
        for i in range(num_iterations):
            # Alternate between symbols
            message = btc_message if i % 2 == 0 else eth_message
            
            start_time = time.perf_counter()
            await client.process_message(message)
            latency_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        stats = self._calculate_statistics(latencies)
        
        # Print results
        self._print_statistics("End-to-End Cache Refresh Performance", stats)
        
        # Validate cache freshness
        btc_cache_age = time.time() - connector.cache_timestamps.get('BTCUSDT', 0)
        eth_cache_age = time.time() - connector.cache_timestamps.get('ETHUSDT', 0)
        
        print("Cache Freshness Validation:")
        print(f"  BTCUSDT cache age: {btc_cache_age*1000:.2f}ms")
        print(f"  ETHUSDT cache age: {eth_cache_age*1000:.2f}ms")
        print(f"  Both caches fresh: {'✅ YES' if btc_cache_age < 0.01 and eth_cache_age < 0.01 else '❌ NO'}\n")
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <10.0ms average latency")
        print(f"  Actual: {stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['average'] < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - stats['average']) / 10.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert stats['average'] < 10.0, \
            f"Average end-to-end latency {stats['average']:.6f}ms exceeds 10ms target"
        assert stats['median'] < 10.0, \
            f"Median end-to-end latency {stats['median']:.6f}ms exceeds 10ms target"
        assert stats['p95'] < 20.0, \
            f"P95 end-to-end latency {stats['p95']:.6f}ms exceeds 20ms acceptable limit"
        
        # Verify cache freshness guarantee
        assert btc_cache_age < 0.01, \
            f"BTCUSDT cache age {btc_cache_age*1000:.2f}ms exceeds 10ms freshness guarantee"
        assert eth_cache_age < 0.01, \
            f"ETHUSDT cache age {eth_cache_age*1000:.2f}ms exceeds 10ms freshness guarantee"
    
    @pytest.mark.asyncio
    async def test_benchmark_cache_under_high_frequency_updates(self, mock_order_book_data):
        """
        Benchmark: Cache performance under high-frequency updates
        
        Simulates realistic HFT scenario with rapid order book updates
        across multiple symbols to validate sustained <10ms performance.
        
        Target: Maintain <10ms latency with 100+ updates/second
        """
        print("\n" + "="*70)
        print("BENCHMARK: Cache Under High-Frequency Updates")
        print("="*70)
        
        # Create connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Test configuration
        num_symbols = 10
        updates_per_symbol = 1000
        total_updates = num_symbols * updates_per_symbol
        
        symbols = [f"SYMBOL{i}USDT" for i in range(num_symbols)]
        
        print(f"Test Configuration:")
        print(f"  - Symbols: {num_symbols}")
        print(f"  - Updates per symbol: {updates_per_symbol:,}")
        print(f"  - Total updates: {total_updates:,}")
        print(f"  - Target rate: >100 updates/second")
        
        # Warmup
        print("\nWarming up...")
        for symbol in symbols[:3]:
            for _ in range(10):
                connector.update_order_book_cache(symbol, mock_order_book_data)
        
        # Benchmark phase
        print("Running benchmark...")
        all_latencies = []
        
        start_total = time.perf_counter()
        
        for symbol in symbols:
            for _ in range(updates_per_symbol):
                start_time = time.perf_counter()
                connector.update_order_book_cache(symbol, mock_order_book_data)
                latency_ms = (time.perf_counter() - start_time) * 1000
                all_latencies.append(latency_ms)
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        # Calculate statistics
        stats = self._calculate_statistics(all_latencies)
        throughput = total_updates / (total_time / 1000)
        
        # Print results
        self._print_statistics("High-Frequency Cache Update Performance", stats)
        
        print(f"Throughput Metrics:")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Throughput: {throughput:.2f} updates/sec")
        print(f"  Average per update: {total_time / total_updates:.6f}ms\n")
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <10.0ms median latency under load")
        print(f"  Actual: {stats['median']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['median'] < 10.0 else '❌ FAILED'}")
        print(f"  Throughput: {throughput:.0f} updates/sec {'✅ >100' if throughput > 100 else '❌ <100'}\n")
        
        # Assertions
        assert stats['median'] < 10.0, \
            f"Median latency {stats['median']:.6f}ms exceeds 10ms target under high-frequency load"
        assert stats['p95'] < 20.0, \
            f"P95 latency {stats['p95']:.6f}ms exceeds 20ms acceptable limit under load"
        assert throughput > 100, \
            f"Throughput {throughput:.2f} updates/sec below 100 updates/sec target"
    
    @pytest.mark.asyncio
    async def test_benchmark_comprehensive_cache_refresh_summary(self, mock_order_book_data):
        """
        Benchmark: Comprehensive cache refresh performance summary
        
        This is the main benchmark test that validates all requirements
        for order book cache refresh performance.
        
        **Validates: Requirement 18.5**
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE ORDER BOOK CACHE REFRESH BENCHMARK")
        print("="*70)
        print("\nTask 26.14: Benchmark order book cache refresh <10ms")
        print("Requirement 18.5: Order book cache <10ms update latency")
        print("="*70 + "\n")
        
        # Create connector
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True
        )
        
        # Create stream client
        cache_update_latencies = []
        
        async def cache_callback(symbol: str, order_book: Dict[str, Any]):
            """Callback that updates cache and measures latency."""
            start_time = time.perf_counter()
            connector.update_order_book_cache(symbol, order_book)
            latency_ms = (time.perf_counter() - start_time) * 1000
            cache_update_latencies.append(latency_ms)
        
        client = BinanceStreamClient(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            callback=cache_callback,
            testnet=True
        )
        
        # Test configuration
        num_iterations = 10000
        num_warmup = 100
        
        print("Test Configuration:")
        print(f"  - Iterations: {num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        print(f"  - Symbols: BTCUSDT, ETHUSDT, BNBUSDT")
        print(f"  - Metrics: WebSocket processing + Cache update")
        
        # Build WebSocket messages
        messages = [
            {'stream': 'btcusdt@depth5', 'data': mock_order_book_data},
            {'stream': 'ethusdt@depth5', 'data': mock_order_book_data},
            {'stream': 'bnbusdt@depth5', 'data': mock_order_book_data}
        ]
        
        # Phase 1: Warmup
        print("\nPhase 1: Warmup")
        for _ in range(num_warmup):
            for message in messages:
                await client.process_message(message)
        cache_update_latencies.clear()  # Clear warmup measurements
        
        # Phase 2: WebSocket Message Processing
        print("Phase 2: Benchmarking WebSocket message processing...")
        ws_latencies = []
        
        for i in range(num_iterations):
            message = messages[i % len(messages)]
            
            start_time = time.perf_counter()
            await client.process_message(message)
            latency_ms = (time.perf_counter() - start_time) * 1000
            ws_latencies.append(latency_ms)
        
        ws_stats = self._calculate_statistics(ws_latencies)
        
        # Phase 3: Cache Update Performance (measured in callback)
        print("Phase 3: Analyzing cache update performance...")
        cache_stats = self._calculate_statistics(cache_update_latencies)
        
        # Phase 4: Cache Retrieval Performance
        print("Phase 4: Benchmarking cache retrieval...")
        retrieval_latencies = []
        
        for _ in range(num_iterations):
            symbol = ["BTCUSDT", "ETHUSDT", "BNBUSDT"][_ % 3]
            
            start_time = time.perf_counter()
            order_book = await connector.get_order_book(symbol)
            latency_ms = (time.perf_counter() - start_time) * 1000
            retrieval_latencies.append(latency_ms)
        
        retrieval_stats = self._calculate_statistics(retrieval_latencies)
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        print("\n1. WEBSOCKET MESSAGE PROCESSING")
        print("-" * 70)
        print(f"Average:  {ws_stats['average']:.6f}ms")
        print(f"Median:   {ws_stats['median']:.6f}ms")
        print(f"P95:      {ws_stats['p95']:.6f}ms")
        print(f"P99:      {ws_stats['p99']:.6f}ms")
        print(f"Max:      {ws_stats['max']:.6f}ms")
        
        print("\n2. CACHE UPDATE PERFORMANCE")
        print("-" * 70)
        print(f"Average:  {cache_stats['average']:.6f}ms")
        print(f"Median:   {cache_stats['median']:.6f}ms")
        print(f"P95:      {cache_stats['p95']:.6f}ms")
        print(f"P99:      {cache_stats['p99']:.6f}ms")
        print(f"Max:      {cache_stats['max']:.6f}ms")
        
        print("\n3. CACHE RETRIEVAL PERFORMANCE")
        print("-" * 70)
        print(f"Average:  {retrieval_stats['average']:.6f}ms")
        print(f"Median:   {retrieval_stats['median']:.6f}ms")
        print(f"P95:      {retrieval_stats['p95']:.6f}ms")
        print(f"P99:      {retrieval_stats['p99']:.6f}ms")
        print(f"Max:      {retrieval_stats['max']:.6f}ms")
        
        print("\n4. END-TO-END LATENCY BREAKDOWN")
        print("-" * 70)
        total_latency = ws_stats['average'] + cache_stats['average']
        print(f"WebSocket processing:  {ws_stats['average']:.6f}ms")
        print(f"Cache update:          {cache_stats['average']:.6f}ms")
        print(f"Total pipeline:        {total_latency:.6f}ms")
        print(f"Cache retrieval:       {retrieval_stats['average']:.6f}ms")
        
        print("\n5. CACHE FRESHNESS VALIDATION")
        print("-" * 70)
        btc_age = time.time() - connector.cache_timestamps.get('BTCUSDT', 0)
        eth_age = time.time() - connector.cache_timestamps.get('ETHUSDT', 0)
        bnb_age = time.time() - connector.cache_timestamps.get('BNBUSDT', 0)
        
        print(f"BTCUSDT cache age: {btc_age*1000:.2f}ms")
        print(f"ETHUSDT cache age: {eth_age*1000:.2f}ms")
        print(f"BNBUSDT cache age: {bnb_age*1000:.2f}ms")
        print(f"All caches fresh (<10ms): {'✅ YES' if max(btc_age, eth_age, bnb_age) < 0.01 else '❌ NO'}")
        
        print("\n6. TARGET VALIDATION")
        print("-" * 70)
        print(f"Requirement 18.5: Order book cache refresh <10ms")
        print(f"  WebSocket processing target: <10.0ms average")
        print(f"  WebSocket processing actual: {ws_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if ws_stats['average'] < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - ws_stats['average']) / 10.0 * 100):.2f}% under target")
        
        print(f"\n  Cache update target: <10.0ms average")
        print(f"  Cache update actual: {cache_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if cache_stats['average'] < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - cache_stats['average']) / 10.0 * 100):.2f}% under target")
        
        print(f"\n  End-to-end target: <10.0ms average")
        print(f"  End-to-end actual: {total_latency:.6f}ms")
        print(f"  Status: {'✅ PASSED' if total_latency < 10.0 else '❌ FAILED'}")
        print(f"  Margin: {((10.0 - total_latency) / 10.0 * 100):.2f}% under target")
        
        print("\n7. PERFORMANCE CHARACTERISTICS")
        print("-" * 70)
        print(f"Iterations tested: {num_iterations:,}")
        print(f"Symbols tested: 3 (BTCUSDT, ETHUSDT, BNBUSDT)")
        print(f"Message format: Binance depth5 WebSocket stream")
        print(f"Consistency (WS): {ws_stats['stdev']:.6f}ms std dev")
        print(f"Consistency (cache): {cache_stats['stdev']:.6f}ms std dev")
        print(f"Consistency (retrieval): {retrieval_stats['stdev']:.6f}ms std dev")
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70 + "\n")
        
        # Final assertions
        assert ws_stats['average'] < 10.0, \
            f"WebSocket processing average latency {ws_stats['average']:.6f}ms exceeds 10ms target (Requirement 18.5)"
        assert ws_stats['median'] < 10.0, \
            f"WebSocket processing median latency {ws_stats['median']:.6f}ms exceeds 10ms target"
        
        assert cache_stats['average'] < 10.0, \
            f"Cache update average latency {cache_stats['average']:.6f}ms exceeds 10ms target (Requirement 18.5)"
        assert cache_stats['median'] < 10.0, \
            f"Cache update median latency {cache_stats['median']:.6f}ms exceeds 10ms target"
        
        assert total_latency < 10.0, \
            f"End-to-end latency {total_latency:.6f}ms exceeds 10ms target"
        
        # P95 should be reasonable (allow some outliers)
        assert ws_stats['p95'] < 20.0, \
            f"WebSocket processing P95 latency {ws_stats['p95']:.6f}ms exceeds 20ms acceptable limit"
        assert cache_stats['p95'] < 20.0, \
            f"Cache update P95 latency {cache_stats['p95']:.6f}ms exceeds 20ms acceptable limit"
        
        # Verify cache freshness guarantee
        max_cache_age = max(btc_age, eth_age, bnb_age)
        assert max_cache_age < 0.01, \
            f"Cache age {max_cache_age*1000:.2f}ms exceeds 10ms freshness guarantee"
        
        # Success message
        print(f"✅ BENCHMARK PASSED: Order book cache refresh <10ms target achieved")
        print(f"✅ WebSocket processing: {ws_stats['average']:.6f}ms")
        print(f"✅ Cache update: {cache_stats['average']:.6f}ms")
        print(f"✅ End-to-end: {total_latency:.6f}ms")
        print(f"✅ Task 26.14 requirements validated\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
