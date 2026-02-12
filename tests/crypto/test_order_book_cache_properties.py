"""
V8 Crypto Module: Property-Based Tests for Order Book Cache Freshness

Tests that order book cache maintains <10ms freshness guarantee.

**Feature: quantmindx-unified-backend, Property 34: Order Book Cache Freshness**
**Validates: Task 24.21, Requirement 18.5**
"""

import pytest
import time
import asyncio
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from src.integrations.crypto.binance_connector import BinanceConnector


class TestOrderBookCacheFreshness:
    """
    Property 34: Order Book Cache Freshness
    
    The order book cache must maintain <10ms freshness guarantee.
    When cache is accessed, if data is older than 10ms, it should
    fetch fresh data from REST API.
    """
    
    @given(
        symbol=st.sampled_from(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT']),
        cache_age_ms=st.floats(min_value=0.0, max_value=50.0)
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_cache_freshness_property(self, symbol, cache_age_ms):
        """
        Property: Cache data older than 10ms should trigger REST fetch.
        
        Given: A symbol and cache age
        When: Cache is accessed
        Then: If cache_age > 10ms, REST API should be called
              If cache_age <= 10ms, cached data should be returned
        """
        # Create fresh connector for each test
        binance_connector = BinanceConnector(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True,
            futures=False
        )
        
        # Setup: Populate cache with known timestamp
        current_time = time.time()
        cache_timestamp = current_time - (cache_age_ms / 1000.0)  # Convert ms to seconds
        
        cached_order_book = {
            'symbol': symbol,
            'bids': [[50000.0, 1.0]],
            'asks': [[50001.0, 1.0]],
            'timestamp': cache_timestamp
        }
        
        binance_connector.order_book_cache[symbol] = cached_order_book
        binance_connector.cache_timestamps[symbol] = cache_timestamp
        
        # Execute: Get order book
        @pytest.mark.timeout(30)
        async def test_cache():
            # Mock REST fetch to track if it was called
            rest_called = False
            original_fetch = binance_connector._fetch_order_book_rest
            
            async def mock_fetch(sym, depth=5):
                nonlocal rest_called
                rest_called = True
                return {
                    'symbol': sym,
                    'bids': [[51000.0, 2.0]],
                    'asks': [[51001.0, 2.0]],
                    'timestamp': time.time()
                }
            
            binance_connector._fetch_order_book_rest = mock_fetch
            
            order_book = await binance_connector.get_order_book(symbol)
            
            # Restore original method
            binance_connector._fetch_order_book_rest = original_fetch
            
            return order_book, rest_called
        
        # Run async test
        loop = asyncio.get_event_loop()
        order_book, rest_called = loop.run_until_complete(test_cache())
        
        # Verify: Cache behavior based on age
        if cache_age_ms > 10.0:
            # Cache is stale, REST should be called
            assert rest_called, f"REST API should be called for cache age {cache_age_ms}ms > 10ms"
            assert order_book['bids'][0][0] == 51000.0, "Should return fresh data from REST"
        else:
            # Cache is fresh, should use cached data
            assert not rest_called, f"REST API should NOT be called for cache age {cache_age_ms}ms <= 10ms"
            assert order_book['bids'][0][0] == 50000.0, "Should return cached data"
    
    @given(
        symbols=st.lists(
            st.sampled_from(['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']),
            min_size=1,
            max_size=5,
            unique=True
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_cache_isolation_property(self, symbols):
        """
        Property: Cache entries for different symbols are isolated.
        
        Given: Multiple symbols with different cache ages
        When: One symbol's cache is updated
        Then: Other symbols' cache should remain unchanged
        """
        # Create fresh connector
        binance_connector = BinanceConnector(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True,
            futures=False
        )
        
        # Setup: Populate cache for all symbols
        current_time = time.time()
        
        for i, symbol in enumerate(symbols):
            cache_timestamp = current_time - (i * 0.001)  # Stagger timestamps
            binance_connector.order_book_cache[symbol] = {
                'symbol': symbol,
                'bids': [[50000.0 + i * 100, 1.0]],
                'asks': [[50001.0 + i * 100, 1.0]],
                'timestamp': cache_timestamp
            }
            binance_connector.cache_timestamps[symbol] = cache_timestamp
        
        # Execute: Update one symbol's cache
        updated_symbol = symbols[0]
        new_order_book = {
            'bids': [[99999.0, 5.0]],
            'asks': [[100000.0, 5.0]]
        }
        
        binance_connector.update_order_book_cache(updated_symbol, new_order_book)
        
        # Verify: Only updated symbol changed
        assert binance_connector.order_book_cache[updated_symbol]['bids'][0][0] == 99999.0
        
        # Other symbols should remain unchanged
        for i, symbol in enumerate(symbols[1:], start=1):
            expected_bid_price = 50000.0 + i * 100
            actual_bid_price = binance_connector.order_book_cache[symbol]['bids'][0][0]
            assert actual_bid_price == expected_bid_price, \
                f"Symbol {symbol} cache should not be affected by update to {updated_symbol}"
    
    @given(
        update_count=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=50, deadline=None)
    def test_cache_update_monotonicity_property(self, update_count):
        """
        Property: Cache timestamps should be monotonically increasing.
        
        Given: Multiple cache updates for the same symbol
        When: Updates are applied sequentially
        Then: Each update should have a timestamp >= previous timestamp
        """
        # Create fresh connector
        binance_connector = BinanceConnector(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True,
            futures=False
        )
        
        symbol = 'BTCUSDT'
        previous_timestamp = 0.0
        
        for i in range(update_count):
            order_book_data = {
                'bids': [[50000.0 + i, 1.0]],
                'asks': [[50001.0 + i, 1.0]]
            }
            
            binance_connector.update_order_book_cache(symbol, order_book_data)
            
            current_timestamp = binance_connector.cache_timestamps[symbol]
            
            # Verify monotonicity
            assert current_timestamp >= previous_timestamp, \
                f"Timestamp should be monotonically increasing: {current_timestamp} >= {previous_timestamp}"
            
            previous_timestamp = current_timestamp
            
            # Small delay to ensure time progression
            time.sleep(0.0001)  # 0.1ms delay
    
    @given(
        depth=st.integers(min_value=1, max_value=20)
    )
    @settings(max_examples=50, deadline=None)
    def test_cache_depth_independence_property(self, depth):
        """
        Property: Cache freshness is independent of order book depth.
        
        Given: Different order book depths
        When: Cache is accessed
        Then: Freshness check should work regardless of depth
        """
        # Create fresh connector
        binance_connector = BinanceConnector(
            api_key="test_api_key",
            api_secret="test_api_secret",
            testnet=True,
            futures=False
        )
        
        symbol = 'BTCUSDT'
        
        # Setup: Fresh cache
        current_time = time.time()
        binance_connector.order_book_cache[symbol] = {
            'symbol': symbol,
            'bids': [[50000.0 - i, 1.0] for i in range(depth)],
            'asks': [[50001.0 + i, 1.0] for i in range(depth)],
            'timestamp': current_time
        }
        binance_connector.cache_timestamps[symbol] = current_time
        
        # Execute: Get order book with different depth
        async def test_depth():
            order_book = await binance_connector.get_order_book(symbol, depth=depth)
            return order_book
        
        loop = asyncio.get_event_loop()
        order_book = loop.run_until_complete(test_depth())
        
        # Verify: Cache hit (fresh data)
        assert order_book['symbol'] == symbol
        assert len(order_book['bids']) == depth
        assert len(order_book['asks']) == depth


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
