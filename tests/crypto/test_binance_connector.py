"""
V8 Crypto Module: Unit Tests for BinanceConnector

Tests REST API methods, HMAC authentication, and order placement.

**Validates: Task 24.20**
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.integrations.crypto.binance_connector import BinanceConnector
from src.integrations.crypto.broker_client import (
    ConnectionError,
    OrderError,
    SymbolNotFoundError
)


@pytest.fixture
def binance_connector():
    """Create BinanceConnector instance for testing."""
    return BinanceConnector(
        api_key="test_api_key",
        api_secret="test_api_secret",
        testnet=True,
        futures=False
    )


@pytest.fixture
def binance_futures_connector():
    """Create BinanceConnector instance for futures testing."""
    return BinanceConnector(
        api_key="test_api_key",
        api_secret="test_api_secret",
        testnet=True,
        futures=True
    )


class TestBinanceConnectorInitialization:
    """Test BinanceConnector initialization."""
    
    def test_spot_testnet_initialization(self, binance_connector):
        """Test spot testnet initialization."""
        assert binance_connector.api_key == "test_api_key"
        assert binance_connector.api_secret == "test_api_secret"
        assert binance_connector.testnet is True
        assert binance_connector.futures is False
        assert binance_connector.base_url == "https://testnet.binance.vision"
        assert binance_connector.ws_url == "wss://testnet.binance.vision/ws"
    
    def test_futures_testnet_initialization(self, binance_futures_connector):
        """Test futures testnet initialization."""
        assert binance_futures_connector.futures is True
        # Testnet uses same URL for both spot and futures
        assert binance_futures_connector.base_url == "https://testnet.binance.vision"
        assert binance_futures_connector.ws_url == "wss://testnet.binance.vision/ws"
    
    def test_production_initialization(self):
        """Test production initialization."""
        connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            testnet=False,
            futures=False
        )
        assert connector.base_url == "https://api.binance.com"
        assert connector.ws_url == "wss://stream.binance.com:9443/ws"


class TestSignatureGeneration:
    """Test HMAC SHA256 signature generation."""
    
    def test_generate_signature(self, binance_connector):
        """Test signature generation with known input."""
        query_string = "symbol=BTCUSDT&side=BUY&type=MARKET&quantity=0.001&timestamp=1234567890"
        signature = binance_connector._generate_signature(query_string)
        
        # Signature should be 64-character hex string
        assert isinstance(signature, str)
        assert len(signature) == 64
        assert all(c in '0123456789abcdef' for c in signature)
    
    def test_signature_consistency(self, binance_connector):
        """Test that same input produces same signature."""
        query_string = "test=value&timestamp=123"
        sig1 = binance_connector._generate_signature(query_string)
        sig2 = binance_connector._generate_signature(query_string)
        
        assert sig1 == sig2
    
    def test_signature_uniqueness(self, binance_connector):
        """Test that different inputs produce different signatures."""
        sig1 = binance_connector._generate_signature("test=value1")
        sig2 = binance_connector._generate_signature("test=value2")
        
        assert sig1 != sig2


class TestGetBalance:
    """Test get_balance method."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_balance_spot_success(self, binance_connector):
        """Test successful balance retrieval for spot account."""
        mock_response = {
            'balances': [
                {'asset': 'USDT', 'free': '1000.0', 'locked': '0.0'},
                {'asset': 'BTC', 'free': '0.5', 'locked': '0.0'}
            ]
        }
        
        with patch.object(binance_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            balance = await binance_connector.get_balance()
            
            assert balance == 1000.0
    
    @pytest.mark.asyncio
    async def test_get_balance_futures_success(self, binance_futures_connector):
        """Test successful balance retrieval for futures account."""
        mock_response = [
            {'asset': 'USDT', 'balance': '5000.0'},
            {'asset': 'BTC', 'balance': '0.0'}
        ]
        
        with patch.object(binance_futures_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            balance = await binance_futures_connector.get_balance()
            
            assert balance == 5000.0
    
    @pytest.mark.asyncio
    async def test_get_balance_connection_error(self, binance_connector):
        """Test balance retrieval with connection error."""
        with patch.object(binance_connector, '_signed_request', new=AsyncMock(side_effect=Exception("Connection failed"))):
            with pytest.raises(ConnectionError, match="Failed to get balance"):
                await binance_connector.get_balance()


class TestPlaceOrder:
    """Test place_order method."""
    
    @pytest.mark.asyncio
    async def test_place_market_order_success(self, binance_connector):
        """Test successful market order placement."""
        mock_response = {
            'orderId': 12345,
            'symbol': 'BTCUSDT',
            'status': 'FILLED',
            'executedQty': '0.001',
            'price': '50000.0'
        }
        
        with patch.object(binance_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            order = await binance_connector.place_order(
                symbol="BTCUSDT",
                volume=0.001,
                direction="buy",
                order_type="market"
            )
            
            assert order['order_id'] == '12345'
            assert order['symbol'] == 'BTCUSDT'
            assert order['direction'] == 'buy'
            assert order['status'] == 'FILLED'
    
    @pytest.mark.asyncio
    async def test_place_limit_order_success(self, binance_connector):
        """Test successful limit order placement."""
        mock_response = {
            'orderId': 67890,
            'symbol': 'ETHUSDT',
            'status': 'NEW',
            'executedQty': '0.0',
            'price': '3000.0'
        }
        
        with patch.object(binance_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            order = await binance_connector.place_order(
                symbol="ETHUSDT",
                volume=0.1,
                direction="sell",
                order_type="limit",
                price=3000.0
            )
            
            assert order['order_id'] == '67890'
            assert order['status'] == 'NEW'
    
    @pytest.mark.asyncio
    async def test_place_limit_order_without_price(self, binance_connector):
        """Test limit order placement without price raises error."""
        with pytest.raises(OrderError, match="Price required for limit orders"):
            await binance_connector.place_order(
                symbol="BTCUSDT",
                volume=0.001,
                direction="buy",
                order_type="limit"
            )
    
    @pytest.mark.asyncio
    async def test_place_order_with_shadow_stops(self, binance_connector):
        """Test order placement with shadow stops."""
        mock_response = {
            'orderId': 11111,
            'symbol': 'BTCUSDT',
            'status': 'FILLED',
            'executedQty': '0.001',
            'price': '50000.0'
        }
        
        with patch.object(binance_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            with patch.object(binance_connector, '_place_shadow_stops', new=AsyncMock()) as mock_shadow:
                order = await binance_connector.place_order(
                    symbol="BTCUSDT",
                    volume=0.001,
                    direction="buy",
                    order_type="market",
                    stop_loss=49000.0,
                    take_profit=51000.0
                )
                
                # Verify shadow stops were called
                mock_shadow.assert_called_once()
                assert order['order_id'] == '11111'


class TestOrderBook:
    """Test order book retrieval and caching."""
    
    @pytest.mark.asyncio
    async def test_get_order_book_cache_miss(self, binance_connector):
        """Test order book retrieval with cache miss."""
        mock_response = {
            'bids': [['50000.0', '1.0'], ['49999.0', '2.0']],
            'asks': [['50001.0', '1.5'], ['50002.0', '2.5']]
        }
        
        with patch.object(binance_connector, '_fetch_order_book_rest', new=AsyncMock(return_value={
            'symbol': 'BTCUSDT',
            'bids': [[50000.0, 1.0], [49999.0, 2.0]],
            'asks': [[50001.0, 1.5], [50002.0, 2.5]],
            'timestamp': 1234567890.0
        })):
            order_book = await binance_connector.get_order_book('BTCUSDT', depth=5)
            
            assert order_book['symbol'] == 'BTCUSDT'
            assert len(order_book['bids']) == 2
            assert len(order_book['asks']) == 2
    
    @pytest.mark.asyncio
    async def test_get_order_book_cache_hit(self, binance_connector):
        """Test order book retrieval with cache hit."""
        import time
        
        # Populate cache
        cached_order_book = {
            'symbol': 'BTCUSDT',
            'bids': [[50000.0, 1.0]],
            'asks': [[50001.0, 1.0]],
            'timestamp': time.time()
        }
        binance_connector.order_book_cache['BTCUSDT'] = cached_order_book
        binance_connector.cache_timestamps['BTCUSDT'] = time.time()
        
        # Should return cached data without REST call
        order_book = await binance_connector.get_order_book('BTCUSDT')
        
        assert order_book == cached_order_book
    
    def test_update_order_book_cache(self, binance_connector):
        """Test order book cache update from WebSocket."""
        order_book_data = {
            'bids': [[50000.0, 1.0]],
            'asks': [[50001.0, 1.0]]
        }
        
        binance_connector.update_order_book_cache('BTCUSDT', order_book_data)
        
        assert 'BTCUSDT' in binance_connector.order_book_cache
        assert binance_connector.order_book_cache['BTCUSDT']['symbol'] == 'BTCUSDT'
        assert 'BTCUSDT' in binance_connector.cache_timestamps


class TestGetPositions:
    """Test get_positions method."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_get_positions_spot_empty(self, binance_connector):
        """Test get_positions for spot account returns empty list."""
        positions = await binance_connector.get_positions()
        
        assert positions == []
    
    @pytest.mark.asyncio
    async def test_get_positions_futures_success(self, binance_futures_connector):
        """Test get_positions for futures account."""
        mock_response = [
            {
                'symbol': 'BTCUSDT',
                'positionAmt': '0.001',
                'entryPrice': '50000.0',
                'markPrice': '50500.0',
                'unRealizedProfit': '0.5'
            },
            {
                'symbol': 'ETHUSDT',
                'positionAmt': '-0.1',
                'entryPrice': '3000.0',
                'markPrice': '2950.0',
                'unRealizedProfit': '5.0'
            },
            {
                'symbol': 'BNBUSDT',
                'positionAmt': '0.0',  # No position
                'entryPrice': '0.0',
                'markPrice': '300.0',
                'unRealizedProfit': '0.0'
            }
        ]
        
        with patch.object(binance_futures_connector, '_signed_request', new=AsyncMock(return_value=mock_response)):
            positions = await binance_futures_connector.get_positions()
            
            # Should only return positions with non-zero amount
            assert len(positions) == 2
            assert positions[0]['symbol'] == 'BTCUSDT'
            assert positions[0]['direction'] == 'buy'
            assert positions[1]['symbol'] == 'ETHUSDT'
            assert positions[1]['direction'] == 'sell'


class TestValidateConnection:
    """Test validate_connection method."""
    
    def test_validate_connection_success(self, binance_connector):
        """Test successful connection validation."""
        with patch.object(binance_connector, 'get_balance', new=AsyncMock(return_value=1000.0)):
            result = binance_connector.validate_connection()
            
            assert result is True
    
    def test_validate_connection_failure(self, binance_connector):
        """Test failed connection validation."""
        with patch.object(binance_connector, 'get_balance', new=AsyncMock(side_effect=Exception("Connection failed"))):
            result = binance_connector.validate_connection()
            
            assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
