"""
V8 Mock MT5 Adapter

Mock implementation of MT5 adapter for development and testing.
Simulates MT5 API without requiring actual MT5 installation.

**Validates: Tasks 25.8-25.12 (Mock version)**
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional

from src.integrations.crypto.broker_client import BrokerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMT5Adapter(BrokerClient):
    """
    Mock MT5 adapter for development and testing.
    
    Simulates MT5 API behavior without requiring actual MT5 installation.
    Useful for:
    - Local development on Linux/Mac
    - Unit testing
    - Integration testing without MT5
    
    **Validates: Tasks 25.8-25.12 (Mock version)**
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize mock MT5 adapter.
        
        Args:
            config: Broker configuration dictionary
        """
        self.account_id = config.get('account_id', 'MOCK_ACCOUNT')
        self.server = config.get('server', 'MockServer')
        self.login = config.get('login', '12345678')
        
        # Mock state
        self.balance = config.get('initial_balance', 10000.0)
        self.equity = self.balance
        self.positions: List[Dict[str, Any]] = []
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 1000
        
        # Mock market data
        self.mock_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 149.50,
            'XAUUSD': 2050.00,
            'BTCUSD': 45000.00,
        }
        
        logger.info(f"MockMT5Adapter initialized: account={self.account_id}, balance=${self.balance}")
    
    async def get_balance(self) -> float:
        """
        Get mock account balance.
        
        Returns:
            Mock balance
            
        **Validates: Task 25.10**
        """
        logger.debug(f"MockMT5: get_balance() -> ${self.balance}")
        return self.balance
    
    async def place_order(
        self,
        symbol: str,
        volume: float,
        direction: str,
        order_type: str = "market",
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place mock order.
        
        Args:
            symbol: Trading symbol
            volume: Lot size
            direction: "buy" or "sell"
            order_type: "market" or "limit"
            price: Limit price (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            Mock order details
            
        **Validates: Task 25.11**
        """
        # Generate order ID
        order_id = str(self.order_counter)
        self.order_counter += 1
        
        # Get mock price
        current_price = self.mock_prices.get(symbol, 1.0)
        fill_price = price if order_type == "limit" and price else current_price
        
        # Create order
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'volume': volume,
            'direction': direction,
            'order_type': order_type,
            'status': 'filled' if order_type == 'market' else 'pending',
            'filled_price': fill_price if order_type == 'market' else None,
            'requested_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timestamp': time.time()
        }
        
        self.orders[order_id] = order
        
        # If market order, create position
        if order_type == 'market':
            position = {
                'symbol': symbol,
                'volume': volume,
                'direction': direction,
                'entry_price': fill_price,
                'current_price': fill_price,
                'profit': 0.0,
                'timestamp': time.time(),
                'order_id': order_id
            }
            self.positions.append(position)
        
        logger.info(
            f"MockMT5: place_order({symbol}, {volume}, {direction}) -> "
            f"Order {order_id} {order['status']}"
        )
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel mock order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if cancelled successfully
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            if order['status'] == 'pending':
                order['status'] = 'cancelled'
                logger.info(f"MockMT5: cancel_order({order_id}) -> Success")
                return True
            else:
                logger.warning(f"MockMT5: cancel_order({order_id}) -> Order not pending")
                return False
        else:
            logger.error(f"MockMT5: cancel_order({order_id}) -> Order not found")
            return False
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """
        Get mock order book.
        
        Args:
            symbol: Trading symbol
            depth: Number of price levels
            
        Returns:
            Mock order book
        """
        current_price = self.mock_prices.get(symbol, 1.0)
        spread = current_price * 0.0001  # 1 pip spread
        
        # Generate mock order book
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = current_price - spread - (i * spread)
            ask_price = current_price + spread + (i * spread)
            
            bids.append([bid_price, 1.0 + i * 0.5])
            asks.append([ask_price, 1.0 + i * 0.5])
        
        order_book = {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'timestamp': time.time()
        }
        
        logger.debug(f"MockMT5: get_order_book({symbol}) -> {len(bids)} levels")
        return order_book
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get mock open positions.
        
        Returns:
            List of mock positions
            
        **Validates: Task 25.12**
        """
        # Update position prices and profits
        for position in self.positions:
            symbol = position['symbol']
            current_price = self.mock_prices.get(symbol, position['entry_price'])
            position['current_price'] = current_price
            
            # Calculate profit
            if position['direction'] == 'buy':
                profit = (current_price - position['entry_price']) * position['volume'] * 100000
            else:
                profit = (position['entry_price'] - current_price) * position['volume'] * 100000
            
            position['profit'] = profit
        
        logger.debug(f"MockMT5: get_positions() -> {len(self.positions)} positions")
        return self.positions.copy()
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get mock order status.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Mock order status
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            return {
                'order_id': order_id,
                'status': order['status'],
                'filled_volume': order['volume'] if order['status'] == 'filled' else 0.0,
                'remaining_volume': 0.0 if order['status'] == 'filled' else order['volume'],
                'average_price': order.get('filled_price', 0.0)
            }
        else:
            return {
                'order_id': order_id,
                'status': 'not_found',
                'filled_volume': 0.0,
                'remaining_volume': 0.0,
                'average_price': 0.0
            }
    
    async def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_volume: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify mock order.
        
        Args:
            order_id: Order identifier
            new_price: New limit price
            new_volume: New volume
            new_stop_loss: New stop loss
            new_take_profit: New take profit
            
        Returns:
            True if modification successful
        """
        if order_id in self.orders:
            order = self.orders[order_id]
            
            if new_price is not None:
                order['requested_price'] = new_price
            if new_volume is not None:
                order['volume'] = new_volume
            if new_stop_loss is not None:
                order['stop_loss'] = new_stop_loss
            if new_take_profit is not None:
                order['take_profit'] = new_take_profit
            
            logger.info(f"MockMT5: modify_order({order_id}) -> Success")
            return True
        else:
            logger.error(f"MockMT5: modify_order({order_id}) -> Order not found")
            return False
    
    def validate_connection(self) -> bool:
        """
        Validate mock connection.
        
        Always returns True for mock adapter.
        
        Returns:
            True (mock always connected)
            
        **Validates: Task 25.9**
        """
        logger.debug("MockMT5: validate_connection() -> True")
        return True
    
    def update_mock_price(self, symbol: str, price: float):
        """
        Update mock price for testing.
        
        Args:
            symbol: Trading symbol
            price: New price
        """
        self.mock_prices[symbol] = price
        logger.debug(f"MockMT5: Updated {symbol} price to {price}")
    
    def close_position(self, order_id: str) -> bool:
        """
        Close mock position.
        
        Args:
            order_id: Order ID of position to close
            
        Returns:
            True if closed successfully
        """
        for i, position in enumerate(self.positions):
            if position.get('order_id') == order_id:
                # Update balance with profit
                self.balance += position['profit']
                self.equity = self.balance
                
                # Remove position
                self.positions.pop(i)
                
                logger.info(
                    f"MockMT5: Closed position {order_id}, "
                    f"profit: ${position['profit']:.2f}, "
                    f"new balance: ${self.balance:.2f}"
                )
                return True
        
        logger.error(f"MockMT5: Position {order_id} not found")
        return False


