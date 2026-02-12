"""
V8 Crypto Module: BrokerClient Abstract Base Class

Defines the unified interface that all broker adapters must implement.
This enables strategy portability across MT5, Binance, dYdX, and Bybit.

**Validates: Requirement 18.1, 18.2**
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BrokerClient(ABC):
    """
    Abstract base class for all broker adapters.
    
    All broker implementations (MT5, Binance, dYdX, Bybit) must implement
    this interface to ensure strategy portability.
    
    **Interface Methods:**
    - get_balance(): Get account balance
    - place_order(): Place market/limit order
    - cancel_order(): Cancel existing order
    - get_order_book(): Get L2 order book data
    - get_positions(): Get open positions
    - get_order_status(): Get order status
    - modify_order(): Modify existing order
    """
    
    @abstractmethod
    async def get_balance(self) -> float:
        """
        Get account balance.
        
        Returns:
            Account balance in base currency (USD/USDT)
            
        Raises:
            ConnectionError: If broker connection fails
            AuthenticationError: If credentials invalid
        """
        pass
    
    @abstractmethod
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
        Place order on broker.
        
        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "EURUSD")
            volume: Position size (lots for MT5, quantity for crypto)
            direction: "buy" or "sell"
            order_type: "market" or "limit"
            price: Limit price (required for limit orders)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            Dictionary with order details:
            {
                "order_id": str,
                "symbol": str,
                "volume": float,
                "direction": str,
                "status": str,
                "filled_price": float,
                "timestamp": float
            }
            
        Raises:
            OrderError: If order placement fails
            InsufficientFundsError: If insufficient balance
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel existing order.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if cancellation successful, False otherwise
            
        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """
        Get order book L2 data.
        
        Args:
            symbol: Trading symbol
            depth: Number of price levels (default: 5)
            
        Returns:
            Dictionary with order book data:
            {
                "symbol": str,
                "bids": [[price, volume], ...],
                "asks": [[price, volume], ...],
                "timestamp": float
            }
            
        Raises:
            SymbolNotFoundError: If symbol invalid
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions.
        
        Returns:
            List of position dictionaries:
            [
                {
                    "symbol": str,
                    "volume": float,
                    "direction": str,
                    "entry_price": float,
                    "current_price": float,
                    "profit": float,
                    "timestamp": float
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dictionary with order status:
            {
                "order_id": str,
                "status": str,  # "pending", "filled", "cancelled", "rejected"
                "filled_volume": float,
                "remaining_volume": float,
                "average_price": float
            }
            
        Raises:
            OrderNotFoundError: If order doesn't exist
        """
        pass
    
    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_volume: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """
        Modify existing order.
        
        Args:
            order_id: Order identifier
            new_price: New limit price (optional)
            new_volume: New volume (optional)
            new_stop_loss: New stop loss (optional)
            new_take_profit: New take profit (optional)
            
        Returns:
            True if modification successful, False otherwise
            
        Raises:
            OrderNotFoundError: If order doesn't exist
            ModificationError: If modification not allowed
        """
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate broker connection and credentials.
        
        Returns:
            True if connection valid, False otherwise
        """
        pass


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class ConnectionError(BrokerError):
    """Raised when broker connection fails."""
    pass


class AuthenticationError(BrokerError):
    """Raised when credentials are invalid."""
    pass


class OrderError(BrokerError):
    """Raised when order placement fails."""
    pass


class InsufficientFundsError(BrokerError):
    """Raised when insufficient balance for order."""
    pass


class OrderNotFoundError(BrokerError):
    """Raised when order doesn't exist."""
    pass


class SymbolNotFoundError(BrokerError):
    """Raised when symbol is invalid."""
    pass


class ModificationError(BrokerError):
    """Raised when order modification fails."""
    pass

