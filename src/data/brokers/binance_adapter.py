"""
V8 Binance Adapters

Wraps BinanceConnector to implement BrokerClient interface.
Provides Spot and Futures adapters for unified broker registry.

**Validates: Tasks 25.13-25.16**
"""

import logging
from typing import Dict, Any, List, Optional

from src.integrations.crypto.broker_client import BrokerClient
from src.integrations.crypto.binance_connector import BinanceConnector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceSpotAdapter(BrokerClient):
    """
    Binance Spot adapter wrapping BinanceConnector.
    
    Implements BrokerClient interface for Binance Spot trading.
    
    **Validates: Tasks 25.13-25.15**
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance Spot adapter.
        
        Args:
            config: Broker configuration dictionary
                Required fields:
                - api_key: Binance API key
                - api_secret: Binance API secret
                Optional fields:
                - testnet: Use testnet (default: False)
        """
        api_key = config.get('api_key')
        api_secret = config.get('api_secret')
        testnet = config.get('testnet', False)
        
        if not api_key or not api_secret:
            raise ValueError("Binance Spot adapter requires api_key and api_secret")
        
        # Create BinanceConnector
        self.connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            futures=False  # Spot trading
        )
        
        self.testnet = testnet
        logger.info(f"BinanceSpotAdapter initialized: testnet={testnet}")
    
    async def get_balance(self) -> float:
        """Get account balance."""
        return await self.connector.get_balance()
    
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
        """Place order."""
        return await self.connector.place_order(
            symbol=symbol,
            volume=volume,
            direction=direction,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        return await self.connector.cancel_order(order_id)
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """Get order book."""
        return await self.connector.get_order_book(symbol, depth)
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions (empty for spot)."""
        return await self.connector.get_positions()
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        return await self.connector.get_order_status(order_id)
    
    async def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_volume: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """Modify order."""
        return await self.connector.modify_order(
            order_id=order_id,
            new_price=new_price,
            new_volume=new_volume,
            new_stop_loss=new_stop_loss,
            new_take_profit=new_take_profit
        )
    
    def validate_connection(self) -> bool:
        """
        Validate Binance connection.
        
        Returns:
            True if connection valid
            
        **Validates: Task 25.15**
        """
        return self.connector.validate_connection()


class BinanceFuturesAdapter(BrokerClient):
    """
    Binance Futures adapter wrapping BinanceConnector.
    
    Implements BrokerClient interface for Binance Futures trading.
    
    **Validates: Task 25.16**
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Binance Futures adapter.
        
        Args:
            config: Broker configuration dictionary
                Required fields:
                - api_key: Binance API key
                - api_secret: Binance API secret
                Optional fields:
                - testnet: Use testnet (default: False)
        """
        api_key = config.get('api_key')
        api_secret = config.get('api_secret')
        testnet = config.get('testnet', False)
        
        if not api_key or not api_secret:
            raise ValueError("Binance Futures adapter requires api_key and api_secret")
        
        # Create BinanceConnector
        self.connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            futures=True  # Futures trading
        )
        
        self.testnet = testnet
        logger.info(f"BinanceFuturesAdapter initialized: testnet={testnet}")
    
    async def get_balance(self) -> float:
        """Get account balance."""
        return await self.connector.get_balance()
    
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
        """Place order."""
        return await self.connector.place_order(
            symbol=symbol,
            volume=volume,
            direction=direction,
            order_type=order_type,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order."""
        return await self.connector.cancel_order(order_id)
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """Get order book."""
        return await self.connector.get_order_book(symbol, depth)
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get positions."""
        return await self.connector.get_positions()
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        return await self.connector.get_order_status(order_id)
    
    async def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_volume: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """Modify order."""
        return await self.connector.modify_order(
            order_id=order_id,
            new_price=new_price,
            new_volume=new_volume,
            new_stop_loss=new_stop_loss,
            new_take_profit=new_take_profit
        )
    
    def validate_connection(self) -> bool:
        """
        Validate Binance connection.
        
        Returns:
            True if connection valid
        """
        return self.connector.validate_connection()


