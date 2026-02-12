"""
V8 MT5 Socket Adapter

Connects to MT5 via Socket Bridge for remote VPS trading.
Enables Python backend on Linux to communicate with MT5 on Windows VPS.

**Validates: Tasks 25.8-25.12 (Socket version)**
"""

import asyncio
import json
import zmq
import zmq.asyncio
import time
import logging
from typing import Dict, Any, List, Optional

from src.integrations.crypto.broker_client import (
    BrokerClient,
    ConnectionError,
    OrderError,
    OrderNotFoundError
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MT5SocketAdapter(BrokerClient):
    """
    MT5 adapter using Socket Bridge for remote connectivity.
    
    Communicates with MT5 on Windows VPS via ZMQ socket bridge.
    Enables:
    - Python backend on Linux
    - MT5 on Windows VPS
    - <5ms latency communication
    
    **Validates: Tasks 25.8-25.12 (Socket version)**
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MT5 socket adapter.
        
        Args:
            config: Broker configuration dictionary
                Required fields:
                - vps_host: VPS IP address or hostname
                - vps_port: Socket server port (default: 5555)
                - account_id: MT5 account number
                Optional fields:
                - timeout: Request timeout in seconds (default: 5.0)
                - max_retries: Maximum connection retries (default: 3)
        """
        self.vps_host = config.get('vps_host', 'localhost')
        self.vps_port = config.get('vps_port', 5555)
        self.account_id = config.get('account_id')
        self.timeout = config.get('timeout', 5.0)
        self.max_retries = config.get('max_retries', 3)
        
        # ZMQ context and socket
        self.context = zmq.asyncio.Context()
        self.socket = None
        self._connected = False
        
        logger.info(
            f"MT5SocketAdapter initialized: "
            f"vps={self.vps_host}:{self.vps_port}, "
            f"account={self.account_id}"
        )
    
    async def _ensure_connection(self):
        """
        Ensure socket connection is established.
        
        Creates socket if not already connected.
        """
        if self.socket is None or self._connected is False:
            await self._connect()
    
    async def _connect(self):
        """
        Connect to MT5 socket server on VPS.
        
        Raises:
            ConnectionError: If connection fails
        """
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            self.socket.setsockopt(zmq.SNDTIMEO, int(self.timeout * 1000))
            self.socket.connect(f"tcp://{self.vps_host}:{self.vps_port}")
            
            self._connected = True
            logger.info(f"Connected to MT5 socket server: {self.vps_host}:{self.vps_port}")
            
        except Exception as e:
            self._connected = False
            logger.error(f"Failed to connect to MT5 socket server: {e}")
            raise ConnectionError(f"Failed to connect to MT5 VPS: {e}")
    
    async def _send_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send request to MT5 socket server.
        
        Args:
            message: Request message dictionary
            
        Returns:
            Response dictionary
            
        Raises:
            ConnectionError: If communication fails
        """
        await self._ensure_connection()
        
        # Add account_id to message
        message['account_id'] = self.account_id
        message['timestamp'] = time.time()
        
        # Send request
        try:
            await self.socket.send_json(message)
            logger.debug(f"Sent request: {message['type']}")
            
            # Receive response
            response = await self.socket.recv_json()
            logger.debug(f"Received response: {response.get('status')}")
            
            # Check for errors
            if response.get('status') == 'error':
                error_msg = response.get('error', 'Unknown error')
                raise ConnectionError(f"MT5 VPS error: {error_msg}")
            
            return response
            
        except zmq.error.Again:
            logger.error("Socket timeout - VPS not responding")
            self._connected = False
            raise ConnectionError("MT5 VPS timeout - check VPS connectivity")
        
        except Exception as e:
            logger.error(f"Socket communication error: {e}")
            self._connected = False
            raise ConnectionError(f"MT5 VPS communication error: {e}")
    
    async def get_balance(self) -> float:
        """
        Get account balance from MT5 VPS.
        
        Returns:
            Account balance
            
        **Validates: Task 25.10**
        """
        message = {
            'type': 'GET_BALANCE'
        }
        
        response = await self._send_request(message)
        balance = response.get('balance', 0.0)
        
        logger.info(f"MT5 VPS balance: ${balance:.2f}")
        return balance
    
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
        Place order on MT5 VPS.
        
        Args:
            symbol: Trading symbol
            volume: Lot size
            direction: "buy" or "sell"
            order_type: "market" or "limit"
            price: Limit price (optional)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            Order details
            
        **Validates: Task 25.11**
        """
        message = {
            'type': 'TRADE_OPEN',
            'symbol': symbol,
            'volume': volume,
            'direction': direction,
            'order_type': order_type,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        
        response = await self._send_request(message)
        
        order = {
            'order_id': str(response.get('order_id')),
            'symbol': symbol,
            'volume': volume,
            'direction': direction,
            'status': response.get('status', 'unknown'),
            'filled_price': response.get('filled_price'),
            'timestamp': time.time()
        }
        
        logger.info(
            f"MT5 VPS order placed: {order['order_id']} "
            f"{symbol} {direction} {volume} lots"
        )
        
        return order
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order on MT5 VPS.
        
        Args:
            order_id: Order identifier
            
        Returns:
            True if cancelled successfully
        """
        message = {
            'type': 'CANCEL_ORDER',
            'order_id': order_id
        }
        
        response = await self._send_request(message)
        success = response.get('success', False)
        
        logger.info(f"MT5 VPS cancel order {order_id}: {'Success' if success else 'Failed'}")
        return success
    
    async def get_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """
        Get order book from MT5 VPS.
        
        Note: MT5 doesn't provide full order book, returns bid/ask only.
        
        Args:
            symbol: Trading symbol
            depth: Number of price levels (ignored for MT5)
            
        Returns:
            Order book with bid/ask
        """
        message = {
            'type': 'GET_QUOTES',
            'symbol': symbol
        }
        
        response = await self._send_request(message)
        
        bid = response.get('bid', 0.0)
        ask = response.get('ask', 0.0)
        
        order_book = {
            'symbol': symbol,
            'bids': [[bid, 1.0]],  # MT5 only provides single bid/ask
            'asks': [[ask, 1.0]],
            'timestamp': time.time()
        }
        
        logger.debug(f"MT5 VPS quotes: {symbol} bid={bid} ask={ask}")
        return order_book
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get open positions from MT5 VPS.
        
        Returns:
            List of positions
            
        **Validates: Task 25.12**
        """
        message = {
            'type': 'GET_POSITIONS'
        }
        
        response = await self._send_request(message)
        positions = response.get('positions', [])
        
        logger.info(f"MT5 VPS positions: {len(positions)} open")
        return positions
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get order status from MT5 VPS.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Order status
        """
        message = {
            'type': 'GET_ORDER_STATUS',
            'order_id': order_id
        }
        
        response = await self._send_request(message)
        
        return {
            'order_id': order_id,
            'status': response.get('status', 'unknown'),
            'filled_volume': response.get('filled_volume', 0.0),
            'remaining_volume': response.get('remaining_volume', 0.0),
            'average_price': response.get('average_price', 0.0)
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
        Modify order on MT5 VPS.
        
        Args:
            order_id: Order identifier
            new_price: New limit price
            new_volume: New volume
            new_stop_loss: New stop loss
            new_take_profit: New take profit
            
        Returns:
            True if modification successful
        """
        message = {
            'type': 'TRADE_MODIFY',
            'order_id': order_id,
            'new_price': new_price,
            'new_volume': new_volume,
            'new_stop_loss': new_stop_loss,
            'new_take_profit': new_take_profit
        }
        
        response = await self._send_request(message)
        success = response.get('success', False)
        
        logger.info(f"MT5 VPS modify order {order_id}: {'Success' if success else 'Failed'}")
        return success
    
    def validate_connection(self) -> bool:
        """
        Validate connection to MT5 VPS.
        
        Returns:
            True if connected, False otherwise
            
        **Validates: Task 25.9**
        """
        try:
            # Try to connect synchronously
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, just check connection status
                return self._connected
            else:
                # If loop not running, try to connect
                loop.run_until_complete(self._connect())
                return self._connected
        except Exception as e:
            logger.error(f"MT5 VPS connection validation failed: {e}")
            return False
    
    def close(self):
        """
        Close socket connection.
        
        Should be called when adapter is no longer needed.
        """
        if self.socket:
            self.socket.close()
            self._connected = False
            logger.info("MT5 socket connection closed")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


