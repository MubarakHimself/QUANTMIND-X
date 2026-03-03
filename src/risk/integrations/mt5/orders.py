"""Order management for MT5 integration."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OrderInfo:
    """Order information model."""
    ticket: int
    symbol: str
    type: str
    volume: float
    price: float
    sl: float
    tp: float
    profit: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "type": self.type,
            "volume": self.volume,
            "price": self.price,
            "sl": self.sl,
            "tp": self.tp,
            "profit": self.profit,
            "status": self.status
        }


class OrderManager:
    """
    MT5 Order Manager for order operations.

    Provides order placement, modification, and retrieval.
    """

    def __init__(self, account_manager=None, fallback_to_simulated: bool = True):
        """
        Initialize Order Manager.

        Args:
            account_manager: MT5 account manager instance
            fallback_to_simulated: Use simulated data when MT5 unavailable
        """
        self._account_manager = account_manager
        self._fallback_to_simulated = fallback_to_simulated
        self._orders: List[OrderInfo] = []

    def get_orders(self, symbol: Optional[str] = None) -> List[OrderInfo]:
        """
        Get open orders, optionally filtered by symbol.

        Args:
            symbol: Optional symbol to filter by

        Returns:
            List of OrderInfo objects
        """
        if not self._account_manager:
            if self._fallback_to_simulated:
                return self._orders
            return []

        try:
            # Get orders from MT5
            # This is a placeholder - actual implementation would call MT5 API
            if symbol:
                return [o for o in self._orders if o.symbol == symbol]
            return self._orders
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            if self._fallback_to_simulated:
                return self._orders
            return []

    def place_order(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float = 0.0,
        tp: float = 0.0
    ) -> Optional[int]:
        """
        Place a new order.

        Args:
            symbol: Trading symbol
            order_type: Order type (buy/sell)
            volume: Order volume in lots
            price: Order price
            sl: Stop loss price
            tp: Take profit price

        Returns:
            Order ticket number or None if failed
        """
        if not self._account_manager:
            if self._fallback_to_simulated:
                ticket = len(self._orders) + 1000
                order = OrderInfo(
                    ticket=ticket,
                    symbol=symbol,
                    type=order_type,
                    volume=volume,
                    price=price,
                    sl=sl,
                    tp=tp,
                    profit=0.0,
                    status="open"
                )
                self._orders.append(order)
                return ticket
            return None

        try:
            # Place order via MT5
            # This is a placeholder - actual implementation would call MT5 API
            logger.warning("Real order placement not implemented, using simulated")
            return None
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None

    def modify_order(
        self,
        ticket: int,
        sl: float = None,
        tp: float = None
    ) -> bool:
        """
        Modify an existing order.

        Args:
            ticket: Order ticket number
            sl: New stop loss price (None to keep current)
            tp: New take profit price (None to keep current)

        Returns:
            True if successful, False otherwise
        """
        if not self._account_manager:
            if self._fallback_to_simulated:
                for order in self._orders:
                    if order.ticket == ticket:
                        if sl is not None:
                            order.sl = sl
                        if tp is not None:
                            order.tp = tp
                        return True
                return False
            return False

        try:
            # Modify order via MT5
            logger.warning("Real order modification not implemented")
            return False
        except Exception as e:
            logger.error(f"Error modifying order: {e}")
            return False

    def close_order(self, ticket: int, lots: Optional[float] = None) -> bool:
        """
        Close an order (or partial close).

        Args:
            ticket: Order ticket number
            lots: Lots to close (None for full close)

        Returns:
            True if successful, False otherwise
        """
        if not self._account_manager:
            if self._fallback_to_simulated:
                for order in self._orders:
                    if order.ticket == ticket:
                        if lots is None or lots >= order.volume:
                            self._orders.remove(order)
                        else:
                            order.volume -= lots
                        return True
                return False
            return False

        try:
            # Close order via MT5
            logger.warning("Real order closure not implemented")
            return False
        except Exception as e:
            logger.error(f"Error closing order: {e}")
            return False
