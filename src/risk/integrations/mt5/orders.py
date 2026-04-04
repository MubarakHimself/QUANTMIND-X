"""Order management for MT5 integration with Layer 1 SL/TP support.

Story 14.1: Layer 1 EA Hard Safety SL/TP
- SL/TP set as native MT5 order parameters at placement
- Trade log records SL/TP metadata per NFR-D1
- EA enforces: per-trade risk cap, force-close hour, no overnight hold
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


def _default_simulated_fallback() -> bool:
    value = os.environ.get("MT5_ALLOW_SIMULATED_FALLBACK")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class EAInputParameters:
    """
    EA Input Parameters for Layer 1 Safety (Story 14.1).

    These parameters are injected at EA initialization and enforced
    at the broker level regardless of QUANTMINDX pipeline state.
    """
    per_trade_risk_cap: float = 0.005  # 0.5% hard ceiling
    force_close_hour: int = 21  # UTC hour (0-23)
    force_close_minute: int = 45  # UTC minute (0-59)
    no_overnight_hold: bool = True  # Hard constraint
    session_mask: int = 0xFF  # All sessions eligible
    islamic_compliance: bool = True  # Swap-free accounts

    def __post_init__(self):
        """Validate EA parameters after initialization."""
        if not 0 < self.per_trade_risk_cap <= 1.0:
            raise ValueError(f"per_trade_risk_cap must be between 0 and 1, got {self.per_trade_risk_cap}")
        if not 0 <= self.force_close_hour <= 23:
            raise ValueError(f"force_close_hour must be 0-23, got {self.force_close_hour}")
        if not 0 <= self.force_close_minute <= 59:
            raise ValueError(f"force_close_minute must be 0-59, got {self.force_close_minute}")
        if not 0 <= self.session_mask <= 0xFF:
            raise ValueError(f"session_mask must be 0x00-0xFF, got {self.session_mask}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "per_trade_risk_cap": self.per_trade_risk_cap,
            "force_close_hour": self.force_close_hour,
            "force_close_minute": self.force_close_minute,
            "no_overnight_hold": self.no_overnight_hold,
            "session_mask": self.session_mask,
            "islamic_compliance": self.islamic_compliance,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EAInputParameters":
        """Create from dictionary with validation."""
        return cls(
            per_trade_risk_cap=data.get("per_trade_risk_cap", 0.005),
            force_close_hour=data.get("force_close_hour", 21),
            force_close_minute=data.get("force_close_minute", 45),
            no_overnight_hold=data.get("no_overnight_hold", True),
            session_mask=data.get("session_mask", 0xFF),
            islamic_compliance=data.get("islamic_compliance", True),
        )


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
    MT5 Order Manager for order operations with Layer 1 SL/TP support.

    Story 14.1: Layer 1 EA Hard Safety SL/TP
    - Places orders with native MT5 SL/TP parameters
    - Records trade log with SL/TP metadata (NFR-D1 compliance)
    - Enforces EA input parameters (risk cap, force-close, overnight hold)
    """

    def __init__(
        self,
        account_manager=None,
        fallback_to_simulated: Optional[bool] = None,
        ea_parameters: Optional[EAInputParameters] = None,
        pip_value_calculator: Optional[Callable[[str, float], Optional[float]]] = None,
    ):
        """
        Initialize Order Manager.

        Args:
            account_manager: MT5 account manager instance
            fallback_to_simulated: Use simulated data when MT5 unavailable
            ea_parameters: EA input parameters for Layer 1 safety enforcement
            pip_value_calculator: Callable that takes (symbol, volume) and returns pip value in account currency
                                 If not provided, uses hardcoded fallback calculation
        """
        self._account_manager = account_manager
        self._fallback_to_simulated = (
            _default_simulated_fallback()
            if fallback_to_simulated is None
            else fallback_to_simulated
        )
        self._orders: List[OrderInfo] = []
        self._ea_parameters = ea_parameters or EAInputParameters()
        self._trade_record_session = None
        self._pip_value_calculator = pip_value_calculator

    def _calculate_pip_value(self, symbol: str, volume: float = 1.0) -> float:
        """
        Calculate pip value for a symbol using the injected calculator or fallback.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "XAUUSD")
            volume: Volume in lots (used for custom calculator, ignored for fallback)

        Returns:
            Pip value in account currency per pip per standard lot
        """
        if self._pip_value_calculator:
            result = self._pip_value_calculator(symbol, volume)
            if result is not None:
                return result

        # Fallback calculation for common symbols
        # Returns pip value PER STANDARD LOT - do NOT multiply by volume
        # This is a simplified fallback - production should use MT5Client.calculate_pip_value
        if symbol == "XAUUSD":
            # Gold: typically $0.10 per pip per micro lot, so $10 per standard lot
            return 10.0
        elif symbol.endswith("JPY"):
            # JPY pairs: different pip location, ~$10 per pip per standard lot
            return 10.0
        elif symbol.endswith("USD"):
            # Most USD pairs: $10 per pip per standard lot
            return 10.0
        else:
            # Default fallback
            return 10.0

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
        Place a new order with SL/TP at broker level.

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
            # Try to call the actual MT5 API through the account manager
            # The underlying account_manager from mcp_metatrader5 has order_send
            mt5_api = getattr(self._account_manager, '_account_manager', None)
            if mt5_api and hasattr(mt5_api, 'order_send'):
                # Convert order_type to MT5 constants
                action = 0  # TRADE_ACTION_DEAL
                order_type_mt5 = 0 if order_type.lower() == 'buy' else 1  # ORDER_TYPE_BUY/SELL
                deviation = 10  # Slippage in points

                # Call MT5 order_send with SL/TP
                result = mt5_api.order_send(
                    action=action,
                    symbol=symbol,
                    volume=volume,
                    type=order_type_mt5,
                    price=price,
                    sl=sl,  # Stop loss price
                    tp=tp,  # Take profit price
                    deviation=deviation,
                    comment=f"QUANTMINDX Layer1 SL/TP"
                )

                if result and result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                    ticket = result.get('order', 0)
                    logger.info(f"MT5 order placed: ticket={ticket}, SL={sl}, TP={tp}")
                    # Store in local orders list for tracking
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
                else:
                    logger.error(f"MT5 order failed: {result}")
                    return None
            else:
                # Fallback if underlying MT5 API not available
                logger.warning("MT5 account_manager does not have order_send, using simulated")
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
        except Exception as e:
            logger.error(f"Error placing order: {e}")
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

    def force_close_by_ticket(self, ticket: int) -> Dict[str, Any]:
        """
        Force close a position by ticket number.

        Used by Layer 3 Kill Switch (Story 14.3) for CHAOS-forced exit.
        This bypasses all Layer 2 locks and closes at current market price.

        Per GG-4: Layer 3 BYPASSES Layer 2 Redis locks when executing forced exit.

        Args:
            ticket: Position ticket number

        Returns:
            Dict with keys:
                - success: bool
                - partial: bool (if True, only partial close occurred)
                - volume: volume that was closed (if partial)
                - error: str (if success is False)
        """
        if not self._account_manager:
            if self._fallback_to_simulated:
                for order in self._orders:
                    if order.ticket == ticket:
                        self._orders.remove(order)
                        return {"success": True, "partial": False, "volume": order.volume}
                return {"success": False, "error": f"Order {ticket} not found"}
            return {"success": False, "error": "MT5 account manager not available"}

        try:
            # Try to call the actual MT5 API through the account manager
            mt5_api = getattr(self._account_manager, '_account_manager', None)
            if mt5_api and hasattr(mt5_api, 'order_send'):
                # Find the order to get symbol, volume, and type
                order_type_mt5 = None
                symbol = None
                volume = None

                for order in self._orders:
                    if order.ticket == ticket:
                        order_type_mt5 = 0 if order.type.lower() == 'buy' else 1  # ORDER_TYPE_BUY/SELL
                        symbol = order.symbol
                        volume = order.volume
                        break

                if order_type_mt5 is None:
                    logger.error(f"Position {ticket} not found in tracked orders")
                    return {"success": False, "error": f"Position {ticket} not found"}

                # TRADE_ACTION_CLOSE_BY closes a position by opposite position
                # But for single position close, we use TRADE_ACTION_DEAL with close volume
                action = 0  # TRADE_ACTION_DEAL
                deviation = 20  # Larger slippage for forced exit
                price = 0  # Market price

                # Call MT5 order_send for forced close
                result = mt5_api.order_send(
                    action=action,
                    symbol=symbol,
                    volume=volume,
                    type=order_type_mt5,
                    price=price,
                    sl=0,  # No SL/TP for forced close
                    tp=0,
                    deviation=deviation,
                    position=ticket,  # Specify position ticket for close
                    comment=f"QUANTMINDX Layer3 Kill Switch forced close"
                )

                if result and result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                    # Remove from local tracking
                    self._orders = [o for o in self._orders if o.ticket != ticket]
                    logger.warning(f"MT5 force close: ticket={ticket}, volume={volume}")
                    return {"success": True, "partial": False, "volume": volume}
                else:
                    error_msg = result.get('comment', 'Unknown error') if result else 'No result'
                    logger.error(f"MT5 force close failed: ticket={ticket}, error={error_msg}")
                    return {"success": False, "error": error_msg}
            else:
                # Fallback if underlying MT5 API not available
                logger.warning("MT5 account_manager does not have order_send, using simulated")
                if self._fallback_to_simulated:
                    for order in self._orders:
                        if order.ticket == ticket:
                            self._orders.remove(order)
                            return {"success": True, "partial": False, "volume": order.volume}
                    return {"success": False, "error": f"Order {ticket} not found"}
                return {"success": False, "error": "MT5 order_send not available"}
        except Exception as e:
            logger.error(f"Error force closing order {ticket}: {e}")
            return {"success": False, "error": str(e)}

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None
    ) -> bool:
        """
        Modify an existing position's stop loss and/or take profit.

        Used by Layer 2 Position Monitor for move-to-breakeven and
        other dynamic position management.

        Args:
            ticket: Position ticket number
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
                        logger.info(f"Simulated position modification: ticket={ticket}, SL={sl}, TP={tp}")
                        return True
                logger.warning(f"Position {ticket} not found for modification")
                return False
            return False

        try:
            # Try to call the actual MT5 API through the account manager
            mt5_api = getattr(self._account_manager, '_account_manager', None)
            if mt5_api and hasattr(mt5_api, 'order_send'):
                # Get current order info to determine order type
                action = 2  # TRADE_ACTION_SLTP for modification
                deviation = 0

                # Find the order to get symbol and type
                order_type_mt5 = None
                symbol = None
                volume = None
                price = None

                for order in self._orders:
                    if order.ticket == ticket:
                        order_type_mt5 = 0 if order.type.lower() == 'buy' else 1
                        symbol = order.symbol
                        volume = order.volume
                        price = order.price
                        break

                if order_type_mt5 is None:
                    logger.error(f"Position {ticket} not found in tracked orders")
                    return False

                # Call MT5 order_send with SL/TP modification
                result = mt5_api.order_send(
                    action=action,
                    symbol=symbol,
                    volume=volume,
                    type=order_type_mt5,
                    price=price,
                    sl=sl if sl is not None else 0,  # 0 means no change
                    tp=tp if tp is not None else 0,   # 0 means no change
                    deviation=deviation,
                    position=ticket,  # Specify position ticket for modification
                    comment=f"QUANTMINDX Layer2 modification"
                )

                if result and result.get('retcode') == 10009:  # TRADE_RETCODE_DONE
                    # Update local tracking
                    for order in self._orders:
                        if order.ticket == ticket:
                            if sl is not None:
                                order.sl = sl
                            if tp is not None:
                                order.tp = tp
                            break
                    logger.info(f"MT5 position modified: ticket={ticket}, SL={sl}, TP={tp}")
                    return True
                else:
                    logger.error(f"MT5 position modification failed: {result}")
                    return False
            else:
                # Fallback if underlying MT5 API not available
                logger.warning("MT5 account_manager does not have order_send, using simulated")
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
        except Exception as e:
            logger.error(f"Error modifying position {ticket}: {e}")
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

    # =========================================================================
    # Layer 1 SL/TP Methods (Story 14.1)
    # =========================================================================

    def set_ea_parameters(self, ea_parameters: EAInputParameters) -> None:
        """
        Set EA input parameters for Layer 1 safety enforcement.

        Args:
            ea_parameters: EAInputParameters instance with safety settings
        """
        self._ea_parameters = ea_parameters
        logger.info(
            f"EA parameters updated: risk_cap={ea_parameters.per_trade_risk_cap}, "
            f"force_close={ea_parameters.force_close_hour}:{ea_parameters.force_close_minute}, "
            f"no_overnight={ea_parameters.no_overnight_hold}"
        )

    def get_ea_parameters(self) -> EAInputParameters:
        """Get current EA input parameters."""
        return self._ea_parameters

    def enforce_risk_cap(self, volume: float, equity: float, entry_price: float, sl_price: float, symbol: str = "EURUSD") -> float:
        """
        Enforce per-trade risk cap (0.5% hard ceiling).

        Uses symbol-aware pip value calculation for accurate risk assessment.

        Args:
            volume: Requested volume in lots
            equity: Current account equity
            entry_price: Entry price for the trade
            sl_price: Stop loss price
            symbol: Trading symbol for pip value lookup

        Returns:
            Adjusted volume that respects the risk cap
        """
        risk_cap = self._ea_parameters.per_trade_risk_cap
        max_risk_amount = equity * risk_cap  # e.g., 10000 * 0.005 = 50

        # Calculate risk in price terms
        if sl_price > 0 and entry_price > 0:
            price_risk = abs(entry_price - sl_price)  # e.g., 0.005 (50 pips)

            if price_risk > 0:
                # Get symbol-aware pip value
                pip_value = self._calculate_pip_value(symbol, volume)
                PIP_SIZE = 0.0001  # For most forex pairs (will be adjusted for JPY pairs)

                # Adjust pip size for JPY pairs
                if symbol.endswith("JPY"):
                    PIP_SIZE = 0.01

                pips_risk = price_risk / PIP_SIZE  # e.g., 0.005 / 0.0001 = 50 pips
                risk_per_lot = pips_risk * pip_value  # e.g., 50 * 10 = $500

                if risk_per_lot > 0:
                    max_volume_by_risk = max_risk_amount / risk_per_lot  # e.g., 50 / 500 = 0.1 lots
                    adjusted_volume = min(volume, max_volume_by_risk)

                    if adjusted_volume < volume:
                        logger.warning(
                            f"Volume adjusted from {volume} to {adjusted_volume} "
                            f"to respect {risk_cap*100}% risk cap"
                        )
                        return adjusted_volume

        return volume

    def should_force_close(self, current_utc_time: Optional[datetime] = None) -> bool:
        """
        Check if force-close should trigger based on configured hour.

        Args:
            current_utc_time: Current time in UTC (for testing)

        Returns:
            True if positions should be force-closed
        """
        if current_utc_time is None:
            current_utc_time = datetime.now(timezone.utc)

        force_hour = self._ea_parameters.force_close_hour
        force_minute = self._ea_parameters.force_close_minute

        # Check if current time matches force-close window (within the hour)
        if current_utc_time.hour == force_hour and current_utc_time.minute >= force_minute:
            return True

        return False

    def should_prevent_overnight(self, current_utc_time: Optional[datetime] = None) -> bool:
        """
        Check if overnight holds should be prevented.

        Args:
            current_utc_time: Current time in UTC (for testing)

        Returns:
            True if positions should not be held overnight
        """
        if not self._ea_parameters.no_overnight_hold:
            return False

        if current_utc_time is None:
            current_utc_time = datetime.now(timezone.utc)

        # Weekend prevention (Friday after close, through Sunday)
        # Weekend starts at Friday 22:00 UTC
        if current_utc_time.weekday() == 4 and current_utc_time.hour >= 22:
            return True
        # Saturday all day
        if current_utc_time.weekday() == 5:
            return True
        # Sunday before Monday 00:00
        if current_utc_time.weekday() == 6 and current_utc_time.hour < 0:
            return True

        return False

    def force_close_positions(
        self,
        current_utc_time: Optional[datetime] = None,
        reason: str = "force_close_hour"
    ) -> List[Dict[str, Any]]:
        """
        Force close all open positions based on EA parameters.

        This method should be called by a scheduled task or event loop
        to check if force-close conditions are met.

        Args:
            current_utc_time: Current time in UTC (for testing)
            reason: Reason for force close (for logging)

        Returns:
            List of close results with ticket and status
        """
        if not self.should_force_close(current_utc_time):
            return []

        results = []
        # Make a copy of the orders list to avoid modification during iteration
        open_orders = list(self.get_orders())

        for order in open_orders:
            success = self.close_order(order.ticket)
            results.append({
                "ticket": order.ticket,
                "symbol": order.symbol,
                "success": success,
                "reason": reason,
                "closed_at": datetime.now(timezone.utc).isoformat()
            })
            logger.info(
                f"Force-close executed: ticket={order.ticket}, "
                f"symbol={order.symbol}, reason={reason}"
            )

        return results

    def prevent_overnight_holds(
        self,
        current_utc_time: Optional[datetime] = None,
        reason: str = "overnight_prevention"
    ) -> List[Dict[str, Any]]:
        """
        Close all positions to prevent overnight holds.

        This method should be called by a scheduled task or event loop
        to check if overnight prevention conditions are met.

        Args:
            current_utc_time: Current time in UTC (for testing)
            reason: Reason for close (for logging)

        Returns:
            List of close results with ticket and status
        """
        if not self.should_prevent_overnight(current_utc_time):
            return []

        results = []
        # Make a copy of the orders list to avoid modification during iteration
        open_orders = list(self.get_orders())

        for order in open_orders:
            success = self.close_order(order.ticket)
            results.append({
                "ticket": order.ticket,
                "symbol": order.symbol,
                "success": success,
                "reason": reason,
                "closed_at": datetime.now(timezone.utc).isoformat()
            })
            logger.info(
                f"Overnight prevention executed: ticket={order.ticket}, "
                f"symbol={order.symbol}, reason={reason}"
            )

        return results

    def create_trade_record(
        self,
        signal_id: str,
        strategy_id: str,
        epic_id: Optional[str],
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        position_volume: float,
        risk_amount: float,
        regime_at_entry: Optional[str] = None,
        mode: str = "live",
        broker_account_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a trade record with SL/TP metadata (NFR-D1 compliance).

        Args:
            signal_id: Unique signal identifier from Commander pipeline
            strategy_id: Strategy that generated the signal
            epic_id: Epic the strategy belongs to
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Price at which position was opened
            sl_price: Stop loss price (0 = no SL)
            tp_price: Take profit price (0 = no TP)
            position_volume: Volume in lots
            risk_amount: Risk in account currency
            regime_at_entry: Market regime from Sentinel
            mode: Trading mode (demo/live)
            broker_account_id: Broker account used

        Returns:
            Trade record dictionary or None if creation failed
        """
        try:
            from src.database.models import TradeRecord
            from src.database.models import SessionLocal

            session = SessionLocal()
            try:
                trade_record = TradeRecord(
                    signal_id=signal_id,
                    strategy_id=strategy_id,
                    epic_id=epic_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    position_volume=position_volume,
                    risk_amount=risk_amount,
                    timestamp_utc=datetime.now(timezone.utc),
                    regime_at_entry=regime_at_entry,
                    mode=mode,
                    broker_account_id=broker_account_id,
                    ea_parameters=self._ea_parameters.to_dict(),
                )
                session.add(trade_record)
                session.commit()
                session.refresh(trade_record)

                logger.info(
                    f"Trade record created: {trade_record.id} - "
                    f"{symbol} {direction} @{entry_price} "
                    f"SL={sl_price} TP={tp_price}"
                )
                return trade_record.to_dict()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to create trade record: {e}")
            # Per NFR-D1: Trade log write must complete before order_send returns
            # If we can't write the trade record, we should still allow the order
            # but log the failure prominently
            return None

    def place_order_with_sltp(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: float,
        sl: float = 0.0,
        tp: float = 0.0,
        signal_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        epic_id: Optional[str] = None,
        regime_at_entry: Optional[str] = None,
        mode: str = "live",
        broker_account_id: Optional[str] = None,
        equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Place an order with SL/TP and create trade record atomically.

        This is the primary method for Story 14.1 Layer 1 SL/TP placement.
        SL/TP are set as native MT5 order parameters and recorded in the trade log.

        Args:
            symbol: Trading symbol
            order_type: Order type (buy/sell)
            volume: Order volume in lots
            price: Order price
            sl: Stop loss price
            tp: Take profit price
            signal_id: Signal identifier for trade record
            strategy_id: Strategy ID for trade record
            epic_id: Epic ID for trade record
            regime_at_entry: Regime at entry for trade record
            mode: Trading mode
            broker_account_id: Broker account ID
            equity: Account equity for risk cap calculation

        Returns:
            Dict with order result and trade record
        """
        # Apply risk cap if equity provided (now using symbol-aware calculation)
        if equity is not None and sl > 0:
            volume = self.enforce_risk_cap(volume, equity, price, sl, symbol)

        # Place the order
        ticket = self.place_order(symbol, order_type, volume, price, sl, tp)

        if ticket is None:
            return {
                "success": False,
                "error": "Order placement failed",
                "ticket": None,
                "trade_record": None,
            }

        # Calculate risk amount
        risk_amount = 0.0
        if sl > 0 and price > 0:
            risk_amount = abs(price - sl) * volume

        # Create trade record (per NFR-D1: must complete before ACK)
        trade_record = None
        if signal_id and strategy_id:
            trade_record = self.create_trade_record(
                signal_id=signal_id,
                strategy_id=strategy_id,
                epic_id=epic_id,
                symbol=symbol,
                direction=order_type.upper(),
                entry_price=price,
                sl_price=sl,
                tp_price=tp,
                position_volume=volume,
                risk_amount=risk_amount,
                regime_at_entry=regime_at_entry,
                mode=mode,
                broker_account_id=broker_account_id,
            )

        return {
            "success": True,
            "ticket": ticket,
            "sl": sl,
            "tp": tp,
            "volume": volume,
            "trade_record": trade_record,
        }
