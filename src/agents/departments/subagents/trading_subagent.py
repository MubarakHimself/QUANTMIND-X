"""
Trading Subagent

Worker agent for the Trading Department.
Responsible for order execution, fill tracking, and trade monitoring.

Model: Haiku (fast, low-cost for worker tasks)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TradingTask:
    """Trading task input data."""
    task_type: str  # order_execution, fill_tracking, trade_monitor
    orders: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class TradingSubAgent:
    """
    Trading subagent for order execution and monitoring.

    Capabilities:
    - Order execution (paper trading)
    - Fill tracking and confirmation
    - Trade monitoring and management
    - Position tracking
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[TradingTask] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize trading subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
        """
        self.agent_id = agent_id
        self.agent_type = "trading"
        self.task = task or TradingTask(task_type="order_execution")
        self.available_tools = available_tools or []
        self.model_tier = "haiku"
        self._open_positions: Dict[str, Any] = {}
        self._pending_orders: List[Dict[str, Any]] = []
        self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        tool_registry = {
            "execute_order": self.execute_order,
            "cancel_order": self.cancel_order,
            "modify_order": self.modify_order,
            "get_positions": self.get_positions,
            "get_orders": self.get_orders,
            "close_position": self.close_position,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def execute_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        volume: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a trading order.

        Args:
            symbol: Trading symbol (e.g., EURUSD)
            order_type: Order type (market, limit, stop)
            side: Order side (buy, sell)
            volume: Order volume in lots
            price: Limit/stop price (optional for market orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            comment: Order comment

        Returns:
            Order execution result
        """
        order_id = f"ORD_{hash(symbol + datetime.now().isoformat()) % 100000}"

        order = {
            "order_id": order_id,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "volume": volume,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "comment": comment,
            "status": "filled",  # Paper trading = instant fill
            "filled_at": datetime.now().isoformat(),
            "filled_price": price or self._get_market_price(symbol),
        }

        self._pending_orders.append(order)

        # If market order, convert to position
        if order_type == "market":
            self._open_positions[order_id] = order

        return {
            "status": "success",
            "order": order,
            "message": f"Order {order_id} executed successfully",
        }

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel a pending order.

        Args:
            order_id: Order identifier

        Returns:
            Cancellation result
        """
        for i, order in enumerate(self._pending_orders):
            if order["order_id"] == order_id and order["status"] == "pending":
                self._pending_orders[i]["status"] = "cancelled"
                return {
                    "status": "success",
                    "order_id": order_id,
                    "message": "Order cancelled successfully",
                }

        return {
            "status": "error",
            "order_id": order_id,
            "message": "Order not found or cannot be cancelled",
        }

    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None,
        new_volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Modify an existing order.

        Args:
            order_id: Order identifier
            new_price: New order price
            new_stop_loss: New stop loss
            new_take_profit: New take profit
            new_volume: New volume

        Returns:
            Modification result
        """
        for order in self._pending_orders:
            if order["order_id"] == order_id:
                if new_price is not None:
                    order["price"] = new_price
                if new_stop_loss is not None:
                    order["stop_loss"] = new_stop_loss
                if new_take_profit is not None:
                    order["take_profit"] = new_take_profit
                if new_volume is not None:
                    order["volume"] = new_volume

                return {
                    "status": "success",
                    "order_id": order_id,
                    "message": "Order modified successfully",
                    "order": order,
                }

        return {
            "status": "error",
            "order_id": order_id,
            "message": "Order not found",
        }

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            Open positions
        """
        positions = list(self._open_positions.values())
        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]

        return {
            "positions": positions,
            "count": len(positions),
            "total_profit": sum(
                (p.get("filled_price", 0) - p.get("entry_price", 0)) * p.get("volume", 0)
                for p in positions
            ),
        }

    def get_orders(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get orders with optional filters.

        Args:
            status: Filter by status (pending, filled, cancelled)
            symbol: Filter by symbol

        Returns:
            Orders list
        """
        orders = self._pending_orders.copy()
        if status:
            orders = [o for o in orders if o.get("status") == status]
        if symbol:
            orders = [o for o in orders if o.get("symbol") == symbol]

        return {
            "orders": orders,
            "count": len(orders),
        }

    def close_position(
        self,
        position_id: str,
        volume: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            position_id: Position identifier
            volume: Volume to close (partial close supported)

        Returns:
            Close result
        """
        if position_id not in self._open_positions:
            return {
                "status": "error",
                "message": f"Position {position_id} not found",
            }

        position = self._open_positions[position_id]
        close_volume = volume or position["volume"]

        # Create closing order
        close_order = {
            "order_id": f"CLOSE_{position_id}",
            "symbol": position["symbol"],
            "type": "market",
            "side": "sell" if position["side"] == "buy" else "buy",
            "volume": close_volume,
            "status": "filled",
            "closed_position_id": position_id,
            "filled_at": datetime.now().isoformat(),
        }

        # Update or remove position
        if close_volume >= position["volume"]:
            del self._open_positions[position_id]
        else:
            position["volume"] -= close_volume

        return {
            "status": "success",
            "close_order": close_order,
            "message": f"Position {position_id} closed",
        }

    def _get_market_price(self, symbol: str) -> float:
        """Get current market price (placeholder)."""
        # In production, this would fetch real market data
        return 1.0850

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "order_execution":
            return self._execute_orders()
        elif task_type == "fill_tracking":
            return self._track_fills()
        elif task_type == "trade_monitor":
            return self._monitor_trades()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _execute_orders(self) -> Dict[str, Any]:
        """Execute pending orders."""
        results = []
        for order_spec in self.task.orders:
            result = self.execute_order(
                symbol=order_spec.get("symbol"),
                order_type=order_spec.get("type", "market"),
                side=order_spec.get("side"),
                volume=order_spec.get("volume", 0.01),
                price=order_spec.get("price"),
                stop_loss=order_spec.get("stop_loss"),
                take_profit=order_spec.get("take_profit"),
            )
            results.append(result)

        return {
            "task_type": "order_execution",
            "status": "completed",
            "results": results,
            "total_orders": len(results),
        }

    def _track_fills(self) -> Dict[str, Any]:
        """Track order fills."""
        return {
            "task_type": "fill_tracking",
            "status": "completed",
            "fills": self._pending_orders,
            "total_fills": len(self._pending_orders),
        }

    def _monitor_trades(self) -> Dict[str, Any]:
        """Monitor open trades."""
        positions = self.get_positions()
        return {
            "task_type": "trade_monitor",
            "status": "completed",
            "positions": positions["positions"],
            "count": positions["count"],
            "total_profit": positions["total_profit"],
        }

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "order_execution",
            "order_cancellation",
            "order_modification",
            "position_tracking",
            "trade_monitoring",
        ]
