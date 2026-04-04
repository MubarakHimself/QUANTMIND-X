"""
Trading Subagent

Worker agent for the Trading Department.
Responsible for order execution, fill tracking, and trade monitoring.

Model: Haiku (fast, low-cost for worker tasks)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Trading system prompts for LLM
TRADING_SYSTEM_PROMPT = """You are an expert trading assistant for paper trading operations.

Your role is to help with:
1. Parsing natural language order requests into structured trade parameters
2. Suggesting trades based on market analysis and risk parameters
3. Explaining current positions and their status in plain language

## Important Rules:
- This is PAPER TRADING ONLY - no real money is involved
- Always confirm that trades are simulated before executing
- Consider risk management: position sizing, stop-loss, take-profit
- Follow the user's risk tolerance and trading strategy
- Never execute trades without user confirmation

## Order Types:
- market: Execute immediately at current price
- limit: Execute at specified price or better
- stop: Execute when price reaches specified level

## Position Management:
- Always suggest appropriate stop-loss and take-profit levels
- Consider risk-to-reward ratio (minimum 1:2 recommended)
- Monitor position size relative to account balance

## Response Format:
When parsing orders, return JSON with:
- symbol: Trading pair (e.g., EURUSD, BTCUSD)
- side: buy or sell
- order_type: market, limit, or stop
- volume: Position size in lots
- price: Limit/stop price (if applicable)
- stop_loss: Stop loss price
- take_profit: Take profit price
- reasoning: Brief explanation of the trade logic
"""

PARSE_ORDER_PROMPT = """Parse the following natural language trading request into structured order parameters.

Return JSON with these fields:
- symbol: Trading pair
- side: buy or sell
- order_type: market, limit, or stop
- volume: Position size in lots (standard lot = 100,000 units)
- price: Limit/stop price (for limit/stop orders)
- stop_loss: Stop loss price
- take_profit: Take profit price
- reasoning: Brief explanation

Example: "Buy 1 lot EURUSD at 1.0850 with stop at 1.0800 and take profit at 1.0950"
-> {symbol: "EURUSD", side: "buy", order_type: "limit", volume: 1.0, price: 1.0850, stop_loss: 1.0800, take_profit: 1.0950, reasoning: "Long position at support level"}

Request: {user_request}
"""

SUGGEST_TRADE_PROMPT = """Based on the following market analysis and current positions, suggest an appropriate trade.

Current positions: {positions}
Market data: {market_data}
Risk parameters: {risk_params}

Consider:
1. Direction (buy/sell) based on trend and analysis
2. Entry price level
3. Stop loss placement
4. Take profit target
5. Position size based on risk parameters

Return JSON with:
- recommendation: "buy", "sell", or "no_trade"
- symbol: Trading pair
- entry_price: Suggested entry price
- stop_loss: Stop loss price
- take_profit: Take profit price
- volume: Suggested volume in lots
- reasoning: Explanation of the recommendation
- risk_assessment: Risk level (low, medium, high)
"""

EXPLAIN_POSITION_PROMPT = """Explain the following trading position in simple, natural language.

Position details: {position_details}

Include:
1. What the position is (long/short)
2. Entry price and current status
3. Profit/loss status
4. Any stop loss or take profit levels
5. Recommendation (hold, close, modify)

Be clear that this is paper trading and no real money is involved.
"""


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
        self._llm_client = None
        self._initialize_tools()
        self._initialize_llm()

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
            "parse_order_request": self.parse_order_request,
            "suggest_trade": self.suggest_trade,
            "explain_position": self.explain_position,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def _initialize_llm(self) -> None:
        """Initialize LLM client for natural language processing."""
        try:
            from src.agents.departments.subagents.llm_utils import get_subagent_client
            self._llm_client, self._llm_model = get_subagent_client()
            logger.info(f"TradingSubAgent: LLM client initialized (model={self._llm_model})")
        except Exception as e:
            logger.warning(f"TradingSubAgent: LLM init failed: {e}")

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = TRADING_SYSTEM_PROMPT,
    ) -> str:
        """
        Call LLM for trading assistance.

        Args:
            user_prompt: User's request
            system_prompt: System instructions

        Returns:
            LLM response text
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.messages.create(
                model=self._llm_model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"TradingSubAgent: LLM call failed: {e}")
            raise

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
            "parse_order_request",
            "suggest_trade",
            "explain_position",
        ]

    def parse_order_request(self, user_request: str) -> Dict[str, Any]:
        """
        Parse natural language order request into structured parameters.

        Args:
            user_request: Natural language trading request

        Returns:
            Parsed order parameters as dict
        """
        logger.info(f"TradingSubAgent: Parsing order request: {user_request[:50]}...")

        if not self._llm_client:
            return {
                "status": "error",
                "error": "LLM client not available",
            }

        try:
            user_prompt = PARSE_ORDER_PROMPT.format(user_request=user_request)

            response = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=TRADING_SYSTEM_PROMPT,
            )

            # Try to parse JSON from response
            try:
                # Find JSON in response
                json_match = None
                for pattern in [r'\{[\s\S]*\}', r'\[[\s\S]*\]']:
                    json_match = response.strip()
                    if json_match:
                        break

                parsed = json.loads(json_match) if json_match else {}

                return {
                    "status": "parsed",
                    "order_params": parsed,
                    "raw_response": response,
                }
            except json.JSONDecodeError:
                # Return raw response if JSON parsing fails
                return {
                    "status": "parsed",
                    "order_params": {},
                    "raw_response": response,
                    "warning": "Could not parse JSON from response",
                }

        except Exception as e:
            logger.error(f"TradingSubAgent: Failed to parse order: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def suggest_trade(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        risk_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Suggest a trade based on market analysis and risk parameters.

        Args:
            market_data: Current market conditions
            risk_params: Risk management parameters

        Returns:
            Trade suggestion with entry, stop-loss, take-profit
        """
        logger.info("TradingSubAgent: Generating trade suggestion")

        if not self._llm_client:
            return {
                "status": "error",
                "error": "LLM client not available",
            }

        positions = self.get_positions()
        market_data = market_data or {}
        risk_params = risk_params or {
            "max_risk_per_trade": 2.0,  # % of account
            "min_risk_reward": 2.0,
        }

        try:
            user_prompt = SUGGEST_TRADE_PROMPT.format(
                positions=json.dumps(positions),
                market_data=json.dumps(market_data),
                risk_params=json.dumps(risk_params),
            )

            response = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=TRADING_SYSTEM_PROMPT,
            )

            # Try to parse JSON from response
            try:
                # Find JSON in response
                suggestion = {}
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        suggestion = json.loads(line)
                        break

                return {
                    "status": "suggested",
                    "suggestion": suggestion,
                    "raw_response": response,
                }
            except json.JSONDecodeError:
                return {
                    "status": "suggested",
                    "suggestion": {},
                    "raw_response": response,
                    "warning": "Could not parse JSON from response",
                }

        except Exception as e:
            logger.error(f"TradingSubAgent: Failed to suggest trade: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def explain_position(self, position_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Explain current position(s) in natural language.

        Args:
            position_id: Specific position ID to explain (optional)

        Returns:
            Position explanation in plain language
        """
        logger.info(f"TradingSubAgent: Explaining position: {position_id or 'all'}")

        if not self._llm_client:
            return {
                "status": "error",
                "error": "LLM client not available",
            }

        positions = self.get_positions()

        if position_id:
            position = self._open_positions.get(position_id)
            if not position:
                return {
                    "status": "error",
                    "error": f"Position {position_id} not found",
                }
            position_details = json.dumps(position)
        else:
            position_details = json.dumps(positions)

        try:
            user_prompt = EXPLAIN_POSITION_PROMPT.format(
                position_details=position_details,
            )

            explanation = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=TRADING_SYSTEM_PROMPT,
            )

            return {
                "status": "explained",
                "explanation": explanation,
                "position_id": position_id,
                "positions_count": positions.get("count", 0),
            }

        except Exception as e:
            logger.error(f"TradingSubAgent: Failed to explain position: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
