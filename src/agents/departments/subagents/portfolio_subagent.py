"""
Portfolio Subagent

Worker agent for the Portfolio Department.
Responsible for portfolio allocation, rebalancing, and performance tracking.

Model: Haiku (fast, low-cost for worker tasks)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PortfolioTask:
    """Portfolio task input data."""
    task_type: str  # allocation, rebalancing, performance, optimization
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    volume: float
    entry_price: float
    current_price: float
    side: str = "buy"

    @property
    def value(self) -> float:
        """Position value."""
        return self.volume * self.current_price * 100000

    @property
    def pnl(self) -> float:
        """Profit/Loss."""
        if self.side == "buy":
            return (self.current_price - self.entry_price) * self.volume * 100000
        return (self.entry_price - self.current_price) * self.volume * 100000


class PortfolioSubAgent:
    """
    Portfolio subagent for portfolio management.

    Capabilities:
    - Portfolio allocation management
    - Rebalancing operations
    - Performance tracking and attribution
    - Risk-adjusted returns calculation
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[PortfolioTask] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize portfolio subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
        """
        self.agent_id = agent_id
        self.agent_type = "portfolio"
        self.task = task or PortfolioTask(task_type="allocation")
        self.available_tools = available_tools or []
        self.model_tier = "haiku"
        self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        tool_registry = {
            "get_allocation": self.get_allocation,
            "calculate_rebalance": self.calculate_rebalance,
            "track_performance": self.track_performance,
            "calculate_returns": self.calculate_returns,
            "optimize_weights": self.optimize_weights,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def get_allocation(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
    ) -> Dict[str, Any]:
        """
        Get current portfolio allocation.

        Args:
            positions: List of open positions
            total_value: Total portfolio value

        Returns:
            Allocation breakdown
        """
        if total_value == 0:
            return {
                "status": "error",
                "message": "Total value cannot be zero",
            }

        allocation = {}
        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            current_price = position.get("current_price", position.get("entry_price", 1.0))

            position_value = volume * current_price * 100000
            allocation[symbol] = {
                "value": round(position_value, 2),
                "weight": round(position_value / total_value, 4),
                "volume": volume,
            }

        return {
            "allocation": allocation,
            "total_value": round(total_value, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def calculate_rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        total_value: float,
    ) -> Dict[str, Any]:
        """
        Calculate rebalancing trades needed.

        Args:
            current_allocation: Current allocation weights
            target_allocation: Target allocation weights
            total_value: Total portfolio value

        Returns:
            Rebalancing trades
        """
        rebalance_trades = []

        # Get all symbols
        all_symbols = set(current_allocation.keys()) | set(target_allocation.keys())

        for symbol in all_symbols:
            current_weight = current_allocation.get(symbol, 0)
            target_weight = target_allocation.get(symbol, 0)
            weight_diff = target_weight - current_weight

            # Calculate trade value
            trade_value = weight_diff * total_value
            trade_volume = abs(trade_value) / 100000  # Convert to lots

            if abs(trade_volume) > 0.01:  # Minimum trade size
                rebalance_trades.append({
                    "symbol": symbol,
                    "side": "buy" if trade_value > 0 else "sell",
                    "volume": round(trade_volume, 2),
                    "current_weight": round(current_weight, 4),
                    "target_weight": round(target_weight, 4),
                    "trade_value": round(trade_value, 2),
                })

        return {
            "rebalance_trades": rebalance_trades,
            "total_trades": len(rebalance_trades),
            "total_rebalance_value": round(sum(t["trade_value"] for t in rebalance_trades), 2),
            "status": "calculated",
        }

    def track_performance(
        self,
        positions: List[Dict[str, Any]],
        period_start_value: float,
        period_end_value: float,
    ) -> Dict[str, Any]:
        """
        Track portfolio performance.

        Args:
            positions: Current positions
            period_start_value: Value at start of period
            period_end_value: Value at end of period

        Returns:
            Performance metrics
        """
        # Calculate returns
        period_return = (period_end_value - period_start_value) / period_start_value if period_start_value > 0 else 0

        # Calculate position-level P&L
        total_pnl = 0
        pnl_by_symbol = {}

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            entry_price = position.get("entry_price", 1.0)
            current_price = position.get("current_price", entry_price)

            position_pnl = (current_price - entry_price) * volume * 100000
            total_pnl += position_pnl
            pnl_by_symbol[symbol] = round(position_pnl, 2)

        return {
            "period_return": round(period_return * 100, 2),
            "absolute_return": round(period_end_value - period_start_value, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_by_symbol": pnl_by_symbol,
            "period_start_value": round(period_start_value, 2),
            "period_end_value": round(period_end_value, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def calculate_returns(
        self,
        equity_curve: List[float],
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate risk-adjusted returns.

        Args:
            equity_curve: List of portfolio values over time
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            Return metrics
        """
        if len(equity_curve) < 2:
            return {
                "status": "error",
                "message": "Insufficient data for return calculation",
            }

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Average return
        avg_return = sum(returns) / len(returns) if returns else 0

        # Volatility (annualized)
        import math
        volatility = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns)) * math.sqrt(252) if returns else 0

        # Sharpe ratio
        sharpe_ratio = (avg_return * 252 - risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "total_return": round(total_return * 100, 2),
            "avg_return": round(avg_return * 100, 4),
            "volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "periods": len(returns),
            "status": "calculated",
        }

    def optimize_weights(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        volatility: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        target_return: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of symbols
            expected_returns: Expected returns per symbol
            volatility: Volatility per symbol
            correlation_matrix: Correlation matrix
            target_return: Target return (optional)

        Returns:
            Optimized weights
        """
        # Simplified equal-weight optimization as placeholder
        # In production, would use mean-variance optimization

        n = len(symbols)
        if n == 0:
            return {
                "status": "error",
                "message": "No symbols provided",
            }

        # Equal weight as baseline
        weights = {symbol: 1.0 / n for symbol in symbols}

        return {
            "weights": weights,
            "method": "equal_weight",
            "expected_return": sum(expected_returns.get(s, 0) * weights[s] for s in symbols),
            "expected_volatility": sum(volatility.get(s, 0) * weights[s] for s in symbols),
            "status": "optimized",
        }

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "allocation":
            return self._execute_allocation()
        elif task_type == "rebalancing":
            return self._execute_rebalancing()
        elif task_type == "performance":
            return self._execute_performance()
        elif task_type == "optimization":
            return self._execute_optimization()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _execute_allocation(self) -> Dict[str, Any]:
        """Execute allocation task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        total_value = params.get("total_value", 10000)
        return self.get_allocation(positions, total_value)

    def _execute_rebalancing(self) -> Dict[str, Any]:
        """Execute rebalancing task."""
        params = self.task.parameters
        current = params.get("current_allocation", {})
        target = params.get("target_allocation", {})
        total_value = params.get("total_value", 10000)
        return self.calculate_rebalance(current, target, total_value)

    def _execute_performance(self) -> Dict[str, Any]:
        """Execute performance tracking task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        start_value = params.get("period_start_value", 10000)
        end_value = params.get("period_end_value", 10500)
        return self.track_performance(positions, start_value, end_value)

    def _execute_optimization(self) -> Dict[str, Any]:
        """Execute optimization task."""
        params = self.task.parameters
        symbols = params.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])
        expected_returns = params.get("expected_returns", {})
        volatility = params.get("volatility", {})
        correlation = params.get("correlation_matrix", None)
        target = params.get("target_return", None)
        return self.optimize_weights(symbols, expected_returns, volatility, correlation, target)

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "allocation_management",
            "rebalancing",
            "performance_tracking",
            "returns_calculation",
            "portfolio_optimization",
        ]
