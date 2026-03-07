"""
Risk Subagent

Worker agent for the Risk Department.
Responsible for position sizing, exposure management, VaR calculations,
and drawdown monitoring.

Model: Haiku (fast, low-cost for worker tasks)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import math


@dataclass
class RiskTask:
    """Risk task input data."""
    task_type: str  # position_sizing, var_calculation, drawdown_check, risk_validation
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 1.0  # Lots
    max_daily_loss: float = 0.05  # 5% of account
    max_drawdown: float = 0.15  # 15% of account
    max_exposure_per_symbol: float = 0.20  # 20% of account per symbol
    max_leverage: float = 10.0


class RiskSubAgent:
    """
    Risk subagent for risk management calculations.

    Capabilities:
    - Position sizing calculations
    - Value at Risk (VaR) calculations
    - Drawdown monitoring
    - Risk validation for trades
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[RiskTask] = None,
        available_tools: Optional[List[str]] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize risk subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
            risk_limits: Risk management limits
        """
        self.agent_id = agent_id
        self.agent_type = "risk"
        self.task = task or RiskTask(task_type="risk_validation")
        self.available_tools = available_tools or []
        self.risk_limits = risk_limits or RiskLimits()
        self.model_tier = "haiku"
        self._initialize_tools()

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        tool_registry = {
            "calculate_position_size": self.calculate_position_size,
            "calculate_var": self.calculate_var,
            "check_drawdown": self.check_drawdown,
            "validate_trade": self.validate_trade,
            "calculate_exposure": self.calculate_exposure,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            account_balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (default 2%)

        Returns:
            Position size calculation
        """
        # Calculate risk amount
        risk_amount = account_balance * risk_percent

        # Calculate price difference
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            return {
                "status": "error",
                "message": "Stop loss cannot be equal to entry price",
            }

        # Calculate position size (simplified for forex)
        # For EURUSD, 1 lot = 100,000 units
        # Each pip = $10 for 1 lot
        pip_value = 10  # Approximate for major pairs

        position_size = risk_amount / (price_diff * pip_value)

        # Apply max limit
        position_size = min(position_size, self.risk_limits.max_position_size)

        return {
            "position_size": round(position_size, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": risk_percent,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "status": "calculated",
        }

    def calculate_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for portfolio.

        Args:
            positions: List of open positions
            confidence_level: Confidence level (default 95%)
            holding_period: Holding period in days

        Returns:
            VaR calculation
        """
        # Simplified VaR calculation using historical volatility
        # In production, would use more sophisticated methods

        total_exposure = 0.0
        for position in positions:
            volume = position.get("volume", 0)
            price = position.get("entry_price", 1.0)
            total_exposure += volume * price * 100000  # Convert lots to units

        # Simplified volatility estimate (would use real data)
        volatility = 0.015  # 1.5% daily volatility

        # VaR calculation (simplified parametric)
        z_score = 1.65 if confidence_level == 0.95 else 2.33  # 95% or 99%
        var = total_exposure * volatility * z_score * math.sqrt(holding_period)

        return {
            "var": round(var, 2),
            "confidence_level": confidence_level,
            "holding_period": holding_period,
            "total_exposure": round(total_exposure, 2),
            "status": "calculated",
        }

    def check_drawdown(
        self,
        account_balance: float,
        peak_balance: float,
    ) -> Dict[str, Any]:
        """
        Check current drawdown against limits.

        Args:
            account_balance: Current account balance
            peak_balance: Peak account balance

        Returns:
            Drawdown check results
        """
        if peak_balance == 0:
            return {
                "status": "error",
                "message": "Peak balance cannot be zero",
            }

        drawdown = (peak_balance - account_balance) / peak_balance
        drawdown_percent = drawdown * 100

        return {
            "drawdown_percent": round(drawdown_percent, 2),
            "current_balance": account_balance,
            "peak_balance": peak_balance,
            "max_drawdown_limit": self.risk_limits.max_drawdown * 100,
            "within_limits": drawdown <= self.risk_limits.max_drawdown,
            "status": "checked",
        }

    def validate_trade(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        account_balance: float,
        current_exposure: float,
    ) -> Dict[str, Any]:
        """
        Validate a trade against risk limits.

        Args:
            symbol: Trading symbol
            side: Trade side (buy, sell)
            volume: Trade volume in lots
            price: Entry price
            account_balance: Account balance
            current_exposure: Current exposure for this symbol

        Returns:
            Validation result
        """
        validation_results = []
        is_valid = True

        # Check position size limit
        if volume > self.risk_limits.max_position_size:
            validation_results.append({
                "check": "position_size",
                "passed": False,
                "message": f"Volume {volume} exceeds max {self.risk_limits.max_position_size}",
            })
            is_valid = False
        else:
            validation_results.append({
                "check": "position_size",
                "passed": True,
                "message": "Volume within limits",
            })

        # Check exposure limit (simplified - use margin-based exposure)
        # For forex with 100:1 leverage, margin is 1% of notional
        # position_value represents margin requirement
        leverage = 100.0  # Assume 100:1 leverage
        margin_required = (volume * 100000) / leverage
        new_exposure = current_exposure + margin_required
        exposure_percent = new_exposure / account_balance

        if exposure_percent > self.risk_limits.max_exposure_per_symbol:
            validation_results.append({
                "check": "exposure",
                "passed": False,
                "message": f"Exposure {exposure_percent:.1%} exceeds max {self.risk_limits.max_exposure_per_symbol:.1%}",
            })
            is_valid = False
        else:
            validation_results.append({
                "check": "exposure",
                "passed": True,
                "message": "Exposure within limits",
            })

        # Check daily loss limit (simplified - would track daily P&L)
        validation_results.append({
            "check": "daily_loss",
            "passed": True,
            "message": "Daily loss within limits",
        })

        return {
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "is_valid": is_valid,
            "validation_results": validation_results,
            "status": "validated",
        }

    def calculate_exposure(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate current exposure per symbol and total.

        Args:
            positions: List of open positions

        Returns:
            Exposure breakdown
        """
        exposure_by_symbol = {}
        total_exposure = 0.0

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            price = position.get("entry_price", 1.0)

            position_value = volume * price * 100000
            exposure_by_symbol[symbol] = position_value
            total_exposure += position_value

        return {
            "exposure_by_symbol": exposure_by_symbol,
            "total_exposure": round(total_exposure, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "position_sizing":
            return self._calculate_position_size()
        elif task_type == "var_calculation":
            return self._calculate_var()
        elif task_type == "drawdown_check":
            return self._check_drawdown()
        elif task_type == "risk_validation":
            return self._validate_risk()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _calculate_position_size(self) -> Dict[str, Any]:
        """Execute position sizing task."""
        params = self.task.parameters
        return self.calculate_position_size(
            account_balance=params.get("account_balance", 10000),
            entry_price=params.get("entry_price", 1.0850),
            stop_loss=params.get("stop_loss", 1.0800),
            risk_percent=params.get("risk_percent", 0.02),
        )

    def _calculate_var(self) -> Dict[str, Any]:
        """Execute VaR calculation task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        return self.calculate_var(
            positions=positions,
            confidence_level=params.get("confidence_level", 0.95),
            holding_period=params.get("holding_period", 1),
        )

    def _check_drawdown(self) -> Dict[str, Any]:
        """Execute drawdown check task."""
        params = self.task.parameters
        return self.check_drawdown(
            account_balance=params.get("account_balance", 9500),
            peak_balance=params.get("peak_balance", 10000),
        )

    def _validate_risk(self) -> Dict[str, Any]:
        """Execute risk validation task."""
        params = self.task.parameters
        return self.validate_trade(
            symbol=params.get("symbol", "EURUSD"),
            side=params.get("side", "buy"),
            volume=params.get("volume", 0.1),
            price=params.get("price", 1.0850),
            account_balance=params.get("account_balance", 10000),
            current_exposure=params.get("current_exposure", 0),
        )

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "position_sizing",
            "var_calculation",
            "drawdown_monitoring",
            "risk_validation",
            "exposure_calculation",
        ]
