"""
Risk and Position Tools

READ-ONLY access for all departments.
No actual risk limits can be set - only calculations and monitoring.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Calculated position size."""
    units: float
    lots: float
    risk_percent: float
    risk_amount: float


@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio."""
    var_95: float  # Value at Risk at 95% confidence
    var_99: float  # Value at Risk at 99% confidence
    max_drawdown: float
    current_drawdown: float
    risk_reward_ratio: float


@dataclass
class DrawdownInfo:
    """Drawdown information."""
    current_drawdown_percent: float
    max_drawdown_percent: float
    drawdown_duration_days: int
    is_in_drawdown: bool


class RiskTools:
    """
    Risk and position calculation tools.

    All methods are READ-ONLY - no actual risk limits can be set.
    Use for monitoring and calculation only.
    """

    def calculate_position_size(
        self,
        account_balance: float,
        risk_percent: float,
        entry_price: float,
        stop_loss_price: float,
    ) -> PositionSize:
        """
        Calculate position size based on risk percentage.

        Args:
            account_balance: Account balance
            risk_percent: Risk percentage (e.g., 1.0 for 1%)
            entry_price: Entry price
            stop_loss_price: Stop loss price

        Returns:
            PositionSize with calculated values
        """
        risk_amount = account_balance * (risk_percent / 100)

        if stop_loss_price == 0:
            stop_distance = 0
        else:
            stop_distance = abs(entry_price - stop_loss_price)

        if stop_distance == 0:
            units = 0
            lots = 0
        else:
            units = risk_amount / stop_distance
            lots = units / 100000  # Standard lot size

        return PositionSize(
            units=round(units, 2),
            lots=round(lots, 2),
            risk_percent=risk_percent,
            risk_amount=round(risk_amount, 2),
        )

    def calculate_var(
        self,
        returns: List[float],
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk (VaR) using historical method.

        Args:
            returns: List of historical returns
            confidence_level: Confidence level (0.95 or 0.99)

        Returns:
            VaR value at specified confidence level
        """
        if not returns:
            return 0.0

        import numpy as np

        sorted_returns = sorted(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[index] if index < len(sorted_returns) else sorted_returns[-1]

        return round(var, 4)

    def calculate_max_drawdown(
        self,
        equity_curve: List[float],
    ) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity_curve: List of equity values

        Returns:
            Maximum drawdown as percentage
        """
        if not equity_curve:
            return 0.0

        max_drawdown = 0.0
        peak = equity_curve[0]

        for value in equity_curve:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak * 100 if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return round(max_drawdown, 2)

    def get_current_drawdown(
        self,
        current_equity: float,
        peak_equity: float,
    ) -> DrawdownInfo:
        """
        Get current drawdown information.

        Args:
            current_equity: Current equity value
            peak_equity: Historical peak equity

        Returns:
            DrawdownInfo with current drawdown details
        """
        if peak_equity == 0:
            drawdown_percent = 0
        else:
            drawdown_percent = (peak_equity - current_equity) / peak_equity * 100

        return DrawdownInfo(
            current_drawdown_percent=round(drawdown_percent, 2),
            max_drawdown_percent=round(drawdown_percent, 2),  # Assuming current is max for now
            drawdown_duration_days=0,  # Would need historical data
            is_in_drawdown=current_equity < peak_equity,
        )

    def calculate_risk_reward_ratio(
        self,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
    ) -> float:
        """
        Calculate risk/reward ratio.

        Args:
            entry_price: Entry price
            take_profit: Take profit level
            stop_loss: Stop loss level

        Returns:
            Risk/reward ratio
        """
        if stop_loss == 0 or stop_loss == entry_price:
            return 0.0

        potential_profit = abs(take_profit - entry_price)
        potential_loss = abs(stop_loss - entry_price)

        if potential_loss == 0:
            return 0.0

        ratio = potential_profit / potential_loss
        return round(ratio, 2)

    def check_risk_limits(
        self,
        position_size: float,
        account_balance: float,
        max_position_percent: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Check if position is within risk limits (READ-ONLY check).

        Args:
            position_size: Position size in base currency
            account_balance: Account balance
            max_position_percent: Maximum position as % of balance

        Returns:
            Dictionary with check results
        """
        position_percent = (position_size / account_balance * 100) if account_balance > 0 else 0

        return {
            "position_percent": round(position_percent, 2),
            "max_allowed_percent": max_position_percent,
            "within_limits": position_percent <= max_position_percent,
            "excess_percent": round(max(0, position_percent - max_position_percent), 2),
        }

    def get_portfolio_risk_summary(
        self,
        positions: List[Dict[str, Any]],
        account_balance: float,
    ) -> Dict[str, Any]:
        """
        Get summary of portfolio risk (READ-ONLY).

        Args:
            positions: List of open positions
            account_balance: Account balance

        Returns:
            Portfolio risk summary
        """
        total_exposure = sum(p.get("exposure", 0) for p in positions)
        total_unrealized_pnl = sum(p.get("unrealized_pnl", 0) for p in positions)

        return {
            "total_positions": len(positions),
            "total_exposure": round(total_exposure, 2),
            "exposure_percent": round((total_exposure / account_balance * 100) if account_balance > 0 else 0, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "pnl_percent": round((total_unrealized_pnl / account_balance * 100) if account_balance > 0 else 0, 2),
        }

    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing.

        Args:
            win_rate: Win rate (0 to 1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount

        Returns:
            Kelly percentage (0 to 100)
        """
        if avg_loss == 0:
            return 0.0

        win_loss_ratio = avg_win / abs(avg_loss)
        kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

        # Kelly can be negative - return 0 in that case
        kelly_percent = max(0, kelly) * 100

        return round(kelly_percent, 2)

    def analyze_risk_metrics(
        self,
        returns: List[float],
        equity_curve: List[float],
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: List of historical returns
            equity_curve: List of equity values

        Returns:
            RiskMetrics with all calculated values
        """
        return RiskMetrics(
            var_95=self.calculate_var(returns, 0.95),
            var_99=self.calculate_var(returns, 0.99),
            max_drawdown=self.calculate_max_drawdown(equity_curve),
            current_drawdown=self.get_current_drawdown(
                equity_curve[-1] if equity_curve else 0,
                max(equity_curve) if equity_curve else 0,
            ).current_drawdown_percent,
            risk_reward_ratio=0.0,  # Would need trade-level data
        )
