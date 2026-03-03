"""
Risk Command Module

Provides risk management commands for the Commander.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RiskCommand:
    """
    Handles risk management commands.

    Manages risk checks, limits, and position sizing
    through the trading system.
    """

    def __init__(self, commander=None):
        """
        Initialize RiskCommand.

        Args:
            commander: Optional Commander instance for delegation
        """
        self._commander = commander
        self._risk_limits: Dict[str, float] = {
            "max_position_size": 10.0,
            "max_daily_loss": 1000.0,
            "max_drawdown": 0.15,
            "max_open_positions": 20
        }
        self._risk_checks_enabled = True

    def check_risk(self, trade_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if a trade passes risk validation.

        Args:
            trade_params: Trade parameters to validate

        Returns:
            Risk check result
        """
        if not self._risk_checks_enabled:
            return {"passed": True, "reason": "Risk checks disabled"}

        # Check position size
        volume = trade_params.get("volume", 0)
        if volume > self._risk_limits["max_position_size"]:
            return {
                "passed": False,
                "reason": f"Position size {volume} exceeds limit {self._risk_limits['max_position_size']}"
            }

        # Check max open positions
        if self._commander:
            active_count = len(getattr(self._commander, 'active_bots', {}))
            if active_count >= self._risk_limits["max_open_positions"]:
                return {
                    "passed": False,
                    "reason": f"Max open positions {self._risk_limits['max_open_positions']} reached"
                }

        return {"passed": True, "reason": "All risk checks passed"}

    def calculate_position_size(self, account_balance: float,
                                risk_per_trade: float,
                                stop_loss_pips: float) -> float:
        """
        Calculate position size based on risk parameters.

        Args:
            account_balance: Account balance
            risk_per_trade: Risk percentage per trade (e.g., 0.02 for 2%)
            stop_loss_pips: Stop loss in pips

        Returns:
            Calculated position size
        """
        if stop_loss_pips <= 0:
            return 0.0

        risk_amount = account_balance * risk_per_trade
        # Simplified pip value calculation
        pip_value = 10.0  # Default for most currency pairs
        position_size = risk_amount / (stop_loss_pips * pip_value)

        # Apply max position size limit
        return min(position_size, self._risk_limits["max_position_size"])

    def set_risk_limit(self, limit_name: str, value: float) -> Dict[str, Any]:
        """
        Set a risk limit value.

        Args:
            limit_name: Name of the risk limit
            value: New limit value

        Returns:
            Result of setting the limit
        """
        if limit_name not in self._risk_limits:
            return {"success": False, "error": f"Unknown limit: {limit_name}"}

        self._risk_limits[limit_name] = value
        logger.info(f"RiskCommand: Set {limit_name} = {value}")

        return {"success": True, "limit_name": limit_name, "value": value}

    def get_risk_limits(self) -> Dict[str, float]:
        """
        Get all risk limits.

        Returns:
            Dictionary of risk limits
        """
        return self._risk_limits.copy()

    def enable_risk_checks(self) -> None:
        """Enable risk checks."""
        self._risk_checks_enabled = True
        logger.info("RiskCommand: Risk checks enabled")

    def disable_risk_checks(self) -> None:
        """Disable risk checks."""
        self._risk_checks_enabled = False
        logger.info("RiskCommand: Risk checks disabled")

    def check_daily_loss(self, current_loss: float) -> Dict[str, Any]:
        """
        Check if daily loss limit has been reached.

        Args:
            current_loss: Current daily loss amount

        Returns:
            Check result
        """
        max_loss = self._risk_limits["max_daily_loss"]

        if abs(current_loss) >= max_loss:
            return {
                "limit_reached": True,
                "current_loss": current_loss,
                "max_loss": max_loss,
                "action": "stop_trading"
            }

        return {
            "limit_reached": False,
            "current_loss": current_loss,
            "max_loss": max_loss,
            "remaining": max_loss - abs(current_loss)
        }

    def check_drawdown(self, current_drawdown: float) -> Dict[str, Any]:
        """
        Check if drawdown limit has been reached.

        Args:
            current_drawdown: Current drawdown percentage

        Returns:
            Check result
        """
        max_drawdown = self._risk_limits["max_drawdown"]

        if current_drawdown >= max_drawdown:
            return {
                "limit_reached": True,
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown,
                "action": "stop_trading"
            }

        return {
            "limit_reached": False,
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "remaining": max_drawdown - current_drawdown
        }
