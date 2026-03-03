"""
Trade Command Module

Provides trade execution commands for the Commander.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TradeCommand:
    """
    Handles trade-related commands and execution.

    Manages trade execution, modification, and cancellation
    through the trading system.
    """

    def __init__(self, commander=None):
        """
        Initialize TradeCommand.

        Args:
            commander: Optional Commander instance for delegation
        """
        self._commander = commander
        self._active_trades: Dict[str, Dict[str, Any]] = {}

    def execute_trade(self, symbol: str, action: str, volume: float,
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a trade command.

        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            volume: Trade volume
            params: Additional trade parameters

        Returns:
            Execution result
        """
        params = params or {}
        trade_id = f"{symbol}_{datetime.now(timezone.utc).timestamp()}"

        trade_data = {
            "id": trade_id,
            "symbol": symbol,
            "action": action,
            "volume": volume,
            "params": params,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        self._active_trades[trade_id] = trade_data

        logger.info(f"TradeCommand: Executing {action} {volume} {symbol}")

        return {
            "success": True,
            "trade_id": trade_id,
            "symbol": symbol,
            "action": action,
            "volume": volume
        }

    def modify_trade(self, trade_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify an existing trade.

        Args:
            trade_id: Trade identifier
            params: Parameters to modify

        Returns:
            Modification result
        """
        if trade_id not in self._active_trades:
            return {"success": False, "error": "Trade not found"}

        trade = self._active_trades[trade_id]
        trade["params"].update(params)
        trade["modified_at"] = datetime.now(timezone.utc).isoformat()

        return {"success": True, "trade_id": trade_id, "params": params}

    def cancel_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Cancel a pending trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Cancellation result
        """
        if trade_id not in self._active_trades:
            return {"success": False, "error": "Trade not found"}

        trade = self._active_trades[trade_id]
        if trade["status"] != "pending":
            return {"success": False, "error": "Cannot cancel non-pending trade"}

        trade["status"] = "cancelled"
        trade["cancelled_at"] = datetime.now(timezone.utc).isoformat()

        return {"success": True, "trade_id": trade_id}

    def get_trade(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """
        Get trade details.

        Args:
            trade_id: Trade identifier

        Returns:
            Trade data or None if not found
        """
        return self._active_trades.get(trade_id)

    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get all active trades.

        Returns:
            List of active trades
        """
        return [
            trade for trade in self._active_trades.values()
            if trade["status"] == "pending"
        ]

    def close_trade(self, trade_id: str) -> Dict[str, Any]:
        """
        Close an existing trade.

        Args:
            trade_id: Trade identifier

        Returns:
            Close result
        """
        if trade_id not in self._active_trades:
            return {"success": False, "error": "Trade not found"}

        trade = self._active_trades[trade_id]
        trade["status"] = "closed"
        trade["closed_at"] = datetime.now(timezone.utc).isoformat()

        return {"success": True, "trade_id": trade_id}
