"""
Command Handler Module

Provides command handling functionality for the Commander.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Handles command processing and dispatching for the trading system.

    Responsible for parsing, validating, and executing commands
    related to trading operations.
    """

    def __init__(self, commander=None):
        """
        Initialize CommandHandler.

        Args:
            commander: Optional Commander instance for delegation
        """
        self._commander = commander
        self._command_history: List[Dict[str, Any]] = []

    def parse_command(self, instruction: str) -> Dict[str, Any]:
        """
        Parse a command instruction into structured format.

        Args:
            instruction: Raw command string

        Returns:
            Parsed command dictionary
        """
        parts = instruction.strip().split()
        if not parts:
            return {"action": "unknown", "params": {}}

        action = parts[0].lower()
        params = {}

        # Parse key=value pairs
        for part in parts[1:]:
            if "=" in part:
                key, value = part.split("=", 1)
                params[key] = value

        return {"action": action, "params": params}

    def validate_command(self, command: Dict[str, Any]) -> bool:
        """
        Validate a parsed command.

        Args:
            command: Parsed command dictionary

        Returns:
            True if valid, False otherwise
        """
        if "action" not in command:
            return False

        valid_actions = {"trade", "risk", "status", "stop", "start", "cancel"}
        return command["action"] in valid_actions

    def execute_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a validated command.

        Args:
            command: Validated command dictionary

        Returns:
            Execution result
        """
        if not self.validate_command(command):
            return {"success": False, "error": "Invalid command"}

        # Record in history
        self._command_history.append({
            "command": command,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        action = command["action"]
        params = command.get("params", {})

        if action == "trade":
            return self._execute_trade_command(params)
        elif action == "risk":
            return self._execute_risk_command(params)
        elif action == "status":
            return self._execute_status_command(params)
        elif action == "stop":
            return self._execute_stop_command(params)
        elif action == "start":
            return self._execute_start_command(params)
        elif action == "cancel":
            return self._execute_cancel_command(params)

        return {"success": False, "error": "Unknown action"}

    def _execute_trade_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade command."""
        return {"success": True, "action": "trade", "params": params}

    def _execute_risk_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a risk command."""
        return {"success": True, "action": "risk", "params": params}

    def _execute_status_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a status command."""
        return {"success": True, "action": "status", "params": params}

    def _execute_stop_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a stop command."""
        return {"success": True, "action": "stop", "params": params}

    def _execute_start_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a start command."""
        return {"success": True, "action": "start", "params": params}

    def _execute_cancel_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a cancel command."""
        return {"success": True, "action": "cancel", "params": params}

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get command execution history.

        Returns:
            List of executed commands
        """
        return self._command_history.copy()

    def clear_history(self) -> None:
        """Clear command history."""
        self._command_history.clear()
