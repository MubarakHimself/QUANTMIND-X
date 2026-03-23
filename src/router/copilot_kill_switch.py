"""
Copilot Kill Switch

Kill switch for Copilot/agent activity - independent from trading kill switch.
This controls AI/agent activity without affecting live trading.

Architecturally INDEPENDENT from src/router/kill_switch.py (Trading Kill Switch).
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# Lazy import to avoid circular dependency
_floor_manager_instance: Optional[Any] = None


def _get_floor_manager():
    """Get FloorManager instance for task cancellation."""
    global _floor_manager_instance
    if _floor_manager_instance is None:
        try:
            from src.agents.departments.floor_manager import FloorManager
            # Get singleton instance if available
            _floor_manager_instance = FloorManager.get_instance() if hasattr(FloorManager, 'get_instance') else None
        except ImportError:
            logger.warning("FloorManager not available for task cancellation")
            _floor_manager_instance = None
    return _floor_manager_instance


class CopilotKillReason(Enum):
    """Reasons for triggering copilot kill switch."""
    MANUAL = "MANUAL"
    API_COMMAND = "API_COMMAND"


@dataclass
class CopilotKillEvent:
    """Record of a copilot kill switch activation."""
    timestamp: datetime
    reason: CopilotKillReason
    activated_by: str
    message: str
    terminated_tasks: List[str] = field(default_factory=list)


class CopilotKillSwitch:
    """
    Kill switch for Copilot/agent activity - independent from trading kill switch.

    This kill switch:
    - Terminates all running FloorManager and department agent tasks
    - Prevents new task starts while active
    - Preserves partial results with "Task interrupted" marker
    - Does NOT affect live trading
    """

    def __init__(self):
        self._active = False
        self._terminated_tasks: List[str] = []
        self._suspended_at_utc: Optional[datetime] = None
        self._activated_by: Optional[str] = None
        self._kill_history: List[CopilotKillEvent] = []

    @property
    def is_active(self) -> bool:
        """Check if kill switch is active."""
        return self._active

    async def activate(self, activator: str = "system") -> Dict[str, Any]:
        """
        Activate copilot kill switch - terminate all running agent tasks.

        Args:
            activator: Who activated the kill switch (e.g., "user", "api")

        Returns:
            Dict with success status and details
        """
        if self._active:
            logger.warning("Copilot kill switch already active")
            return {
                "success": True,
                "already_active": True,
                "suspended_at_utc": self._suspended_at_utc.isoformat() if self._suspended_at_utc else None
            }

        self._active = True
        self._suspended_at_utc = datetime.utcnow()
        self._activated_by = activator

        logger.warning(f"🚨 COPILOT KILL SWITCH ACTIVATED by {activator}")

        # Cancel active department agent tasks (Story 5.6)
        floor_manager = _get_floor_manager()
        if floor_manager and hasattr(floor_manager, '_cancel_active_tasks'):
            self._terminated_tasks = floor_manager._cancel_active_tasks()
            logger.info(f"Cancelled {len(self._terminated_tasks)} active tasks")

        # Clean up draft nodes in graph memory (Story 5.6)
        await self.cleanup_draft_nodes()

        # Create event record
        event = CopilotKillEvent(
            timestamp=self._suspended_at_utc,
            reason=CopilotKillReason.MANUAL if activator == "user" else CopilotKillReason.API_COMMAND,
            activated_by=activator,
            message="Copilot kill switch activated - agent activity suspended",
            terminated_tasks=self._terminated_tasks.copy()
        )
        self._kill_history.append(event)

        return {
            "success": True,
            "suspended_at_utc": self._suspended_at_utc.isoformat(),
            "activated_by": activator,
            "terminated_tasks": self._terminated_tasks
        }

    async def resume(self) -> Dict[str, Any]:
        """
        Resume copilot - reactivate agent system.

        Returns:
            Dict with success status and details
        """
        if not self._active:
            logger.warning("Copilot kill switch not active, nothing to resume")
            return {
                "success": True,
                "not_active": True
            }

        self._active = False
        self._terminated_tasks.clear()
        self._activated_by = None
        resumed_at = datetime.utcnow()

        logger.info(f"🚀 COPILOT KILL SWITCH RESUMED at {resumed_at.isoformat()}")

        return {
            "success": True,
            "resumed_at_utc": resumed_at.isoformat()
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get current kill switch status.

        Returns:
            Dict with current status
        """
        return {
            "active": self._active,
            "suspended_at_utc": self._suspended_at_utc.isoformat() if self._suspended_at_utc else None,
            "activated_by": self._activated_by,
            "terminated_tasks_count": len(self._terminated_tasks)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get kill switch activation history."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "reason": e.reason.value,
                "activated_by": e.activated_by,
                "message": e.message,
                "terminated_tasks": e.terminated_tasks
            }
            for e in self._kill_history
        ]

    async def cleanup_draft_nodes(self) -> Dict[str, Any]:
        """
        Clean up or mark draft nodes in graph memory when kill switch activates.

        This prevents partial state changes from being committed to graph memory.

        Returns:
            Dict with cleanup status
        """
        if not self._active:
            return {"success": True, "not_active": True, "cleaned_nodes": 0}

        # Try to mark draft nodes as interrupted
        try:
            from src.memory.graph.facade import GraphMemoryFacade
            facade = GraphMemoryFacade()
            # Mark all draft sessions as interrupted
            cleaned = await facade.mark_draft_nodes_interrupted(self._suspended_at_utc)
            logger.info(f"Marked {cleaned} draft nodes as interrupted")
            return {"success": True, "cleaned_nodes": cleaned}
        except ImportError:
            logger.warning("GraphMemoryFacade not available for draft node cleanup")
            return {"success": True, "cleaned_nodes": 0, "not_available": True}
        except Exception as e:
            logger.error(f"Error cleaning up draft nodes: {e}")
            return {"success": False, "error": str(e)}


# Global singleton instance
_copilot_kill_switch: Optional[CopilotKillSwitch] = None


def get_copilot_kill_switch() -> CopilotKillSwitch:
    """Get or create the global copilot kill switch instance."""
    global _copilot_kill_switch
    if _copilot_kill_switch is None:
        _copilot_kill_switch = CopilotKillSwitch()
    return _copilot_kill_switch
