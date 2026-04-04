"""
Weekday Parameter Guard
=======================

Architecturally enforced weekday parameter change blocking.
Rejects any bot parameter changes on weekdays (Monday–Thursday).

Hard-block notification: "Weekday parameter updates are not permitted — weekend cycle only."

Reference: Story 8.13 (8-13-workflow-4-weekend-update-cycle) AC5
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Callable, Awaitable

logger = logging.getLogger(__name__)


class WeekdayBlockError(Exception):
    """Raised when a parameter change is attempted on a weekday."""

    def __init__(self, message: Optional[str] = None):
        if message is None:
            message = self._default_message()
        self.message = message
        super().__init__(self.message)

    def _default_message(self) -> str:
        """Generate default block message with current time context."""
        now = datetime.now(timezone.utc)
        day_name = now.strftime("%A")
        current_time = now.strftime("%H:%M GMT")
        return (
            f"Weekday parameter updates are not permitted — weekend cycle only. "
            f"Current time: {day_name} {current_time}. "
            f"Next allowed update window: Friday 21:00 GMT."
        )


@dataclass
class ParameterChange:
    """Represents a proposed bot parameter change."""
    bot_id: str
    parameter_name: str
    old_value: any
    new_value: any
    proposed_at: datetime
    proposed_by: str = "unknown"  # Agent or workflow that proposed the change


class WeekdayParameterGuard:
    """
    Weekday parameter change guard — architecturally enforced.

    Rejects any bot parameter changes on weekdays (Monday–Thursday).
    This is a HARD block - no exceptions allowed.

    Integration points:
    - Commander.set_bot_parameters()
    - RoutingMatrix.update_bot_config()
    - AlphaForge promotion workflow
    - Copilot natural language parameter changes
    - Workflow 1/2/3 parameter adjustment steps
    """

    # Weekdays that are blocked (Monday=0 through Thursday=3)
    BLOCKED_WEEKDAYS = [0, 1, 2, 3]

    # Weekend days (Friday=4, Saturday=5, Sunday=6)
    ALLOWED_WEEKEND_DAYS = [4, 5, 6]

    def __init__(self):
        self._enabled = True
        self._custom_block_message: Optional[str] = None
        self._registered_entry_points: List[str] = []
        logger.info("WeekdayParameterGuard initialized with weekday hard block")

    def enable(self) -> None:
        """Enable the weekday guard."""
        self._enabled = True
        logger.info("WeekdayParameterGuard enabled")

    def disable(self) -> None:
        """Disable the weekday guard (for testing only)."""
        self._enabled = False
        logger.warning("WeekdayParameterGuard disabled (testing mode)")

    def is_enabled(self) -> bool:
        """Check if guard is enabled."""
        return self._enabled

    def set_custom_block_message(self, message: str) -> None:
        """Set a custom block message."""
        self._custom_block_message = message

    def register_entry_point(self, entry_point: str) -> None:
        """Register an entry point for tracking."""
        if entry_point not in self._registered_entry_points:
            self._registered_entry_points.append(entry_point)
            logger.info(f"Registered weekday guard entry point: {entry_point}")

    def is_change_allowed(self, utc_now: Optional[datetime] = None) -> bool:
        """
        Check if parameter change is allowed at current time.

        Args:
            utc_now: Optional datetime to check (defaults to now)

        Returns:
            True if change is allowed (weekend), False if blocked (weekday)
        """
        if not self._enabled:
            return True

        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        # Check if today is a blocked weekday
        return utc_now.weekday() not in self.BLOCKED_WEEKDAYS

    def check_and_reject(self, change: ParameterChange) -> None:
        """
        Check if parameter change is allowed and reject if not.

        Args:
            change: The proposed parameter change

        Raises:
            WeekdayBlockError: If change is attempted on a weekday
        """
        if not self.is_enabled():
            return

        utc_now = change.proposed_at or datetime.now(timezone.utc)

        if not self.is_change_allowed(utc_now):
            logger.warning(
                f"Weekday parameter change BLOCKED: bot={change.bot_id}, "
                f"param={change.parameter_name}, proposed_by={change.proposed_by}, "
                f"time={utc_now.strftime('%A %H:%M GMT')}"
            )
            raise WeekdayBlockError(self._custom_block_message)

    def reject_change(
        self,
        bot_id: str,
        parameter_name: str,
        proposed_by: str = "unknown",
        utc_now: Optional[datetime] = None
    ) -> None:
        """
        Reject a parameter change with hard-block notification.

        Args:
            bot_id: Bot ID
            parameter_name: Parameter being changed
            proposed_by: Who/what proposed the change
            utc_now: Optional time to check

        Raises:
            WeekdayBlockError: Always raises with block message
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        logger.warning(
            f"Weekday parameter change BLOCKED: bot={bot_id}, "
            f"param={parameter_name}, proposed_by={proposed_by}, "
            f"time={utc_now.strftime('%A %H:%M GMT')}"
        )

        raise WeekdayBlockError(self._custom_block_message)

    def get_block_status(self, utc_now: Optional[datetime] = None) -> dict:
        """
        Get current block status for UI display.

        Args:
            utc_now: Optional datetime to check

        Returns:
            Dict with block status information
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        is_blocked = not self.is_change_allowed(utc_now)
        day_name = utc_now.strftime("%A")

        # Calculate next allowed window
        if is_blocked:
            # Next allowed is Friday 21:00 GMT
            days_until_friday = (4 - utc_now.weekday()) % 7
            if days_until_friday == 0 and utc_now.hour >= 21:
                days_until_friday = 7  # After Friday 21:00, wait for next week
            elif days_until_friday == 0:
                next_allowed = utc_now.replace(hour=21, minute=0, second=0, microsecond=0)
            else:
                next_friday = utc_now.replace(day=utc_now.day + days_until_friday, hour=21, minute=0, second=0, microsecond=0)
                next_allowed = next_friday
        else:
            next_allowed = None

        return {
            "is_blocked": is_blocked,
            "current_day": day_name,
            "current_time": utc_now.strftime("%H:%M GMT"),
            "next_allowed_window": next_allowed.isoformat() if next_allowed else None,
            "message": (
                "Weekday parameter updates are not permitted — weekend cycle only."
                if is_blocked
                else "Parameter changes allowed (weekend)"
            ),
            "registered_entry_points": self._registered_entry_points,
        }


# ============= Singleton Factory =============
_guard_instance: Optional[WeekdayParameterGuard] = None


def get_weekday_parameter_guard() -> WeekdayParameterGuard:
    """Get singleton instance of WeekdayParameterGuard."""
    global _guard_instance
    if _guard_instance is None:
        _guard_instance = WeekdayParameterGuard()
    return _guard_instance


# ============= Integration Helper =============
async def guard_parameter_change(
    bot_id: str,
    parameter_name: str,
    proposed_by: str = "unknown"
) -> None:
    """
    Guard a parameter change entry point.

    Call this at the start of any parameter change function to enforce
    weekday blocking.

    Args:
        bot_id: Bot ID
        parameter_name: Parameter being changed
        proposed_by: Who/what proposed the change

    Raises:
        WeekdayBlockError: If change is attempted on a weekday
    """
    guard = get_weekday_parameter_guard()
    change = ParameterChange(
        bot_id=bot_id,
        parameter_name=parameter_name,
        old_value=None,
        new_value=None,
        proposed_at=datetime.now(timezone.utc),
        proposed_by=proposed_by,
    )
    guard.check_and_reject(change)
