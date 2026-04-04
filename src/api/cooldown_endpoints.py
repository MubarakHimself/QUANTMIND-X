"""
Inter-Session Cooldown REST API Endpoints

Provides HTTP fallback for cooldown phase status when WebSocket is unavailable.
Used by InterSessionCooldownPanel frontend component to display:
- Current session -> next session transition
- Countdown timer (hours:minutes)
- Intelligence window status (ACTIVE/SLEEPING)
- Actions blocked indicator
- Progress bar showing cooldown elapsed/remaining

Primary updates come via Redis pub/sub, but HTTP polling is used as a
fallback or for initial state hydration.

Story 16.3: Inter-Session Cooldown Window (10:00-13:00 GMT)
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Optional
import logging

from src.events.cooldown import (
    CooldownState,
    CooldownPhase,
    COOLDOWN_STATE_TO_PHASE,
    InterSessionCooldownStateEvent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading/cooldown", tags=["trading", "cooldown"])


# Module-level cooldown state singleton
# This is set by the cooldown subsystem during startup
_cooldown_machine: Optional["InterSessionCooldownOrchestrator"] = None


def set_cooldown_machine(cooldown_machine: "InterSessionCooldownOrchestrator") -> None:
    """
    Set the global InterSessionCooldownOrchestrator instance for HTTP endpoint access.

    Called during server startup by the cooldown subsystem initialization.
    """
    global _cooldown_machine
    _cooldown_machine = cooldown_machine
    logger.info(f"InterSessionCooldownOrchestrator instance set for HTTP endpoints: {id(cooldown_machine)}")


def get_cooldown_machine() -> Optional["InterSessionCooldownOrchestrator"]:
    """Get the global InterSessionCooldownOrchestrator instance."""
    return _cooldown_machine


def _is_within_cooldown_window() -> bool:
    """
    Check if current UTC time is within the 10:00-13:00 GMT cooldown window.

    Returns:
        True if within cooldown window, False otherwise
    """
    now = datetime.now(timezone.utc)
    current_hour = now.hour
    # Cooldown window: 10:00-13:00 GMT
    return 10 <= current_hour < 13


def _compute_time_remaining(window_end: Optional[datetime]) -> tuple[int, int]:
    """
    Compute hours and minutes remaining until window_end.

    Args:
        window_end: The cooldown window end datetime

    Returns:
        Tuple of (hours_remaining, minutes_remaining)
    """
    if not window_end:
        return 0, 0

    now = datetime.now(timezone.utc)
    if now >= window_end:
        return 0, 0

    delta = window_end - now
    total_seconds = int(delta.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return hours, minutes


def _compute_progress(window_start: Optional[datetime], window_end: Optional[datetime]) -> float:
    """
    Compute cooldown progress as a percentage (0.0 to 1.0).

    Args:
        window_start: The cooldown window start datetime
        window_end: The cooldown window end datetime

    Returns:
        Progress as a float between 0.0 and 1.0
    """
    if not window_start or not window_end:
        return 0.0

    now = datetime.now(timezone.utc)
    if now <= window_start:
        return 0.0
    if now >= window_end:
        return 1.0

    total_duration = (window_end - window_start).total_seconds()
    elapsed = (now - window_start).total_seconds()

    return min(1.0, max(0.0, elapsed / total_duration))


@router.get("/status")
async def get_cooldown_status():
    """
    Returns the current Inter-Session Cooldown status for UI display.

    This endpoint provides a synchronous HTTP fallback for when Redis pub/sub
    is unavailable. The InterSessionCooldownPanel uses this for polling fallback
    and initial state hydration.

    Response:
        is_active: Whether cooldown is currently active (within 10:00-13:00 GMT window)
        session_transition: "LONDON -> NEW_YORK" during cooldown, None otherwise
        cooldown_end_time: ISO timestamp when cooldown ends (13:00 GMT), None if not active
        hours_remaining: Whole hours until cooldown end
        minutes_remaining: Remaining minutes after hours
        current_session: "LONDON" (closing session)
        next_session: "NEW_YORK" (incoming session)
        intelligence_window_active: True when cooldown is in progress
        actions_blocked: True when new trades are blocked during cooldown
        progress: Float 0.0-1.0 representing cooldown elapsed fraction
        state: Internal CooldownState string
        step_name: Human-readable current step name
        current_step: Step number (0-4)
        window_start: ISO timestamp when cooldown started
        window_end: ISO timestamp when cooldown ends
        ny_roster_locked: Whether NY roster has been locked
    """
    try:
        cooldown_machine = get_cooldown_machine()

        if cooldown_machine is None:
            # Cooldown subsystem not initialized — determine state from time
            in_window = _is_within_cooldown_window()

            if in_window:
                # Within cooldown window but orchestrator not running
                # Compute expected window times for today
                now = datetime.now(timezone.utc)
                window_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
                window_end = now.replace(hour=13, minute=0, second=0, microsecond=0)
                hours_remaining, minutes_remaining = _compute_time_remaining(window_end)
                progress = _compute_progress(window_start, window_end)

                return {
                    "is_active": True,
                    "session_transition": "LONDON -> NEW_YORK",
                    "cooldown_end_time": window_end.isoformat(),
                    "hours_remaining": hours_remaining,
                    "minutes_remaining": minutes_remaining,
                    "current_session": "LONDON",
                    "next_session": "NEW_YORK",
                    "intelligence_window_active": True,
                    "actions_blocked": True,
                    "progress": progress,
                    "state": "pending",
                    "step_name": "Cooldown Active",
                    "current_step": 0,
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "ny_roster_locked": False,
                }
            else:
                # Outside cooldown window
                return {
                    "is_active": False,
                    "session_transition": None,
                    "cooldown_end_time": None,
                    "hours_remaining": 0,
                    "minutes_remaining": 0,
                    "current_session": None,
                    "next_session": None,
                    "intelligence_window_active": False,
                    "actions_blocked": False,
                    "progress": 0.0,
                    "state": "outside_window",
                    "step_name": None,
                    "current_step": 0,
                    "window_start": None,
                    "window_end": None,
                    "ny_roster_locked": False,
                }

        # Orchestrator is available — get real state
        state_event = cooldown_machine.get_current_state_event()
        window_start = state_event.window_start
        window_end = state_event.window_end

        # Determine if cooldown is currently active
        is_active = cooldown_machine.is_running or (
            cooldown_machine.state == CooldownState.PENDING and _is_within_cooldown_window()
        )

        # Compute time remaining
        hours_remaining, minutes_remaining = _compute_time_remaining(window_end)

        # Compute progress
        progress = _compute_progress(window_start, window_end)

        # Determine session transition
        session_transition = None
        current_session = None
        next_session = None

        if is_active:
            session_transition = "LONDON -> NEW_YORK"
            current_session = "LONDON"
            next_session = "NEW_YORK"

        # Determine actions blocked
        # Actions are blocked when cooldown is running (not PENDING, not COMPLETED)
        actions_blocked = cooldown_machine.is_running

        # Determine intelligence window active
        intelligence_window_active = cooldown_machine.is_running or (
            cooldown_machine.state == CooldownState.PENDING and _is_within_cooldown_window()
        )

        return {
            "is_active": is_active,
            "session_transition": session_transition,
            "cooldown_end_time": window_end.isoformat() if window_end else None,
            "hours_remaining": hours_remaining,
            "minutes_remaining": minutes_remaining,
            "current_session": current_session,
            "next_session": next_session,
            "intelligence_window_active": intelligence_window_active,
            "actions_blocked": actions_blocked,
            "progress": progress,
            "state": state_event.state.value,
            "step_name": state_event.step_name,
            "current_step": state_event.current_step,
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "ny_roster_locked": state_event.ny_roster_locked,
        }

    except Exception as e:
        logger.error(f"Error getting cooldown status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cooldown status: {str(e)}")


@router.get("/state-event")
async def get_cooldown_state_event():
    """
    Returns the full InterSessionCooldownStateEvent for detailed UI display.

    Used when UI needs additional metadata beyond the basic /status response.
    """
    try:
        cooldown_machine = get_cooldown_machine()

        if cooldown_machine is None:
            raise HTTPException(status_code=503, detail="Cooldown subsystem not initialized")

        state_event = cooldown_machine.get_current_state_event()

        return {
            "state": state_event.state.value,
            "current_step": state_event.current_step,
            "step_name": state_event.step_name,
            "window_start": state_event.window_start.isoformat() if state_event.window_start else None,
            "window_end": state_event.window_end.isoformat() if state_event.window_end else None,
            "ny_roster_locked": state_event.ny_roster_locked,
            "timestamp_utc": state_event.timestamp_utc.isoformat() if state_event.timestamp_utc else None,
            "metadata": state_event.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cooldown state event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cooldown state event: {str(e)}")
