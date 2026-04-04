"""
Trading Tilt REST API Endpoints

Provides HTTP fallback for tilt phase status when WebSocket is unavailable.
Used by SessionTimeline frontend component to display current tilt phase
with color coding (IDLE/LOCK/WAIT/RE_RANK/ACTIVATE).

Primary updates come via WebSocket (Redis pub/sub), but HTTP polling
is used as a fallback or for initial state hydration.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone
from typing import Optional
import logging

from src.events.tilt import TiltState, TiltPhase, TiltPhaseEvent, TILT_STATE_TO_PHASE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trading/tilt", tags=["trading", "tilt"])


# Module-level tilt state singleton
# This is set by the tilt subsystem during startup
_tilt_machine: Optional["TiltStateMachine"] = None


def set_tilt_machine(tilt_machine: "TiltStateMachine") -> None:
    """
    Set the global TiltStateMachine instance for HTTP endpoint access.

    Called during server startup by the tilt subsystem initialization.
    """
    global _tilt_machine
    _tilt_machine = tilt_machine
    logger.info(f"TiltStateMachine instance set for HTTP endpoints: {id(tilt_machine)}")


def get_tilt_machine() -> Optional["TiltStateMachine"]:
    """Get the global TiltStateMachine instance."""
    return _tilt_machine


@router.get("/status")
async def get_tilt_status():
    """
    Returns the current tilt phase status for UI display.

    This endpoint provides a synchronous HTTP fallback for when WebSocket
    is unavailable. The SessionTimeline uses WebSocket as primary (live updates)
    and falls back to polling this endpoint every 10 seconds.

    Response:
        phase: Current phase (IDLE, LOCK, WAIT, RE_RANK, ACTIVATE)
        time_in_phase_seconds: Seconds since entering current phase (approximate)
        next_transition: Estimated seconds until next transition (if in transition)
        regime_context: Closing -> Incoming session transition if active
        session_name: Current session name (closing session if in transition)
        state: Internal TiltState string (idle, lock, signal, wait, re_rank, activate, suspended)
    """
    try:
        tilt_machine = get_tilt_machine()

        if tilt_machine is None:
            # Tilt not initialized yet — return IDLE state
            return {
                "phase": "IDLE",
                "time_in_phase_seconds": 0,
                "next_transition": None,
                "regime_context": None,
                "session_name": None,
                "state": "idle",
            }

        # Get current phase event from tilt machine
        phase_event = tilt_machine.get_current_phase_event()

        # Calculate time in phase (approximate)
        time_in_phase = 0
        if phase_event.timestamp_utc:
            delta = datetime.now(timezone.utc) - phase_event.timestamp_utc
            time_in_phase = int(delta.total_seconds())

        # Determine next transition estimate
        next_transition = None
        if phase_event.phase != TiltPhase.ACTIVATE and phase_event.regime_persistence_timer > 0:
            # In WAIT phase — next transition is regime_persistence_timer seconds
            next_transition = phase_event.regime_persistence_timer

        # Build regime context
        regime_context = None
        if phase_event.closing_session and phase_event.incoming_session:
            regime_context = f"{phase_event.closing_session} -> {phase_event.incoming_session}"

        return {
            "phase": phase_event.phase.value if phase_event.phase else "IDLE",
            "time_in_phase_seconds": time_in_phase,
            "next_transition": next_transition,
            "regime_context": regime_context,
            "session_name": phase_event.closing_session or None,
            "state": phase_event.state.value if phase_event.state else "idle",
        }

    except Exception as e:
        logger.error(f"Error getting tilt status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tilt status: {str(e)}")


@router.get("/phase-event")
async def get_tilt_phase_event():
    """
    Returns the full TiltPhaseEvent for detailed UI display.

    Used when UI needs additional metadata beyond the basic /status response.
    """
    try:
        tilt_machine = get_tilt_machine()

        if tilt_machine is None:
            raise HTTPException(status_code=503, detail="Tilt subsystem not initialized")

        phase_event = tilt_machine.get_current_phase_event()

        return {
            "phase": phase_event.phase.value,
            "state": phase_event.state.value,
            "closing_session": phase_event.closing_session,
            "incoming_session": phase_event.incoming_session,
            "regime_persistence_timer": phase_event.regime_persistence_timer,
            "timestamp_utc": phase_event.timestamp_utc.isoformat() if phase_event.timestamp_utc else None,
            "metadata": phase_event.metadata,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tilt phase event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tilt phase event: {str(e)}")


# Regime persistence timer constant (matches the value in src/events/tilt.py)
REGIME_PERSISTENCE_SECONDS = 1800


@router.get("/regime-timer")
async def get_regime_timer():
    """
    Returns the current regime persistence timer state for UI display.

    Provides a dedicated endpoint for the Regime Persistence Timer Display
    in SessionTimeline, returning all data needed for the countdown badge.

    Response:
        regime_timer_seconds: Seconds remaining in WAIT phase (0 if not in WAIT)
        regime_timer_max: Maximum timer value (1800 seconds = 30 minutes)
        regime_name: Name of the incoming regime (incoming_session)
        action_pending: The next action to be taken (RE_RANK, ACTIVATE, etc.)
        next_action: Human-readable description of the pending action
    """
    try:
        tilt_machine = get_tilt_machine()

        if tilt_machine is None:
            return {
                "regime_timer_seconds": 0,
                "regime_timer_max": REGIME_PERSISTENCE_SECONDS,
                "regime_name": None,
                "action_pending": None,
                "next_action": None,
            }

        phase_event = tilt_machine.get_current_phase_event()
        timer_seconds = phase_event.regime_persistence_timer

        # Determine the next action based on current phase
        action_pending = None
        next_action = None

        if timer_seconds > 0:
            # In WAIT phase - next action is RE_RANK
            action_pending = "RE_RANK"
            next_action = "QUEUE RE-RANK"
        elif phase_event.phase.value in ("LOCK", "SIGNAL"):
            # Between LOCK and WAIT - regime confirmation pending
            action_pending = "WAIT"
            next_action = "REGIME CONFIRM"
        elif phase_event.phase.value == "RE_RANK":
            action_pending = "ACTIVATE"
            next_action = "ACTIVATE BOTS"

        return {
            "regime_timer_seconds": timer_seconds,
            "regime_timer_max": REGIME_PERSISTENCE_SECONDS,
            "regime_name": phase_event.incoming_session or None,
            "action_pending": action_pending,
            "next_action": next_action,
        }

    except Exception as e:
        logger.error(f"Error getting regime timer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get regime timer: {str(e)}")
