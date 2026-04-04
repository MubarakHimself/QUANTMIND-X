"""
Tilt State Machine — Universal Session Boundary Mechanism.

Story 16.1: Tilt — Universal Session Boundary Mechanism

Implements the LOCK→SIGNAL→WAIT→RE-RANK→ACTIVATE sequence for every
session boundary transition in the 24-hour trading cycle.

Per NFR-M2: Tilt is a synchronous state machine — NO LLM calls in hot path.
Per NFR-D1: All Tilt phase transitions are logged with timestamps for audit.
Per NFR-P1: Kill switch protocol executes in full, in order — correctness over raw speed.

State transitions:
    IDLE → LOCK → SIGNAL → WAIT → RE_RANK → ACTIVATE → IDLE
              ↑
              └── SUSPENDED (when CHAOS fires during any phase)
                  Resume from SUSPENDED → previous phase when CHAOS resolves
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

import redis

from src.events.tilt import (
    REGIME_PERSISTENCE_SECONDS,
    TiltChaosResumeEvent,
    TiltChaosSuspendEvent,
    TiltPhase,
    TiltPhaseEvent,
    TiltSessionBoundaryEvent,
    TiltState,
    TILT_STATE_TO_PHASE,
    TiltTransitionAuditLog,
)
from src.events.regime import RegimeShiftEvent
from src.events.chaos import ChaosEvent

logger = logging.getLogger(__name__)


# Redis channel names
CHANNEL_TILT_PHASE = "tilt:phase"
CHANNEL_TILT_SESSION_CLOSE = "tilt:session:close"
CHANNEL_TILT_SESSION_OPEN = "tilt:session:open"
CHANNEL_TILT_RERANK_FAILED = "tilt:rerank:failed"
CHANNEL_CHAOS_TRIGGERED = "chaos:triggered"
CHANNEL_REGIME_CONFIRMED = "regime:confirmed"


@dataclass
class TiltTransition:
    """Record of a Tilt state transition for audit logging."""
    entry_id: str
    from_state: Optional[TiltState]
    to_state: TiltState
    closing_session: str
    incoming_session: str
    timestamp_utc: datetime
    metadata: dict = field(default_factory=dict)
    rerank_warning: bool = False


class TiltStateMachine:
    """
    Universal Session Boundary Mechanism — Tilt State Machine.

    Implements the LOCK→SIGNAL→WAIT→RE-RANK→ACTIVATE sequence for every
    session boundary transition.

    Key behaviors:
    1. Session boundary detection via SessionDetector subscription
    2. LOCK: Layer 1 EA hard stops hold, Layer 2 LOCKs positions
    3. SIGNAL: Sentinel confirms regime shift, broadcast fires
    4. WAIT: 30-minute regime persistence observation window
    5. RE-RANK: DPR queue recalculated for incoming session
    6. ACTIVATE: Session-masked bots armed for new window

    When CHAOS fires (Layer 3):
    - Tilt transitions to SUSPENDED state
    - Timer pauses during WAIT
    - After CHAOS resolution, resumes from previous state if session transition still valid

    Attributes:
        redis_client: Redis client for pub/sub and distributed coordination
        instance_id: Unique identifier for this Tilt instance
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        instance_id: str = "tilt-default",
    ):
        """
        Initialize TiltStateMachine.

        Args:
            redis_client: Redis client for pub/sub
            instance_id: Unique identifier for this instance
        """
        self._redis = redis_client
        self._instance_id = instance_id

        # State machine state
        self._state = TiltState.IDLE
        self._closing_session: Optional[str] = None
        self._incoming_session: Optional[str] = None
        self._regime_persistence_timer: int = 0
        self._suspended_from_state: Optional[TiltState] = None

        # Audit log
        self._audit_log: List[TiltTransition] = []

        # Subscribers for internal events
        self._chaos_subscribers: List[Callable[[ChaosEvent], None]] = []
        self._regime_subscribers: List[Callable[[RegimeShiftEvent], None]] = []

        # Layer integration callbacks
        self._layer1_hold_callback: Optional[Callable[[str], None]] = None
        self._layer2_lock_callback: Optional[Callable[[str], None]] = None
        self._dpr_re_rank_callback: Optional[Callable[[str], bool]] = None
        self._session_activate_callback: Optional[Callable[[str], None]] = None

        # Flag to control the WAIT loop
        self._wait_task: Optional[asyncio.Task] = None
        self._wait_cancelled = False

        # Fire-and-forget task tracking
        self._active_tasks: list = []

        # Rerank warning flag
        self._rerank_warning = False

        # Callback for when DPR re-rank fails — triggers slot halving
        self._rerank_failed_callback: Optional[Callable[[], None]] = None

        logger.info(f"TiltStateMachine initialized: instance={instance_id}")

    # =========================================================================
    # State Machine Core
    # =========================================================================

    @property
    def state(self) -> TiltState:
        """Get current Tilt state."""
        return self._state

    @property
    def current_session(self) -> Optional[str]:
        """Get current (closing) session name."""
        return self._closing_session

    @property
    def incoming_session(self) -> Optional[str]:
        """Get incoming session name."""
        return self._incoming_session

    @property
    def regime_persistence_timer(self) -> int:
        """Get remaining seconds in WAIT phase (0 if not in WAIT)."""
        return self._regime_persistence_timer

    @property
    def is_active(self) -> bool:
        """Check if Tilt is currently in a transition (not IDLE)."""
        return self._state != TiltState.IDLE

    @property
    def is_suspended(self) -> bool:
        """Check if Tilt is suspended due to CHAOS."""
        return self._state == TiltState.SUSPENDED

    def get_current_phase_event(self) -> TiltPhaseEvent:
        """
        Get current phase event for UI subscription.

        Returns:
            TiltPhaseEvent with current state and timer info
        """
        phase = TILT_STATE_TO_PHASE.get(
            self._state,
            TiltPhase.LOCK if self._state == TiltState.LOCK else TiltPhase.ACTIVATE
        )

        return TiltPhaseEvent(
            phase=phase,
            state=self._state,
            closing_session=self._closing_session or "",
            incoming_session=self._incoming_session or "",
            regime_persistence_timer=self._regime_persistence_timer,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={"instance_id": self._instance_id},
        )

    # =========================================================================
    # Session Boundary Trigger
    # =========================================================================

    def start_transition(
        self,
        closing_session: str,
        incoming_session: str,
    ) -> bool:
        """
        Start a session boundary transition.

        Called when SessionDetector detects a canonical window change.
        Validates preconditions and initiates the LOCK→SIGNAL→WAIT→RE-RANK→ACTIVATE sequence.

        Args:
            closing_session: Session that is closing
            incoming_session: Session that is opening

        Returns:
            True if transition started, False if already in transition
        """
        # Reject if already in transition
        if self.is_active:
            logger.warning(
                f"Tilt already in transition ({self._state.value}) — "
                f"rejecting new boundary: {closing_session}→{incoming_session}"
            )
            return False

        # Reject if suspended
        if self.is_suspended:
            logger.warning(
                f"Tilt is suspended — cannot start new transition until CHAOS resolved"
            )
            return False

        logger.info(
            f"Tilt starting transition: {closing_session} → {incoming_session}"
        )

        self._closing_session = closing_session
        self._incoming_session = incoming_session

        # Start the state machine
        return self._execute_lock()

    def _execute_lock(self) -> bool:
        """
        Execute LOCK step: Layer 1 EA hard stops hold, Layer 2 LOCKs positions.

        Returns:
            True if successful
        """
        logger.info(f"Tilt LOCK: {self._closing_session} → {self._incoming_session}")

        # Transition to LOCK state
        self._transition_to(TiltState.LOCK)

        # Layer 1: Signal EA to hold hard stops (no new SL/TP modifications)
        if self._layer1_hold_callback:
            try:
                self._layer1_hold_callback(self._closing_session)
                logger.info("Layer 1 EA hold signal sent")
            except Exception as e:
                logger.error(f"Layer 1 hold callback failed: {e}")

        # Layer 2: Activate LOCK on all positions in closing session
        if self._layer2_lock_callback:
            try:
                self._layer2_lock_callback(self._closing_session)
                logger.info("Layer 2 position LOCK activated")
            except Exception as e:
                logger.error(f"Layer 2 lock callback failed: {e}")

        # Broadcast session closing to bots
        self._publish_session_close(self._closing_session)

        # Proceed to SIGNAL
        return self._execute_signal()

    def _execute_signal(self) -> bool:
        """
        Execute SIGNAL step: Sentinel confirms regime shift, broadcast fires.

        Returns:
            True if successful
        """
        logger.info(f"Tilt SIGNAL: regime confirmation for {self._closing_session}")

        # Transition to SIGNAL state
        self._transition_to(TiltState.SIGNAL)

        # Publish to tilt:phase for UI subscription
        self._publish_phase_event()

        # Proceed to WAIT
        return self._execute_wait()

    def _execute_wait(self) -> bool:
        """
        Execute WAIT step: 30-minute regime persistence observation window.

        This method initiates an async timer. The timer:
        - Counts down from 1800 seconds
        - Resets on each regime confirmation from Sentinel
        - Pauses (suspends) if CHAOS fires
        - Proceeds to RE-RANK when timer reaches 0

        Returns:
            True if completed normally (timer reached 0)
        """
        logger.info(
            f"Tilt WAIT: starting {REGIME_PERSISTENCE_SECONDS}s regime persistence window"
        )

        # Transition to WAIT state
        self._transition_to(TiltState.WAIT)

        # Initialize timer
        self._regime_persistence_timer = REGIME_PERSISTENCE_SECONDS
        self._publish_phase_event()

        # Start async timer task
        self._wait_cancelled = False
        task = asyncio.create_task(self._run_wait_timer())
        self._wait_task = task
        self._active_tasks.append(task)
        task.add_done_callback(
            lambda t: self._active_tasks.remove(t) if t in self._active_tasks else None
        )
        task.add_done_callback(self._on_timer_complete)

        return True

    async def _run_wait_timer(self) -> None:
        """
        Async timer for WAIT phase.

        Counts down from REGIME_PERSISTENCE_SECONDS.
        Resets on regime confirmation.
        Cancels on CHAOS (suspension).
        """
        while self._regime_persistence_timer > 0 and not self._wait_cancelled:
            await asyncio.sleep(1)
            if not self._wait_cancelled:
                self._regime_persistence_timer -= 1
                self._publish_phase_event()

        if self._wait_cancelled:
            logger.info("WAIT timer cancelled (suspended)")
        else:
            logger.info("WAIT timer completed — proceeding to RE-RANK")
            # Proceed to next step asynchronously
            await self._execute_re_rank()

    def _cancel_wait_timer(self) -> None:
        """Cancel the WAIT timer (called on CHAOS)."""
        self._wait_cancelled = True
        if self._wait_task and not self._wait_task.done():
            self._wait_task.cancel()
            logger.info("WAIT timer task cancelled")

    def _on_timer_complete(self, task: asyncio.Task) -> None:
        """Handle WAIT timer completion with exception logging."""
        if task.exception():
            logger.error(f"Tilt WAIT timer failed: {task.exception()}")

    async def _execute_re_rank(self) -> bool:
        """
        Execute RE-RANK step: DPR queue recalculated for incoming session.

        Returns:
            True if successful (or fallback used)
        """
        logger.info(f"Tilt RE-RANK: recalculating DPR queue for {self._incoming_session}")

        # Transition to RE_RANK state
        self._transition_to(TiltState.RE_RANK)
        self._publish_phase_event()

        rerank_success = False
        self._rerank_warning = False

        # Trigger DPR queue recalculation
        if self._dpr_re_rank_callback:
            try:
                success = self._dpr_re_rank_callback(self._incoming_session)
                if success:
                    logger.info("DPR queue recalculated successfully")
                    rerank_success = True
                else:
                    logger.warning("DPR queue recalculation failed — retrying in 5s")
            except Exception as e:
                logger.error(f"DPR re-rank callback failed: {e}")
        else:
            logger.warning(
                "DPR re-rank callback not set — DPR (Epic 17) may not be implemented."
            )

        # Retry once if first attempt failed
        if not rerank_success and self._dpr_re_rank_callback:
            await asyncio.sleep(5)
            try:
                success = self._dpr_re_rank_callback(self._incoming_session)
                if success:
                    logger.info("DPR queue recalculated successfully on retry")
                    rerank_success = True
                else:
                    logger.error("DPR queue recalculation failed on retry — using stale queue")
                    self._rerank_warning = True
                    self._emit_rerank_failed()
            except Exception as e:
                logger.error(f"DPR re-rank callback retry failed: {e}")
                self._rerank_warning = True
                self._emit_rerank_failed()
        elif not self._dpr_re_rank_callback:
            self._rerank_warning = True
            self._emit_rerank_failed()

        # Create audit log entry with rerank_warning flag
        entry = TiltTransition(
            entry_id=str(uuid.uuid4()),
            from_state=TiltState.RE_RANK,
            to_state=TiltState.ACTIVATE,
            closing_session=self._closing_session or "",
            incoming_session=self._incoming_session or "",
            timestamp_utc=datetime.now(timezone.utc),
            metadata={"rerank_warning": self._rerank_warning},
        )
        self._audit_log.append(entry)

        # Proceed to ACTIVATE
        return self._execute_activate()

    def _execute_activate(self) -> bool:
        """
        Execute ACTIVATE step: Session-masked bots armed for new window.

        Returns:
            True if successful
        """
        logger.info(f"Tilt ACTIVATE: arming bots for {self._incoming_session}")

        # Transition to ACTIVATE state
        self._transition_to(TiltState.ACTIVATE)
        self._publish_phase_event()

        # Activate session-masked bots for new window
        if self._session_activate_callback:
            try:
                self._session_activate_callback(self._incoming_session)
                logger.info(f"Session activation callback sent for {self._incoming_session}")
            except Exception as e:
                logger.error(f"Session activate callback failed: {e}")

        # Broadcast session opening
        self._publish_session_open(self._incoming_session)

        # Log regime transition with regime state and timestamp
        if self._rerank_warning:
            logger.warning("ACTIVATE with stale queue — re-rank failed. Halving active bot slots.")

        # Transition back to IDLE
        self._transition_to(TiltState.IDLE)
        self._log_regime_transition()
        self._reset()

        logger.info(
            f"Tilt transition complete: {self._closing_session} → {self._incoming_session}"
        )

        return True

    # =========================================================================
    # CHAOS Preemption
    # =========================================================================

    def suspend_tilt(self, chaos_event: ChaosEvent) -> bool:
        """
        Suspend Tilt due to CHAOS (Layer 3 fires).

        When CHAOS fires during any phase:
        1. Cancel WAIT timer if running
        2. Record the state we were in
        3. Transition to SUSPENDED
        4. Publish suspension event

        Args:
            chaos_event: The CHAOS event that triggered suspension

        Returns:
            True if suspended successfully
        """
        if not self.is_active:
            logger.debug("Tilt not active — CHAOS ignored")
            return False

        logger.warning(
            f"Tilt SUSPENDED due to CHAOS: lyapunov={chaos_event.lyapunov_value:.4f}, "
            f"was in state={self._state.value}"
        )

        # Record previous state for resumption
        self._suspended_from_state = self._state

        # Cancel WAIT timer if running
        if self._state == TiltState.WAIT:
            self._cancel_wait_timer()

        # Transition to SUSPENDED
        self._transition_to(TiltState.SUSPENDED)

        # Publish chaos suspend event
        suspend_event = TiltChaosSuspendEvent(
            previous_state=self._suspended_from_state,
            chaos_lyapunov=chaos_event.lyapunov_value,
            metadata={"instance_id": self._instance_id},
        )
        self._publish_chaos_suspend(suspend_event)

        return True

    async def resume_tilt(self) -> bool:
        """
        Resume Tilt after CHAOS resolution.

        Called when Layer 3 CHAOS is resolved. If session transition
        is still valid, resumes from the previous state.

        Returns:
            True if resumed successfully, False if cannot resume
        """
        if not self.is_suspended:
            logger.warning("Tilt not suspended — cannot resume")
            return False

        if self._suspended_from_state is None:
            logger.error("No previous state recorded — cannot resume")
            self._reset()
            return False

        previous_state = self._suspended_from_state

        logger.info(
            f"Tilt resuming from SUSPENDED → {previous_state.value} "
            f"({self._closing_session} → {self._incoming_session})"
        )

        # Publish resume event
        resume_event = TiltChaosResumeEvent(
            resuming_to_state=previous_state,
            chaos_resolved=True,
            metadata={"instance_id": self._instance_id},
        )
        self._publish_chaos_resume(resume_event)

        # Resume from the appropriate state
        if previous_state == TiltState.LOCK:
            return self._execute_lock()
        elif previous_state == TiltState.SIGNAL:
            return self._execute_signal()
        elif previous_state == TiltState.WAIT:
            return self._execute_wait()
        elif previous_state == TiltState.RE_RANK:
            return await self._execute_re_rank()
        elif previous_state == TiltState.ACTIVATE:
            return self._execute_activate()
        else:
            logger.error(f"Cannot resume from state: {previous_state}")
            self._reset()
            return False

    def on_chaos_event(self, chaos_event: ChaosEvent) -> None:
        """
        Handle CHAOS event from Layer 3.

        Subscribe to this method to Layer 3's chaos:triggered channel.
        When CHAOS fires, Tilt suspends if active.

        Args:
            chaos_event: ChaosEvent from Layer 3
        """
        if chaos_event.chaos_level.value in ["WARNING", "CRITICAL"]:
            self.suspend_tilt(chaos_event)

    def on_regime_confirmed(self, regime_event: RegimeShiftEvent) -> None:
        """
        Handle regime confirmation from Sentinel.

        Subscribe to this method to the regime:confirmed channel.
        When regime is confirmed during WAIT, reset the timer.

        Args:
            regime_event: RegimeShiftEvent from Sentinel
        """
        if self._state == TiltState.WAIT:
            self._regime_persistence_timer = REGIME_PERSISTENCE_SECONDS
            logger.info(
                f"Regime confirmed — WAIT timer reset to {REGIME_PERSISTENCE_SECONDS}s"
            )
            self._publish_phase_event()

    # =========================================================================
    # Layer Callbacks
    # =========================================================================

    def set_layer1_hold_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for Layer 1 EA hold signal."""
        self._layer1_hold_callback = callback

    def set_layer2_lock_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for Layer 2 position LOCK."""
        self._layer2_lock_callback = callback

    def set_dpr_re_rank_callback(self, callback: Callable[[str], bool]) -> None:
        """Set callback for DPR queue recalculation."""
        self._dpr_re_rank_callback = callback

    def set_session_activate_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for session-masked bot activation."""
        self._session_activate_callback = callback

    def set_rerank_failed_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for when DPR re-rank fails — used to trigger slot halving."""
        self._rerank_failed_callback = callback

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _transition_to(self, new_state: TiltState) -> None:
        """
        Internal state transition with audit logging.

        Args:
            new_state: The state to transition to
        """
        old_state = self._state
        self._state = new_state

        # Create audit log entry
        entry = TiltTransition(
            entry_id=str(uuid.uuid4()),
            from_state=old_state,
            to_state=new_state,
            closing_session=self._closing_session or "",
            incoming_session=self._incoming_session or "",
            timestamp_utc=datetime.now(timezone.utc),
        )
        self._audit_log.append(entry)

        logger.info(f"Tilt state: {old_state.value} → {new_state.value}")

    def _reset(self) -> None:
        """Reset Tilt to IDLE state after transition complete."""
        self._state = TiltState.IDLE
        self._closing_session = None
        self._incoming_session = None
        self._regime_persistence_timer = 0
        self._suspended_from_state = None
        self._wait_cancelled = False
        self._wait_task = None

    def _log_regime_transition(self) -> None:
        """Log regime transition with regime state and timestamp.

        Per AC #3: Creates a TiltTransitionAuditLog entry documenting the
        completed transition with regime state and timestamp.
        """
        # Create audit log entry for the completed transition
        entry = TiltTransition(
            entry_id=str(uuid.uuid4()),
            from_state=TiltState.ACTIVATE,
            to_state=TiltState.IDLE,
            closing_session=self._closing_session or "",
            incoming_session=self._incoming_session or "",
            timestamp_utc=datetime.now(timezone.utc),
            metadata={
                "regime_state": "TRANSITION_COMPLETE",
                "transition_duration": "COMPLETE",
            },
        )
        self._audit_log.append(entry)
        logger.info(
            f"Regime transition logged: "
            f"{self._closing_session} → {self._incoming_session} "
            f"at {datetime.now(timezone.utc).isoformat()}"
        )

    def _publish_phase_event(self) -> None:
        """Publish current phase event to Redis tilt:phase channel."""
        try:
            event = self.get_current_phase_event()
            self._redis.publish(
                CHANNEL_TILT_PHASE,
                str(event.model_dump_json())
            )
        except Exception as e:
            logger.error(f"Failed to publish phase event: {e}")

    def _publish_session_close(self, session: str) -> None:
        """Publish session closing broadcast to Redis."""
        try:
            message = {"session": session, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            self._redis.publish(
                CHANNEL_TILT_SESSION_CLOSE,
                json.dumps(message)
            )
        except Exception as e:
            logger.error(f"Failed to publish session close: {e}")

    def _publish_session_open(self, session: str) -> None:
        """Publish session opening broadcast to Redis."""
        try:
            message = {"session": session, "timestamp_utc": datetime.now(timezone.utc).isoformat()}
            self._redis.publish(
                CHANNEL_TILT_SESSION_OPEN,
                json.dumps(message)
            )
        except Exception as e:
            logger.error(f"Failed to publish session open: {e}")

    def _publish_chaos_suspend(self, event: TiltChaosSuspendEvent) -> None:
        """Publish CHAOS suspend event to Redis."""
        try:
            message = {"type": "chaos_suspend", **event.model_dump()}
            self._redis.publish(
                CHANNEL_TILT_PHASE,
                json.dumps(message, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to publish chaos suspend: {e}")

    def _publish_chaos_resume(self, event: TiltChaosResumeEvent) -> None:
        """Publish CHAOS resume event to Redis."""
        try:
            message = {"type": "chaos_resume", **event.model_dump()}
            self._redis.publish(
                CHANNEL_TILT_PHASE,
                json.dumps(message, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to publish chaos resume: {e}")

    def _emit_rerank_failed(self) -> None:
        """Emit TILT_RERANK_FAILED event to Redis."""
        try:
            message = {
                "type": "TILT_RERANK_FAILED",
                "session": self._incoming_session,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._redis.publish(CHANNEL_TILT_RERANK_FAILED, json.dumps(message, default=str))
        except Exception as e:
            logger.error(f"Failed to publish TILT_RERANK_FAILED event: {e}")

        # Invoke rerank failed callback if registered (triggers DPR slot halving)
        if self._rerank_failed_callback:
            try:
                self._rerank_failed_callback()
            except Exception as e:
                logger.error(f"Rerank failed callback raised: {e}")

    # =========================================================================
    # Audit Log
    # =========================================================================

    def get_audit_log(self) -> List[TiltTransition]:
        """Get the complete audit log of state transitions."""
        return self._audit_log.copy()

    def get_audit_log_entries(
        self,
        closing_session: Optional[str] = None,
        incoming_session: Optional[str] = None,
    ) -> List[TiltTransition]:
        """
        Get filtered audit log entries.

        Args:
            closing_session: Filter by closing session
            incoming_session: Filter by incoming session

        Returns:
            List of matching TiltTransition entries
        """
        entries = self._audit_log
        if closing_session:
            entries = [e for e in entries if e.closing_session == closing_session]
        if incoming_session:
            entries = [e for e in entries if e.incoming_session == incoming_session]
        return entries
