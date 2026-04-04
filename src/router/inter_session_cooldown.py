"""
Inter-Session Cooldown Orchestrator — 10:00–13:00 GMT London Close to NY Open.

Story 16.3: Inter-Session Cooldown Window

Implements the 4-step intelligence and preparation sequence that runs
between London close (12:00 GMT) and NY open (13:00 GMT):

Step 1 (10:00-10:30): London Session Scoring
  - DPR finalises London performer scores
  - Best London bot eligible for NY T1 queue

Step 2 (10:30-11:30): Paper Recovery Review
  - TIER_1 paper bots reviewed (quarantined from SSL)
  - Recovery path confirmed or extended

Step 3 (11:30-12:40): NY Queue Order via DPR + Tier Remix
  - Hybrid NY queue = best London performer + recovery candidate + T2 fresh candidates

Step 4 (12:40-13:00): System Health/SQS/Sentinel Pre-Check
  - SVSS cache validated
  - SQS warm-up if Monday
  - Sentinel regime confirmed for NY open

At 13:00 GMT:
  - NY session roster is locked
  - Tilt ACTIVATE fires for NY open

Per NFR-M2: InterSessionCooldownOrchestrator is a synchronous workflow
orchestrator — NO LLM calls in hot path.
Per NFR-D1: All cooldown state transitions logged with timestamps for audit.
Per NFR-P1: Kill switch protocol executes in full — correctness over raw speed.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, time
from typing import Callable, Dict, List, Optional, Any

import redis

from src.events.cooldown import (
    COOLDOWN_STATE_TO_PHASE,
    CooldownPhase,
    CooldownPhaseEvent,
    CooldownState,
    InterSessionCooldownCompletionEvent,
    InterSessionCooldownStateEvent,
    NYQueueCandidate,
    STEP_WINDOWS,
)
from src.events.tilt import TiltPhase, TiltPhaseEvent

logger = logging.getLogger(__name__)


# Redis channel names
CHANNEL_COOLDOWN_PHASE = "cooldown:phase"
CHANNEL_COOLDOWN_LONDON_SCORE_COMPLETE = "cooldown:london_score_complete"
CHANNEL_COOLDOWN_PAPER_REVIEW_COMPLETE = "cooldown:paper_review_complete"
CHANNEL_COOLDOWN_QUEUE_READY = "cooldown:queue_ready"
CHANNEL_COOLDOWN_PRECHECK_COMPLETE = "cooldown:precheck_complete"
CHANNEL_COOLDOWN_COMPLETED = "cooldown:completed"
CHANNEL_TILT_PHASE = "tilt:phase"

# Cooldown window times (GMT)
COOLDOWN_START_HOUR = 10
COOLDOWN_START_MINUTE = 0
COOLDOWN_END_HOUR = 13
COOLDOWN_END_MINUTE = 0


@dataclass
class CooldownTransition:
    """Record of a cooldown state transition for audit logging."""
    entry_id: str
    from_state: Optional[CooldownState]
    to_state: CooldownState
    step_name: str
    timestamp_utc: datetime
    step_started_at: Optional[datetime] = None
    step_deadline: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


class InterSessionCooldownOrchestrator:
    """
    Orchestrates the Inter-Session Cooldown 4-step sequence.

    The Inter-Session Cooldown runs from 10:00-13:00 GMT, executing
    intelligence and preparation work between London close and NY open.

    Key behaviors:
    1. Triggered by SessionDetector at 10:00 GMT (London Mid transition)
    2. Executes 4 steps in sequence with time-bounded windows
    3. Publishes phase events to Redis for UI subscription
    4. Locks NY roster and triggers Tilt ACTIVATE at 13:00 GMT

    Step timeouts:
    - Step 1 exceeds window: Log warning, proceed to Step 2
    - Step 2 exceeds window: Log warning, proceed to Step 3
    - Step 3 exceeds window: Proceed to Step 4 with remaining time
    - Step 4 exceeds window: Proceed to completion

    Attributes:
        redis_client: Redis client for pub/sub and distributed coordination
        instance_id: Unique identifier for this instance
    """

    # Step names for logging and events
    STEP_NAMES = {
        CooldownState.STEP_1_SCORING: "London Session Scoring",
        CooldownState.STEP_2_PAPER_RECOVERY: "Paper Recovery Review",
        CooldownState.STEP_3_QUEUE_BUILD: "NY Queue Order via DPR + Tier Remix",
        CooldownState.STEP_4_HEALTH_CHECK: "System Health/SQS/Sentinel Pre-Check",
    }

    def __init__(
        self,
        redis_client: redis.Redis,
        instance_id: str = "cooldown-default",
    ):
        """
        Initialize InterSessionCooldownOrchestrator.

        Args:
            redis_client: Redis client for pub/sub
            instance_id: Unique identifier for this instance
        """
        self._redis = redis_client
        self._instance_id = instance_id

        # State machine state
        self._state = CooldownState.PENDING
        self._window_start: Optional[datetime] = None
        self._window_end: Optional[datetime] = None
        self._ny_roster_locked = False

        # Audit log
        self._audit_log: List[CooldownTransition] = []

        # Step timing
        self._current_step_start: Optional[datetime] = None

        # Callbacks for integrations (set by external code)
        self._dpr_scoring_callback: Optional[Callable[[], Dict[str, Any]]] = None
        self._dpr_queue_callback: Optional[Callable[[], List[NYQueueCandidate]]] = None
        self._ssl_paper_review_callback: Optional[Callable[[], Dict[str, Any]]] = None
        self._svss_health_callback: Optional[Callable[[], Dict[str, Any]]] = None
        self._sqs_warmup_callback: Optional[Callable[[bool], None]] = None  # bool = is_monday
        self._sentinel_regime_callback: Optional[Callable[[], Dict[str, Any]]] = None
        self._tilt_activate_callback: Optional[Callable[[], None]] = None
        self._roster_lock_callback: Optional[Callable[[], None]] = None

        # Tilt subscription flag
        self._tilt_activate_confirmed = False

        logger.info(f"InterSessionCooldownOrchestrator initialized: instance={instance_id}")

    # =========================================================================
    # State Machine Core
    # =========================================================================

    @property
    def state(self) -> CooldownState:
        """Get current cooldown state."""
        return self._state

    @property
    def is_running(self) -> bool:
        """Check if cooldown is currently running (not PENDING or COMPLETED)."""
        return self._state not in (CooldownState.PENDING, CooldownState.COMPLETED)

    @property
    def ny_roster_locked(self) -> bool:
        """Check if NY roster has been locked."""
        return self._ny_roster_locked

    @property
    def window_start(self) -> Optional[datetime]:
        """Get cooldown window start time."""
        return self._window_start

    @property
    def window_end(self) -> Optional[datetime]:
        """Get cooldown window end time."""
        return self._window_end

    @property
    def step_deadlines(self) -> Dict[CooldownState, datetime]:
        """
        Get step deadlines based on window start time.

        Returns:
            Dictionary mapping each step state to its deadline datetime
        """
        if not self._window_start:
            return {}
        from datetime import timedelta
        deadlines = {}
        for state, (start_min, end_min) in STEP_WINDOWS.items():
            deadlines[state] = self._window_start + timedelta(minutes=end_min)
        return deadlines

    def get_current_state_event(self) -> InterSessionCooldownStateEvent:
        """
        Get current state event for UI subscription.

        Returns:
            InterSessionCooldownStateEvent with current state info
        """
        step_name = self.STEP_NAMES.get(
            self._state,
            "Pending" if self._state == CooldownState.PENDING else "Completed"
        )

        # Determine current step number
        if self._state == CooldownState.PENDING:
            current_step = 0
        elif self._state == CooldownState.COMPLETED:
            current_step = 4
        else:
            state_to_step = {
                CooldownState.STEP_1_SCORING: 1,
                CooldownState.STEP_2_PAPER_RECOVERY: 2,
                CooldownState.STEP_3_QUEUE_BUILD: 3,
                CooldownState.STEP_4_HEALTH_CHECK: 4,
            }
            current_step = state_to_step.get(self._state, 0)

        return InterSessionCooldownStateEvent(
            state=self._state,
            current_step=current_step,
            step_name=step_name,
            window_start=self._window_start or datetime.now(timezone.utc),
            window_end=self._window_end or datetime.now(timezone.utc),
            ny_roster_locked=self._ny_roster_locked,
            timestamp_utc=datetime.now(timezone.utc),
            metadata={"instance_id": self._instance_id},
        )

    def get_current_phase_event(self) -> CooldownPhaseEvent:
        """
        Get current phase event for Redis pub/sub.

        Returns:
            CooldownPhaseEvent with current phase info
        """
        phase = COOLDOWN_STATE_TO_PHASE.get(
            self._state,
            CooldownPhase.STEP_1 if self._state == CooldownState.PENDING else CooldownPhase.COMPLETED
        )
        step_name = self.STEP_NAMES.get(
            self._state,
            "Pending" if self._state == CooldownState.PENDING else "Completed"
        )

        return CooldownPhaseEvent(
            phase=phase,
            step_name=step_name,
            started_at=self._current_step_start or datetime.now(timezone.utc),
            completed_at=None if self.is_running else datetime.now(timezone.utc),
            results={},
            timestamp_utc=datetime.now(timezone.utc),
        )

    # =========================================================================
    # Start Cooldown
    # =========================================================================

    def start(self) -> bool:
        """
        Start the Inter-Session Cooldown sequence.

        Called when SessionDetector detects transition to Inter-Session Cooldown
        window (10:00 GMT).

        Returns:
            True if cooldown started, False if already running or outside window
        """
        if self.is_running:
            logger.warning(
                f"Cooldown already running ({self._state.value}) — rejecting start"
            )
            return False

        # Validate we're within the cooldown window (10:00-13:00 GMT)
        now = datetime.now(timezone.utc)
        current_time = now.time()
        window_start_time = time(COOLDOWN_START_HOUR, COOLDOWN_START_MINUTE)
        window_end_time = time(COOLDOWN_END_HOUR, COOLDOWN_END_MINUTE)

        # Check if current time is within window
        in_window = window_start_time <= current_time <= window_end_time
        if not in_window:
            logger.warning(
                f"Cooldown start() called outside 10:00-13:00 GMT window "
                f"(current: {current_time}) — should be triggered by SessionDetector. "
                f"Accepting anyway for resilience."
            )
            # Per NFR-P1: proceed anyway — correctness over timing enforcement

        logger.info("Starting Inter-Session Cooldown sequence")

        # Record window times
        self._window_start = now
        # Calculate window end: 13:00 GMT today
        window_end_today = datetime(
            year=now.year,
            month=now.month,
            day=now.day,
            hour=COOLDOWN_END_HOUR,
            minute=COOLDOWN_END_MINUTE,
            tzinfo=timezone.utc
        )
        # If 13:00 has passed today, use tomorrow
        if window_end_today <= now:
            from datetime import timedelta
            window_end_today += timedelta(days=1)
        self._window_end = window_end_today

        # Transition to Step 1
        self._transition_to(CooldownState.STEP_1_SCORING)

        # Execute Step 1
        return self._execute_step_1()

    def _execute_step_1(self) -> bool:
        """
        Execute Step 1: London Session Scoring (10:00-10:30).

        DPR finalises London performer scores.
        Best London bot eligible for NY T1 queue.

        Returns:
            True if step completed (or timed out and proceeded to next step)
        """
        logger.info("Step 1: London Session Scoring starting")

        self._transition_to(CooldownState.STEP_1_SCORING)
        self._current_step_start = datetime.now(timezone.utc)

        step_results = {}

        # Call DPR scoring callback if available
        if self._dpr_scoring_callback:
            try:
                step_results = self._dpr_scoring_callback()
                logger.info(f"Step 1 DPR scoring completed: {step_results}")
            except Exception as e:
                logger.error(f"Step 1 DPR scoring failed: {e}")
                step_results = {"error": str(e)}
        else:
            # DPR not yet implemented — stub
            logger.warning(
                "DPR scoring callback not set — DPR (Epic 17) may not be implemented. "
                "Using stub scoring."
            )
            step_results = self._stub_london_scoring()

        # Check for timeout - proceed to next step if exceeded
        if self._check_step_timeout(CooldownState.STEP_1_SCORING):
            step_results["timeout"] = True

        # Publish step 1 completion
        self._publish_step_completion(
            CooldownPhase.STEP_1,
            "London Session Scoring",
            step_results
        )

        # Proceed to Step 2
        return self._execute_step_2()

    def _execute_step_2(self) -> bool:
        """
        Execute Step 2: Paper Recovery Review (10:30-11:30).

        TIER_1 paper bots reviewed (quarantined from SSL).
        Recovery path confirmed or extended.

        Returns:
            True if step completed
        """
        logger.info("Step 2: Paper Recovery Review starting")

        self._transition_to(CooldownState.STEP_2_PAPER_RECOVERY)
        self._current_step_start = datetime.now(timezone.utc)

        step_results = {}

        # Call SSL paper review callback if available
        if self._ssl_paper_review_callback:
            try:
                step_results = self._ssl_paper_review_callback()
                logger.info(f"Step 2 paper recovery review completed: {step_results}")
            except Exception as e:
                logger.error(f"Step 2 paper recovery review failed: {e}")
                step_results = {"error": str(e)}
        else:
            # SSL not yet implemented — stub
            logger.warning(
                "SSL paper review callback not set — SSL (Epic 18) may not be implemented. "
                "Using stub paper review."
            )
            step_results = self._stub_paper_review()

        # Check for timeout - proceed to next step if exceeded
        if self._check_step_timeout(CooldownState.STEP_2_PAPER_RECOVERY):
            step_results["timeout"] = True

        # Publish step 2 completion
        self._publish_step_completion(
            CooldownPhase.STEP_2,
            "Paper Recovery Review",
            step_results
        )

        # Proceed to Step 3
        return self._execute_step_3()

    def _execute_step_3(self) -> bool:
        """
        Execute Step 3: NY Queue Order via DPR + Tier Remix (11:30-12:40).

        Build hybrid NY queue:
        - Queue positions 1-2: Best London performer + TIER_1 recovery candidate
        - Queue positions 3-N: TIER_3 DPR-ranked bots
        - After TIER_3: TIER_2 fresh candidates from AlphaForge

        Returns:
            True if step completed
        """
        logger.info("Step 3: NY Queue Order via DPR + Tier Remix starting")

        self._transition_to(CooldownState.STEP_3_QUEUE_BUILD)
        self._current_step_start = datetime.now(timezone.utc)

        step_results = {}

        # Call DPR queue callback if available
        if self._dpr_queue_callback:
            try:
                queue_candidates = self._dpr_queue_callback()
                step_results = {
                    "queue_candidates": [c.model_dump() for c in queue_candidates],
                    "count": len(queue_candidates)
                }
                logger.info(f"Step 3 NY queue built: {len(queue_candidates)} candidates")
            except Exception as e:
                logger.error(f"Step 3 NY queue build failed: {e}")
                step_results = {"error": str(e)}
        else:
            # DPR not yet implemented — stub
            logger.warning(
                "DPR queue callback not set — DPR (Epic 17) may not be implemented. "
                "Using stub queue assembly."
            )
            step_results = self._stub_ny_queue_build()

        # Check for timeout - proceed to next step if exceeded
        if self._check_step_timeout(CooldownState.STEP_3_QUEUE_BUILD):
            step_results["timeout"] = True

        # Publish step 3 completion
        self._publish_step_completion(
            CooldownPhase.STEP_3,
            "NY Queue Order via DPR + Tier Remix",
            step_results
        )

        # Proceed to Step 4
        return self._execute_step_4()

    def _execute_step_4(self) -> bool:
        """
        Execute Step 4: System Health/SQS/Sentinel Pre-Check (12:40-13:00).

        - SVSS cache validated
        - SQS warm-up if Monday
        - Sentinel regime confirmed for NY open

        Returns:
            True if step completed
        """
        logger.info("Step 4: System Health/SQS/Sentinel Pre-Check starting")

        self._transition_to(CooldownState.STEP_4_HEALTH_CHECK)
        self._current_step_start = datetime.now(timezone.utc)

        step_results = {}

        # SVSS health check
        if self._svss_health_callback:
            try:
                svss_results = self._svss_health_callback()
                step_results["svss"] = svss_results
                logger.info(f"Step 4 SVSS health check: {svss_results}")
            except Exception as e:
                logger.error(f"Step 4 SVSS health check failed: {e}")
                step_results["svss"] = {"error": str(e)}
        else:
            # SVSS not yet implemented — stub
            logger.warning(
                "SVSS health callback not set — SVSS (Epic 15) may not be implemented. "
                "Using stub health check."
            )
            step_results["svss"] = self._stub_svss_health()

        # SQS warm-up check (Monday only)
        is_monday = datetime.now(timezone.utc).weekday() == 0
        if self._sqs_warmup_callback:
            try:
                self._sqs_warmup_callback(is_monday)
                step_results["sqs_warmup"] = {"triggered": is_monday}
                logger.info(f"Step 4 SQS warm-up: triggered={is_monday}")
            except Exception as e:
                logger.error(f"Step 4 SQS warm-up failed: {e}")
                step_results["sqs_warmup"] = {"error": str(e)}
        else:
            # Stub SQS warm-up
            step_results["sqs_warmup"] = {"triggered": is_monday, "stub": True}

        # Sentinel regime confirmation
        if self._sentinel_regime_callback:
            try:
                sentinel_results = self._sentinel_regime_callback()
                step_results["sentinel"] = sentinel_results
                logger.info(f"Step 4 Sentinel regime: {sentinel_results}")
            except Exception as e:
                logger.error(f"Step 4 Sentinel regime check failed: {e}")
                step_results["sentinel"] = {"error": str(e)}
        else:
            # Stub Sentinel regime check
            step_results["sentinel"] = self._stub_sentinel_regime()

        # Check for timeout - proceed to completion if exceeded
        if self._check_step_timeout(CooldownState.STEP_4_HEALTH_CHECK):
            step_results["timeout"] = True

        # Publish step 4 completion
        self._publish_step_completion(
            CooldownPhase.STEP_4,
            "System Health/SQS/Sentinel Pre-Check",
            step_results
        )

        # Complete cooldown
        return self._complete()

    def _complete(self) -> bool:
        """
        Complete the Inter-Session Cooldown sequence.

        - Lock NY roster
        - Trigger Tilt ACTIVATE for NY open
        - Publish completion event

        Returns:
            True
        """
        logger.info("Inter-Session Cooldown completing")

        # Lock NY roster
        if self._roster_lock_callback:
            try:
                self._roster_lock_callback()
                logger.info("NY roster locked")
            except Exception as e:
                logger.error(f"NY roster lock failed: {e}")

        self._ny_roster_locked = True

        # Trigger Tilt ACTIVATE for NY open
        tilt_activated = False
        if self._tilt_activate_callback:
            try:
                self._tilt_activate_callback()
                tilt_activated = True
                logger.info("Tilt ACTIVATE triggered for NY open")
            except Exception as e:
                logger.error(f"Tilt ACTIVATE callback failed: {e}")

        # Transition to COMPLETED
        self._transition_to(CooldownState.COMPLETED)

        # Publish completion event
        completion_event = InterSessionCooldownCompletionEvent(
            completed_at=datetime.now(timezone.utc),
            ny_roster_locked=self._ny_roster_locked,
            roster_summary={"state": "locked"},
            tilt_activate_triggered=tilt_activated,
        )
        self._publish_completion(completion_event)

        logger.info(
            f"Inter-Session Cooldown complete: roster_locked={self._ny_roster_locked}, "
            f"tilt_activate={tilt_activated}"
        )

        return True

    # =========================================================================
    # Tilt ACTIVATE Integration
    # =========================================================================

    def on_tilt_activate(self, event: TiltPhaseEvent) -> None:
        """
        Handle Tilt ACTIVATE confirmation.

        Subscribe to tilt:phase channel to receive ACTIVATE confirmation.
        This is used to coordinate cooldown completion with Tilt.

        Args:
            event: TiltPhaseEvent from tilt:phase channel
        """
        if event.phase == TiltPhase.ACTIVATE:
            logger.info(f"Tilt ACTIVATE confirmed for session: {event.incoming_session}")
            self._tilt_activate_confirmed = True

    # =========================================================================
    # Stub Implementations (when DPR/SSL/SVSS not yet implemented)
    # =========================================================================

    def _stub_london_scoring(self) -> Dict[str, Any]:
        """
        Stub London session scoring when DPR not yet implemented.

        Returns:
            Stub scoring results
        """
        return {
            "stub": True,
            "london_performers": [],
            "best_london_bot": None,
            "note": "DPR (Epic 17) not yet implemented — stub scoring"
        }

    def _stub_paper_review(self) -> Dict[str, Any]:
        """
        Stub paper recovery review when SSL not yet implemented.

        Returns:
            Stub paper review results
        """
        return {
            "stub": True,
            "tier_1_paper_bots_reviewed": 0,
            "recovery_confirmed": 0,
            "recovery_extended": 0,
            "note": "SSL (Epic 18) not yet implemented — stub review"
        }

    def _stub_ny_queue_build(self) -> Dict[str, Any]:
        """
        Stub NY queue build when DPR not yet implemented.

        Returns stub queue demonstrating hybrid NY queue composition:
        - Queue positions 1-2: Best London performer + TIER_1 recovery candidate
        - Queue positions 3-N: TIER_3 DPR-ranked bots
        - After TIER_3: TIER_2 fresh candidates from AlphaForge
        """
        # Build stub queue candidates demonstrating proper composition
        stub_candidates = [
            NYQueueCandidate(
                bot_id="stub-london-performer-1",
                queue_position=1,
                source="london_performer",
                tier=1,
                dpr_score=0.85,
                session_specialist=True,
                consecutive_paper_wins=0,
            ),
            NYQueueCandidate(
                bot_id="stub-recovery-candidate-1",
                queue_position=2,
                source="recovery_candidate",
                tier=1,
                dpr_score=0.78,
                session_specialist=False,
                consecutive_paper_wins=2,
            ),
            NYQueueCandidate(
                bot_id="stub-tier3-dpr-1",
                queue_position=3,
                source="tier_3_dpr",
                tier=3,
                dpr_score=0.72,
                session_specialist=False,
                consecutive_paper_wins=0,
            ),
            NYQueueCandidate(
                bot_id="stub-tier3-dpr-2",
                queue_position=4,
                source="tier_3_dpr",
                tier=3,
                dpr_score=0.68,
                session_specialist=False,
                consecutive_paper_wins=0,
            ),
            NYQueueCandidate(
                bot_id="stub-tier2-fresh-1",
                queue_position=5,
                source="tier_2_fresh",
                tier=2,
                dpr_score=0.65,
                session_specialist=False,
                consecutive_paper_wins=0,
            ),
        ]

        return {
            "stub": True,
            "queue_candidates": [c.model_dump() for c in stub_candidates],
            "composition": {
                "london_performer": 1,
                "recovery_candidate": 1,
                "tier_3_dpr": 2,
                "tier_2_fresh": 1,
            },
            "note": "DPR (Epic 17) not yet implemented — stub queue"
        }

    def _stub_svss_health(self) -> Dict[str, Any]:
        """
        Stub SVSS health check when SVSS not yet implemented.

        Returns:
            Stub health results (passes with warning)
        """
        return {
            "stub": True,
            "status": "pass",
            "cache_validated": False,
            "note": "SVSS (Epic 15) not yet implemented — stub health check"
        }

    def _stub_sentinel_regime(self) -> Dict[str, Any]:
        """
        Stub Sentinel regime confirmation.

        Returns:
            Stub regime results
        """
        return {
            "stub": True,
            "regime": "UNKNOWN",
            "ny_open_ready": True,
            "note": "Sentinel regime check stub"
        }

    # =========================================================================
    # State Transitions
    # =========================================================================

    def _transition_to(self, new_state: CooldownState) -> None:
        """
        Internal state transition with audit logging.

        Args:
            new_state: The state to transition to
        """
        old_state = self._state
        self._state = new_state

        step_name = self.STEP_NAMES.get(
            new_state,
            "Pending" if new_state == CooldownState.PENDING else "Completed"
        )

        # Calculate step deadline from window start + STEP_WINDOWS offset
        step_deadline = None
        if self._window_start and new_state in STEP_WINDOWS:
            from datetime import timedelta
            window_offset_minutes = STEP_WINDOWS[new_state][1]  # end of window
            step_deadline = self._window_start + timedelta(minutes=window_offset_minutes)

        # Create audit log entry
        entry = CooldownTransition(
            entry_id=str(uuid.uuid4()),
            from_state=old_state,
            to_state=new_state,
            step_name=step_name,
            timestamp_utc=datetime.now(timezone.utc),
            step_started_at=self._current_step_start,
            step_deadline=step_deadline,
        )
        self._audit_log.append(entry)

        logger.info(f"Cooldown state: {old_state.value} → {new_state.value}")

        # Publish phase event
        self._publish_phase_event()

    def _check_step_timeout(self, step_state: CooldownState) -> bool:
        """
        Check if current step has exceeded its time window.

        Per Subtask 1.5: If step exceeds window, log warning and proceed to next step.

        Args:
            step_state: The current step state

        Returns:
            True if step timed out, False if still within window
        """
        if step_state not in STEP_WINDOWS:
            return False

        # If window_start is not set, we can't check timeout
        if not self._window_start:
            return False

        from datetime import timedelta

        # Calculate deadline
        deadline = self._window_start + timedelta(minutes=STEP_WINDOWS[step_state][1])
        now = datetime.now(timezone.utc)

        if now > deadline:
            elapsed = (now - (self._current_step_start or now)).total_seconds() / 60
            window_minutes = STEP_WINDOWS[step_state][1] - STEP_WINDOWS[step_state][0]
            logger.warning(
                f"Step {step_state.value} exceeded window "
                f"(elapsed: {elapsed:.1f}min, window: {window_minutes}min) — proceeding to next step"
            )
            return True
        return False

    # =========================================================================
    # Redis Pub/Sub
    # =========================================================================

    def _publish_phase_event(self) -> None:
        """Publish current phase event to Redis cooldown:phase channel."""
        try:
            event = self.get_current_phase_event()
            self._redis.publish(
                CHANNEL_COOLDOWN_PHASE,
                json.dumps(event.to_redis_message(), default=str)
            )
        except Exception as e:
            logger.error(f"Failed to publish phase event: {e}")

    def _publish_step_completion(
        self,
        phase: CooldownPhase,
        step_name: str,
        results: Dict[str, Any]
    ) -> None:
        """
        Publish step completion event to appropriate Redis channel.

        Args:
            phase: The phase that completed
            step_name: Human-readable step name
            results: Step-specific results
        """
        try:
            event = CooldownPhaseEvent(
                phase=phase,
                step_name=step_name,
                started_at=self._current_step_start or datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                results=results,
                timestamp_utc=datetime.now(timezone.utc),
            )

            # Publish to appropriate channel
            channel_map = {
                CooldownPhase.STEP_1: CHANNEL_COOLDOWN_LONDON_SCORE_COMPLETE,
                CooldownPhase.STEP_2: CHANNEL_COOLDOWN_PAPER_REVIEW_COMPLETE,
                CooldownPhase.STEP_3: CHANNEL_COOLDOWN_QUEUE_READY,
                CooldownPhase.STEP_4: CHANNEL_COOLDOWN_PRECHECK_COMPLETE,
            }

            channel = channel_map.get(phase, CHANNEL_COOLDOWN_PHASE)
            self._redis.publish(
                channel,
                json.dumps(event.to_redis_message(), default=str)
            )

            # Also publish to main cooldown:phase channel
            self._redis.publish(
                CHANNEL_COOLDOWN_PHASE,
                json.dumps(event.to_redis_message(), default=str)
            )

        except Exception as e:
            logger.error(f"Failed to publish step completion: {e}")

    def _publish_completion(self, event: InterSessionCooldownCompletionEvent) -> None:
        """
        Publish cooldown completion event to Redis.

        Args:
            event: The completion event
        """
        try:
            message = event.model_dump()
            message["timestamp_utc"] = event.timestamp_utc.isoformat()
            self._redis.publish(
                CHANNEL_COOLDOWN_COMPLETED,
                json.dumps(message, default=str)
            )
        except Exception as e:
            logger.error(f"Failed to publish completion event: {e}")

    # =========================================================================
    # Callbacks Setters
    # =========================================================================

    def set_dpr_scoring_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        """Set callback for DPR London scoring."""
        self._dpr_scoring_callback = callback

    def set_dpr_queue_callback(self, callback: Callable[[], List[NYQueueCandidate]]) -> None:
        """Set callback for DPR NY queue building."""
        self._dpr_queue_callback = callback

    def set_ssl_paper_review_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        """Set callback for SSL paper recovery review."""
        self._ssl_paper_review_callback = callback

    def set_svss_health_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        """Set callback for SVSS health check."""
        self._svss_health_callback = callback

    def set_sqs_warmup_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback for SQS warm-up (is_monday: bool)."""
        self._sqs_warmup_callback = callback

    def set_sentinel_regime_callback(self, callback: Callable[[], Dict[str, Any]]) -> None:
        """Set callback for Sentinel regime confirmation."""
        self._sentinel_regime_callback = callback

    def set_tilt_activate_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for Tilt ACTIVATE."""
        self._tilt_activate_callback = callback

    def set_roster_lock_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for NY roster lock."""
        self._roster_lock_callback = callback

    # =========================================================================
    # Audit Log
    # =========================================================================

    def get_audit_log(self) -> List[CooldownTransition]:
        """Get the complete audit log of state transitions."""
        return self._audit_log.copy()
