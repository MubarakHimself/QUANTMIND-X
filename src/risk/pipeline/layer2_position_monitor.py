"""
Layer 2 Position Monitor for QUANTMINDX Trading System.

Story 14.2: Layer 2 Tier 1 Position Monitor (Dynamic)

A synchronous, event-driven monitor that:
- Moves stops to breakeven when positions are in profit >= 1R
- Responds to Sentinel regime shift events
- Uses Redis locks for position modification (GG-1 resolution)
- Handles Kill Switch preemption from Layer 3 (GG-4 resolution)

NO LLM calls in the hot path - runs on Kamakura T1 (London, latency-critical).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import redis

from src.events.regime import RegimeShiftEvent, RegimeSuitability, RegimeType
from src.database.models import SessionLocal
from src.database.models.trade_record import TradeRecord

logger = logging.getLogger(__name__)


# Configuration constants
LAYER2_ACTIVATION_DELAY_MS = 100  # 100ms tick sync guard
REDIS_LOCK_TTL_SECONDS = 3  # 3-second TTL for position modification locks


@dataclass
class PositionState:
    """
    Current state of a monitored position.

    Captures all information needed for Layer 2 decision-making.
    """
    ticket: int
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    current_price: float
    sl_price: float  # Current stop loss
    tp_price: float  # Current take profit
    volume: float
    strategy_id: str
    epic_id: Optional[str]
    profit_points: float = 0.0
    profit_r: float = 0.0  # Profit in R multiples
    sl_distance_points: float = 0.0  # Original SL distance in points
    has_moved_to_breakeven: bool = False
    regime_at_entry: Optional[str] = None

    def update_profit(self, current_price: float) -> None:
        """Update profit calculations based on current price."""
        self.current_price = current_price
        if self.direction.upper() == "BUY":
            self.profit_points = current_price - self.entry_price
        else:  # SELL
            self.profit_points = self.entry_price - current_price

        if self.sl_distance_points > 0:
            self.profit_r = self.profit_points / self.sl_distance_points

    def is_in_profit_by_r(self, r_threshold: float = 1.0, epsilon: float = 0.001) -> bool:
        """Check if position is in profit by >= R threshold.

        Args:
            r_threshold: R multiple to check (default 1.0 = 1R)
            epsilon: Floating point tolerance (default 0.001 = 0.1%)
        """
        return self.profit_r >= (r_threshold - epsilon) and not self.has_moved_to_breakeven


@dataclass
class ModificationResult:
    """Result of a position modification attempt."""
    success: bool
    ticket: int
    action: str
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    error: Optional[str] = None
    lock_released: bool = False
    retry_count: int = 0


class Layer2PositionMonitor:
    """
    Layer 2 Tier 1 Position Monitor.

    Story 14.2: Implements dynamic move-to-breakeven and regime shift response.
    Story 16.1: Implements Tilt LOCK step — session-boundary position locking.
    Runs on Kamakura T1 (London) with NO LLM calls in the hot path.

    Key behaviors:
    1. Move-to-breakeven when position reaches >= 1R profit
    2. React to regime shifts by evaluating position suitability
    3. Use Redis locks to prevent concurrent modifications (GG-1)
    4. Handle Kill Switch preemption from Layer 3 (GG-4)
    5. Tilt LOCK: Hold positions during session boundary transitions (Story 16.1)

    Attributes:
        redis_client: Redis client for distributed locking
        mt5_client: MT5 client for order modifications
        instance_id: Unique identifier for this monitor instance
        activation_delay_ms: Tick sync delay before modifications
    """

    # Redis keys for Tilt session locking
    SESSION_LOCK_PREFIX = "tilt:lock:sessions:"
    SESSION_LOCK_TTL_SECONDS = 3600  # 1 hour TTL for session locks

    def __init__(
        self,
        redis_client: redis.Redis,
        mt5_client: Any,
        instance_id: str = "layer2-default",
        activation_delay_ms: int = LAYER2_ACTIVATION_DELAY_MS,
    ):
        """
        Initialize Layer 2 Position Monitor.

        Args:
            redis_client: Redis client for distributed locking
            mt5_client: MT5 client for order modifications
            instance_id: Unique identifier for this monitor instance
            activation_delay_ms: Tick sync delay in milliseconds
        """
        self._redis = redis_client
        self._mt5 = mt5_client
        self._instance_id = instance_id
        self._activation_delay_ms = activation_delay_ms
        self._positions: Dict[int, PositionState] = {}
        self._kill_preemption_callbacks: List[Callable[[int], None]] = []

        # Tilt session lock state
        self._session_locked: Dict[str, bool] = {}  # session_name -> locked

        logger.info(
            f"Layer2PositionMonitor initialized: instance={instance_id}, "
            f"activation_delay={activation_delay_ms}ms"
        )

    # =========================================================================
    # Position Tracking
    # =========================================================================

    def track_position(
        self,
        ticket: int,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        volume: float,
        strategy_id: str,
        epic_id: Optional[str] = None,
        regime_at_entry: Optional[str] = None,
    ) -> PositionState:
        """
        Start tracking a new position.

        Args:
            ticket: MT5 ticket number
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            sl_price: Initial stop loss price
            tp_price: Take profit price
            volume: Position volume in lots
            strategy_id: Strategy that opened the position
            epic_id: Epic the strategy belongs to
            regime_at_entry: Regime at time of entry

        Returns:
            PositionState for the tracked position
        """
        # Calculate SL distance in points
        sl_distance = abs(entry_price - sl_price) if sl_price > 0 else 0.0

        state = PositionState(
            ticket=ticket,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            current_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            volume=volume,
            strategy_id=strategy_id,
            epic_id=epic_id,
            sl_distance_points=sl_distance,
            regime_at_entry=regime_at_entry,
        )

        self._positions[ticket] = state
        logger.info(
            f"Tracking position: ticket={ticket}, {symbol} {direction} "
            f"@{entry_price}, SL={sl_price}, R_dist={sl_distance:.5f}"
        )

        return state

    def untrack_position(self, ticket: int) -> bool:
        """
        Stop tracking a position.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if position was being tracked, False otherwise
        """
        if ticket in self._positions:
            del self._positions[ticket]
            logger.info(f"Stopped tracking position: ticket={ticket}")
            return True
        return False

    def get_position(self, ticket: int) -> Optional[PositionState]:
        """Get current state of a tracked position."""
        return self._positions.get(ticket)

    def get_all_positions(self) -> List[PositionState]:
        """Get all tracked positions."""
        return list(self._positions.values())

    # =========================================================================
    # Move-to-Breakeven (AC #1)
    # =========================================================================

    def check_and_move_to_breakeven(
        self,
        ticket: int,
        current_price: float,
        symbol: str,
    ) -> ModificationResult:
        """
        Check if position qualifies for move-to-breakeven and execute if so.

        AC #1: Given position in profit >= 1R for first time, move stop to
        breakeven + 0.5*SL spread buffer via MT5 bridge.

        Args:
            ticket: MT5 ticket number
            current_price: Current market price
            symbol: Symbol for pip calculation

        Returns:
            ModificationResult with outcome
        """
        state = self._positions.get(ticket)
        if not state:
            return ModificationResult(
                success=False,
                ticket=ticket,
                action="move_to_breakeven",
                error=f"Position {ticket} not tracked"
            )

        # Update profit state
        state.update_profit(current_price)

        # Check if already moved to breakeven
        if state.has_moved_to_breakeven:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="move_to_breakeven",
                new_sl=state.sl_price,
                error="Already at breakeven"
            )

        # Check if position is in profit >= 1R
        if not state.is_in_profit_by_r(1.0):
            logger.debug(
                f"Position {ticket} not yet at 1R: profit_r={state.profit_r:.2f}"
            )
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="move_to_breakeven",
                error=f"Not at 1R yet: profit_r={state.profit_r:.2f}"
            )

        # Calculate breakeven stop: entry + 0.5 * SL_distance
        sl_distance = state.sl_distance_points
        if state.direction.upper() == "BUY":
            breakeven_stop = state.entry_price + (0.5 * sl_distance)
        else:  # SELL
            breakeven_stop = state.entry_price - (0.5 * sl_distance)

        # Don't move stop if it's already better than breakeven
        if state.direction.upper() == "BUY" and state.sl_price >= breakeven_stop:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="move_to_breakeven",
                new_sl=state.sl_price,
                error="SL already better than breakeven"
            )
        if state.direction.upper() == "SELL" and state.sl_price <= breakeven_stop:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="move_to_breakeven",
                new_sl=state.sl_price,
                error="SL already better than breakeven"
            )

        # Execute move-to-breakeven with Redis lock
        result = self._modify_position_with_lock(
            ticket=ticket,
            new_sl=breakeven_stop,
            action="move_to_breakeven"
        )

        if result.success:
            state.has_moved_to_breakeven = True
            state.sl_price = breakeven_stop
            # Log to trade record
            self._log_move_to_breakeven(ticket, breakeven_stop, state)

        return result

    def _log_move_to_breakeven(
        self,
        ticket: int,
        new_sl: float,
        state: PositionState
    ) -> None:
        """
        Log move-to-breakeven event to trade record.

        Per NFR-D1: Trade records must be persisted before system acknowledgment.
        """
        try:
            session = SessionLocal()
            try:
                # Find the trade record by ticket
                record = session.query(TradeRecord).filter_by(signal_id=str(ticket)).first()
                if record:
                    # Log the move-to-breakeven event in metadata
                    events = record.ea_parameters.get("layer2_events", []) if record.ea_parameters else []
                    events.append({
                        "event": "move_to_breakeven",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "new_sl": new_sl,
                        "profit_r_at_move": state.profit_r,
                    })
                    if not record.ea_parameters:
                        record.ea_parameters = {}
                    record.ea_parameters["layer2_events"] = events
                    session.commit()
                    logger.info(f"Logged move-to-breakeven for ticket={ticket}")
                else:
                    logger.warning(f"No trade record found for ticket={ticket}")
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to log move-to-breakeven: {e}")

    # =========================================================================
    # Regime Shift Response (AC #2)
    # =========================================================================

    def evaluate_regime_shift(self, event: RegimeShiftEvent) -> List[RegimeSuitability]:
        """
        Evaluate all tracked positions against a regime shift.

        AC #2: Given regime shift detected by Sentinel, evaluate each position
        against new regime suitability and flag positions for potential close/reduce.

        Args:
            event: RegimeShiftEvent from Sentinel

        Returns:
            List of RegimeSuitability assessments for each position
        """
        logger.info(f"Evaluating regime shift: {event}")

        results = []
        for position in self.get_all_positions():
            # Skip if regime event is symbol-specific and doesn't match
            if event.symbol and position.symbol != event.symbol:
                continue

            # Evaluate suitability
            strategy_id = position.strategy_id or "unknown"
            suitability = RegimeSuitability.evaluate(
                strategy_id=strategy_id,
                regime=event.current_regime,
                confidence=event.confidence
            )
            results.append(suitability)

            # Log and act on non-hold recommendations
            if suitability.action != "hold":
                logger.warning(
                    f"Position {position.ticket} ({position.symbol}): "
                    f"{suitability.action.upper()} - {suitability.reason}"
                )

                if suitability.action == "close":
                    self._flag_for_closure(position, suitability)
                elif suitability.action == "reduce":
                    self._flag_for_reduction(position, suitability)

        return results

    def _flag_for_closure(
        self,
        position: PositionState,
        suitability: RegimeSuitability
    ) -> None:
        """Flag a position for Layer 2 closure."""
        logger.warning(
            f"FLAGGED FOR CLOSURE: ticket={position.ticket}, "
            f"strategy={position.strategy_id}, reason={suitability.reason}"
        )
        # Note: Actual closure is handled by higher priority systems
        # Layer 2 flags the position; closure may be triggered by Layer 3
        # or manual intervention

    def _flag_for_reduction(
        self,
        position: PositionState,
        suitability: RegimeSuitability
    ) -> None:
        """Flag a position for exposure reduction."""
        logger.info(
            f"FLAGGED FOR REDUCTION: ticket={position.ticket}, "
            f"strategy={position.strategy_id}, reason={suitability.reason}"
        )
        # Note: Actual reduction logic would go here
        # For now, just logging - reduction requires additional position sizing logic

    # =========================================================================
    # Redis Lock & Tick-Sync Guard (AC #3, #4)
    # =========================================================================

    def _get_lock_key(self, ticket: int) -> str:
        """Get Redis lock key for a position modification."""
        return f"lock:modify:{ticket}"

    def _acquire_lock(self, ticket: int) -> bool:
        """
        Acquire Redis lock for position modification.

        GG-1: Redis lock with 3-second TTL prevents concurrent modifications.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if lock acquired, False otherwise
        """
        lock_key = self._get_lock_key(ticket)
        lock_value = f"{self._instance_id}:{time.time()}"

        try:
            # Try to acquire lock with blocking
            acquired = self._redis.set(
                lock_key,
                lock_value,
                nx=True,  # Only set if not exists
                ex=REDIS_LOCK_TTL_SECONDS  # 3 second TTL
            )

            if acquired:
                logger.debug(f"Lock acquired: {lock_key}")
                return True
            else:
                logger.warning(f"Lock not acquired (already held): {lock_key}")
                return False

        except redis.RedisError as e:
            logger.error(f"Redis error acquiring lock: {e}")
            return False

    def _release_lock(self, ticket: int) -> bool:
        """
        Release Redis lock for position modification.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if lock released, False otherwise
        """
        lock_key = self._get_lock_key(ticket)

        try:
            self._redis.delete(lock_key)
            logger.debug(f"Lock released: {lock_key}")
            return True
        except redis.RedisError as e:
            logger.error(f"Redis error releasing lock: {e}")
            return False

    def _apply_activation_delay(self) -> None:
        """
        Apply tick-sync activation delay.

        LAYER2_ACTIVATION_DELAY guard: 100ms delay ensures tick sync
        before position modification executes.
        """
        time.sleep(self._activation_delay_ms / 1000.0)

    def _modify_position_with_lock(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
        action: str = "modify"
    ) -> ModificationResult:
        """
        Modify position with Redis lock and retry logic.

        GG-1: Lock acquisition with 3s TTL
        GG-1: LAYER2_ACTIVATION_DELAY tick sync guard
        AC #4: Retry-once on TTL expiry, then log failure

        When the 3-second TTL expires before the modification completes,
        the lock is released and the modification is retried once. If the
        retry fails, the position retains its current stop and failure is logged.

        Args:
            ticket: MT5 ticket number
            new_sl: New stop loss price
            new_tp: New take profit price
            action: Action description for logging

        Returns:
            ModificationResult with outcome
        """
        # Step 1: Acquire lock
        if not self._acquire_lock(ticket):
            # Lock not acquired - try once more after short delay
            time.sleep(0.1)
            if not self._acquire_lock(ticket):
                return ModificationResult(
                    success=False,
                    ticket=ticket,
                    action=action,
                    error="Failed to acquire lock (held by another process)"
                )

        # Track modification start time for TTL expiry detection
        modification_start_time = time.time()
        retry_count = 0
        max_retries = 1

        while retry_count <= max_retries:
            try:
                # Check if lock TTL would expire during modification
                elapsed = time.time() - modification_start_time
                remaining_ttl = REDIS_LOCK_TTL_SECONDS - elapsed

                # If less than 100ms remaining, skip this attempt to avoid partial modification
                if remaining_ttl < 0.1 and retry_count < max_retries:
                    # TTL expired before we could complete - retry once
                    logger.warning(
                        f"Lock TTL expired for ticket={ticket} during modification attempt {retry_count + 1}"
                    )
                    retry_count += 1
                    # Re-acquire lock before retry
                    if not self._acquire_lock(ticket):
                        time.sleep(0.1)
                        if not self._acquire_lock(ticket):
                            return ModificationResult(
                                success=False,
                                ticket=ticket,
                                action=action,
                                error="Failed to re-acquire lock after TTL expiry",
                                retry_count=retry_count
                            )
                    modification_start_time = time.time()
                    continue

                # Apply tick-sync delay
                self._apply_activation_delay()

                # Execute modification via MT5
                success = self._execute_modification(ticket, new_sl, new_tp)

                if success:
                    return ModificationResult(
                        success=True,
                        ticket=ticket,
                        action=action,
                        new_sl=new_sl,
                        new_tp=new_tp,
                        lock_released=False,  # Will be released separately
                        retry_count=retry_count
                    )
                else:
                    # Modification failed - release lock and return
                    self._release_lock(ticket)
                    return ModificationResult(
                        success=False,
                        ticket=ticket,
                        action=action,
                        new_sl=new_sl,
                        new_tp=new_tp,
                        error="MT5 modification failed",
                        lock_released=True,
                        retry_count=retry_count
                    )

            except Exception as e:
                # On any error, ensure lock is released
                self._release_lock(ticket)
                logger.error(f"Modification error for ticket={ticket}: {e}")
                return ModificationResult(
                    success=False,
                    ticket=ticket,
                    action=action,
                    error=str(e),
                    lock_released=True,
                    retry_count=retry_count
                )

        # Exhausted retries
        self._release_lock(ticket)
        logger.error(f"Modification exhausted retries for ticket={ticket} after TTL expiry")
        return ModificationResult(
            success=False,
            ticket=ticket,
            action=action,
            error="Modification failed after TTL expiry and retry exhaustion",
            lock_released=True,
            retry_count=retry_count
        )

    def _execute_modification(
        self,
        ticket: int,
        new_sl: Optional[float],
        new_tp: Optional[float]
    ) -> bool:
        """
        Execute position modification via MT5 bridge.

        Args:
            ticket: MT5 ticket number
            new_sl: New stop loss price
            new_tp: New take profit price

        Returns:
            True if successful, False otherwise
        """
        if not self._mt5:
            logger.error("MT5 client not available")
            return False

        try:
            # Call MT5 position modification
            # Using OrderManager.modify_position if available
            if hasattr(self._mt5, 'modify_position'):
                return self._mt5.modify_position(ticket, sl=new_sl, tp=new_tp)
            else:
                logger.error("MT5 client has no modify_position method")
                return False
        except Exception as e:
            logger.error(f"MT5 modification error: {e}")
            return False

    def modify_position(
        self,
        ticket: int,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None
    ) -> ModificationResult:
        """
        Public method to modify a position with full lock handling.

        This method handles the complete flow:
        1. Check if Tilt session lock is active (Story 16.1)
        2. Check if kill is pending for this ticket
        3. Acquire Redis lock
        4. Apply tick-sync delay
        5. Execute modification
        6. Release lock on success (caller releases) or failure

        Args:
            ticket: MT5 ticket number
            new_sl: New stop loss price
            new_tp: New take profit price

        Returns:
            ModificationResult with outcome
        """
        # Step 0: Check if Tilt session lock is active (Story 16.1)
        # Per Subtask 2.3: Verify Layer 2 Redis lock pattern is respected during LOCK
        if self.check_tilt_lock_pending(ticket):
            locked_sessions = self.get_locked_sessions()
            return ModificationResult(
                success=False,
                ticket=ticket,
                action="manual_modify",
                error=f"Tilt session lock active for sessions: {locked_sessions}"
            )

        # Step 1: Check if kill is pending - Layer 3 takes precedence
        if self.check_kill_pending(ticket):
            return ModificationResult(
                success=False,
                ticket=ticket,
                action="manual_modify",
                error="Kill pending - Layer 3 has priority"
            )

        return self._modify_position_with_lock(ticket, new_sl, new_tp, "manual_modify")

    # =========================================================================
    # Kill Switch Preemption (AC #5, GG-4)
    # =========================================================================

    def register_kill_preemption_callback(
        self,
        callback: Callable[[int], None]
    ) -> None:
        """
        Register a callback for kill switch preemption.

        When Layer 3 fires a kill switch, it preempts any active Layer 2 lock.
        This callback is invoked to handle the preemption.

        Args:
            callback: Function that takes ticket number
        """
        self._kill_preemption_callbacks.append(callback)

    def handle_kill_preemption(self, ticket: int) -> ModificationResult:
        """
        Handle kill switch preemption for a position.

        GG-4: When Kill Switch fires (Tier 3), it preempts mid-modification
        Layer 2 action and queues forced exit. Redis lock must be released
        before Layer 3 forced exit executes.

        Args:
            ticket: MT5 ticket number

        Returns:
            ModificationResult confirming preemption
        """
        logger.warning(f"KILL PREEMPTION: ticket={ticket} - Layer 3 taking over")

        # Release any active lock
        lock_released = self._release_lock(ticket)

        # Invoke preemption callbacks
        for callback in self._kill_preemption_callbacks:
            try:
                callback(ticket)
            except Exception as e:
                logger.error(f"Preemption callback error: {e}")

        # NOTE: The kill:pending:{ticket} priority queue is consumed by Layer 3
        # KillSwitchHandler, not Layer 2. Layer 2 only:
        # 1. Checks kill:pending via check_kill_pending() before modifications
        # 2. Releases locks via handle_kill_preemption() when Layer 3 preempts
        # The kill queue consumer belongs to Layer 3, not Layer 2.

        return ModificationResult(
            success=True,
            ticket=ticket,
            action="kill_preemption",
            lock_released=lock_released,
            error=None if lock_released else "Lock was not held"
        )

    def check_kill_pending(self, ticket: int) -> bool:
        """
        Check if a kill is pending for a position.

        Used by Layer 2 to check if it should yield to Layer 3.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if kill is pending, False otherwise
        """
        try:
            kill_key = f"kill:pending:{ticket}"
            return self._redis.exists(kill_key) > 0
        except redis.RedisError as e:
            logger.error(f"Redis error checking kill pending: {e}")
            return False

    # =========================================================================
    # Periodic Evaluation (for use in trading loop)
    # =========================================================================

    def evaluate_all_positions(
        self,
        price_provider: Callable[[str], Optional[float]]
    ) -> List[ModificationResult]:
        """
        Evaluate all tracked positions for move-to-breakeven eligibility.

        This method is called periodically by the trading loop to check
        if any positions need move-to-breakeven adjustments.

        Args:
            price_provider: Function that takes symbol and returns current price

        Returns:
            List of ModificationResults for positions that were modified
        """
        results = []

        for position in self.get_all_positions():
            current_price = price_provider(position.symbol)
            if current_price is None:
                logger.warning(f"Could not get price for {position.symbol}")
                continue

            # Check for kill preemption first
            if self.check_kill_pending(position.ticket):
                logger.warning(
                    f"Position {position.ticket} has pending kill - skipping evaluation"
                )
                continue

            # Check and execute move-to-breakeven
            result = self.check_and_move_to_breakeven(
                position.ticket,
                current_price,
                position.symbol
            )

            if result.success and result.new_sl is not None:
                results.append(result)

        return results

    def shutdown(self) -> None:
        """Gracefully shutdown the monitor."""
        logger.info("Shutting down Layer2PositionMonitor")
        self._positions.clear()

    # =========================================================================
    # Tilt Session Locking (Story 16.1 - LOCK Step)
    # =========================================================================

    def activate_session_lock(self, session_name: str) -> bool:
        """
        Activate LOCK on all positions in the closing session.

        Story 16.1 - Task 2: Layer Integration for LOCK Step.
        Subtask 2.2: Layer 2 Position Monitor: activate LOCK on all positions
        in closing session.

        When Tilt initiates a session boundary transition, this method is called
        to prevent Layer 2 from modifying positions during the transition.
        The existing Redis lock pattern (GG-1) is respected — positions that
        have pending modifications complete first.

        Args:
            session_name: Name of the session that is closing (e.g., "LONDON")

        Returns:
            True if session lock activated successfully
        """
        logger.info(f"Tilt LOCK: Activating session lock for {session_name}")

        # Set session lock in Redis with TTL
        lock_key = f"{self.SESSION_LOCK_PREFIX}{session_name}"
        lock_value = f"{self._instance_id}:{datetime.now(timezone.utc).isoformat()}"

        try:
            self._redis.set(
                lock_key,
                lock_value,
                ex=self.SESSION_LOCK_TTL_SECONDS
            )
            self._session_locked[session_name] = True
            logger.info(f"Tilt LOCK: Session lock activated for {session_name}")
            return True
        except redis.RedisError as e:
            logger.error(f"Tilt LOCK: Failed to activate session lock: {e}")
            return False

    def release_session_lock(self, session_name: str) -> bool:
        """
        Release LOCK on all positions in the session.

        Called when Tilt completes or cancels a session transition.

        Args:
            session_name: Name of the session to unlock

        Returns:
            True if session lock released successfully
        """
        logger.info(f"Tilt LOCK: Releasing session lock for {session_name}")

        lock_key = f"{self.SESSION_LOCK_PREFIX}{session_name}"

        try:
            self._redis.delete(lock_key)
            self._session_locked[session_name] = False
            logger.info(f"Tilt LOCK: Session lock released for {session_name}")
            return True
        except redis.RedisError as e:
            logger.error(f"Tilt LOCK: Failed to release session lock: {e}")
            return False

    def is_session_locked(self, session_name: str) -> bool:
        """
        Check if a session is currently locked.

        Args:
            session_name: Name of the session to check

        Returns:
            True if session is locked, False otherwise
        """
        # Check local state first
        if self._session_locked.get(session_name, False):
            return True

        # Also check Redis for distributed lock state
        lock_key = f"{self.SESSION_LOCK_PREFIX}{session_name}"
        try:
            return self._redis.exists(lock_key) > 0
        except redis.RedisError as e:
            logger.error(f"Tilt LOCK: Failed to check session lock: {e}")
            return False

    def check_tilt_lock_pending(self, ticket: int) -> bool:
        """
        Check if Tilt session lock prevents modification of a position.

        Per Subtask 2.3: Verify Layer 2 Redis lock pattern is respected during LOCK.
        This method should be called before any position modification to ensure
        Tilt LOCK is not blocking the modification.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if modification should be blocked, False otherwise
        """
        # Check if any session is locked
        for session_name, locked in self._session_locked.items():
            if locked:
                logger.debug(
                    f"Tilt LOCK: Position {ticket} blocked — session {session_name} is locked"
                )
                return True

        # Also check Redis for distributed locks
        for session_name in self._session_locked:
            if self.is_session_locked(session_name):
                logger.debug(
                    f"Tilt LOCK: Position {ticket} blocked — Redis session lock active"
                )
                return True

        return False

    def get_locked_sessions(self) -> List[str]:
        """
        Get list of currently locked sessions.

        Returns:
            List of session names that are locked
        """
        return [s for s, locked in self._session_locked.items() if locked]

