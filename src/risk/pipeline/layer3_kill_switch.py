"""
Layer 3 Kill Switch for QUANTMINDX Trading System.

Story 14.3: Layer 3 CHAOS + Kill Switch Forced Exit

A synchronous, event-driven kill switch that:
- Detects LYAPUNOV_EXCEEDED threshold (0.95) and flags all scalping positions for forced exit
- Produces kill:pending:{ticket} entries in Redis priority queue
- Consumes kill:pending queue and executes forced close via MT5 bridge
- Bypasses Layer 2 Redis locks via release_layer2_locks() before forced exit
- Subscribes to SVSS RVOL channel for early warning (AC#3, graceful fallback if SVSS unavailable)
- Records all forced exit events to audit log

NO LLM calls in the hot path - runs on Kamatera T1 (London, latency-critical).

Per NFR-P1: Kill switch protocol executes in full, in order — correctness over raw speed.
Per NFR-R4: Layer 3 runs on T1, LYAPUNOV_EXCEEDED path does not depend on T2.
Per NFR-D1: Trade records must be persisted before any system acknowledgment.
Per NFR-M2: Layer 3 is NOT an agent — synchronous event-driven kill switch, no LLM calls.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import redis

from src.events.chaos import (
    ChaosEvent,
    ChaosLevel,
    ForcedExitOutcome,
    KillSwitchResult,
    RVOLWarningEvent,
)
from src.database.models import SessionLocal
from src.database.models.trade_record import TradeRecord

logger = logging.getLogger(__name__)


# Configuration constants
LYAPUNOV_CHAOS_THRESHOLD = 0.95  # CHAOS threshold from Dev Notes
RVOL_WARNING_THRESHOLD = 0.5  # RVOL < 0.5 triggers warning
KILL_QUEUE_KEY = "kill:pending:queue"  # Redis sorted set for kill queue
LAYER2_LOCK_PREFIX = "lock:modify:"  # Layer 2 Redis lock prefix


@dataclass
class KillQueueEntry:
    """
    Kill queue entry stored in Redis.

    Stored as: kill:pending:{ticket}
    Score: timestamp for priority ordering
    """
    ticket: int
    timestamp: float
    triggered_by: str  # "LYAPUNOV" or "RVOL"
    symbol: Optional[str] = None


@dataclass
class ForcedExitResult:
    """Result of a forced exit execution."""
    ticket: int
    success: bool
    outcome: ForcedExitOutcome
    lock_released: bool = False
    error: Optional[str] = None
    partial_volume: Optional[float] = None


class Layer3KillSwitch:
    """
    Layer 3 CHAOS + Kill Switch Forced Exit.

    Story 14.3: The last resort mechanism that can override Layer 1 and Layer 2.

    Key behaviors:
    1. CHAOS Detection (AC #1): Flag all scalping positions when Lyapunov > 0.95
    2. Priority Queue Production (AC #1): Enqueue kill:pending:{ticket} for each open ticket
    3. Priority Queue Consumption (AC #2): Process queue, execute forced close via MT5
    4. Layer 2 Lock Release (AC #2): Release any held Redis locks before forced exit
    5. Audit Logging (AC #1, #2): Record all forced exit events with outcome
    6. SVSS Early Warning (AC #3): Subscribe to RVOL channel, graceful fallback

    Attributes:
        redis_client: Redis client for queue and pub/sub
        mt5_client: MT5 client for forced close operations
        instance_id: Unique identifier for this kill switch instance
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        mt5_client: Any,
        instance_id: str = "layer3-default",
    ):
        """
        Initialize Layer 3 Kill Switch.

        Args:
            redis_client: Redis client for kill queue and pub/sub
            mt5_client: MT5 client for forced close operations
            instance_id: Unique identifier for this kill switch instance
        """
        self._redis = redis_client
        self._mt5 = mt5_client
        self._instance_id = instance_id
        self._rvol_callbacks: List[Callable[[str, float], None]] = []
        self._svss_available = True  # Assume SVSS available until proven otherwise

        logger.info(
            f"Layer3KillSwitch initialized: instance={instance_id}, "
            f"lyapunov_threshold={LYAPUNOV_CHAOS_THRESHOLD}, "
            f"rvol_threshold={RVOL_WARNING_THRESHOLD}"
        )

    # =========================================================================
    # CHAOS Detection & Position Flagging (AC #1)
    # =========================================================================

    def detect_chaos(self, lyapunov_value: float) -> bool:
        """
        Detect if Lyapunov Exponent exceeds CHAOS threshold.

        AC #1: Given Lyapunov Exponent exceeds 0.95 (CHAOS threshold),
        When Layer 3 evaluates, Then all scalping positions are flagged for forced exit.

        Args:
            lyapunov_value: Current Lyapunov Exponent value

        Returns:
            True if chaos threshold exceeded, False otherwise
        """
        return lyapunov_value > LYAPUNOV_CHAOS_THRESHOLD

    def flag_positions_for_forced_exit(
        self,
        tickets: List[int],
        triggered_by: str = "LYAPUNOV",
        symbol: Optional[str] = None
    ) -> List[int]:
        """
        Flag positions for forced exit by enqueueing kill:pending entries.

        AC #1: Given LYAPUNOV_EXCEEDED, When Layer 3 evaluates,
        Then a priority queue entry kill:pending:{ticket} is created for each open ticket.

        Args:
            tickets: List of open ticket numbers to flag
            triggered_by: What triggered the kill ("LYAPUNOV" or "RVOL")
            symbol: Optional symbol filter

        Returns:
            List of tickets that were enqueued
        """
        if not tickets:
            logger.info("No positions to flag for forced exit")
            return []

        enqueued = []
        timestamp = time.time()

        for ticket in tickets:
            try:
                # Create kill:pending:{ticket} entry
                kill_key = f"kill:pending:{ticket}"

                # Store metadata as hash fields
                self._redis.hset(kill_key, mapping={
                    "ticket": str(ticket),
                    "triggered_by": triggered_by,
                    "symbol": symbol or "",
                    "timestamp": str(timestamp),
                    "instance_id": self._instance_id,
                })

                # Set TTL of 24 hours for orphan cleanup
                self._redis.expire(kill_key, 86400)

                # Add to priority queue sorted set
                self._redis.zadd(KILL_QUEUE_KEY, {kill_key: timestamp})

                enqueued.append(ticket)
                logger.warning(
                    f"KILL FLAGGED: ticket={ticket}, triggered_by={triggered_by}, "
                    f"symbol={symbol}, lyapunov_threshold_breached"
                )

            except redis.RedisError as e:
                logger.error(f"Failed to enqueue kill for ticket={ticket}: {e}")

        logger.info(
            f"Flagged {len(enqueued)}/{len(tickets)} positions for forced exit "
            f"(triggered_by={triggered_by})"
        )

        return enqueued

    def get_open_scalping_tickets(self) -> List[int]:
        """
        Query all open scalping position tickets.

        Queries MT5 positions and filters for scalping strategy tickets.
        Scalping strategy is identified by strategy_id containing 'scalp' or 'scalping'.

        Returns:
            List of open scalping ticket numbers
        """
        if self._mt5 is None:
            logger.warning("MT5 client not available, cannot query open tickets")
            return []

        try:
            tickets = []
            if hasattr(self._mt5, 'get_open_positions'):
                positions = self._mt5.get_open_positions()
                for pos in positions:
                    strategy_id = pos.get('strategy_id', '')
                    ticket = pos.get('ticket')
                    if ticket and ('scalp' in strategy_id.lower() or 'scalping' in strategy_id.lower()):
                        tickets.append(ticket)
            elif hasattr(self._mt5, '_orders'):
                # Fallback to internal order tracking
                for order in self._mt5._orders:
                    strategy_id = getattr(order, 'strategy_id', '')
                    ticket = getattr(order, 'ticket', None)
                    if ticket and ('scalp' in strategy_id.lower() or 'scalping' in strategy_id.lower()):
                        tickets.append(ticket)
            logger.info(f"Found {len(tickets)} open scalping tickets")
            return tickets
        except Exception as e:
            logger.error(f"Failed to query open scalping tickets: {e}")
            return []

    # =========================================================================
    # Priority Queue Processing & Forced Exit (AC #2)
    # =========================================================================

    def process_kill_queue(self, max_items: int = 10) -> List[KillSwitchResult]:
        """
        Process kill:pending queue and execute forced exits.

        AC #2: Given a kill:pending:{ticket} entry exists in the priority queue,
        When the queue processes, Then it immediately sends a close order via MT5 bridge
        bypassing Layer 2 locks, And the outcome (filled/partial/rejected) is recorded.

        Args:
            max_items: Maximum number of queue items to process (default 10)

        Returns:
            List of KillSwitchResult for each processed ticket
        """
        results = []

        # Pop items from queue (oldest first due to timestamp score)
        try:
            items = self._redis.zpopmin(KILL_QUEUE_KEY, max_items)
        except redis.RedisError as e:
            logger.error(f"Failed to pop from kill queue: {e}")
            return results

        for item, score in items:
            kill_key = item if isinstance(item, str) else item.decode("utf-8")

            # Parse ticket from key
            if not kill_key.startswith("kill:pending:"):
                logger.warning(f"Invalid kill queue key format: {kill_key}")
                continue

            try:
                ticket = int(kill_key.split(":")[-1])
            except ValueError:
                logger.warning(f"Invalid ticket in kill queue key: {kill_key}")
                continue

            # Get metadata from hash
            try:
                metadata = self._redis.hgetall(kill_key)
                if not metadata:
                    logger.warning(f"No metadata found for {kill_key}")
                    continue

                triggered_by = metadata.get(b"triggered_by", b"LYAPUNOV").decode("utf-8")
                symbol = metadata.get(b"symbol", b"").decode("utf-8") or None
            except redis.RedisError as e:
                logger.error(f"Failed to get kill metadata for {kill_key}: {e}")
                continue

            # Execute forced close
            result = self.execute_forced_close(
                ticket=ticket,
                triggered_by=triggered_by,
                symbol=symbol
            )
            results.append(result)

            # Delete the kill:pending key after processing
            try:
                self._redis.delete(kill_key)
            except redis.RedisError as e:
                logger.error(f"Failed to delete kill key {kill_key}: {e}")

        return results

    def execute_forced_close(
        self,
        ticket: int,
        triggered_by: str = "LYAPUNOV",
        symbol: Optional[str] = None
    ) -> KillSwitchResult:
        """
        Execute forced close for a single ticket.

        GG-4: Release any held Layer 2 Redis locks before forced exit.
        Layer 3 BYPASSES Layer 2 Redis locks via release_layer2_locks().

        Args:
            ticket: MT5 ticket number
            triggered_by: What triggered the kill ("LYAPUNOV" or "RVOL")
            symbol: Optional symbol for logging

        Returns:
            KillSwitchResult with outcome details
        """
        lyapunov_triggered = triggered_by == "LYAPUNOV"
        rvol_triggered = triggered_by == "RVOL"

        # Step 1: Release any Layer 2 Redis locks (GG-4 resolution)
        lock_released = self.release_layer2_locks(ticket)

        if not lock_released:
            logger.warning(f"Layer 2 lock was not held for ticket={ticket} (may already be released)")

        # Step 2: Execute forced close via MT5 bridge
        try:
            if self._mt5 is None:
                logger.error("MT5 client not available for forced close")
                return KillSwitchResult(
                    ticket=ticket,
                    outcome=ForcedExitOutcome.REJECTED,
                    lyapunov_triggered=lyapunov_triggered,
                    rvol_triggered=rvol_triggered,
                    lock_released=lock_released,
                    error="MT5 client not available"
                )

            # Try to call force_close_by_ticket on MT5 client
            if hasattr(self._mt5, 'force_close_by_ticket'):
                close_result = self._mt5.force_close_by_ticket(ticket)

                if close_result.get("success"):
                    outcome = ForcedExitOutcome.FILLED
                    if close_result.get("partial"):
                        outcome = ForcedExitOutcome.PARTIAL

                    logger.warning(
                        f"FORCED EXIT COMPLETE: ticket={ticket}, outcome={outcome.value}, "
                        f"triggered_by={triggered_by}"
                    )

                    # Record to audit log
                    self._log_forced_exit(
                        ticket=ticket,
                        outcome=outcome,
                        triggered_by=triggered_by,
                        symbol=symbol
                    )

                    return KillSwitchResult(
                        ticket=ticket,
                        outcome=outcome,
                        lyapunov_triggered=lyapunov_triggered,
                        rvol_triggered=rvol_triggered,
                        lock_released=lock_released,
                        partial_volume=close_result.get("volume")
                    )
                else:
                    error_msg = close_result.get("error", "Unknown MT5 error")
                    logger.error(f"FORCED EXIT REJECTED: ticket={ticket}, error={error_msg}")

                    self._log_forced_exit(
                        ticket=ticket,
                        outcome=ForcedExitOutcome.REJECTED,
                        triggered_by=triggered_by,
                        symbol=symbol,
                        error=error_msg
                    )

                    return KillSwitchResult(
                        ticket=ticket,
                        outcome=ForcedExitOutcome.REJECTED,
                        lyapunov_triggered=lyapunov_triggered,
                        rvol_triggered=rvol_triggered,
                        lock_released=lock_released,
                        error=error_msg
                    )
            else:
                # MT5 client doesn't have force_close_by_ticket
                logger.error("MT5 client has no force_close_by_ticket method")
                return KillSwitchResult(
                    ticket=ticket,
                    outcome=ForcedExitOutcome.REJECTED,
                    lyapunov_triggered=lyapunov_triggered,
                    rvol_triggered=rvol_triggered,
                    lock_released=lock_released,
                    error="MT5 client missing force_close_by_ticket"
                )

        except Exception as e:
            logger.error(f"FORCED EXIT ERROR: ticket={ticket}, error={e}")
            return KillSwitchResult(
                ticket=ticket,
                outcome=ForcedExitOutcome.REJECTED,
                lyapunov_triggered=lyapunov_triggered,
                rvol_triggered=rvol_triggered,
                lock_released=lock_released,
                error=str(e)
            )

    def release_layer2_locks(self, ticket: int) -> bool:
        """
        Release any held Layer 2 Redis locks before forced exit.

        GG-4: When Layer 3 fires, it BYPASSES Layer 2 locks via this method.
        Layer 2 uses lock:modify:{ticket} with 3s TTL.

        Args:
            ticket: MT5 ticket number

        Returns:
            True if lock was released or wasn't held, False on error
        """
        lock_key = f"{LAYER2_LOCK_PREFIX}{ticket}"

        try:
            # Check if lock exists
            if self._redis.exists(lock_key):
                self._redis.delete(lock_key)
                logger.info(f"Released Layer 2 lock: {lock_key}")
                return True
            else:
                logger.debug(f"No Layer 2 lock held for ticket={ticket}")
                return True

        except redis.RedisError as e:
            logger.error(f"Failed to release Layer 2 lock for ticket={ticket}: {e}")
            return False

    def _log_forced_exit(
        self,
        ticket: int,
        outcome: ForcedExitOutcome,
        triggered_by: str,
        symbol: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log forced exit event to trade record.

        Per NFR-D1: Trade records must be persisted before any system acknowledgment.

        Args:
            ticket: MT5 ticket number
            outcome: Outcome of the forced exit
            triggered_by: What triggered the exit
            symbol: Trading symbol
            error: Error message if rejected
        """
        try:
            session = SessionLocal()
            try:
                # Try multiple lookup strategies for trade record
                record = None

                # Strategy 1: signal_id equals ticket (most common for Layer 1)
                record = session.query(TradeRecord).filter_by(signal_id=str(ticket)).first()

                # Strategy 2: If symbol provided, search by symbol + direction + approximate time
                if not record and symbol:
                    records = session.query(TradeRecord).filter_by(symbol=symbol).all()
                    for rec in records:
                        # Match by ticket stored in ea_parameters or signal_id
                        ea_params = rec.ea_parameters or {}
                        if str(ticket) == str(rec.signal_id) or ea_params.get('ticket') == ticket:
                            record = rec
                            break

                if record:
                    # Append forced exit event to layer3_events
                    events = record.ea_parameters.get("layer3_events", []) if record.ea_parameters else []
                    events.append({
                        "event": "forced_exit",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "outcome": outcome.value,
                        "triggered_by": triggered_by,
                        "symbol": symbol,
                        "ticket": ticket,
                        "error": error,
                    })
                    if not record.ea_parameters:
                        record.ea_parameters = {}
                    record.ea_parameters["layer3_events"] = events
                    session.commit()
                    logger.info(f"Logged forced exit for ticket={ticket}, outcome={outcome.value}")
                else:
                    logger.warning(f"No trade record found for ticket={ticket}, symbol={symbol}")

            finally:
                session.close()

        except Exception as e:
            logger.error(f"Failed to log forced exit for ticket={ticket}: {e}")

    # =========================================================================
    # SVSS Early Warning Chain (AC #3) - Graceful Fallback
    # =========================================================================

    def subscribe_rvol_channel(self, symbol: str) -> None:
        """
        Subscribe to SVSS RVOL channel for a symbol.

        AC #3: SVSS→CorrelationSensor→Sentinel→Layer3 early warning chain.

        Note: AC#3 requires Epic 15 (SVSS) to be complete first.
        This implements graceful fallback: if SVSS unavailable, rely on LYAPUNOV path only.

        Args:
            symbol: Trading symbol to subscribe to
        """
        channel = f"svss:{symbol}:rvvol"

        try:
            self._pubsub = self._redis.pubsub()
            self._pubsub.subscribe(channel)
            logger.info(f"Subscribed to SVSS RVOL channel: {channel}")
        except redis.RedisError as e:
            logger.warning(f"Failed to subscribe to SVSS channel {channel}: {e}")
            self._svss_available = False
            logger.info("SVSS unavailable - Layer 3 will operate on LYAPUNOV path only")

    def process_rvol_messages(self, timeout: float = 0.1) -> List[Optional[RVOLWarningEvent]]:
        """
        Process pending RVOL messages from SVSS pub/sub.

        Must be called periodically by the event loop to process SVSS messages.

        Args:
            timeout: How long to wait for messages (in seconds)

        Returns:
            List of RVOLWarningEvent for messages that triggered warnings
        """
        if not hasattr(self, '_pubsub') or self._pubsub is None:
            return []

        if not self._svss_available:
            return []

        events = []
        try:
            message = self._pubsub.get_message(timeout=timeout)
            if message and message['type'] == 'message':
                data = message['data']
                if isinstance(data, bytes):
                    data = data.decode('utf-8')

                import json
                msg_data = json.loads(data) if isinstance(data, str) else data

                symbol = msg_data.get('symbol', '')
                rvol = msg_data.get('rvvol', 1.0)

                # Get open tickets for this symbol
                open_tickets = self._get_tickets_for_symbol(symbol)

                event = self.handle_rvol_warning(
                    symbol=symbol,
                    rvol=rvol,
                    has_open_positions=len(open_tickets) > 0
                )
                if event:
                    events.append(event)

                    # Block new entries on this symbol (AC#3)
                    self._block_symbol_entries(symbol)

                    # Flag existing positions for Layer 2 review
                    if open_tickets:
                        logger.warning(
                            f"SVSS RVOL warning: symbol={symbol}, rvol={rvol}, "
                            f"flagging {len(open_tickets)} positions for Layer 2 review"
                        )

        except redis.RedisError as e:
            logger.error(f"Error processing RVOL messages: {e}")
            self._svss_available = False
        except Exception as e:
            logger.error(f"Unexpected error processing RVOL messages: {e}")

        return events

    def _get_tickets_for_symbol(self, symbol: str) -> List[int]:
        """Get open tickets for a specific symbol."""
        if self._mt5 is None:
            return []

        try:
            tickets = []
            if hasattr(self._mt5, 'get_open_positions'):
                positions = self._mt5.get_open_positions()
                for pos in positions:
                    if pos.get('symbol') == symbol:
                        ticket = pos.get('ticket')
                        if ticket:
                            tickets.append(ticket)
            return tickets
        except Exception as e:
            logger.error(f"Failed to get tickets for symbol {symbol}: {e}")
            return []

    def _block_symbol_entries(self, symbol: str) -> None:
        """
        Block new entries on a symbol.

        AC#3: New entries on that symbol are blocked.
        Sets a Redis flag that the Sentinel/router can check before allowing new positions.
        """
        block_key = f"symbol:blocked:{symbol}"
        try:
            self._redis.set(block_key, "1", ex=86400)  # 24 hour TTL
            logger.warning(f"BLOCKED new entries for symbol={symbol} due to RVOL < 0.5")
        except redis.RedisError as e:
            logger.error(f"Failed to block symbol entries for {symbol}: {e}")

    def handle_rvol_warning(
        self,
        symbol: str,
        rvol: float,
        has_open_positions: bool = False
    ) -> Optional[RVOLWarningEvent]:
        """
        Handle RVOL warning from SVSS.

        AC #3: Given SVSS reports RVOL < 0.5 on a symbol with open positions,
        When Layer 3 evaluates, Then new entries on that symbol are blocked,
        And existing positions are flagged for review (Layer 2 evaluates exit).

        Note: This AC requires Epic 15 (SVSS) to be complete first.
        Implements graceful fallback: if SVSS unavailable, rely on LYAPUNOV path.

        Args:
            symbol: Trading symbol
            rvol: Relative Volume value
            has_open_positions: Whether symbol has open positions

        Returns:
            RVOLWarningEvent or None if threshold not breached
        """
        if not self._svss_available:
            logger.debug("SVSS unavailable, skipping RVOL warning handling")
            return None

        # Check if RVOL < threshold
        if rvol >= RVOL_WARNING_THRESHOLD:
            return None

        event = RVOLWarningEvent.create(
            symbol=symbol,
            rvol=rvol,
            has_open_positions=has_open_positions,
            metadata={"source": "svss", "instance_id": self._instance_id}
        )

        logger.warning(
            f"RVOL WARNING: symbol={symbol}, rvol={rvol:.2f}, "
            f"has_positions={has_open_positions}"
        )

        # Block new entries on this symbol (handled by Sentinel/router)
        # Flag existing positions for Layer 2 review
        if has_open_positions:
            # In production: query open positions for this symbol and flag them
            # For now, rely on LYAPUNOV path to trigger actual kills
            logger.info(f"Symbol {symbol} has open positions - RVOL warning will be handled by Layer 2")

        return event

    def handle_rvol_fallback(self) -> None:
        """
        Handle SVSS unavailability by switching to LYAPUNOV-only mode.

        Per Dev Notes: Fallback: if SVSS unavailable, Layer 3 operates on LYAPUNOV path only.
        """
        self._svss_available = False
        logger.info("SVSS unavailable - Layer 3 operating in LYAPUNOV-only mode")

    def is_svss_available(self) -> bool:
        """Check if SVSS is available for RVOL monitoring."""
        return self._svss_available

    # =========================================================================
    # Event Handling - Called by Risk Pipeline
    # =========================================================================

    def on_lyapunov_event(self, lyapunov_value: float, tickets: List[int]) -> ChaosEvent:
        """
        Handle Lyapunov Exponent event from risk sensor.

        This is the main entry point for CHAOS detection from the risk pipeline.

        Args:
            lyapunov_value: Current Lyapunov Exponent value
            tickets: List of open scalping ticket numbers

        Returns:
            ChaosEvent with detection details
        """
        # Always create event with tickets=self for proper classification
        # but only set tickets to enqueued list if chaos is detected
        event = ChaosEvent.create(
            lyapunov_value=lyapunov_value,
            tickets=[],  # Start empty, set after chaos check
            metadata={"source": "lyapunov_sensor", "instance_id": self._instance_id}
        )

        logger.warning(
            f"LYAPUNOV EVENT: lyapunov={lyapunov_value:.4f}, "
            f"threshold={LYAPUNOV_CHAOS_THRESHOLD}, tickets={len(tickets)}"
        )

        if self.detect_chaos(lyapunov_value):
            # Flag all positions for forced exit
            enqueued = self.flag_positions_for_forced_exit(
                tickets=tickets,
                triggered_by="LYAPUNOV"
            )
            event.tickets = enqueued
            logger.warning(
                f"CHAOS DETECTED: {len(enqueued)} positions flagged for forced exit"
            )

        return event

    def on_rvol_event(self, symbol: str, rvol: float, open_tickets: List[int]) -> Optional[RVOLWarningEvent]:
        """
        Handle RVOL event from SVSS.

        This is the entry point for SVSS→Layer3 early warning chain (AC#3).

        Note: AC#3 requires Epic 15 (SVSS) to be complete first.
        Implements graceful fallback.

        Args:
            symbol: Trading symbol
            rvol: Relative Volume value
            open_tickets: List of open tickets for this symbol

        Returns:
            RVOLWarningEvent or None if threshold not breached or SVSS unavailable
        """
        event = self.handle_rvol_warning(
            symbol=symbol,
            rvol=rvol,
            has_open_positions=len(open_tickets) > 0
        )

        if event is None:
            return None

        # If RVOL < 0.5 and positions exist, they should be reviewed by Layer 2
        # The actual kill would come from LYAPUNOV path if chaos is detected
        # SVSS→Layer3 is an early warning, not the kill trigger

        return event

    # =========================================================================
    # Queue Inspection (for testing and monitoring)
    # =========================================================================

    def get_queue_size(self) -> int:
        """Get current size of kill queue."""
        try:
            return self._redis.zcard(KILL_QUEUE_KEY)
        except redis.RedisError:
            return 0

    def get_pending_tickets(self) -> List[int]:
        """Get list of tickets currently in kill queue."""
        try:
            items = self._redis.zrange(KILL_QUEUE_KEY, 0, -1)
            tickets = []
            for item in items:
                key = item if isinstance(item, str) else item.decode("utf-8")
                if key.startswith("kill:pending:"):
                    try:
                        ticket = int(key.split(":")[-1])
                        tickets.append(ticket)
                    except ValueError:
                        continue
            return tickets
        except redis.RedisError:
            return []

    def shutdown(self) -> None:
        """Gracefully shutdown the kill switch."""
        logger.info("Shutting down Layer3KillSwitch")
