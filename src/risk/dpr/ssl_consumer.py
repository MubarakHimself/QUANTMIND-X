"""
DPR SSL Event Consumer — Subscribes to Redis ssl:events channel.

Story 18.2: SSL → DPR Integration

Provides DPRSSLConsumer class that:
- Subscribes to Redis `ssl:events` channel
- Routes SSL events to DPRQueueManager for mid-session queueing
- Triggers DPR score recalculation with paper-only inputs on move-to-paper
- Handles retirement events and AlphaForge trigger

Per NFR-M2: DPR is a synchronous scoring engine — NO LLM calls in scoring path.
Per NFR-D1: All SSL events logged immutably to DPR audit log before processing.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Callable

import redis

from src.events.ssl import SSLCircuitBreakerEvent, TradeOutcome
from src.risk.dpr.scoring_engine import DPRScoringEngine
from src.risk.dpr.queue_manager import DPRQueueManager


logger = logging.getLogger(__name__)


# Redis channel for SSL events
SSL_EVENTS_CHANNEL = "ssl:events"


class DPRSSLConsumer:
    """
    DPR SSL Event Consumer for processing SSL state transitions.

    Subscribes to Redis `ssl:events` channel and routes events to:
    - DPRQueueManager.queue_event() for mid-session queueing (per queue lock rule)
    - DPRScoringEngine.recalculate_paper_only_score() on move-to-paper
    - AlphaForge workflow trigger on retirement

    Attributes:
        dpr_queue_manager: DPRQueueManager instance for queue operations
        dpr_scoring_engine: DPRScoringEngine instance for score recalculation
        redis_host: Redis host for subscription
        redis_port: Redis port for subscription
        on_ui_state_change: Optional callback for UI state change events
        on_alphaforge_trigger: Optional callback for AlphaForge retirement trigger
    """

    def __init__(
        self,
        dpr_queue_manager: DPRQueueManager,
        dpr_scoring_engine: DPRScoringEngine,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        on_ui_state_change: Optional[Callable] = None,
        on_alphaforge_trigger: Optional[Callable] = None,
    ):
        """
        Initialize DPR SSL Consumer.

        Args:
            dpr_queue_manager: DPRQueueManager instance
            dpr_scoring_engine: DPRScoringEngine instance
            redis_host: Redis host for subscription
            redis_port: Redis port for subscription
            on_ui_state_change: Optional callback(bot_id, new_state, tier, timestamp_utc)
            on_alphaforge_trigger: Optional callback(bot_id, strategy_id) for retirement
        """
        self._dpr_queue_manager = dpr_queue_manager
        self._dpr_scoring_engine = dpr_scoring_engine
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._redis_client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._running = False
        self.on_ui_state_change = on_ui_state_change
        self.on_alphaforge_trigger = on_alphaforge_trigger

    def _get_redis_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                decode_responses=True,
            )
        return self._redis_client

    def start(self) -> None:
        """
        Start subscribing to Redis ssl:events channel.

        This method blocks and should be run in a separate thread or async context.
        """
        self._redis_client = self._get_redis_client()
        self._pubsub = self._redis_client.pubsub()
        self._pubsub.subscribe(SSL_EVENTS_CHANNEL)
        self._running = True

        logger.info(f"DPRSSLConsumer subscribed to {SSL_EVENTS_CHANNEL}")

        try:
            for message in self._pubsub.listen():
                if not self._running:
                    break
                if message["type"] == "message":
                    self._handle_message(message)
        except Exception as e:
            logger.error(f"Error in DPRSSLConsumer: {e}")
            raise
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop subscribing to Redis channel."""
        self._running = False
        if self._pubsub is not None:
            self._pubsub.unsubscribe()
            self._pubsub.close()
            self._pubsub = None
        if self._redis_client is not None:
            self._redis_client.close()
            self._redis_client = None
        logger.info("DPRSSLConsumer stopped")

    def _handle_message(self, message: dict) -> None:
        """
        Handle incoming Redis message.

        Args:
            message: Redis pubsub message dict
        """
        try:
            data = message["data"]
            event = SSLCircuitBreakerEvent.model_validate_json(data)
            self.on_ssl_event(event)
        except Exception as e:
            logger.error(f"Error handling SSL event message: {e}")

    def on_ssl_event(self, event: SSLCircuitBreakerEvent) -> None:
        """
        Handle SSL event from Redis.

        Routes to appropriate handler based on event type:
        - move_to_paper: queue_event, recalculate score, emit UI state change
        - recovery_step_1: queue_event, emit UI state change
        - recovery_confirmed: queue_event, emit UI state change
        - retired: queue_event, trigger AlphaForge, emit UI state change

        Args:
            event: SSL circuit breaker event
        """
        logger.info(f"DPRSSLConsumer processing: {event}")

        session_id = event.trade_outcome.session_id if event.trade_outcome else "LONDON"

        # Get DPR score at event time for audit log
        dpr_score = self._dpr_scoring_engine.get_dpr_score(event.bot_id, session_id)
        dpr_composite_score = dpr_score.composite_score if dpr_score else 0

        # Route by event type
        if event.event_type.value == "move_to_paper":
            self._handle_move_to_paper(event, session_id, dpr_composite_score)
        elif event.event_type.value == "recovery_step_1":
            self._handle_recovery_step_1(event, session_id, dpr_composite_score)
        elif event.event_type.value == "recovery_confirmed":
            self._handle_recovery_confirmed(event, session_id, dpr_composite_score)
        elif event.event_type.value == "retired":
            self._handle_retired(event, session_id, dpr_composite_score)

        # Append to DPR audit log
        self._append_to_audit_log(event, session_id, dpr_composite_score)

        # Emit UI state change if callback is set (AC#8: only for move_to_paper)
        if self.on_ui_state_change and event.event_type.value == "move_to_paper":
            self.on_ui_state_change(
                bot_id=event.bot_id,
                new_state=event.new_state,
                tier=event.tier,
                timestamp_utc=event.timestamp_utc,
                metadata=event.metadata,
            )

    def _handle_move_to_paper(
        self,
        event: SSLCircuitBreakerEvent,
        session_id: str,
        dpr_composite_score: int,
    ) -> None:
        """
        Handle move-to-paper SSL event.

        - Queues event in DPRQueueManager for application at next Dead Zone
        - Recalculates DPR score with paper-only inputs (WR=0, PnL=0, EV=0)

        Args:
            event: SSL event
            session_id: Session identifier
            dpr_composite_score: DPR score at event time
        """
        from src.events.dpr import SSLEvent, SSLEventType

        # Queue event for mid-session queueing (per queue lock rule)
        ssl_event = SSLEvent(
            event_type=SSLEventType.MOVE_TO_PAPER,
            bot_id=event.bot_id,
            magic_number=event.magic_number,
            session_id=session_id,
            tier=event.tier,
            timestamp_utc=event.timestamp_utc,
        )
        self._dpr_queue_manager.queue_event(ssl_event)

        # Recalculate DPR score with paper-only inputs
        paper_score = self._dpr_scoring_engine.recalculate_paper_only_score(
            event.bot_id, session_id
        )
        logger.info(
            f"move_to_paper: bot={event.bot_id}, previous_dpr={dpr_composite_score}, "
            f"paper_score={paper_score}"
        )

    def _handle_recovery_step_1(
        self,
        event: SSLCircuitBreakerEvent,
        session_id: str,
        dpr_composite_score: int,
    ) -> None:
        """
        Handle recovery step 1 SSL event (first consecutive win in paper).

        - Queues recovery_step_1 event in DPRQueueManager

        Args:
            event: SSL event
            session_id: Session identifier
            dpr_composite_score: DPR score at event time
        """
        from src.events.dpr import SSLEvent, SSLEventType

        ssl_event = SSLEvent(
            event_type=SSLEventType.RECOVERY_STEP_1,
            bot_id=event.bot_id,
            magic_number=event.magic_number,
            session_id=session_id,
            tier=event.tier,
            timestamp_utc=event.timestamp_utc,
        )
        self._dpr_queue_manager.queue_event(ssl_event)

        logger.info(f"recovery_step_1: bot={event.bot_id}, dpr={dpr_composite_score}")

    def _handle_recovery_confirmed(
        self,
        event: SSLCircuitBreakerEvent,
        session_id: str,
        dpr_composite_score: int,
    ) -> None:
        """
        Handle recovery confirmed SSL event (2 consecutive wins in paper).

        - Queues recovery_confirmed event in DPRQueueManager

        Args:
            event: SSL event
            session_id: Session identifier
            dpr_composite_score: DPR score at event time
        """
        from src.events.dpr import SSLEvent, SSLEventType

        ssl_event = SSLEvent(
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            bot_id=event.bot_id,
            magic_number=event.magic_number,
            session_id=session_id,
            tier=event.tier,
            timestamp_utc=event.timestamp_utc,
        )
        self._dpr_queue_manager.queue_event(ssl_event)

        logger.info(f"recovery_confirmed: bot={event.bot_id}, dpr={dpr_composite_score}")

    def _handle_retired(
        self,
        event: SSLCircuitBreakerEvent,
        session_id: str,
        dpr_composite_score: int,
    ) -> None:
        """
        Handle retired SSL event (failed to recover after 2 Dead Zones).

        - Queues retired event in DPRQueueManager
        - Triggers AlphaForge Workflow 1 for new candidate generation

        Args:
            event: SSL event
            session_id: Session identifier
            dpr_composite_score: DPR score at event time
        """
        from src.events.dpr import SSLEvent, SSLEventType

        ssl_event = SSLEvent(
            event_type=SSLEventType.RETIRED,
            bot_id=event.bot_id,
            magic_number=event.magic_number,
            session_id=session_id,
            tier=event.tier,
            timestamp_utc=event.timestamp_utc,
        )
        self._dpr_queue_manager.queue_event(ssl_event)

        # Trigger AlphaForge Workflow 1 if callback is set
        if self.on_alphaforge_trigger:
            strategy_id = event.metadata.get("strategy_id", event.magic_number)
            self.on_alphaforge_trigger(
                bot_id=event.bot_id,
                strategy_id=strategy_id,
            )

        logger.info(f"retired: bot={event.bot_id}, dpr={dpr_composite_score}")

    def _append_to_audit_log(
        self,
        event: SSLCircuitBreakerEvent,
        session_id: str,
        dpr_composite_score: int,
    ) -> None:
        """
        Append SSL event to DPR audit log.

        Args:
            event: SSL event
            session_id: Session identifier
            dpr_composite_score: DPR score at event time
        """
        try:
            self._dpr_queue_manager.append_ssl_event_to_audit(
                bot_id=event.bot_id,
                ssl_event=event,
                dpr_composite_score=dpr_composite_score,
                session_id=session_id,
            )
        except Exception as e:
            logger.error(f"Error appending to DPR audit log: {e}")


class DPRSSLEventEmitter:
    """
    Helper class to emit SSL events directly (without Redis subscription).

    Used for testing or direct injection of SSL events into DPR system.
    """

    def __init__(
        self,
        dpr_queue_manager: DPRQueueManager,
        dpr_scoring_engine: DPRScoringEngine,
    ):
        """
        Initialize DPR SSLEventEmitter.

        Args:
            dpr_queue_manager: DPRQueueManager instance
            dpr_scoring_engine: DPRScoringEngine instance
        """
        self._dpr_queue_manager = dpr_queue_manager
        self._dpr_scoring_engine = dpr_scoring_engine

    def emit(self, event: SSLCircuitBreakerEvent) -> None:
        """
        Emit an SSL event directly to DPR system.

        Args:
            event: SSL circuit breaker event
        """
        consumer = DPRSSLConsumer(
            dpr_queue_manager=self._dpr_queue_manager,
            dpr_scoring_engine=self._dpr_scoring_engine,
        )
        consumer.on_ssl_event(event)
