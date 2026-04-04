"""
SSL Circuit Breaker — Per-Bot Consecutive Loss Counter & Paper Rotation.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Provides SSLCircuitBreaker class for:
- Tracking consecutive losses per bot (keyed on Magic Number)
- Firing circuit breaker when thresholds are breached (2 for scalping, 3 for ORB)
- Managing paper rotation and recovery

Per NFR-M2: SSL is a synchronous state machine — NO LLM calls in hot path.
Per NFR-D1: All SSL state transitions logged immutably before Redis publish.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import select, or_

from src.database.models import BotCircuitBreaker, SessionLocal
from src.risk.ssl.state import SSLCircuitBreakerState, SSLState, BotTier, is_valid_transition
from src.events.ssl import SSLCircuitBreakerEvent, SSLEventType


logger = logging.getLogger(__name__)


# Bot type thresholds
class BotType(str, Enum):
    """Bot strategy types for threshold determination."""
    SCALPING = "scalping"
    ORB = "orb"


# Loss thresholds by bot type
LOSS_THRESHOLDS: Dict[BotType, int] = {
    BotType.SCALPING: 2,
    BotType.ORB: 3,
}

# Redis channel for SSL events
SSL_EVENTS_CHANNEL = "ssl:events"

# Failure type constants for 3-loss-in-a-day rule (Story 18.3)
FAIL_S02 = "FAIL-S02"  # 3 losses in a day - halt for rest of day
FAIL_S03 = "FAIL-S03"  # 3 days with 3-loss trigger in week - quarantine for following week


class SSLCircuitBreaker:
    """
    SSL Circuit Breaker for per-bot consecutive loss tracking and paper rotation.

    Tracks consecutive losses keyed on Magic Number and triggers paper rotation
    when thresholds are breached.

    State Machine:
        LIVE → (2/3 consecutive losses) → PAPER
        PAPER → (2 consecutive paper wins) → RECOVERY
        RECOVERY → (DPR evaluation at Dead Zone) → LIVE
        PAPER → (fails to recover after limits) → RETIRED

    Attributes:
        db_session: SQLAlchemy session for database access
        redis_host: Redis host for event publishing
        redis_port: Redis port for event publishing
    """

    def __init__(
        self,
        db_session: Optional[Session] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        """
        Initialize SSL Circuit Breaker.

        Args:
            db_session: SQLAlchemy session (creates new if None)
            redis_host: Redis host for event publishing
            redis_port: Redis port for event publishing
        """
        self._db_session = db_session
        self._redis_host = redis_host
        self._redis_port = redis_port
        self._state_manager = None  # Lazy initialization
        self._redis_client = None  # Lazy Redis client
        # Story 18.3: 3-loss-in-a-day tracking
        self.daily_loss_count: Dict[str, int] = {}  # bot_id -> loss count today (resets at session boundary)
        self.weekly_3loss_trigger_count: Dict[str, int] = {}  # bot_id -> days with 3-loss trigger this week
        self._last_session_date: Optional[str] = None  # Track session date for daily reset
        self._current_week_start: Optional[datetime] = None  # Track week start for weekly reset

    @property
    def db_session(self) -> Session:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session

    @property
    def state_manager(self) -> SSLCircuitBreakerState:
        """Get or create state manager."""
        if self._state_manager is None:
            self._state_manager = SSLCircuitBreakerState(db_session=self.db_session)
        return self._state_manager

    def _get_bot_type(self, bot_id: str) -> BotType:
        """
        Determine bot type (scalping vs ORB) for threshold selection.

        Bot type is determined by strategy variant assigned to the bot.
        Checks BotManifest for strategy_type, or bot_tags for ORB indicators.

        Args:
            bot_id: Bot identifier

        Returns:
            BotType.SCALPING or BotType.ORB
        """
        try:
            from src.database.models import BotManifest, BotLifecycleLog
            from sqlalchemy import or_

            # First, try to get strategy type from BotManifest
            result = self.db_session.execute(
                select(BotManifest).where(BotManifest.bot_name == bot_id)
            ).scalar_one_or_none()

            if result and result.strategy_type:
                strategy = result.strategy_type.lower()
                if 'orb' in strategy or 'opening_range' in strategy:
                    return BotType.ORB
                elif 'scalp' in strategy:
                    return BotType.SCALPING

            # Fallback: check bot_tags for ORB-related tags
            tag_result = self.db_session.execute(
                select(BotLifecycleLog).where(
                    BotLifecycleLog.bot_id == bot_id,
                    or_(
                        BotLifecycleLog.to_tag.contains('orb'),
                        BotLifecycleLog.notes.contains('orb')
                    )
                ).limit(1)
            ).scalar_one_or_none()

            if tag_result is not None:
                return BotType.ORB

            # Default to scalping
            return BotType.SCALPING

        except Exception as e:
            logger.warning(f"Error determining bot type for {bot_id}: {e}, defaulting to SCALPING")
            return BotType.SCALPING

    def _get_threshold(self, bot_id: str) -> int:
        """
        Get loss threshold for a bot based on its type.

        Args:
            bot_id: Bot identifier

        Returns:
            Loss threshold (2 for scalping, 3 for ORB)
        """
        bot_type = self._get_bot_type(bot_id)
        return LOSS_THRESHOLDS[bot_type]

    def _determine_tier(self, bot_id: str) -> BotTier:
        """
        Determine paper trading tier for a bot.

        TIER_1: Bot has @primal tag (has lived before)
        TIER_2: Bot is fresh from AlphaForge (no live history)

        Args:
            bot_id: Bot identifier

        Returns:
            BotTier.TIER_1 or BotTier.TIER_2
        """
        # Check if bot has lived before (has @primal tag or has live history)
        # For now, check if bot has previous consecutive losses record
        # A more complete implementation would check bot_tags table
        try:
            from src.database.models import BotLifecycleLog

            # Check if bot has any lifecycle log entries indicating live trading
            result = self.db_session.execute(
                select(BotLifecycleLog).where(
                    BotLifecycleLog.bot_id == bot_id,
                    BotLifecycleLog.to_tag == '@primal'
                ).limit(1)
            )
            has_live_history = result.scalar_one_or_none() is not None

            if has_live_history:
                return BotTier.TIER_1
            else:
                return BotTier.TIER_2

        except Exception as e:
            logger.warning(f"Error checking live history for bot {bot_id}: {e}, defaulting to TIER_2")
            return BotTier.TIER_2

    def increment_consecutive_losses(self, bot_id: str, magic_number: str) -> int:
        """
        Increment consecutive loss counter for a bot.

        Persists to bot_circuit_breaker table.

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number

        Returns:
            New consecutive loss count
        """
        try:
            record = self.state_manager._get_record(bot_id)

            if record is None:
                # Create new record
                record = BotCircuitBreaker(
                    bot_id=bot_id,
                    magic_number=magic_number,
                    consecutive_losses=1,
                    state=SSLState.LIVE.value,
                )
                self.db_session.add(record)
            else:
                record.consecutive_losses = (record.consecutive_losses or 0) + 1
                record.magic_number = magic_number
                record.updated_at = datetime.now(timezone.utc)

            self.db_session.commit()
            new_count = record.consecutive_losses

            logger.debug(f"Incremented consecutive losses for bot {bot_id}: {new_count}")
            return new_count

        except Exception as e:
            logger.error(f"Error incrementing consecutive losses for bot {bot_id}: {e}")
            self.db_session.rollback()
            return 0

    def reset_consecutive_losses(self, bot_id: str, magic_number: str) -> None:
        """
        Reset consecutive loss counter to 0 for a bot.

        Persists to bot_circuit_breaker table.

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number
        """
        try:
            record = self.state_manager._get_record(bot_id)

            if record is None:
                # Create new record with 0 losses
                record = BotCircuitBreaker(
                    bot_id=bot_id,
                    magic_number=magic_number,
                    consecutive_losses=0,
                    state=SSLState.LIVE.value,
                )
                self.db_session.add(record)
            else:
                record.consecutive_losses = 0
                record.magic_number = magic_number
                record.updated_at = datetime.now(timezone.utc)

            self.db_session.commit()
            logger.debug(f"Reset consecutive losses for bot {bot_id}")

        except Exception as e:
            logger.error(f"Error resetting consecutive losses for bot {bot_id}: {e}")
            self.db_session.rollback()

    def evaluate_circuit_breaker(self, bot_id: str) -> bool:
        """
        Evaluate if circuit breaker should fire for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            True if threshold breached and circuit breaker should fire
        """
        current_state = self.state_manager.get_state(bot_id)

        # Don't fire if already in paper or retired
        if current_state in (SSLState.PAPER, SSLState.RETIRED):
            return False

        consecutive_losses = self.state_manager.get_consecutive_losses(bot_id)
        threshold = self._get_threshold(bot_id)

        return consecutive_losses >= threshold

    def move_to_paper_only(self, bot_id: str, tier: BotTier) -> SSLCircuitBreakerEvent:
        """
        Move a bot to paper-only mode.

        Removes @primal tag, applies @paper_only tag, sets tier.

        Args:
            bot_id: Bot identifier
            tier: Paper trading tier (TIER_1 or TIER_2)

        Returns:
            SSLCircuitBreakerEvent for the transition
        """
        previous_state = self.state_manager.get_state(bot_id)
        magic_number = self.state_manager.get_magic_number(bot_id) or ""
        consecutive_losses = self.state_manager.get_consecutive_losses(bot_id)

        # Update state
        self.state_manager.update_state(
            bot_id=bot_id,
            new_state=SSLState.PAPER,
            tier=tier,
            paper_entry_timestamp=datetime.now(timezone.utc),
        )

        # Update bot tags
        self._remove_primal_tag(bot_id)
        self._add_paper_only_tag(bot_id)

        # Notify Trading Department of SSL circuit break
        self._schedule_ssl_paper_review_dispatch(
            bot_id=bot_id,
            magic_number=magic_number,
            tier=tier,
            ssl_state={"state": SSLState.PAPER.value, "tier": tier.value},
        )

        # Create event
        event = SSLCircuitBreakerEvent(
            bot_id=bot_id,
            magic_number=magic_number,
            event_type=SSLEventType.MOVE_TO_PAPER,
            consecutive_losses=consecutive_losses,
            tier=tier.value,
            previous_state=previous_state,
            new_state=SSLState.PAPER,
            metadata={
                "bot_type": self._get_bot_type(bot_id).value,
                "threshold_breached": self._get_threshold(bot_id),
            }
        )

        # Emit event to Redis
        self._emit_event(event)

        logger.info(f"Circuit breaker fired for bot {bot_id}, moved to {tier.value} paper")
        return event

    def _increment_recovery_win_count(self, bot_id: str) -> int:
        """
        Increment recovery win count for a bot in paper tier.

        Args:
            bot_id: Bot identifier

        Returns:
            New recovery win count
        """
        try:
            record = self.state_manager._get_record(bot_id)
            if record is None:
                return 0

            record.recovery_win_count = (record.recovery_win_count or 0) + 1
            record.updated_at = datetime.now(timezone.utc)
            self.db_session.commit()

            return record.recovery_win_count

        except Exception as e:
            logger.error(f"Error incrementing recovery win count for bot {bot_id}: {e}")
            self.db_session.rollback()
            return 0

    def evaluate_recovery(self, bot_id: str) -> bool:
        """
        Evaluate if a bot in paper tier is ready for recovery.

        A bot is ready for recovery if:
        - State is PAPER
        - Tier is TIER_1
        - Recovery win count >= 2

        Args:
            bot_id: Bot identifier

        Returns:
            True if bot is eligible for recovery promotion
        """
        state = self.state_manager.get_state(bot_id)
        if state != SSLState.PAPER:
            return False

        tier = self.state_manager.get_tier(bot_id)
        if tier != BotTier.TIER_1:
            return False

        recovery_win_count = self.state_manager.get_recovery_win_count(bot_id)
        return recovery_win_count >= 2

    def promote_to_live(self, bot_id: str) -> SSLCircuitBreakerEvent:
        """
        Promote a bot from paper/recovery to live trading.

        Removes @paper_only tag, applies @primal_remount tag,
        resets CONSECUTIVE_LOSSES to 0.

        Args:
            bot_id: Bot identifier

        Returns:
            SSLCircuitBreakerEvent for the transition
        """
        previous_state = self.state_manager.get_state(bot_id)
        magic_number = self.state_manager.get_magic_number(bot_id) or ""
        consecutive_losses = self.state_manager.get_consecutive_losses(bot_id)

        # Update state
        self.state_manager.update_state(
            bot_id=bot_id,
            new_state=SSLState.LIVE,
            consecutive_losses=0,
            tier=None,
            recovery_win_count=0,
        )

        # Update bot tags
        self._remove_paper_only_tag(bot_id)
        self._add_primal_remount_tag(bot_id)

        # Create event
        event = SSLCircuitBreakerEvent(
            bot_id=bot_id,
            magic_number=magic_number,
            event_type=SSLEventType.RECOVERY_CONFIRMED,
            consecutive_losses=0,
            previous_state=previous_state,
            new_state=SSLState.LIVE,
            metadata={"recovery_promoted": True}
        )

        # Emit event to Redis
        self._emit_event(event)

        logger.info(f"Bot {bot_id} promoted to live trading")
        return event

    def on_trade_close(
        self,
        bot_id: str,
        magic_number: str,
        is_win: bool,
    ) -> Optional[SSLCircuitBreakerEvent]:
        """
        Handle trade close event for SSL evaluation.

        This is the main entry point for SSL circuit breaker evaluation.

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number
            is_win: Whether the trade was a win

        Returns:
            SSLCircuitBreakerEvent if a state transition occurred, None otherwise
        """
        current_state = self.state_manager.get_state(bot_id)

        if is_win:
            # Handle win
            if current_state == SSLState.PAPER:
                # Increment recovery win count
                new_recovery_count = self._increment_recovery_win_count(bot_id)

                # Check if recovery threshold reached
                if new_recovery_count >= 2:
                    tier = self.state_manager.get_tier(bot_id)
                    if tier == BotTier.TIER_1:
                        # Transition to recovery state
                        self.state_manager.update_state(
                            bot_id=bot_id,
                            new_state=SSLState.RECOVERY,
                            recovery_win_count=new_recovery_count,
                        )

                        event = SSLCircuitBreakerEvent(
                            bot_id=bot_id,
                            magic_number=magic_number,
                            event_type=SSLEventType.RECOVERY_STEP_1,
                            consecutive_losses=0,
                            previous_state=SSLState.PAPER,
                            new_state=SSLState.RECOVERY,
                            recovery_win_count=new_recovery_count,
                        )
                        self._emit_event(event)
                        return event

            # Reset consecutive losses on win
            self.reset_consecutive_losses(bot_id, magic_number)
            return None

        else:
            # Handle loss
            if current_state == SSLState.RECOVERY:
                # Reset recovery win count on loss during recovery
                self.state_manager.reset_recovery_state(bot_id)
                self.state_manager.update_state(
                    bot_id=bot_id,
                    new_state=SSLState.PAPER,  # Go back to paper
                )

                event = SSLCircuitBreakerEvent(
                    bot_id=bot_id,
                    magic_number=magic_number,
                    event_type=SSLEventType.MOVE_TO_PAPER,
                    consecutive_losses=self.state_manager.get_consecutive_losses(bot_id),
                    previous_state=SSLState.RECOVERY,
                    new_state=SSLState.PAPER,
                    tier=BotTier.TIER_1.value,
                )
                self._emit_event(event)
                return event

            # Increment consecutive losses
            new_count = self.increment_consecutive_losses(bot_id, magic_number)

            # Story 18.3: Check 3-loss-in-a-day rule (independent of consecutive SSL)
            # This is checked AFTER incrementing consecutive losses but BEFORE
            # the consecutive SSL circuit breaker check, since they are independent rules.
            loss_3_day_event = self._check_and_handle_3_loss_in_day(bot_id, magic_number)
            if loss_3_day_event is not None:
                # Bot was halted for day or quarantined for week - don't continue with SSL check
                return loss_3_day_event

            # Check if circuit breaker should fire
            threshold = self._get_threshold(bot_id)
            if new_count >= threshold:
                tier = self._determine_tier(bot_id)
                return self.move_to_paper_only(bot_id, tier)

            return None

    def get_ssl_state(self, bot_id: str) -> SSLState:
        """
        Get current SSL state for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Current SSL state
        """
        return self.state_manager.get_state(bot_id)

    def get_consecutive_losses(self, bot_id: str) -> int:
        """
        Get consecutive loss count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Consecutive loss count
        """
        return self.state_manager.get_consecutive_losses(bot_id)

    def get_tier(self, bot_id: str) -> Optional[BotTier]:
        """
        Get paper trading tier for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            BotTier or None if not in paper
        """
        return self.state_manager.get_tier(bot_id)

    def get_recovery_candidates(self) -> List[str]:
        """
        Get all TIER_1 paper bots eligible for recovery.

        Returns:
            List of bot IDs eligible for recovery
        """
        return self.state_manager.get_tier_1_recovery_candidates()

    def get_paper_candidates(self) -> List[str]:
        """
        Get all bots currently in paper tier.

        Returns:
            List of bot IDs in paper tier
        """
        return self.state_manager.get_all_paper_bots()

    def get_tier_2_paper_bots(self) -> List[str]:
        """
        Get all TIER_2 paper bots.

        Returns:
            List of bot IDs in TIER_2 paper
        """
        return self.state_manager.get_tier_2_paper_bots()

    @property
    def redis_client(self):
        """Get or create Redis client (lazy initialization)."""
        if self._redis_client is None:
            import redis
            self._redis_client = redis.Redis(
                host=self._redis_host,
                port=self._redis_port,
                decode_responses=True,
            )
        return self._redis_client

    def _emit_event(self, event: SSLCircuitBreakerEvent) -> bool:
        """
        Emit SSL event to Redis channel.

        Args:
            event: SSL circuit breaker event

        Returns:
            True if event was emitted successfully
        """
        try:
            self.redis_client.publish(SSL_EVENTS_CHANNEL, event.to_redis_message())
            logger.debug(f"Emitted SSL event to {SSL_EVENTS_CHANNEL}: {event}")
            return True

        except Exception as e:
            logger.error(f"Failed to emit SSL event to Redis: {e}")
            return False

    def _add_bot_tag(self, bot_id: str, tag: str) -> bool:
        """
        Add a tag to a bot.

        Args:
            bot_id: Bot identifier
            tag: Tag to add (e.g., '@paper_only', '@primal_remount')

        Returns:
            True if tag was added successfully
        """
        try:
            from src.database.models import BotLifecycleLog

            log_entry = BotLifecycleLog(
                bot_id=bot_id,
                from_tag="",
                to_tag=tag,
                reason="SSL state transition",
                timestamp=datetime.now(timezone.utc),
                triggered_by="SSL_CIRCUIT_BREAKER",
            )
            self.db_session.add(log_entry)
            self.db_session.commit()
            return True

        except Exception as e:
            logger.error(f"Error adding tag {tag} to bot {bot_id}: {e}")
            self.db_session.rollback()
            return False

    def _remove_bot_tag(self, bot_id: str, tag: str) -> bool:
        """
        Remove a tag from a bot.

        Args:
            bot_id: Bot identifier
            tag: Tag to remove (e.g., '@primal')

        Returns:
            True if tag was removed successfully
        """
        try:
            from src.database.models import BotLifecycleLog

            log_entry = BotLifecycleLog(
                bot_id=bot_id,
                from_tag=tag,
                to_tag="",
                reason="SSL state transition",
                timestamp=datetime.now(timezone.utc),
                triggered_by="SSL_CIRCUIT_BREAKER",
            )
            self.db_session.add(log_entry)
            self.db_session.commit()
            return True

        except Exception as e:
            logger.error(f"Error removing tag {tag} from bot {bot_id}: {e}")
            self.db_session.rollback()
            return False

    def _add_paper_only_tag(self, bot_id: str) -> bool:
        """Add @paper_only tag to a bot."""
        return self._add_bot_tag(bot_id, "@paper_only")

    def _remove_paper_only_tag(self, bot_id: str) -> bool:
        """Remove @paper_only tag from a bot."""
        return self._remove_bot_tag(bot_id, "@paper_only")

    def _add_primal_remount_tag(self, bot_id: str) -> bool:
        """Add @primal_remount tag to a bot."""
        return self._add_bot_tag(bot_id, "@primal_remount")

    def _remove_primal_tag(self, bot_id: str) -> bool:
        """Remove @primal tag from a bot."""
        return self._remove_bot_tag(bot_id, "@primal")

    def evaluate_retirement(self, bot_id: str) -> bool:
        """
        Evaluate if a bot should be retired.

        AC#5: Given a bot in TIER_1 paper fails to recover
        (does not achieve 2 consecutive wins within the recovery window),
        When the SSL evaluates at the second Dead Zone after the bot entered TIER_1 paper,
        Then the original strategy variant is marked RETIRED.

        Recovery window = 2 Dead Zone evaluations after entering paper.
        This means if paper_entry_timestamp was at Dead Zone N, retirement is evaluated at Dead Zone N+2.

        Args:
            bot_id: Bot identifier

        Returns:
            True if bot should be retired
        """
        from datetime import datetime, timezone, timedelta

        state = self.state_manager.get_state(bot_id)
        tier = self.state_manager.get_tier(bot_id)
        paper_entry_ts = self.state_manager.get_paper_entry_timestamp(bot_id)
        recovery_win_count = self.state_manager.get_recovery_win_count(bot_id)

        # Only evaluate TIER_1 paper bots
        if state != SSLState.PAPER or tier != BotTier.TIER_1:
            return False

        # Check if recovery window has passed (2 Dead Zone evaluations)
        # Dead Zone is approximately 4 hours (London: 10:30, NY: 15:30)
        # For simplicity, we use 8 hours as 2 Dead Zone window
        if paper_entry_ts is None:
            return False

        now = datetime.now(timezone.utc)
        hours_since_paper_entry = (now - paper_entry_ts).total_seconds() / 3600

        # 2 Dead Zones = ~8 hours (conservative estimate)
        recovery_window_hours = 8

        if hours_since_paper_entry < recovery_window_hours:
            return False

        # Check if recovery win count is still < 2 (failed to recover)
        return recovery_win_count < 2

    def mark_strategy_retired(self, bot_id: str, reason: str = "SSL failure to recover from paper") -> bool:
        """
        Mark strategy as retired in strategy library.

        AC#5: The original strategy variant is marked RETIRED in the strategy library,
        And a retirement diagnosis report is generated (Q17-Q20 lifecycle report framework):
        strategy ID, retirement date, cumulative P&L, reason = "SSL failure to recover from paper"

        Args:
            bot_id: Bot identifier
            reason: Retirement reason

        Returns:
            True if successful
        """
        try:
            from src.database.models import BotManifest, BotLifecycleLog
            from datetime import datetime, timezone

            # Get bot's strategy info from manifest
            manifest = self.db_session.query(BotManifest).filter(
                BotManifest.bot_name == bot_id
            ).scalar_one_or_none()

            if manifest is None:
                logger.warning(f"No manifest found for bot {bot_id} to mark strategy retired")
                return False

            # Log retirement in lifecycle
            log_entry = BotLifecycleLog(
                bot_id=bot_id,
                from_tag="",
                to_tag="@retired",
                reason=reason,
                timestamp=datetime.now(timezone.utc),
                triggered_by="SSL_CIRCUIT_BREAKER",
                notes=f"Strategy {manifest.strategy_type} retired due to SSL failure to recover",
            )
            self.db_session.add(log_entry)

            # Update bot status to retired
            manifest.status = "retired"
            self.db_session.commit()

            logger.info(f"Marked strategy {manifest.strategy_type} for bot {bot_id} as RETIRED")
            return True

        except Exception as e:
            logger.error(f"Error marking strategy retired for bot {bot_id}: {e}")
            self.db_session.rollback()
            return False

    def trigger_alphaforge_workflow_1(self, bot_id: str) -> Optional[str]:
        """
        Trigger AlphaForge Workflow 1 to generate new candidate variant.

        AC#5: AlphaForge Workflow 1 is triggered: a new candidate variant is generated
        from the same strategy source.

        Args:
            bot_id: Bot identifier

        Returns:
            New candidate bot_id if triggered successfully, None otherwise
        """
        try:
            # Get the original strategy from bot's manifest
            from src.database.models import BotManifest

            manifest = self.db_session.query(BotManifest).filter(
                BotManifest.bot_name == bot_id
            ).scalar_one_or_none()

            if manifest is None:
                logger.warning(f"No manifest found for bot {bot_id} to trigger AlphaForge")
                return None

            # Emit AlphaForge trigger via department mail system
            try:
                from src.agents.departments.department_mail import DepartmentMailService

                mail_service = DepartmentMailService()
                mail_service.send_message(
                    from_department="TRADING",
                    to_department="RESEARCH",
                    subject="ALPHAFORGE_WORKFLOW_1_TRIGGER",
                    body={
                        "bot_id": bot_id,
                        "strategy_type": manifest.strategy_type,
                        "trigger_reason": "SSL failure to recover from paper",
                        "source_story": "18.2",
                    },
                    priority="HIGH",
                )
                logger.info(
                    f"AlphaForge Workflow 1 triggered via department mail for bot {bot_id}, "
                    f"strategy: {manifest.strategy_type}"
                )
            except Exception as mail_err:
                # Fallback: emit to Redis for external AlphaForge consumer
                logger.warning(f"Department mail failed ({mail_err}), emitting to Redis fallback")
                import redis
                redis_client = redis.Redis(host=self._redis_host, port=self._redis_port, decode_responses=True)
                payload = json.dumps({
                    "type": "ALPHAFORGE_WORKFLOW_1",
                    "bot_id": bot_id,
                    "strategy_type": manifest.strategy_type,
                    "trigger_reason": "SSL failure to recover from paper",
                })
                redis_client.publish("alphaforge:workflow_triggers", payload)
                redis_client.close()

            # Return the strategy_id that would be used for new variant generation
            return f"{manifest.strategy_type}_v2"

        except Exception as e:
            logger.error(f"Error triggering AlphaForge for bot {bot_id}: {e}")
            return None

    def emit_retirement_event(
        self,
        bot_id: str,
        dpr_composite_score: int,
    ) -> Optional[SSLCircuitBreakerEvent]:
        """
        Emit retirement event for a bot that failed to recover.

        AC#5: The retirement and new candidate events are both emitted to DPR
        for queue audit trail.

        Args:
            bot_id: Bot identifier
            dpr_composite_score: DPR score at time of retirement

        Returns:
            SSLCircuitBreakerEvent for the retirement
        """
        previous_state = self.state_manager.get_state(bot_id)
        magic_number = self.state_manager.get_magic_number(bot_id) or ""
        consecutive_losses = self.state_manager.get_consecutive_losses(bot_id)

        # Update state to retired
        self.state_manager.update_state(
            bot_id=bot_id,
            new_state=SSLState.RETIRED,
        )

        # Create and emit retirement event
        event = SSLCircuitBreakerEvent(
            bot_id=bot_id,
            magic_number=magic_number,
            event_type=SSLEventType.RETIRED,
            consecutive_losses=consecutive_losses,
            tier=BotTier.TIER_1.value,
            previous_state=previous_state,
            new_state=SSLState.RETIRED,
            metadata={
                "dpr_score_at_retirement": dpr_composite_score,
                "reason": "SSL failure to recover from paper",
            }
        )

        self._emit_event(event)

        logger.info(f"Retirement event emitted for bot {bot_id}")
        return event

    def on_dead_zone_evaluation(self, bot_id: str, dpr_composite_score: int) -> Optional[SSLCircuitBreakerEvent]:
        """
        Handle Dead Zone evaluation for a bot.

        Called by the Inter-Session Cooldown at each Dead Zone (11:30 GMT).
        Evaluates if TIER_1 paper bots should be retired.

        Args:
            bot_id: Bot identifier
            dpr_composite_score: Current DPR score

        Returns:
            SSLCircuitBreakerEvent if retirement occurred, None otherwise
        """
        if self.evaluate_retirement(bot_id):
            # Mark strategy as retired
            self.mark_strategy_retired(bot_id)

            # Trigger AlphaForge Workflow 1
            new_candidate_id = self.trigger_alphaforge_workflow_1(bot_id)

            # Emit retirement event
            event = self.emit_retirement_event(bot_id, dpr_composite_score)

            if event and new_candidate_id:
                event.metadata["new_candidate_id"] = new_candidate_id

            return event

        return None

    # Story 18.3: 3-Loss-in-a-Day Rule Methods

    def _check_week_boundary(self) -> None:
        """
        Check if we've crossed a week boundary (Monday 00:00 UTC) and reset weekly counters if needed.

        Weekly counters reset every Monday at 00:00 UTC.
        """
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        # Calculate start of current week (Monday 00:00 UTC)
        days_since_monday = now.weekday()
        week_start = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)

        if self._current_week_start is None or self._current_week_start < week_start:
            # New week - reset weekly counters
            if self._current_week_start is not None:
                logger.info(f"Week boundary crossed, resetting weekly_3loss_trigger_count")
            self.weekly_3loss_trigger_count.clear()
            self._current_week_start = week_start

    def _reset_daily_loss_count(self, bot_id: str) -> None:
        """
        Reset daily loss count for a bot.

        Called at session boundary (new session = fresh count).

        Args:
            bot_id: Bot identifier
        """
        if bot_id in self.daily_loss_count:
            self.daily_loss_count[bot_id] = 0
            logger.debug(f"Reset daily_loss_count for bot {bot_id}")

    def reset_all_daily_loss_counts(self) -> None:
        """
        Reset all daily loss counts for session boundary.

        Called when a new session starts.
        """
        self.daily_loss_count.clear()
        self._last_session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Reset all daily_loss_count for new session")

    def _increment_daily_loss_count(self, bot_id: str) -> int:
        """
        Increment daily loss count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            New daily loss count
        """
        current = self.daily_loss_count.get(bot_id, 0)
        self.daily_loss_count[bot_id] = current + 1
        return self.daily_loss_count[bot_id]

    def _increment_weekly_3loss_trigger_count(self, bot_id: str) -> int:
        """
        Increment weekly 3-loss trigger count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            New weekly trigger count
        """
        self._check_week_boundary()
        current = self.weekly_3loss_trigger_count.get(bot_id, 0)
        self.weekly_3loss_trigger_count[bot_id] = current + 1
        return self.weekly_3loss_trigger_count[bot_id]

    def halt_bot_for_day(self, bot_id: str, magic_number: str) -> Optional[SSLCircuitBreakerEvent]:
        """
        Halt a bot for the rest of the day due to 3 losses in a day.

        This is a temporary halt (FAIL-S02) - bot resumes trading next session/day.
        Does NOT transition through the SSL state machine (LIVE -> PAPER -> etc).

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number

        Returns:
            SSLCircuitBreakerEvent for the halt
        """
        previous_state = self.state_manager.get_state(bot_id)
        daily_count = self.daily_loss_count.get(bot_id, 0)

        logger.warning(
            f"FAIL-S02: Bot {bot_id} halted for rest of day due to 3 losses in a day "
            f"(daily_loss_count={daily_count})"
        )

        # Create halt event for DPR scoring
        event = SSLCircuitBreakerEvent(
            bot_id=bot_id,
            magic_number=magic_number,
            event_type=SSLEventType.HALT_FOR_DAY,
            consecutive_losses=self.state_manager.get_consecutive_losses(bot_id),
            previous_state=previous_state,
            new_state=previous_state,  # State doesn't change, just temporarily halted
            metadata={
                "failure_type": FAIL_S02,
                "daily_loss_count": daily_count,
                "halt_reason": "3 losses in a day",
                "bot_type": self._get_bot_type(bot_id).value,
            }
        )

        self._emit_event(event)
        return event

    def quarantine_bot_for_week(self, bot_id: str, magic_number: str) -> Optional[SSLCircuitBreakerEvent]:
        """
        Quarantine a bot for the following week due to 3 days with 3-loss trigger.

        This is a week-long quarantine (FAIL-S03) triggered after a bot has had
        3 days with 3-loss-in-a-day events within the same week.
        Does NOT transition through the SSL state machine.

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number

        Returns:
            SSLCircuitBreakerEvent for the quarantine
        """
        previous_state = self.state_manager.get_state(bot_id)
        weekly_count = self.weekly_3loss_trigger_count.get(bot_id, 0)

        logger.warning(
            f"FAIL-S03: Bot {bot_id} quarantined for week due to {weekly_count} days "
            f"with 3-loss-in-a-day triggers this week"
        )

        # Create quarantine event for DPR scoring
        event = SSLCircuitBreakerEvent(
            bot_id=bot_id,
            magic_number=magic_number,
            event_type=SSLEventType.QUARANTINE_FOR_WEEK,
            consecutive_losses=self.state_manager.get_consecutive_losses(bot_id),
            previous_state=previous_state,
            new_state=previous_state,  # State doesn't change, just quarantined
            metadata={
                "failure_type": FAIL_S03,
                "weekly_3loss_trigger_count": weekly_count,
                "quarantine_reason": "3 days with 3-loss-in-a-day trigger in same week",
                "bot_type": self._get_bot_type(bot_id).value,
            }
        )

        self._emit_event(event)
        return event

    def _check_and_handle_3_loss_in_day(
        self, bot_id: str, magic_number: str
    ) -> Optional[SSLCircuitBreakerEvent]:
        """
        Check if 3 losses in a day threshold is reached and handle accordingly.

        Called on each loss for a bot. Checks:
        1. If daily_loss_count >= 3: halt bot for day (FAIL-S02) and increment weekly counter
        2. If weekly_3loss_trigger_count >= 3: quarantine bot for week (FAIL-S03)

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number

        Returns:
            SSLCircuitBreakerEvent if halt or quarantine triggered, None otherwise
        """
        # Check week boundary first
        self._check_week_boundary()

        # Increment daily loss count
        daily_count = self._increment_daily_loss_count(bot_id)

        if daily_count >= 3:
            # Halt bot for rest of day
            halt_event = self.halt_bot_for_day(bot_id, magic_number)

            # Increment weekly 3-loss trigger count
            weekly_count = self._increment_weekly_3loss_trigger_count(bot_id)

            # Check if we need to quarantine for the week
            if weekly_count >= 3:
                quarantine_event = self.quarantine_bot_for_week(bot_id, magic_number)
                return quarantine_event

            return halt_event

        return None

    def get_daily_loss_count(self, bot_id: str) -> int:
        """
        Get daily loss count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Current daily loss count
        """
        return self.daily_loss_count.get(bot_id, 0)

    def get_weekly_3loss_trigger_count(self, bot_id: str) -> int:
        """
        Get weekly 3-loss trigger count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Current weekly trigger count
        """
        self._check_week_boundary()
        return self.weekly_3loss_trigger_count.get(bot_id, 0)

    def _schedule_ssl_paper_review_dispatch(
        self,
        bot_id: str,
        magic_number: str,
        tier: BotTier,
        ssl_state: Dict[str, Any],
    ) -> None:
        """
        Schedule FloorManager dispatch for SSL paper review.

        Args:
            bot_id: Bot identifier
            magic_number: MT5 magic number
            tier: Paper trading tier
            ssl_state: SSL state dict
        """
        try:
            asyncio.get_event_loop().create_task(
                self._dispatch_ssl_paper_review(
                    bot_id=bot_id,
                    magic_number=magic_number,
                    tier=tier,
                    ssl_state=ssl_state,
                )
            )
        except RuntimeError:
            # No event loop running (e.g., during testing), schedule it anyway
            asyncio.create_task(
                self._dispatch_ssl_paper_review(
                    bot_id=bot_id,
                    magic_number=magic_number,
                    tier=tier,
                    ssl_state=ssl_state,
                )
            )

    async def _dispatch_ssl_paper_review(
        self,
        bot_id: str,
        magic_number: str,
        tier: BotTier,
        ssl_state: Dict[str, Any],
    ) -> None:
        """
        Dispatch SSL_PAPER_REVIEW task to FloorManager for Trading Department analysis.

        Args:
            bot_id: Bot that was moved to paper-only
            magic_number: MT5 magic number
            tier: Paper trading tier
            ssl_state: SSL state dict
        """
        try:
            from src.agents.departments.floor_manager import get_floor_manager

            fm = get_floor_manager()
            await fm.dispatch(
                to_dept="trading",
                task_type="SSL_PAPER_REVIEW",
                payload={
                    "bot_id": bot_id,
                    "magic_number": magic_number,
                    "symbol": None,  # Not available in SSL context
                    "strategy_id": None,  # Not available in SSL context
                    "trigger": "ssl_circuit_break",
                    "ssl_state": ssl_state,
                    "instruction": (
                        "This bot was moved to @paper_only due to SSL circuit break. "
                        "Retrieve its trade history and backtest reports. "
                        "Assess whether the SSL issue is transient or indicates a deeper problem. "
                        "If recoverable, produce improvement brief for Development Dept."
                    ),
                },
                priority="medium",
            )
        except Exception as e:
            logger.warning(f"Failed to dispatch SSL_PAPER_REVIEW to FloorManager: {e}")

    def close(self):
        """Close database session and Redis client if we created them."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
        if self._state_manager is not None:
            self._state_manager.close()
            self._state_manager = None
        if self._redis_client is not None:
            self._redis_client.close()
            self._redis_client = None
