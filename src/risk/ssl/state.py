"""
SSL State Machine and Persistence.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Provides SSL state management, persistence to bot_circuit_breaker table,
and state transition guards.

Per NFR-M2: SSL is a synchronous state machine — NO LLM calls in hot path.
Per NFR-D1: All SSL state transitions logged immutably with timestamps.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Tuple
import logging

from sqlalchemy import select, and_
from sqlalchemy.orm import Session

from src.database.models import BotCircuitBreaker, SessionLocal
from src.events.ssl import SSLState, BotTier


logger = logging.getLogger(__name__)


# Valid state transitions
VALID_TRANSITIONS: Dict[SSLState, Tuple[SSLState, ...]] = {
    SSLState.LIVE: (SSLState.PAPER,),
    SSLState.PAPER: (SSLState.RECOVERY, SSLState.RETIRED),
    SSLState.RECOVERY: (SSLState.LIVE, SSLState.PAPER),
    SSLState.RETIRED: (),  # Terminal state - no transitions allowed
}


def is_valid_transition(from_state: SSLState, to_state: SSLState) -> bool:
    """
    Check if a state transition is valid.

    Args:
        from_state: Current state
        to_state: Target state

    Returns:
        True if transition is valid, False otherwise
    """
    if from_state not in VALID_TRANSITIONS:
        return False
    return to_state in VALID_TRANSITIONS[from_state]


class SSLCircuitBreakerState:
    """
    SSL Circuit Breaker state persistence layer.

    Manages SSL state transitions and persistence to the bot_circuit_breaker table.

    State Machine:
        LIVE → (2/3 consecutive losses) → PAPER
        PAPER → (2 consecutive paper wins) → RECOVERY
        RECOVERY → (DPR evaluation at Dead Zone) → LIVE
        PAPER → (fails to recover after limits) → RETIRED

    Attributes:
        db_session: SQLAlchemy session for database access
    """

    # Threshold defaults
    SCALPING_LOSS_THRESHOLD = 2
    ORB_LOSS_THRESHOLD = 3

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize SSL state manager.

        Args:
            db_session: SQLAlchemy session (creates new if None)
        """
        self._db_session = db_session

    @property
    def db_session(self) -> Session:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session

    def get_state(self, bot_id: str) -> SSLState:
        """
        Get current SSL state for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Current SSL state (defaults to LIVE if not found)
        """
        record = self._get_record(bot_id)
        if record is None:
            return SSLState.LIVE

        state_value = getattr(record, 'state', None)
        if state_value is None:
            return SSLState.LIVE

        try:
            return SSLState(state_value)
        except ValueError:
            logger.warning(f"Invalid SSL state {state_value} for bot {bot_id}, defaulting to LIVE")
            return SSLState.LIVE

    def get_tier(self, bot_id: str) -> Optional[BotTier]:
        """
        Get paper trading tier for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            BotTier.TIER_1, BotTier.TIER_2, or None if not in paper
        """
        record = self._get_record(bot_id)
        if record is None:
            return None

        tier_value = getattr(record, 'tier', None)
        if tier_value is None:
            return None

        try:
            return BotTier(tier_value)
        except ValueError:
            return None

    def get_consecutive_losses(self, bot_id: str) -> int:
        """
        Get consecutive loss count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Consecutive loss count (0 if not found)
        """
        record = self._get_record(bot_id)
        if record is None:
            return 0
        return getattr(record, 'consecutive_losses', 0)

    def get_recovery_win_count(self, bot_id: str) -> int:
        """
        Get recovery win count for a bot in paper tier.

        Args:
            bot_id: Bot identifier

        Returns:
            Recovery win count (0 if not in recovery)
        """
        record = self._get_record(bot_id)
        if record is None:
            return 0
        return getattr(record, 'recovery_win_count', 0)

    def get_magic_number(self, bot_id: str) -> Optional[str]:
        """
        Get magic number for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Magic number string or None
        """
        record = self._get_record(bot_id)
        if record is None:
            return None
        return getattr(record, 'magic_number', None)

    def get_paper_entry_timestamp(self, bot_id: str) -> Optional[datetime]:
        """
        Get timestamp when bot entered paper tier.

        Args:
            bot_id: Bot identifier

        Returns:
            Timestamp or None if not in paper
        """
        record = self._get_record(bot_id)
        if record is None:
            return None
        return getattr(record, 'paper_entry_timestamp', None)

    def _get_record(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """
        Get bot circuit breaker record.

        Args:
            bot_id: Bot identifier

        Returns:
            BotCircuitBreaker record or None
        """
        try:
            result = self.db_session.execute(
                select(BotCircuitBreaker).where(BotCircuitBreaker.bot_id == bot_id)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching SSL state for bot {bot_id}: {e}")
            return None

    def update_state(
        self,
        bot_id: str,
        new_state: SSLState,
        magic_number: Optional[str] = None,
        tier: Optional[BotTier] = None,
        consecutive_losses: Optional[int] = None,
        recovery_win_count: Optional[int] = None,
        paper_entry_timestamp: Optional[datetime] = None,
        force: bool = False,
    ) -> bool:
        """
        Update SSL state for a bot.

        Args:
            bot_id: Bot identifier
            new_state: New SSL state
            magic_number: MT5 magic number
            tier: Paper trading tier
            consecutive_losses: Consecutive loss count
            recovery_win_count: Recovery win count
            paper_entry_timestamp: When bot entered paper tier
            force: If True, bypasses transition validation (for manual resurrection)

        Returns:
            True if update succeeded
        """
        try:
            record = self._get_record(bot_id)

            if record is None:
                # Create new record
                record = BotCircuitBreaker(
                    bot_id=bot_id,
                    consecutive_losses=consecutive_losses or 0,
                    magic_number=magic_number,
                    tier=tier.value if tier else None,
                    state=new_state.value,
                    recovery_win_count=recovery_win_count or 0,
                    paper_entry_timestamp=paper_entry_timestamp,
                )
                self.db_session.add(record)
            else:
                # Validate transition (skip if force=True for manual resurrection)
                if not force:
                    current_state = self.get_state(bot_id)
                    if not is_valid_transition(current_state, new_state):
                        logger.warning(
                            f"Invalid SSL transition {current_state.value} -> {new_state.value} for bot {bot_id}"
                        )
                        return False

                # Update fields
                if consecutive_losses is not None:
                    record.consecutive_losses = consecutive_losses
                if magic_number is not None:
                    record.magic_number = magic_number
                if tier is not None:
                    record.tier = tier.value
                if new_state is not None:
                    record.state = new_state.value
                if recovery_win_count is not None:
                    record.recovery_win_count = recovery_win_count
                if paper_entry_timestamp is not None:
                    record.paper_entry_timestamp = paper_entry_timestamp
                elif new_state == SSLState.PAPER and paper_entry_timestamp is None:
                    # Set paper entry timestamp when entering paper
                    record.paper_entry_timestamp = datetime.now(timezone.utc)

            record.updated_at = datetime.now(timezone.utc)
            self.db_session.commit()
            return True

        except Exception as e:
            logger.error(f"Error updating SSL state for bot {bot_id}: {e}")
            self.db_session.rollback()
            return False

    def reset_recovery_state(self, bot_id: str) -> bool:
        """
        Reset recovery win count when a bot gets a loss during recovery.

        Args:
            bot_id: Bot identifier

        Returns:
            True if reset succeeded
        """
        try:
            record = self._get_record(bot_id)
            if record is None:
                return False

            record.recovery_win_count = 0
            record.updated_at = datetime.now(timezone.utc)
            self.db_session.commit()
            return True

        except Exception as e:
            logger.error(f"Error resetting recovery state for bot {bot_id}: {e}")
            self.db_session.rollback()
            return False

    def get_all_bots_in_state(self, state: SSLState) -> list:
        """
        Get all bots in a specific SSL state.

        Args:
            state: SSL state to filter by

        Returns:
            List of bot IDs in the specified state
        """
        try:
            result = self.db_session.execute(
                select(BotCircuitBreaker).where(
                    BotCircuitBreaker.state == state.value
                )
            )
            records = result.scalars().all()
            return [r.bot_id for r in records]
        except Exception as e:
            logger.error(f"Error fetching bots in state {state.value}: {e}")
            return []

    def get_tier_1_recovery_candidates(self) -> list:
        """
        Get TIER_1 paper bots eligible for recovery evaluation.

        A bot is a recovery candidate if:
        - State is PAPER
        - Tier is TIER_1
        - Recovery win count >= 2

        Returns:
            List of bot IDs eligible for recovery
        """
        try:
            result = self.db_session.execute(
                select(BotCircuitBreaker).where(
                    and_(
                        BotCircuitBreaker.state == SSLState.PAPER.value,
                        BotCircuitBreaker.tier == BotTier.TIER_1.value,
                        BotCircuitBreaker.recovery_win_count >= 2
                    )
                )
            )
            records = result.scalars().all()
            return [r.bot_id for r in records]
        except Exception as e:
            logger.error(f"Error fetching TIER_1 recovery candidates: {e}")
            return []

    def get_tier_2_paper_bots(self) -> list:
        """
        Get all TIER_2 paper bots.

        Returns:
            List of bot IDs in TIER_2 paper
        """
        try:
            result = self.db_session.execute(
                select(BotCircuitBreaker).where(
                    and_(
                        BotCircuitBreaker.state == SSLState.PAPER.value,
                        BotCircuitBreaker.tier == BotTier.TIER_2.value
                    )
                )
            )
            records = result.scalars().all()
            return [r.bot_id for r in records]
        except Exception as e:
            logger.error(f"Error fetching TIER_2 paper bots: {e}")
            return []

    def get_all_paper_bots(self) -> list:
        """
        Get all bots in paper tier (TIER_1 and TIER_2).

        Returns:
            List of bot IDs in paper tier
        """
        return self.get_all_bots_in_state(SSLState.PAPER)

    def close(self):
        """Close database session if we created it."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
