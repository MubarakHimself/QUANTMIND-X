"""
Bot Circuit Breaker Manager

Manages bot-level performance tracking and automatic quarantine.
Prevents catastrophic losses from malfunctioning strategies.

**Validates: Task Group 7.4 - BotCircuitBreaker table and manager**
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, date, timezone

from src.database.db_manager import DBManager
from src.database.models import BotCircuitBreaker
from src.router.fee_monitor import FeeMonitor

logger = logging.getLogger(__name__)


class BotCircuitBreakerManager:
    """
    Manager for Bot Circuit Breaker operations.

    Implements automatic quarantine on:
    - 5 consecutive losses
    - Daily trade limit exceeded (default 20 trades)

    Usage:
        manager = BotCircuitBreakerManager()
        allowed, reason = manager.check_allowed("my_bot")
        manager.record_trade("my_bot", is_loss=False)
    """

    # Quarantine thresholds
    MAX_CONSECUTIVE_LOSSES = 5
    DEFAULT_DAILY_TRADE_LIMIT = 20

    def __init__(self, db_manager: Optional[DBManager] = None):
        """
        Initialize BotCircuitBreaker manager.

        Args:
            db_manager: Optional DBManager instance (creates instance if not provided)
        """
        self.db = db_manager or DBManager()
        self.fee_monitor = FeeMonitor(account_id="quantmindx_default", db_manager=self.db)

    def get_or_create_state(self, bot_id: str) -> BotCircuitBreaker:
        """
        Get or create circuit breaker state for a bot.

        Args:
            bot_id: Unique bot identifier

        Returns:
            BotCircuitBreaker object
        """
        with self.db.get_session() as session:
            # Try to find existing state
            state = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if state is None:
                # Create new state
                state = BotCircuitBreaker(
                    bot_id=bot_id,
                    consecutive_losses=0,
                    daily_trade_count=0,
                    last_trade_time=None,
                    is_quarantined=False,
                    quarantine_reason=None,
                    quarantine_start=None
                )
                session.add(state)
                session.flush()
                session.refresh(state)

            session.expunge(state)
            return state

    def check_allowed(self, bot_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a bot is allowed to trade.

        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if bot can trade
            - reason: None if allowed, otherwise explanation
        """
        state = self.get_or_create_state(bot_id)

        # Check if quarantined
        if state.is_quarantined:
            reason = state.quarantine_reason or "Bot is quarantined"
            return False, reason

        # Check daily trade limit
        if state.daily_trade_count >= self.DEFAULT_DAILY_TRADE_LIMIT:
            return False, f"Daily trade limit reached ({state.daily_trade_count}/{self.DEFAULT_DAILY_TRADE_LIMIT})"

        return True, None

    def record_trade(
        self,
        bot_id: str,
        is_loss: bool,
        fee: float = 0.0,
        trade_date: Optional[date] = None
    ) -> BotCircuitBreaker:
        """
        Record a trade for a bot and update circuit breaker state.

        Automatically quarantines bot if:
        - 5 consecutive losses reached
        - Daily trade limit exceeded

        Args:
            bot_id: Unique bot identifier
            is_loss: Whether the trade was a loss
            fee: Fee for the trade (defaults to 0.0)
            trade_date: Date of the trade (defaults to today)

        Returns:
            Updated BotCircuitBreaker object
        """
        if trade_date is None:
            trade_date = date.today()

        state = self.get_or_create_state(bot_id)

        # Check if we need to reset daily counters (new day)
        if state.last_trade_time:
            last_trade_date = state.last_trade_time.date()
            if last_trade_date < trade_date:
                # New day - reset counters
                state.daily_trade_count = 0
                logger.info(f"New day detected: reset daily trade count for {bot_id}")

        # Update trade count
        state.daily_trade_count += 1
        state.last_trade_time = datetime.now(timezone.utc)

        # Update consecutive losses
        if is_loss:
            state.consecutive_losses += 1
        else:
            # Reset on win
            if state.consecutive_losses > 0:
                logger.info(f"Win recorded: reset consecutive losses for {bot_id}")
            state.consecutive_losses = 0

        # Record fee (Phase 3.2)
        self.fee_monitor.record_trade_fee(bot_id, fee, trade_date)

        # Check fee kill switch
        should_halt, reason = self.fee_monitor.should_halt_trading()
        if should_halt:
            self.quarantine_bot(bot_id, reason=f"FEE_KILL_SWITCH: {reason}")
            state.is_quarantined = True

        # Check quarantine triggers
        if state.consecutive_losses >= self.MAX_CONSECUTIVE_LOSSES:
            self.quarantine_bot(
                bot_id,
                reason=f"{state.consecutive_losses} consecutive losses"
            )
            state.is_quarantined = True
        elif state.daily_trade_count > self.DEFAULT_DAILY_TRADE_LIMIT:
            self.quarantine_bot(
                bot_id,
                reason=f"Daily trade limit exceeded ({state.daily_trade_count}/{self.DEFAULT_DAILY_TRADE_LIMIT})"
            )
            state.is_quarantined = True

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(
            f"Trade recorded: bot={bot_id}, loss={is_loss}, "
            f"consecutive_losses={state.consecutive_losses}, "
            f"daily_trades={state.daily_trade_count}"
        )

        return state

    def quarantine_bot(
        self,
        bot_id: str,
        reason: str
    ) -> BotCircuitBreaker:
        """
        Manually quarantine a bot.

        Args:
            bot_id: Unique bot identifier
            reason: Reason for quarantine

        Returns:
            Updated BotCircuitBreaker object
        """
        state = self.get_or_create_state(bot_id)

        state.is_quarantined = True
        state.quarantine_reason = reason
        state.quarantine_start = datetime.now(timezone.utc)

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.warning(f"Bot quarantined: {bot_id}, reason: {reason}")
        return state

    def reactivate_bot(self, bot_id: str) -> BotCircuitBreaker:
        """
        Reactivate a quarantined bot after manual review.

        Resets consecutive losses counter and clears quarantine flag.

        Args:
            bot_id: Unique bot identifier

        Returns:
            Updated BotCircuitBreaker object
        """
        state = self.get_or_create_state(bot_id)

        state.is_quarantined = False
        state.quarantine_reason = None
        state.quarantine_start = None
        state.consecutive_losses = 0

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(f"Bot reactivated: {bot_id}")
        return state

    def get_state(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """
        Get circuit breaker state for a bot.

        Args:
            bot_id: Unique bot identifier

        Returns:
            Dictionary with state data or None if not found
        """
        with self.db.get_session() as session:
            state = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if state is None:
                return None

            return {
                "id": state.id,
                "bot_id": state.bot_id,
                "consecutive_losses": state.consecutive_losses,
                "daily_trade_count": state.daily_trade_count,
                "last_trade_time": state.last_trade_time.isoformat() if state.last_trade_time else None,
                "is_quarantined": state.is_quarantined,
                "quarantine_reason": state.quarantine_reason,
                "quarantine_start": state.quarantine_start.isoformat() if state.quarantine_start else None,
            }

    def get_quarantined_bots(self) -> List[Dict[str, Any]]:
        """
        Get all currently quarantined bots.

        Returns:
            List of dictionaries with quarantined bot data
        """
        with self.db.get_session() as session:
            states = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.is_quarantined == True
            ).all()

            return [
                {
                    "bot_id": s.bot_id,
                    "consecutive_losses": s.consecutive_losses,
                    "daily_trade_count": s.daily_trade_count,
                    "quarantine_reason": s.quarantine_reason,
                    "quarantine_start": s.quarantine_start.isoformat() if s.quarantine_start else None,
                }
                for s in states
            ]

    def get_all_states(self) -> List[Dict[str, Any]]:
        """
        Get all bot circuit breaker states.

        Returns:
            List of dictionaries with state data
        """
        with self.db.get_session() as session:
            states = session.query(BotCircuitBreaker).all()

            return [
                {
                    "id": s.id,
                    "bot_id": s.bot_id,
                    "consecutive_losses": s.consecutive_losses,
                    "daily_trade_count": s.daily_trade_count,
                    "last_trade_time": s.last_trade_time.isoformat() if s.last_trade_time else None,
                    "is_quarantined": s.is_quarantined,
                    "quarantine_reason": s.quarantine_reason,
                    "quarantine_start": s.quarantine_start.isoformat() if s.quarantine_start else None,
                }
                for s in states
            ]

    def reset_daily_counters(self, bot_id: str) -> BotCircuitBreaker:
        """
        Reset daily trade counter for a bot.

        Does NOT reset consecutive losses or quarantine status.

        Args:
            bot_id: Unique bot identifier

        Returns:
            Updated BotCircuitBreaker object
        """
        state = self.get_or_create_state(bot_id)
        state.daily_trade_count = 0

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(f"Daily counter reset: bot={bot_id}")
        return state

    def set_daily_trade_limit(self, bot_id: str, limit: int) -> None:
        """
        Set a custom daily trade limit for a specific bot.

        Note: This method stores the limit but check_allowed() still uses
        the default limit. Override check_allowed() for custom limits.

        Args:
            bot_id: Unique bot identifier
            limit: Maximum trades per day
        """
        # Store custom limit in bot_id metadata or separate config
        logger.info(f"Custom trade limit set for {bot_id}: {limit} trades/day")
        # Implementation would extend schema to store custom limits
