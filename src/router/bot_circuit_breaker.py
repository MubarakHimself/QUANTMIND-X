"""
Bot Circuit Breaker Manager

Manages bot-level performance tracking and automatic quarantine.
Prevents catastrophic losses from malfunctioning strategies.

**Validates: Task Group 7.4 - BotCircuitBreaker table and manager**
**Configurable Loss Thresholds (S3-6.5):**
- Personal Book (scalping): 2 consecutive losses
- Prop Firm Book (ORB): 3 consecutive losses
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime, date, timezone, timedelta
from enum import Enum

from src.database.db_manager import DBManager
from src.database.models import BotCircuitBreaker
from src.router.fee_monitor import FeeMonitor
from src.router.decline_recovery import DeclineRecoveryEngine, DeclineState

logger = logging.getLogger(__name__)


class AccountBook(str, Enum):
    """Account book type for determining loss thresholds."""
    PERSONAL = "personal"
    PROP_FIRM = "prop_firm"


# Default thresholds by book type
LOSS_THRESHOLDS = {
    AccountBook.PERSONAL: 2,      # Scalping: 2 consecutive losses (S3-6.5)
    AccountBook.PROP_FIRM: 3,     # ORB / prop firms: 3 consecutive losses (S3-6.5)
}
DEFAULT_DAILY_TRADE_LIMIT = 20
THREE_LOSS_WEEKLY_THRESHOLD = 3  # S3-11: 3 separate days with 3-loss streak → quarantine week


class BotCircuitBreakerManager:
    """
    Manager for Bot Circuit Breaker operations.

    Implements automatic quarantine based on configurable thresholds by book type:
    - Personal Book (scalping): 2 consecutive losses (S3-6.5)
    - Prop Firm Book (ORB): 3 consecutive losses (S3-6.5)
    - Daily trade limit: 20 trades (configurable)

    Usage:
        manager = BotCircuitBreakerManager()
        allowed, reason = manager.check_allowed("my_bot")
        manager.record_trade("my_bot", is_loss=False)

        # Or with account book type:
        manager = BotCircuitBreakerManager(account_book=AccountBook.PROP_FIRM)
        manager.record_trade("prop_bot", is_loss=True)  # Triggers at 3 losses
    """

    def __init__(
        self,
        db_manager: Optional[DBManager] = None,
        account_id: Optional[str] = None,
        account_balance: Optional[float] = None,
        account_book: Optional[AccountBook] = None,
        account_book_str: Optional[str] = None
    ):
        """
        Initialize BotCircuitBreaker manager.

        Args:
            db_manager: Optional DBManager instance (creates instance if not provided)
            account_id: Real account identifier for fee tracking
            account_balance: Current account balance for fee calculations
            account_book: Account book type as AccountBook enum (PERSONAL or PROP_FIRM)
            account_book_str: Account book type as string ("personal" or "prop_firm")
        """
        self.db = db_manager or DBManager()
        # Use provided account_id and balance, or fall back to defaults
        self.account_id = account_id or "quantmindx_default"
        self.account_balance = account_balance or 1000.0

        # Handle both enum and string inputs
        if account_book is not None:
            self.account_book = account_book
        elif account_book_str is not None:
            self.account_book = AccountBook(account_book_str)
        else:
            self.account_book = AccountBook.PERSONAL

        # Set threshold based on book type
        self.max_consecutive_losses = LOSS_THRESHOLDS.get(
            self.account_book,
            LOSS_THRESHOLDS[AccountBook.PERSONAL]
        )
        self.default_daily_trade_limit = DEFAULT_DAILY_TRADE_LIMIT

        self.fee_monitor = FeeMonitor(
            account_id=self.account_id,
            db_manager=self.db,
            account_balance=self.account_balance
        )

        # Section 8.2: Decline and Recovery integration
        self.decline_recovery_engine = DeclineRecoveryEngine()

        logger.info(
            f"BotCircuitBreakerManager initialized: "
            f"account_book={self.account_book.value}, "
            f"max_consecutive_losses={self.max_consecutive_losses}"
        )

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
                    quarantine_start=None,
                    # S3-11: 3-Loss-in-a-Row Circuit Breaker fields
                    daily_loss_streak_days=0,
                    last_loss_streak_date=None,
                )
                session.add(state)
                session.flush()
                session.refresh(state)

            session.expunge(state)
            return state

    def _is_new_week(self, last_date: Optional[date], current_date: date) -> bool:
        """
        Check if current_date falls in a new week compared to last_date.

        A new week is detected when last_date is None OR more than 7 days ago.

        Args:
            last_date: The reference date to compare against
            current_date: The current date to check

        Returns:
            True if current_date is in a new week period
        """
        if last_date is None:
            return True
        return (current_date - last_date).days > 7

    def check_allowed(self, bot_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a bot is allowed to trade.

        S3-11 / Section 7.4 checks:
        - Daily block: if bot hit 3 consecutive losses today (last_loss_streak_date == today)
        - Weekly quarantine: if daily_loss_streak_days >= 3 within current week

        Returns:
            Tuple of (is_allowed, reason)
            - is_allowed: True if bot can trade
            - reason: None if allowed, otherwise explanation
        """
        state = self.get_or_create_state(bot_id)
        today = date.today()

        # Check if quarantined - use bool() to handle SQLAlchemy Column type
        if bool(state.is_quarantined):  # type: ignore[arg-type]
            # S3-11 weekly quarantine: block until new week
            quarantine_start = state.quarantine_start  # type: ignore[attr-defined]
            if quarantine_start is not None:
                quarantine_week_start = quarantine_start.date()
                # If quarantine_start is not today and it's still the same week
                # (i.e., quarantine was set earlier this week), block for rest of week
                if quarantine_week_start != today and not self._is_new_week(quarantine_week_start, today):
                    reason = str(state.quarantine_reason) if state.quarantine_reason else "Bot is quarantined for the week"  # type: ignore[arg-type]
                    return False, reason
            reason = str(state.quarantine_reason) if state.quarantine_reason else "Bot is quarantined"  # type: ignore[arg-type]
            return False, reason

        # S3-11: Daily 3-loss-in-a-row block (non-quarantined case)
        # Block if last_loss_streak_date is today (already triggered 3-loss today)
        last_loss_streak_date = state.last_loss_streak_date  # type: ignore[attr-defined]
        if last_loss_streak_date is not None:
            last_streak_day = last_loss_streak_date.date()
            if last_streak_day == today and int(state.daily_loss_streak_days) > 0:  # type: ignore[arg-type]
                return False, "3 consecutive losses triggered today: blocked for rest of day"

        # Check daily trade limit - use int() to handle SQLAlchemy Column type
        if int(state.daily_trade_count) >= self.default_daily_trade_limit:  # type: ignore[arg-type]
            return False, f"Daily trade limit reached ({int(state.daily_trade_count)}/{self.default_daily_trade_limit})"  # type: ignore[arg-type]

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

        # Check if we need to reset daily counters (new day) - handle None and Column types
        last_trade_time = state.last_trade_time  # type: ignore[attr-defined]
        if last_trade_time is not None:
            last_trade_date = last_trade_time.date()
            if last_trade_date < trade_date:
                # New day - reset counters
                state.daily_trade_count = 0  # type: ignore[assignment]
                logger.info(f"New day detected: reset daily trade count for {bot_id}")

        # Update trade count - use SQLAlchemy's Python-side attribute access
        state.daily_trade_count = int(state.daily_trade_count) + 1  # type: ignore[assignment]
        state.last_trade_time = datetime.now(timezone.utc)  # type: ignore[assignment]

        # Update consecutive losses
        if is_loss:
            state.consecutive_losses = int(state.consecutive_losses) + 1  # type: ignore[assignment]
        else:
            # Reset on win
            if int(state.consecutive_losses) > 0:  # type: ignore[arg-type]
                logger.info(f"Win recorded: reset consecutive losses for {bot_id}")
            state.consecutive_losses = 0  # type: ignore[assignment]

        # Record fee (Phase 3.2)
        self.fee_monitor.record_trade_fee(bot_id, fee, trade_date)

        # Check fee kill switch
        should_halt, reason = self.fee_monitor.should_halt_trading()
        if should_halt:
            self.quarantine_bot(bot_id, reason=f"FEE_KILL_SWITCH: {reason}")
            state.is_quarantined = True  # type: ignore[assignment]

        # S3-11: 3-Loss-in-a-Row Circuit Breaker
        # Check if this is a new streak day (first 3-loss trigger of the day)
        last_loss_streak_date = state.last_loss_streak_date  # type: ignore[attr-defined]
        last_streak_day = last_loss_streak_date.date() if last_loss_streak_date else None

        if is_loss and int(state.consecutive_losses) >= 3:  # type: ignore[arg-type]
            if last_streak_day is None or last_streak_day < trade_date:
                # S3-11: Reset weekly counter if we're in a new week
                if last_streak_day is not None and self._is_new_week(last_streak_day, trade_date):
                    state.daily_loss_streak_days = 0  # type: ignore[assignment]
                    logger.info(f"S3-11 new week detected: reset weekly counter for {bot_id}")

                # First 3-loss trigger of the day - increment daily streak counter
                state.daily_loss_streak_days = int(state.daily_loss_streak_days) + 1  # type: ignore[assignment]
                state.last_loss_streak_date = datetime.now(timezone.utc)  # type: ignore[assignment]
                logger.info(
                    f"S3-11 daily loss streak triggered: bot={bot_id}, "
                    f"daily_loss_streak_days={int(state.daily_loss_streak_days)}, "
                    f"trade_date={trade_date}"
                )

                # S3-11 weekly quarantine: if 3+ streak days in current week, quarantine for rest of week
                if int(state.daily_loss_streak_days) >= THREE_LOSS_WEEKLY_THRESHOLD:  # type: ignore[arg-type]
                    reason = (
                        f"S3-11 weekly quarantine: {int(state.daily_loss_streak_days)} "
                        f"3-loss streak days this week (threshold: {THREE_LOSS_WEEKLY_THRESHOLD})"
                    )
                    self.quarantine_bot(bot_id, reason=reason)
                    state.is_quarantined = True  # type: ignore[assignment]
                    logger.warning(f"S3-11 weekly quarantine activated: {bot_id}, reason: {reason}")

                    # Section 8.2: Flag bot for decline and recovery workflow
                    # Check if this is the 3rd day with 3 losses (the threshold trigger)
                    if int(state.daily_loss_streak_days) == THREE_LOSS_WEEKLY_THRESHOLD:
                        # Get current decline state
                        current_state = self.decline_recovery_engine.get_decline_state(bot_id)
                        if current_state == DeclineState.NORMAL:
                            # Flag the bot for decline recovery
                            self.decline_recovery_engine.flag_bot(
                                bot_id=bot_id,
                                reason=reason,
                                regime_state="UNKNOWN",  # Would need HMM integration for actual regime
                                performance_delta=0.0,  # Would need actual performance calculation
                            )
                            logger.info(f"S3-11: Bot {bot_id} flagged for decline recovery")

        # Check quarantine triggers
        if int(state.consecutive_losses) >= self.max_consecutive_losses:  # type: ignore[arg-type]
            self.quarantine_bot(
                bot_id,
                reason=f"{int(state.consecutive_losses)} consecutive losses (threshold: {self.max_consecutive_losses})"  # type: ignore[arg-type]
            )
            state.is_quarantined = True  # type: ignore[assignment]
        elif int(state.daily_trade_count) > self.default_daily_trade_limit:  # type: ignore[arg-type]
            self.quarantine_bot(
                bot_id,
                reason=f"Daily trade limit exceeded ({int(state.daily_trade_count)}/{self.default_daily_trade_limit})"  # type: ignore[arg-type]
            )
            state.is_quarantined = True  # type: ignore[assignment]

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(
            f"Trade recorded: bot={bot_id}, loss={is_loss}, "
            f"consecutive_losses={int(state.consecutive_losses)}, "  # type: ignore[arg-type]
            f"daily_trades={int(state.daily_trade_count)}"  # type: ignore[arg-type]
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

        state.is_quarantined = True  # type: ignore[assignment]
        state.quarantine_reason = reason  # type: ignore[attr-defined]
        state.quarantine_start = datetime.now(timezone.utc)  # type: ignore[assignment]

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
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

        state.is_quarantined = False  # type: ignore[assignment]
        state.quarantine_reason = None  # type: ignore[attr-defined]
        state.quarantine_start = None  # type: ignore[assignment]
        state.consecutive_losses = 0  # type: ignore[assignment]
        # S3-11: Reset 3-loss streak tracking on reactivation
        state.daily_loss_streak_days = 0  # type: ignore[assignment]
        state.last_loss_streak_date = None  # type: ignore[assignment]

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
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
                "consecutive_losses": int(state.consecutive_losses),  # type: ignore[arg-type]
                "daily_trade_count": int(state.daily_trade_count),  # type: ignore[arg-type]
                "last_trade_time": state.last_trade_time.isoformat() if state.last_trade_time else None,  # type: ignore[attr-defined]
                "is_quarantined": bool(state.is_quarantined),  # type: ignore[arg-type]
                "quarantine_reason": state.quarantine_reason,  # type: ignore[attr-defined]
                "quarantine_start": state.quarantine_start.isoformat() if state.quarantine_start else None,  # type: ignore[attr-defined]
                # S3-11: 3-Loss-in-a-Row Circuit Breaker fields
                "daily_loss_streak_days": int(state.daily_loss_streak_days),  # type: ignore[arg-type]
                "last_loss_streak_date": state.last_loss_streak_date.isoformat() if state.last_loss_streak_date else None,  # type: ignore[attr-defined]
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
                    "consecutive_losses": int(s.consecutive_losses),  # type: ignore[arg-type]
                    "daily_trade_count": int(s.daily_trade_count),  # type: ignore[arg-type]
                    "quarantine_reason": s.quarantine_reason,  # type: ignore[attr-defined]
                    "quarantine_start": s.quarantine_start.isoformat() if s.quarantine_start else None,  # type: ignore[attr-defined]
                    # S3-11: 3-Loss-in-a-Row Circuit Breaker fields
                    "daily_loss_streak_days": int(s.daily_loss_streak_days),  # type: ignore[arg-type]
                    "last_loss_streak_date": s.last_loss_streak_date.isoformat() if s.last_loss_streak_date else None,  # type: ignore[attr-defined]
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
                    "consecutive_losses": int(s.consecutive_losses),  # type: ignore[arg-type]
                    "daily_trade_count": int(s.daily_trade_count),  # type: ignore[arg-type]
                    "last_trade_time": s.last_trade_time.isoformat() if s.last_trade_time else None,  # type: ignore[attr-defined]
                    "is_quarantined": bool(s.is_quarantined),  # type: ignore[arg-type]
                    "quarantine_reason": s.quarantine_reason,  # type: ignore[attr-defined]
                    "quarantine_start": s.quarantine_start.isoformat() if s.quarantine_start else None,  # type: ignore[attr-defined]
                    # S3-11: 3-Loss-in-a-Row Circuit Breaker fields
                    "daily_loss_streak_days": int(s.daily_loss_streak_days),  # type: ignore[arg-type]
                    "last_loss_streak_date": s.last_loss_streak_date.isoformat() if s.last_loss_streak_date else None,  # type: ignore[attr-defined]
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
        state.daily_trade_count = 0  # type: ignore[assignment]

        # Save to database
        with self.db.get_session() as session:
            state.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]
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
    
    def update_account_balance(self, new_balance: float) -> None:
        """
        Update the account balance used for fee calculations.
        
        Args:
            new_balance: New account balance
        """
        self.account_balance = new_balance
        self.fee_monitor.account_balance = new_balance
        logger.info(f"Updated account balance to ${new_balance:.2f} for fee monitoring")
    
    def update_account_id(self, new_account_id: str) -> None:
        """
        Update the account ID used for fee tracking.

        Args:
            new_account_id: New account identifier
        """
        self.account_id = new_account_id
        # Create new fee monitor with updated account ID
        self.fee_monitor = FeeMonitor(
            account_id=self.account_id,
            db_manager=self.db,
            account_balance=self.account_balance
        )
        logger.info(f"Updated account ID to {new_account_id} for fee monitoring")

    def update_account_book(self, account_book: AccountBook) -> None:
        """
        Update the account book type and adjust loss threshold accordingly.

        Args:
            account_book: New account book type (PERSONAL or PROP_FIRM)
        """
        old_threshold = self.max_consecutive_losses
        self.account_book = account_book
        self.max_consecutive_losses = LOSS_THRESHOLDS.get(
            account_book,
            LOSS_THRESHOLDS[AccountBook.PERSONAL]
        )
        logger.info(
            f"Updated account_book to {account_book.value}: "
            f"loss threshold {old_threshold} -> {self.max_consecutive_losses}"
        )


def get_loss_threshold(account_book: AccountBook) -> int:
    """
    Get the configured loss threshold for a given account book type.

    Args:
        account_book: The account book type

    Returns:
        Number of consecutive losses before quarantine
    """
    return LOSS_THRESHOLDS.get(account_book, LOSS_THRESHOLDS[AccountBook.PERSONAL])


def get_threshold_config() -> Dict[str, int]:
    """
    Get the complete threshold configuration.

    Returns:
        Dictionary mapping book types to their loss thresholds
    """
    return {book.value: threshold for book, threshold in LOSS_THRESHOLDS.items()}
