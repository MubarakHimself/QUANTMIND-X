"""
Account Monitor for Progressive Kill Switch System

Implements Tier 3 protection by monitoring account-level loss limits
and triggering stops at daily and weekly thresholds.

**Validates: Phase 3 - Tier 3 Account-Level Protection**
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Dict, List, Any, TYPE_CHECKING

from src.router.alert_manager import AlertManager, AlertLevel, get_alert_manager

if TYPE_CHECKING:
    from mcp_mt5.alert_service import AlertService

logger = logging.getLogger(__name__)

# Comment 4: DB Manager for database access
_DB_SESSION = None


def _get_db_session():
    """Get or create database session for DB access."""
    global _DB_SESSION
    if _DB_SESSION is None:
        try:
            from src.database.engine import SessionLocal
            _DB_SESSION = SessionLocal()
        except Exception as e:
            logger.warning(f"Could not load DB session: {e}")
    return _DB_SESSION


@dataclass
class AccountState:
    """
    Tracks account-level loss state.

    Attributes:
        account_id: Unique account identifier
        initial_balance: Starting balance for tracking
        daily_pnl: Today's profit/loss
        weekly_pnl: This week's profit/loss
        last_reset_date: When daily counters were last reset
        week_start: Start of the current week
        daily_stop_triggered: Whether daily stop is active
        weekly_stop_triggered: Whether weekly stop is active
        daily_trades: Number of trades today
        weekly_trades: Number of trades this week
    """
    account_id: str
    initial_balance: float = 10000.0
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    last_reset_date: Optional[date] = None
    week_start: Optional[date] = None
    daily_stop_triggered: bool = False
    weekly_stop_triggered: bool = False
    daily_trades: int = 0
    weekly_trades: int = 0

    @property
    def daily_loss_pct(self) -> float:
        """Calculate daily loss percentage."""
        if self.initial_balance <= 0:
            return 0.0
        return abs(self.daily_pnl) / self.initial_balance if self.daily_pnl < 0 else 0.0

    @property
    def weekly_loss_pct(self) -> float:
        """Calculate weekly loss percentage."""
        if self.initial_balance <= 0:
            return 0.0
        return abs(self.weekly_pnl) / self.initial_balance if self.weekly_pnl < 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "account_id": self.account_id,
            "initial_balance": self.initial_balance,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_loss_pct": self.daily_loss_pct,
            "weekly_loss_pct": self.weekly_loss_pct,
            "last_reset_date": self.last_reset_date.isoformat() if self.last_reset_date else None,
            "week_start": self.week_start.isoformat() if self.week_start else None,
            "daily_stop_triggered": self.daily_stop_triggered,
            "weekly_stop_triggered": self.weekly_stop_triggered,
            "daily_trades": self.daily_trades,
            "weekly_trades": self.weekly_trades
        }


class AccountMonitor:
    """
    Monitors account-level loss limits and triggers protective stops.

    Tier 3 Protection:
    - Triggers daily stop at MAX_DAILY_LOSS_PCT (default 3%)
    - Triggers weekly stop at MAX_WEEKLY_LOSS_PCT (default 10%)
    - Auto-resets daily counters at midnight
    - Auto-resets weekly counters at week start

    Thresholds:
    - MAX_DAILY_LOSS_PCT = 0.03 (3%)
    - MAX_WEEKLY_LOSS_PCT = 0.10 (10%)

    Usage:
        monitor = AccountMonitor(alert_manager)
        monitor.record_trade_pnl("account_123", -50.0, 10000.0)
        allowed = monitor.is_account_allowed("account_123")
    """

    # Default thresholds (can be overridden via config)
    MAX_DAILY_LOSS_PCT = 0.03  # 3%
    MAX_WEEKLY_LOSS_PCT = 0.10  # 10%

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        max_daily_loss_pct: float = MAX_DAILY_LOSS_PCT,
        max_weekly_loss_pct: float = MAX_WEEKLY_LOSS_PCT,
        email_alerts: bool = True,
        sms_alerts: bool = False
    ):
        """
        Initialize AccountMonitor.

        Args:
            alert_manager: AlertManager for raising alerts
            max_daily_loss_pct: Maximum daily loss (default 3%)
            max_weekly_loss_pct: Maximum weekly loss (default 10%)
            email_alerts: Enable email notifications
            sms_alerts: Enable SMS notifications (future)
        """
        self.alert_manager = alert_manager or get_alert_manager()
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.email_alerts = email_alerts
        self.sms_alerts = sms_alerts

        # Account states indexed by account_id
        self.account_states: Dict[str, AccountState] = {}

        logger.info(
            f"AccountMonitor initialized: "
            f"daily_limit={max_daily_loss_pct:.1%}, "
            f"weekly_limit={max_weekly_loss_pct:.1%}, "
            f"email={email_alerts}, sms={sms_alerts}"
        )

    def _get_or_create_state(
        self,
        account_id: str,
        initial_balance: Optional[float] = None
    ) -> AccountState:
        """
        Get or create account state.
        
        Comment 4: First tries to load from DB, falls back to in-memory.
        """
        # Try to load from DB first (Comment 4: DB persistence)
        db_state = self._load_from_db(account_id)
        if db_state:
            self.account_states[account_id] = db_state
            # Update balance if provided
            if initial_balance is not None and initial_balance > 0:
                self.account_states[account_id].initial_balance = initial_balance
            return self.account_states[account_id]
        
        # Fall back to in-memory
        if account_id not in self.account_states:
            self.account_states[account_id] = AccountState(
                account_id=account_id,
                initial_balance=initial_balance or 10000.0
            )

        state = self.account_states[account_id]

        # Update initial balance if provided
        if initial_balance is not None and initial_balance > 0:
            state.initial_balance = initial_balance

        return state
    
    def _load_from_db(self, account_id: str) -> Optional[AccountState]:
        """
        Load account state from database.
        
        Comment 4: Read state from DB instead of in-memory dicts only.
        """
        try:
            session = _get_db_session()
            if session is None:
                return None
                
            from src.database.models import AccountLossState
            db_record = session.query(AccountLossState).filter_by(
                account_id=account_id
            ).first()
            
            if db_record:
                state = AccountState(
                    account_id=db_record.account_id,
                    initial_balance=db_record.initial_balance,
                    daily_pnl=db_record.daily_pnl,
                    weekly_pnl=db_record.weekly_pnl,
                    last_reset_date=date.fromisoformat(db_record.last_reset_date) if db_record.last_reset_date else None,
                    week_start=date.fromisoformat(db_record.week_start) if db_record.week_start else None,
                    daily_stop_triggered=db_record.daily_stop_triggered,
                    weekly_stop_triggered=db_record.weekly_stop_triggered
                )
                return state
        except Exception as e:
            logger.debug(f"Could not load account state from DB: {e}")
        
        return None
    
    def _save_to_db(self, state: AccountState) -> None:
        """
        Save account state to database.
        
        Comment 4: Write state to DB for persistence.
        """
        try:
            session = _get_db_session()
            if session is None:
                return
                
            from src.database.models import AccountLossState
            db_record = session.query(AccountLossState).filter_by(
                account_id=state.account_id
            ).first()
            
            if db_record:
                db_record.initial_balance = state.initial_balance
                db_record.daily_pnl = state.daily_pnl
                db_record.weekly_pnl = state.weekly_pnl
                db_record.last_reset_date = state.last_reset_date.isoformat() if state.last_reset_date else None
                db_record.week_start = state.week_start.isoformat() if state.week_start else None
                db_record.daily_stop_triggered = state.daily_stop_triggered
                db_record.weekly_stop_triggered = state.weekly_stop_triggered
            else:
                db_record = AccountLossState(
                    account_id=state.account_id,
                    initial_balance=state.initial_balance,
                    daily_pnl=state.daily_pnl,
                    weekly_pnl=state.weekly_pnl,
                    last_reset_date=state.last_reset_date.isoformat() if state.last_reset_date else None,
                    week_start=state.week_start.isoformat() if state.week_start else None,
                    daily_stop_triggered=state.daily_stop_triggered,
                    weekly_stop_triggered=state.weekly_stop_triggered
                )
                session.add(db_record)
            
            session.commit()
        except Exception as e:
            logger.warning(f"Could not save account state to DB: {e}")

    def _check_and_reset_counters(self, state: AccountState) -> None:
        """Check and reset daily/weekly counters if needed."""
        today = date.today()

        # Initialize last_reset_date if needed
        if state.last_reset_date is None:
            state.last_reset_date = today
            state.week_start = self._get_week_start(today)
            return

        # Check for new day
        if state.last_reset_date < today:
            logger.info(f"New day detected for account {state.account_id}: resetting daily counters")
            state.daily_pnl = 0.0
            state.daily_trades = 0
            state.daily_stop_triggered = False
            state.last_reset_date = today

        # Check for new week
        current_week_start = self._get_week_start(today)
        if state.week_start is None or state.week_start < current_week_start:
            logger.info(f"New week detected for account {state.account_id}: resetting weekly counters")
            state.weekly_pnl = 0.0
            state.weekly_trades = 0
            state.weekly_stop_triggered = False
            state.week_start = current_week_start

    def _get_week_start(self, d: date) -> date:
        """Get the Monday of the week containing date d."""
        return d - timedelta(days=d.weekday())

    def record_trade_pnl(
        self,
        account_id: str,
        pnl: float,
        account_balance: Optional[float] = None
    ) -> bool:
        """
        Record trade P&L for an account.

        Args:
            account_id: Account identifier
            pnl: Profit/loss amount (negative for loss)
            account_balance: Current account balance (optional)

        Returns:
            True if a stop was triggered, False otherwise
        """
        state = self._get_or_create_state(account_id, account_balance)
        self._check_and_reset_counters(state)

        # Update balances
        if account_balance is not None and account_balance > 0:
            state.initial_balance = account_balance

        # Update P&L
        state.daily_pnl += pnl
        state.weekly_pnl += pnl
        state.daily_trades += 1
        state.weekly_trades += 1

        # Check weekly stop first (higher priority)
        if state.weekly_loss_pct >= self.max_weekly_loss_pct:
            if not state.weekly_stop_triggered:
                self._trigger_weekly_stop(account_id, state, state.weekly_loss_pct)
            # Comment 4: Save state to DB after stop trigger
            self._save_to_db(state)
            return True

        # Check daily stop
        if state.daily_loss_pct >= self.max_daily_loss_pct:
            if not state.daily_stop_triggered:
                self._trigger_daily_stop(account_id, state, state.daily_loss_pct)
            # Comment 4: Save state to DB after stop trigger
            self._save_to_db(state)
            return True

        # Raise warning alerts based on threshold
        self._check_warning_thresholds(account_id, state)
        
        # Comment 4: Save state to DB periodically
        self._save_to_db(state)

        return False

    def _check_warning_thresholds(self, account_id: str, state: AccountState) -> None:
        """Check and raise warning alerts at intermediate thresholds."""
        # Daily threshold warning
        daily_threshold_pct = (state.daily_loss_pct / self.max_daily_loss_pct) * 100
        if daily_threshold_pct >= 50:  # YELLOW or higher
            self.alert_manager.raise_alert(
                tier=3,
                message=f"Account {account_id} daily loss at {state.daily_loss_pct:.1%}",
                threshold_pct=daily_threshold_pct,
                source="account",
                metadata={
                    "account_id": account_id,
                    "daily_pnl": state.daily_pnl,
                    "daily_loss_pct": state.daily_loss_pct,
                    "limit_type": "daily"
                }
            )

        # Weekly threshold warning
        weekly_threshold_pct = (state.weekly_loss_pct / self.max_weekly_loss_pct) * 100
        if weekly_threshold_pct >= 50:  # YELLOW or higher
            self.alert_manager.raise_alert(
                tier=3,
                message=f"Account {account_id} weekly loss at {state.weekly_loss_pct:.1%}",
                threshold_pct=weekly_threshold_pct,
                source="account",
                metadata={
                    "account_id": account_id,
                    "weekly_pnl": state.weekly_pnl,
                    "weekly_loss_pct": state.weekly_loss_pct,
                    "limit_type": "weekly"
                }
            )

    def _trigger_daily_stop(
        self,
        account_id: str,
        state: AccountState,
        loss_pct: float
    ) -> None:
        """Trigger daily stop for an account."""
        state.daily_stop_triggered = True

        # Raise RED alert
        self.alert_manager.raise_alert(
            tier=3,
            message=f"DAILY STOP triggered for {account_id}: {loss_pct:.1%} loss",
            threshold_pct=100.0,
            source="account",
            metadata={
                "account_id": account_id,
                "loss_pct": loss_pct,
                "limit_type": "daily",
                "daily_pnl": state.daily_pnl
            }
        )

        # Send notification
        self._send_notification(account_id, "daily", loss_pct)

        logger.critical(
            f"🚨 DAILY STOP triggered for account {account_id}: "
            f"{loss_pct:.1%} loss (limit: {self.max_daily_loss_pct:.1%})"
        )

    def _trigger_weekly_stop(
        self,
        account_id: str,
        state: AccountState,
        loss_pct: float
    ) -> None:
        """Trigger weekly stop for an account."""
        state.weekly_stop_triggered = True

        # Raise BLACK alert (weekly is more severe)
        self.alert_manager.raise_alert(
            tier=3,
            message=f"WEEKLY STOP triggered for {account_id}: {loss_pct:.1%} loss",
            threshold_pct=100.0,
            source="account",
            metadata={
                "account_id": account_id,
                "loss_pct": loss_pct,
                "limit_type": "weekly",
                "weekly_pnl": state.weekly_pnl
            }
        )

        # Send notification
        self._send_notification(account_id, "weekly", loss_pct)

        logger.critical(
            f"🚨🚨 WEEKLY STOP triggered for account {account_id}: "
            f"{loss_pct:.1%} loss (limit: {self.max_weekly_loss_pct:.1%})"
        )

    def _send_notification(
        self,
        account_id: str,
        stop_type: str,
        loss_pct: float
    ) -> None:
        """Send notification for stop trigger."""
        if not self.email_alerts:
            return

        # Log for SMS integration (future)
        if self.sms_alerts:
            logger.info(
                f"[SMS ALERT] Account {account_id}: {stop_type.upper()} stop at {loss_pct:.1%}"
            )

        # AlertService will handle email via the alert_manager
        logger.info(
            f"[EMAIL] Account {account_id}: {stop_type.upper()} stop notification sent"
        )

    def is_account_allowed(self, account_id: str) -> bool:
        """
        Check if an account is allowed to trade.

        Args:
            account_id: Account identifier

        Returns:
            True if account can trade, False if stop triggered
        """
        state = self.account_states.get(account_id)
        if state is None:
            return True

        # Check and reset counters first
        self._check_and_reset_counters(state)

        # Check stops
        if state.weekly_stop_triggered or state.daily_stop_triggered:
            return False

        return True

    def get_stop_status(self, account_id: str) -> Dict[str, Any]:
        """Get stop status for an account."""
        state = self.account_states.get(account_id)
        if state is None:
            return {
                "account_id": account_id,
                "daily_stop": False,
                "weekly_stop": False,
                "daily_loss_pct": 0.0,
                "weekly_loss_pct": 0.0
            }

        self._check_and_reset_counters(state)

        return {
            "account_id": account_id,
            "daily_stop": state.daily_stop_triggered,
            "weekly_stop": state.weekly_stop_triggered,
            "daily_loss_pct": state.daily_loss_pct,
            "weekly_loss_pct": state.weekly_loss_pct,
            "daily_pnl": state.daily_pnl,
            "weekly_pnl": state.weekly_pnl
        }

    def reset_account_stops(self, account_id: str) -> bool:
        """
        Reset stop flags for an account (after manual review).

        Args:
            account_id: Account identifier

        Returns:
            True if reset, False if account not found
        """
        state = self.account_states.get(account_id)
        if state is None:
            return False

        state.daily_stop_triggered = False
        state.weekly_stop_triggered = False

        # Clear related alerts
        self.alert_manager.clear_alerts_by_source("account")

        logger.info(f"Stops reset for account: {account_id}")
        return True

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all account states."""
        return {
            account_id: state.to_dict()
            for account_id, state in self.account_states.items()
        }

    def get_accounts_at_risk(self, threshold_pct: float = 50.0) -> List[Dict[str, Any]]:
        """
        Get accounts approaching stop limits.

        Args:
            threshold_pct: Minimum threshold percentage to include

        Returns:
            List of at-risk account states
        """
        at_risk = []
        for account_id, state in self.account_states.items():
            self._check_and_reset_counters(state)

            daily_pct = (state.daily_loss_pct / self.max_daily_loss_pct) * 100
            weekly_pct = (state.weekly_loss_pct / self.max_weekly_loss_pct) * 100

            if daily_pct >= threshold_pct or weekly_pct >= threshold_pct:
                at_risk.append({
                    "account_id": account_id,
                    "daily_threshold_pct": daily_pct,
                    "weekly_threshold_pct": weekly_pct,
                    "daily_stop": state.daily_stop_triggered,
                    "weekly_stop": state.weekly_stop_triggered
                })

        return at_risk


# Global singleton instance
_global_account_monitor: Optional[AccountMonitor] = None


def get_account_monitor() -> AccountMonitor:
    """Get or create the global AccountMonitor instance."""
    global _global_account_monitor
    if _global_account_monitor is None:
        _global_account_monitor = AccountMonitor()
    return _global_account_monitor


def reset_account_monitor() -> None:
    """Reset the global AccountMonitor instance (for testing)."""
    global _global_account_monitor
    _global_account_monitor = None
