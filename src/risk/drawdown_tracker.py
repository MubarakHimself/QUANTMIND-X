"""
3/5/7 Risk Framework - Tiered Drawdown Tracker

Tracks daily/weekly/monthly drawdown limits:
- Max 3% daily drawdown
- Max 5% weekly drawdown
- Max 7% monthly drawdown

Drawdown formula: (peak_balance - current_balance) / peak_balance
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date, timedelta
from typing import Optional, Tuple, Dict, Any

from src.risk.config import (
    DAILY_DRAWDOWN_LIMIT,
    WEEKLY_DRAWDOWN_LIMIT,
    MONTHLY_DRAWDOWN_LIMIT,
)

logger = logging.getLogger(__name__)


@dataclass
class DrawdownState:
    """Internal state for drawdown tracking."""
    daily_start_balance: float = 0.0
    weekly_start_balance: float = 0.0
    monthly_start_balance: float = 0.0
    peak_balance: float = 0.0
    last_reset_date: Optional[date] = None
    last_reset_week: Optional[int] = None
    last_reset_month: Optional[int] = None


class DrawdownTracker:
    """
    Tracks tiered drawdown limits (3/5/7 framework).

    Drawdown is calculated from peak balance:
    - Daily drawdown: measured from start of day peak
    - Weekly drawdown: measured from start of week peak
    - Monthly drawdown: measured from start of month peak

    Attributes:
        daily_limit: Maximum daily drawdown (default 3%)
        weekly_limit: Maximum weekly drawdown (default 5%)
        monthly_limit: Maximum monthly drawdown (default 7%)

    Usage:
        tracker = DrawdownTracker()
        tracker.record_trade(pnl=-50.0)
        allowed, reason = tracker.is_drawdown_exceeded()
        daily_pct, weekly_pct, monthly_pct = tracker.get_current_drawdown()
    """

    def __init__(
        self,
        daily_limit: float = DAILY_DRAWDOWN_LIMIT,
        weekly_limit: float = WEEKLY_DRAWDOWN_LIMIT,
        monthly_limit: float = MONTHLY_DRAWDOWN_LIMIT,
        initial_balance: float = 10000.0
    ):
        """
        Initialize DrawdownTracker.

        Args:
            daily_limit: Maximum daily drawdown (default 3%)
            weekly_limit: Maximum weekly drawdown (default 5%)
            monthly_limit: Maximum monthly drawdown (default 7%)
            initial_balance: Starting balance for tracking
        """
        self.daily_limit = daily_limit
        self.weekly_limit = weekly_limit
        self.monthly_limit = monthly_limit

        self._state = DrawdownState(
            daily_start_balance=initial_balance,
            weekly_start_balance=initial_balance,
            monthly_start_balance=initial_balance,
            peak_balance=initial_balance,
        )

        now = datetime.now(timezone.utc)
        self._state.last_reset_date = now.date()
        self._state.last_reset_week = self._get_week_number(now)
        self._state.last_reset_month = now.month

        logger.info(
            f"DrawdownTracker initialized: "
            f"daily={daily_limit:.1%}, weekly={weekly_limit:.1%}, monthly={monthly_limit:.1%}, "
            f"initial_balance={initial_balance}"
        )

    def _get_week_number(self, dt: datetime) -> int:
        """Get ISO week number."""
        return dt.isocalendar()[1]

    def _get_month_start(self, d: date) -> date:
        """Get the first day of the month containing date d."""
        return date(d.year, d.month, 1)

    def _reset_if_new_period(self) -> None:
        """Check and reset counters on day/week/month boundary."""
        now = datetime.now(timezone.utc)
        today = now.date()
        current_week = self._get_week_number(now)
        current_month = now.month

        # Check for new day
        if self._state.last_reset_date is None or self._state.last_reset_date < today:
            if self._state.last_reset_date is not None:
                logger.info(
                    f"New day detected: resetting daily counters "
                    f"(was {self._state.last_reset_date}, now {today})"
                )
            self._state.daily_start_balance = self._state.peak_balance
            self._state.last_reset_date = today

        # Check for new week (Monday)
        if self._state.last_reset_week is None or self._state.last_reset_week < current_week:
            if self._state.last_reset_week is not None:
                logger.info(
                    f"New week detected: resetting weekly counters "
                    f"(was week {self._state.last_reset_week}, now week {current_week})"
                )
            self._state.weekly_start_balance = self._state.peak_balance
            self._state.last_reset_week = current_week

        # Check for new month
        if self._state.last_reset_month is None or self._state.last_reset_month < current_month:
            if self._state.last_reset_month is not None:
                logger.info(
                    f"New month detected: resetting monthly counters "
                    f"(was month {self._state.last_reset_month}, now month {current_month})"
                )
            self._state.monthly_start_balance = self._state.peak_balance
            self._state.last_reset_month = current_month

    def record_trade(self, pnl: float, current_balance: Optional[float] = None) -> None:
        """
        Record a trade and update drawdown counters.

        Args:
            pnl: Profit/loss from the trade (negative for loss)
            current_balance: Current account balance (optional, will update peak if provided)
        """
        self._reset_if_new_period()

        # Update peak balance if current balance is provided and higher
        if current_balance is not None and current_balance > 0:
            if current_balance > self._state.peak_balance:
                self._state.peak_balance = current_balance
                # Also update period start balances to new peak
                self._state.daily_start_balance = current_balance
                self._state.weekly_start_balance = current_balance
                self._state.monthly_start_balance = current_balance
                logger.debug(f"New peak balance: {current_balance}")

    def get_current_drawdown(self) -> Tuple[float, float, float]:
        """
        Get current drawdown percentages for all periods.

        Returns:
            Tuple of (daily_pct, weekly_pct, monthly_pct) as decimals
        """
        self._reset_if_new_period()

        peak = self._state.peak_balance
        if peak <= 0:
            return 0.0, 0.0, 0.0

        # Drawdown = how far we've fallen from peak
        # period_start_balance is the balance at the start of the period
        daily_drawdown = max(0.0, (peak - self._state.daily_start_balance) / peak)
        weekly_drawdown = max(0.0, (peak - self._state.weekly_start_balance) / peak)
        monthly_drawdown = max(0.0, (peak - self._state.monthly_start_balance) / peak)

        return daily_drawdown, weekly_drawdown, monthly_drawdown

    def is_drawdown_exceeded(self) -> Tuple[bool, str]:
        """
        Check if any drawdown limit is exceeded.

        Returns:
            Tuple of (exceeded, which_limit) where which_limit is
            "daily", "weekly", "monthly", or "" if no limit exceeded
        """
        self._reset_if_new_period()

        daily_pct, weekly_pct, monthly_pct = self.get_current_drawdown()

        if daily_pct >= self.daily_limit:
            logger.warning(
                f"Daily drawdown exceeded: {daily_pct:.2%} >= {self.daily_limit:.2%}"
            )
            return True, "daily"

        if weekly_pct >= self.weekly_limit:
            logger.warning(
                f"Weekly drawdown exceeded: {weekly_pct:.2%} >= {self.weekly_limit:.2%}"
            )
            return True, "weekly"

        if monthly_pct >= self.monthly_limit:
            logger.warning(
                f"Monthly drawdown exceeded: {monthly_pct:.2%} >= {self.monthly_limit:.2%}"
            )
            return True, "monthly"

        return False, ""

    def get_drawdown_status(self) -> Dict[str, Any]:
        """
        Get detailed drawdown status.

        Returns:
            Dictionary with drawdown information
        """
        self._reset_if_new_period()

        daily_pct, weekly_pct, monthly_pct = self.get_current_drawdown()
        exceeded, limit = self.is_drawdown_exceeded()

        return {
            "daily_drawdown_pct": daily_pct,
            "weekly_drawdown_pct": weekly_pct,
            "monthly_drawdown_pct": monthly_pct,
            "daily_limit": self.daily_limit,
            "weekly_limit": self.weekly_limit,
            "monthly_limit": self.monthly_limit,
            "is_exceeded": exceeded,
            "exceeded_limit": limit,
            "peak_balance": self._state.peak_balance,
            "daily_start_balance": self._state.daily_start_balance,
            "weekly_start_balance": self._state.weekly_start_balance,
            "monthly_start_balance": self._state.monthly_start_balance,
        }

    def reset_counters(self) -> None:
        """Manually reset all counters (use with caution)."""
        self._state.last_reset_date = date.today()
        self._state.last_reset_week = self._get_week_number(datetime.now(timezone.utc))
        self._state.last_reset_month = datetime.now(timezone.utc).month
        logger.info("Drawdown counters manually reset")
