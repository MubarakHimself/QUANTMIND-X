"""
House Money State Manager

Manages daily P&L tracking and dynamic risk adjustment based on house money effect.
Increases risk when trading with profits, decreases when down.

**Validates: Task Group 7.3 - HouseMoneyState table and manager**
"""

import logging
from typing import Optional, Dict, Any
from datetime import date, datetime

from src.database.manager import DatabaseManager
from src.database.models import HouseMoneyState

logger = logging.getLogger(__name__)


class HouseMoneyManager:
    """
    Manager for House Money State operations.

    Implements the house money effect:
    - 1.0x risk: Baseline (breakeven or small profit/loss)
    - 1.5x risk: Aggressive (up > 5% - trading with house money)
    - 0.5x risk: Conservative (down > 3% - preserve capital)

    Usage:
        manager = HouseMoneyManager()
        manager.update_pnl("12345", 5000.0)  # $5000 profit
        multiplier = manager.get_risk_multiplier("12345")
    """

    # Risk thresholds
    PROFIT_THRESHOLD_PCT = 0.05  # 5% profit triggers aggressive mode
    LOSS_THRESHOLD_PCT = 0.03    # 3% loss triggers conservative mode
    PRESERVATION_TARGET_PCT = 0.08  # 8% triggers preservation mode

    # Risk multipliers
    MULTIPLIER_BASELINE = 1.0
    MULTIPLIER_AGGRESSIVE = 1.5
    MULTIPLIER_CONSERVATIVE = 0.5

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize HouseMoney manager.

        Args:
            db_manager: Optional DatabaseManager instance (creates singleton if not provided)
        """
        self.db = db_manager or DatabaseManager()

    def get_or_create_state(self, account_id: str, state_date: Optional[date] = None) -> HouseMoneyState:
        """
        Get or create house money state for an account.

        Args:
            account_id: MT5 account number
            state_date: Date for the state (defaults to today)

        Returns:
            HouseMoneyState object
        """
        if state_date is None:
            state_date = date.today()

        with self.db.get_session() as session:
            # Try to find existing state
            state = session.query(HouseMoneyState).filter(
                HouseMoneyState.account_id == account_id,
                HouseMoneyState.date == state_date.isoformat()
            ).first()

            if state is None:
                # Create new state
                state = HouseMoneyState(
                    account_id=account_id,
                    daily_start_balance=0.0,
                    current_pnl=0.0,
                    high_water_mark=0.0,
                    risk_multiplier=self.MULTIPLIER_BASELINE,
                    is_preservation_mode=False,
                    date=state_date.isoformat()
                )
                session.add(state)
                session.flush()
                session.refresh(state)

            session.expunge(state)
            return state

    def update_pnl(
        self,
        account_id: str,
        pnl_amount: float,
        current_balance: Optional[float] = None
    ) -> HouseMoneyState:
        """
        Update P&L for an account and recalculate risk multiplier.

        Args:
            account_id: MT5 account number
            pnl_amount: Profit/loss amount to add (positive for profit, negative for loss)
            current_balance: Optional current balance for calculating percentage

        Returns:
            Updated HouseMoneyState object
        """
        state = self.get_or_create_state(account_id)

        # Update P&L and high water mark
        state.current_pnl += pnl_amount

        if current_balance is not None:
            equity = current_balance + state.current_pnl
            state.high_water_mark = max(state.high_water_mark, equity)

        # Calculate P&L percentage
        if state.daily_start_balance > 0:
            pnl_pct = state.current_pnl / state.daily_start_balance
        else:
            pnl_pct = 0.0

        # Update risk multiplier based on P&L
        state.risk_multiplier = self._calculate_multiplier(pnl_pct)

        # Check preservation mode
        if pnl_pct >= self.PRESERVATION_TARGET_PCT:
            state.is_preservation_mode = True

        # Save to database
        with self.db.get_session() as session:
            # Merge to update existing record
            state.updated_at = datetime.utcnow()
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(
            f"House money updated: account={account_id}, "
            f"pnl=${state.current_pnl:.2f}, multiplier={state.risk_multiplier:.2f}x"
        )

        return state

    def set_daily_start(self, account_id: str, start_balance: float) -> HouseMoneyState:
        """
        Set the daily start balance for an account.

        Should be called at the start of each trading day.

        Args:
            account_id: MT5 account number
            start_balance: Starting balance for the day

        Returns:
            Updated HouseMoneyState object
        """
        state = self.get_or_create_state(account_id)
        state.daily_start_balance = start_balance
        state.high_water_mark = start_balance

        # Save to database
        with self.db.get_session() as session:
            session.merge(state)
            session.flush()
            session.refresh(state)
            session.expunge(state)

        logger.info(f"Daily start set: account={account_id}, balance=${start_balance:.2f}")
        return state

    def get_risk_multiplier(self, account_id: str) -> float:
        """
        Get the current risk multiplier for an account.

        Args:
            account_id: MT5 account number

        Returns:
            Risk multiplier (0.5, 1.0, or 1.5)
        """
        state = self.get_or_create_state(account_id)
        return state.risk_multiplier

    def get_state(self, account_id: str, state_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
        """
        Get house money state for an account.

        Args:
            account_id: MT5 account number
            state_date: Date for the state (defaults to today)

        Returns:
            Dictionary with state data or None if not found
        """
        if state_date is None:
            state_date = date.today()

        with self.db.get_session() as session:
            state = session.query(HouseMoneyState).filter(
                HouseMoneyState.account_id == account_id,
                HouseMoneyState.date == state_date.isoformat()
            ).first()

            if state is None:
                return None

            return {
                "id": state.id,
                "account_id": state.account_id,
                "daily_start_balance": state.daily_start_balance,
                "current_pnl": state.current_pnl,
                "high_water_mark": state.high_water_mark,
                "risk_multiplier": state.risk_multiplier,
                "is_preservation_mode": state.is_preservation_mode,
                "date": state.date,
            }

    def reset_daily(self, account_id: str) -> None:
        """
        Reset house money state for a new trading day.

        Clears P&L and resets risk multiplier to baseline.
        Should be called at the start of each new trading day.

        Args:
            account_id: MT5 account number
        """
        state = self.get_or_create_state(account_id)

        with self.db.get_session() as session:
            state.current_pnl = 0.0
            state.risk_multiplier = self.MULTIPLIER_BASELINE
            state.is_preservation_mode = False
            state.updated_at = datetime.utcnow()
            session.merge(state)
            session.flush()

        logger.info(f"House money reset: account={account_id}")

    def _calculate_multiplier(self, pnl_pct: float) -> float:
        """
        Calculate risk multiplier based on P&L percentage.

        Args:
            pnl_pct: P&L as percentage (0.05 = 5% profit)

        Returns:
            Risk multiplier
        """
        # Aggressive mode: Trading with house money
        if pnl_pct >= self.PROFIT_THRESHOLD_PCT:
            return self.MULTIPLIER_AGGRESSIVE

        # Conservative mode: Preserve capital
        if pnl_pct <= -self.LOSS_THRESHOLD_PCT:
            return self.MULTIPLIER_CONSERVATIVE

        # Baseline mode
        return self.MULTIPLIER_BASELINE

    def is_preservation_mode(self, account_id: str) -> bool:
        """
        Check if preservation mode is active for an account.

        Args:
            account_id: MT5 account number

        Returns:
            True if preservation mode is active
        """
        state = self.get_or_create_state(account_id)
        return state.is_preservation_mode

    def get_all_states(self, state_date: Optional[date] = None) -> list[Dict[str, Any]]:
        """
        Get all house money states for a specific date.

        Args:
            state_date: Date to query (defaults to today)

        Returns:
            List of dictionaries with state data
        """
        if state_date is None:
            state_date = date.today()

        with self.db.get_session() as session:
            states = session.query(HouseMoneyState).filter(
                HouseMoneyState.date == state_date.isoformat()
            ).all()

            return [
                {
                    "id": s.id,
                    "account_id": s.account_id,
                    "daily_start_balance": s.daily_start_balance,
                    "current_pnl": s.current_pnl,
                    "high_water_mark": s.high_water_mark,
                    "risk_multiplier": s.risk_multiplier,
                    "is_preservation_mode": s.is_preservation_mode,
                    "date": s.date,
                }
                for s in states
            ]
