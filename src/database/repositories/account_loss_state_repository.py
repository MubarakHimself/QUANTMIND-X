"""
Account Loss State Repository.

Provides database operations for account loss state tracking (Tier 3 protection).
"""

from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import AccountLossState


class AccountLossStateRepository(BaseRepository[AccountLossState]):
    """Repository for AccountLossState database operations."""

    model = AccountLossState

    def get_by_account_id(self, account_id: str) -> Optional[AccountLossState]:
        """Get loss state by account ID."""
        with self.get_session() as session:
            state = session.query(AccountLossState).filter(
                AccountLossState.account_id == account_id
            ).first()
            if state is not None:
                session.expunge(state)
            return state

    def get_active_stops(self) -> List[AccountLossState]:
        """Get all accounts with active stops."""
        with self.get_session() as session:
            states = session.query(AccountLossState).filter(
                (AccountLossState.daily_stop_triggered == True) |
                (AccountLossState.weekly_stop_triggered == True)
            ).all()
            for state in states:
                session.expunge(state)
            return states

    def get_daily_stop_accounts(self) -> List[AccountLossState]:
        """Get accounts with daily stop triggered."""
        with self.get_session() as session:
            states = session.query(AccountLossState).filter(
                AccountLossState.daily_stop_triggered == True
            ).all()
            for state in states:
                session.expunge(state)
            return states

    def get_weekly_stop_accounts(self) -> List[AccountLossState]:
        """Get accounts with weekly stop triggered."""
        with self.get_session() as session:
            states = session.query(AccountLossState).filter(
                AccountLossState.weekly_stop_triggered == True
            ).all()
            for state in states:
                session.expunge(state)
            return states

    def create(
        self,
        account_id: str,
        initial_balance: float = 10000.0
    ) -> AccountLossState:
        """Create a new account loss state."""
        return super().create(
            account_id=account_id,
            initial_balance=initial_balance,
            daily_pnl=0.0,
            weekly_pnl=0.0,
            daily_stop_triggered=False,
            weekly_stop_triggered=False
        )

    def update_pnl(
        self,
        account_id: str,
        daily_pnl: float,
        weekly_pnl: float,
        last_reset_date: str = None,
        week_start: str = None
    ) -> Optional[AccountLossState]:
        """Update PnL for an account."""
        with self.get_session() as session:
            state = session.query(AccountLossState).filter(
                AccountLossState.account_id == account_id
            ).first()

            if state is None:
                return None

            state.daily_pnl = daily_pnl
            state.weekly_pnl = weekly_pnl
            if last_reset_date:
                state.last_reset_date = last_reset_date
            if week_start:
                state.week_start = week_start

            session.flush()
            session.refresh(state)
            session.expunge(state)
            return state

    def trigger_daily_stop(self, account_id: str) -> Optional[AccountLossState]:
        """Trigger daily stop for an account."""
        with self.get_session() as session:
            state = session.query(AccountLossState).filter(
                AccountLossState.account_id == account_id
            ).first()

            if state is None:
                return None

            state.daily_stop_triggered = True
            session.flush()
            session.refresh(state)
            session.expunge(state)
            return state

    def trigger_weekly_stop(self, account_id: str) -> Optional[AccountLossState]:
        """Trigger weekly stop for an account."""
        with self.get_session() as session:
            state = session.query(AccountLossState).filter(
                AccountLossState.account_id == account_id
            ).first()

            if state is None:
                return None

            state.weekly_stop_triggered = True
            session.flush()
            session.refresh(state)
            session.expunge(state)
            return state

    def reset_daily(self, account_id: str) -> Optional[AccountLossState]:
        """Reset daily stop for an account."""
        with self.get_session() as session:
            state = session.query(AccountLossState).filter(
                AccountLossState.account_id == account_id
            ).first()

            if state is None:
                return None

            state.daily_pnl = 0.0
            state.daily_stop_triggered = False
            session.flush()
            session.refresh(state)
            session.expunge(state)
            return state
