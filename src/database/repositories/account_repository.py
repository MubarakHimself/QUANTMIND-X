"""
Account Repository.

Provides database operations for prop firm accounts.
"""

from datetime import date
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.database.engine import Session
from src.database.models import PropFirmAccount


class AccountRepository:
    """Repository for PropFirmAccount database operations."""

    def __init__(self):
        """Initialize the account repository."""
        pass

    @contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_by_account_id(self, account_id: str) -> Optional[PropFirmAccount]:
        """
        Retrieve a prop firm account by account ID.

        Args:
            account_id: MT5 account number

        Returns:
            PropFirmAccount object or None if not found
        """
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == account_id
            ).first()
            if account is not None:
                session.expunge(account)
            return account

    def create(
        self,
        account_id: str,
        firm_name: str,
        daily_loss_limit_pct: float = 5.0,
        hard_stop_buffer_pct: float = 1.0,
        target_profit_pct: float = 8.0,
        min_trading_days: int = 5
    ) -> PropFirmAccount:
        """
        Create a new prop firm account.

        Args:
            account_id: MT5 account number
            firm_name: Name of prop firm
            daily_loss_limit_pct: Maximum daily loss percentage
            hard_stop_buffer_pct: Safety buffer percentage
            target_profit_pct: Profit target percentage
            min_trading_days: Minimum trading days required

        Returns:
            Created PropFirmAccount object
        """
        with self.get_session() as session:
            account = PropFirmAccount(
                firm_name=firm_name,
                account_id=account_id,
                daily_loss_limit_pct=daily_loss_limit_pct,
                hard_stop_buffer_pct=hard_stop_buffer_pct,
                target_profit_pct=target_profit_pct,
                min_trading_days=min_trading_days
            )
            session.add(account)
            session.flush()
            session.refresh(account)
            session.expunge(account)
            return account

    def get_all(self) -> List[PropFirmAccount]:
        """Get all prop firm accounts."""
        with self.get_session() as session:
            accounts = session.query(PropFirmAccount).all()
            for account in accounts:
                session.expunge(account)
            return accounts

    def update(self, account: PropFirmAccount) -> PropFirmAccount:
        """Update an existing account."""
        with self.get_session() as session:
            session.merge(account)
            session.flush()
            session.refresh(account)
            session.expunge(account)
            return account

    def delete(self, account_id: str) -> bool:
        """Delete an account by ID."""
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == account_id
            ).first()
            if account:
                session.delete(account)
                return True
            return False
