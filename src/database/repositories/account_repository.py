"""
Account Repository

Provides data access methods for PropFirmAccount model.
"""

from typing import Optional
from sqlalchemy.orm import Session

from ..models import PropFirmAccount
from ..engine import Session as SessionFactory


class AccountRepository:
    """
    Repository for PropFirmAccount data access.

    Handles all database operations related to prop firm accounts.
    """

    def __init__(self, session: Optional[Session] = None):
        """Initialize repository with optional session."""
        self._session = session

    @property
    def session(self) -> Session:
        """Get the session (creates new if not provided)."""
        if self._session is not None:
            return self._session
        return SessionFactory()

    def get(self, account_id: str) -> Optional[PropFirmAccount]:
        """
        Retrieve a prop firm account by account ID.

        Args:
            account_id: MT5 account number

        Returns:
            PropFirmAccount object or None if not found
        """
        account = self.session.query(PropFirmAccount).filter(
            PropFirmAccount.account_id == account_id
        ).first()
        if account is not None:
            self.session.expunge(account)
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
        account = PropFirmAccount(
            firm_name=firm_name,
            account_id=account_id,
            daily_loss_limit_pct=daily_loss_limit_pct,
            hard_stop_buffer_pct=hard_stop_buffer_pct,
            target_profit_pct=target_profit_pct,
            min_trading_days=min_trading_days
        )
        self.session.add(account)
        self.session.flush()
        self.session.refresh(account)
        self.session.expunge(account)
        return account

    def get_by_id(self, account_id: int) -> Optional[PropFirmAccount]:
        """
        Retrieve a prop firm account by primary key.

        Args:
            account_id: Primary key ID

        Returns:
            PropFirmAccount object or None if not found
        """
        account = self.session.query(PropFirmAccount).filter(
            PropFirmAccount.id == account_id
        ).first()
        if account is not None:
            self.session.expunge(account)
        return account
