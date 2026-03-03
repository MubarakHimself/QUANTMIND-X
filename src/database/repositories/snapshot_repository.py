"""
Snapshot Repository

Provides data access methods for DailySnapshot model.
"""

from datetime import datetime, date
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from ..models import PropFirmAccount, DailySnapshot
from ..engine import Session as SessionFactory


class SnapshotRepository:
    """
    Repository for DailySnapshot data access.

    Handles all database operations related to daily account snapshots.
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

    def save(
        self,
        account_id: str,
        equity: float,
        balance: float,
        snapshot_date: Optional[str] = None
    ) -> DailySnapshot:
        """
        Save or update a daily snapshot for an account.

        Implements upsert behavior: creates new snapshot if none exists
        for the date, otherwise updates the existing one.

        Args:
            account_id: MT5 account number or string ID
            equity: Current equity value
            balance: Current balance value
            snapshot_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Created or updated DailySnapshot object
        """
        if snapshot_date is None:
            snapshot_date = date.today().isoformat()

        # Get account by account_id string
        account = self.session.query(PropFirmAccount).filter(
            PropFirmAccount.account_id == str(account_id)
        ).first()

        # Auto-create account if it doesn't exist
        if account is None:
            account = PropFirmAccount(
                firm_name="Unknown",
                account_id=str(account_id)
            )
            self.session.add(account)
            self.session.flush()

        # Check if snapshot exists for this date
        snapshot = self.session.query(DailySnapshot).filter(
            DailySnapshot.account_id == account.id,
            DailySnapshot.date == snapshot_date
        ).first()

        if snapshot is None:
            # Create new snapshot
            snapshot = DailySnapshot(
                account_id=account.id,
                date=snapshot_date,
                daily_start_balance=balance,
                high_water_mark=max(equity, balance),
                current_equity=equity,
                daily_drawdown_pct=0.0,
                is_breached=False
            )
            self.session.add(snapshot)
        else:
            # Update existing snapshot
            snapshot.current_equity = equity
            snapshot.high_water_mark = max(snapshot.high_water_mark, equity)
            snapshot.snapshot_timestamp = datetime.utcnow()

            # Recalculate drawdown
            if snapshot.daily_start_balance > 0:
                snapshot.daily_drawdown_pct = (
                    (snapshot.daily_start_balance - equity) /
                    snapshot.daily_start_balance * 100
                )

        self.session.flush()
        self.session.refresh(snapshot)
        self.session.expunge(snapshot)
        return snapshot

    def get(
        self,
        account_id: str,
        snapshot_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a daily snapshot for an account.

        Args:
            account_id: MT5 account number
            snapshot_date: Date string (YYYY-MM-DD), defaults to today

        Returns:
            Dictionary with snapshot data or None if not found
        """
        if snapshot_date is None:
            snapshot_date = date.today().isoformat()

        account = self.session.query(PropFirmAccount).filter(
            PropFirmAccount.account_id == str(account_id)
        ).first()

        if account is None:
            return None

        snapshot = self.session.query(DailySnapshot).filter(
            DailySnapshot.account_id == account.id,
            DailySnapshot.date == snapshot_date
        ).first()

        if snapshot is None:
            return None

        # Return dictionary to avoid DetachedInstanceError
        return {
            'id': snapshot.id,
            'account_id': snapshot.account_id,
            'date': snapshot.date,
            'daily_start_balance': snapshot.daily_start_balance,
            'high_water_mark': snapshot.high_water_mark,
            'current_equity': snapshot.current_equity,
            'daily_drawdown_pct': snapshot.daily_drawdown_pct,
            'is_breached': snapshot.is_breached,
            'snapshot_timestamp': snapshot.snapshot_timestamp
        }

    def get_latest(self, account_id: str) -> Optional[DailySnapshot]:
        """
        Retrieve the most recent daily snapshot for an account.

        Args:
            account_id: MT5 account number or string ID

        Returns:
            Latest DailySnapshot object or None if not found
        """
        account = self.session.query(PropFirmAccount).filter(
            PropFirmAccount.account_id == str(account_id)
        ).first()

        if account is None:
            return None

        snapshot = self.session.query(DailySnapshot).filter(
            DailySnapshot.account_id == account.id
        ).order_by(DailySnapshot.date.desc()).first()

        if snapshot is None:
            return None

        # Detach from session to avoid DetachedInstanceError
        self.session.expunge(snapshot)
        return snapshot

    def get_drawdown(self, account_id: str) -> float:
        """
        Calculate current daily drawdown percentage.

        Args:
            account_id: MT5 account number

        Returns:
            Daily drawdown percentage (e.g., 2.5 for 2.5% drawdown)
            Returns 0.0 if account or snapshot not found
        """
        snapshot = self.get(account_id)
        if snapshot is None:
            return 0.0

        return snapshot['daily_drawdown_pct']

    def get_start_balance(self, account_id: str) -> float:
        """
        Get the daily start balance for an account.

        Args:
            account_id: MT5 account number

        Returns:
            Daily start balance, or 0.0 if not found
        """
        snapshot = self.get(account_id)
        if snapshot is None:
            return 0.0

        return snapshot['daily_start_balance']
