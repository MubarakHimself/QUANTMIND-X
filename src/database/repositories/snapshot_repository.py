"""
Snapshot Repository.

Provides database operations for daily account snapshots.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from src.database.engine import Session
from src.database.models import PropFirmAccount, DailySnapshot


class SnapshotRepository:
    """Repository for DailySnapshot database operations."""

    def __init__(self):
        """Initialize the snapshot repository."""
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

    def save_daily_snapshot(
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

        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                account = PropFirmAccount(
                    firm_name="Unknown",
                    account_id=str(account_id)
                )
                session.add(account)
                session.flush()

            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id,
                DailySnapshot.date == snapshot_date
            ).first()

            if snapshot is None:
                snapshot = DailySnapshot(
                    account_id=account.id,
                    date=snapshot_date,
                    daily_start_balance=balance,
                    high_water_mark=max(equity, balance),
                    current_equity=equity,
                    daily_drawdown_pct=0.0,
                    is_breached=False
                )
                session.add(snapshot)
            else:
                snapshot.current_equity = equity
                snapshot.high_water_mark = max(snapshot.high_water_mark, equity)
                snapshot.snapshot_timestamp = datetime.utcnow()

                if snapshot.daily_start_balance > 0:
                    snapshot.daily_drawdown_pct = (
                        (snapshot.daily_start_balance - equity) /
                        snapshot.daily_start_balance * 100
                    )

            session.flush()
            session.refresh(snapshot)
            session.expunge(snapshot)
            return snapshot

    def get_by_date(
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

        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                return None

            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id,
                DailySnapshot.date == snapshot_date
            ).first()

            if snapshot is None:
                return None

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
        """Retrieve the most recent daily snapshot for an account."""
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                return None

            snapshot = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id
            ).order_by(DailySnapshot.date.desc()).first()

            if snapshot is None:
                return None

            session.expunge(snapshot)
            return snapshot

    def get_drawdown(self, account_id: str) -> float:
        """Calculate current daily drawdown percentage."""
        snapshot = self.get_by_date(account_id)
        if snapshot is None:
            return 0.0
        return snapshot['daily_drawdown_pct']

    def get_start_balance(self, account_id: str) -> float:
        """Get the daily start balance for an account."""
        snapshot = self.get_by_date(account_id)
        if snapshot is None:
            return 0.0
        return snapshot['daily_start_balance']

    def get_all_for_account(self, account_id: str) -> List[DailySnapshot]:
        """Get all snapshots for an account."""
        with self.get_session() as session:
            account = session.query(PropFirmAccount).filter(
                PropFirmAccount.account_id == str(account_id)
            ).first()

            if account is None:
                return []

            snapshots = session.query(DailySnapshot).filter(
                DailySnapshot.account_id == account.id
            ).order_by(DailySnapshot.date.desc()).all()

            for snapshot in snapshots:
                session.expunge(snapshot)
            return snapshots
