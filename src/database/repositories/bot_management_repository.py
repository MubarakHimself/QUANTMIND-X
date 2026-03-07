"""
Bot Management Repository.

Provides database operations for bot-related models (BotCloneHistory, DailyFeeTracking, ImportedEA, BotLifecycleLog).
"""

from typing import Optional, List
from datetime import datetime
from src.database.repositories.base_repository import BaseRepository
from src.database.models import BotCloneHistory, DailyFeeTracking, ImportedEA, BotLifecycleLog
from src.database.models.base import TradingMode


class BotCloneHistoryRepository(BaseRepository[BotCloneHistory]):
    """Repository for BotCloneHistory database operations."""

    model = BotCloneHistory

    def get_by_original_bot(self, original_bot_id: str, limit: int = 100) -> List[BotCloneHistory]:
        """Get clone history for an original bot."""
        with self.get_session() as session:
            records = session.query(BotCloneHistory).filter(
                BotCloneHistory.original_bot_id == original_bot_id
            ).order_by(BotCloneHistory.created_at.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_clone_bot(self, clone_bot_id: str) -> Optional[BotCloneHistory]:
        """Get clone history by clone bot ID."""
        with self.get_session() as session:
            record = session.query(BotCloneHistory).filter(
                BotCloneHistory.clone_bot_id == clone_bot_id
            ).first()
            if record is not None:
                session.expunge(record)
            return record


class DailyFeeTrackingRepository(BaseRepository[DailyFeeTracking]):
    """Repository for DailyFeeTracking database operations."""

    model = DailyFeeTracking

    def get_by_account_id(self, account_id: str, limit: int = 100) -> List[DailyFeeTracking]:
        """Get fee tracking records by account ID."""
        with self.get_session() as session:
            records = session.query(DailyFeeTracking).filter(
                DailyFeeTracking.account_id == account_id
            ).order_by(DailyFeeTracking.date.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_date(self, date_str: str, limit: int = 100) -> List[DailyFeeTracking]:
        """Get fee tracking records by date."""
        return self.filter_by(limit=limit, date=date_str)

    def get_by_account_and_date(self, account_id: str, date_str: str) -> Optional[DailyFeeTracking]:
        """Get fee tracking for a specific account and date."""
        with self.get_session() as session:
            record = session.query(DailyFeeTracking).filter(
                DailyFeeTracking.account_id == account_id,
                DailyFeeTracking.date == date_str
            ).first()
            if record is not None:
                session.expunge(record)
            return record

    def get_kill_switch_activated(self, limit: int = 100) -> List[DailyFeeTracking]:
        """Get records where kill switch was activated."""
        with self.get_session() as session:
            records = session.query(DailyFeeTracking).filter(
                DailyFeeTracking.kill_switch_activated == True
            ).order_by(DailyFeeTracking.date.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records


class ImportedEARepository(BaseRepository[ImportedEA]):
    """Repository for ImportedEA database operations."""

    model = ImportedEA

    def get_by_filename(self, filename: str, limit: int = 100) -> List[ImportedEA]:
        """Get imported EAs by filename."""
        return self.filter_by(limit=limit, ea_filename=filename)

    def get_by_checksum(self, checksum: str) -> Optional[ImportedEA]:
        """Get imported EA by checksum."""
        with self.get_session() as session:
            record = session.query(ImportedEA).filter(
                ImportedEA.checksum == checksum
            ).first()
            if record is not None:
                session.expunge(record)
            return record

    def get_by_status(self, status: str, limit: int = 100) -> List[ImportedEA]:
        """Get imported EAs by status."""
        return self.filter_by(limit=limit, status=status)

    def get_by_bot_manifest(self, manifest_id: int, limit: int = 100) -> List[ImportedEA]:
        """Get imported EAs linked to a bot manifest."""
        with self.get_session() as session:
            records = session.query(ImportedEA).filter(
                ImportedEA.bot_manifest_id == manifest_id
            ).order_by(ImportedEA.imported_at.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_new_and_updated(self, limit: int = 100) -> List[ImportedEA]:
        """Get EAs with new or updated status."""
        with self.get_session() as session:
            records = session.query(ImportedEA).filter(
                ImportedEA.status.in_(['new', 'updated'])
            ).order_by(ImportedEA.imported_at.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records


class BotLifecycleLogRepository(BaseRepository[BotLifecycleLog]):
    """Repository for BotLifecycleLog database operations."""

    model = BotLifecycleLog

    def get_by_bot_id(self, bot_id: str, limit: int = 100) -> List[BotLifecycleLog]:
        """Get lifecycle logs for a specific bot."""
        with self.get_session() as session:
            records = session.query(BotLifecycleLog).filter(
                BotLifecycleLog.bot_id == bot_id
            ).order_by(BotLifecycleLog.timestamp.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_from_tag(self, from_tag: str, limit: int = 100) -> List[BotLifecycleLog]:
        """Get lifecycle logs by source tag."""
        return self.filter_by(limit=limit, from_tag=from_tag)

    def get_by_to_tag(self, to_tag: str, limit: int = 100) -> List[BotLifecycleLog]:
        """Get lifecycle logs by destination tag."""
        return self.filter_by(limit=limit, to_tag=to_tag)

    def get_by_triggered_by(self, triggered_by: str, limit: int = 100) -> List[BotLifecycleLog]:
        """Get lifecycle logs by trigger source."""
        return self.filter_by(limit=limit, triggered_by=triggered_by)

    def get_by_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> List[BotLifecycleLog]:
        """Get lifecycle logs within a date range."""
        with self.get_session() as session:
            records = session.query(BotLifecycleLog).filter(
                BotLifecycleLog.timestamp >= start,
                BotLifecycleLog.timestamp <= end
            ).order_by(BotLifecycleLog.timestamp.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records
