"""
DPR Score History — Persistence for DPR Score Audit and Week-over-Week Delta.

Story 17.1: DPR Composite Score Calculation

Provides:
- DPRScoreAuditLog: SQLAlchemy model for score persistence
- DPRScoreHistory: Data access layer for score history

Per NFR-D1: All DPR score calculations logged before any system acknowledgment.
"""

from datetime import datetime, timezone
from typing import List, Optional
from dataclasses import dataclass

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, JSON, Index
from sqlalchemy.orm import Session

from src.database.models.base import Base


class DPRScoreAuditLog(Base):
    """
    DPR Score Audit Log for immutable score calculation records.

    Persisted to SQLite before any system acknowledgment per NFR-D1.

    Attributes:
        id: Primary key
        bot_id: Bot identifier
        session_id: Session identifier (e.g., "LONDON", "NY_AM")
        scoring_window: Time window for scoring ("session", "fortnight")
        win_rate_score: Win rate component score (0-100)
        pnl_score: PnL component score (0-100)
        consistency_score: Consistency component score (0-100)
        ev_per_trade_score: EV per trade component score (0-100)
        composite_score: Final composite score (0-100)
        is_tied: Whether this score resulted in a tie
        tie_break_winner: Bot ID of tie-break winner if applicable
        specialist_boost_applied: Whether SESSION_SPECIALIST boost was applied
        session_concern_flag: Whether SESSION_CONCERN flag was set
        timestamp_utc: When calculation occurred
        metadata_json: Additional JSON context
    """
    __tablename__ = 'dpr_score_audit_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    bot_id = Column(String(100), nullable=False, index=True)
    session_id = Column(String(50), nullable=False, index=True)
    scoring_window = Column(String(20), nullable=False, default="session")
    win_rate_score = Column(Float, nullable=False)
    pnl_score = Column(Float, nullable=False)
    consistency_score = Column(Float, nullable=False)
    ev_per_trade_score = Column(Float, nullable=False)
    composite_score = Column(Integer, nullable=False, index=True)
    is_tied = Column(Boolean, nullable=False, default=False)
    tie_break_winner = Column(String(100), nullable=True)
    specialist_boost_applied = Column(Boolean, nullable=False, default=False)
    session_concern_flag = Column(Boolean, nullable=False, default=False)
    timestamp_utc = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    metadata_json = Column("metadata_json", JSON, nullable=True)

    __table_args__ = (
        Index('idx_dpr_audit_bot_session', 'bot_id', 'session_id'),
        Index('idx_dpr_audit_timestamp', 'timestamp_utc'),
    )

    def __repr__(self):
        return (
            f"<DPRScoreAuditLog(id={self.id}, bot={self.bot_id}, "
            f"session={self.session_id}, score={self.composite_score})>"
        )


@dataclass
class DPRScoreHistoryRecord:
    """
    DPR score history record for week-over-week delta calculation.

    Attributes:
        bot_id: Bot identifier
        session_id: Session identifier
        composite_score: Score at this point in time
        timestamp_utc: When score was recorded
    """
    bot_id: str
    session_id: str
    composite_score: int
    timestamp_utc: datetime


class DPRScoreHistory:
    """
    Data access layer for DPR score history.

    Provides methods for persisting and retrieving score history
    for week-over-week delta calculations.
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize DPR Score History.

        Args:
            db_session: SQLAlchemy session
        """
        self._db_session = db_session

    @property
    def db_session(self) -> Session:
        """Get or create database session."""
        if self._db_session is None:
            from src.database.models import SessionLocal
            self._db_session = SessionLocal()
        return self._db_session

    def persist_score(
        self,
        bot_id: str,
        session_id: str,
        composite_score: int,
        component_scores: dict,
        scoring_window: str = "session",
        is_tied: bool = False,
        tie_break_winner: Optional[str] = None,
        specialist_boost_applied: bool = False,
        session_concern_flag: bool = False,
        metadata: Optional[dict] = None,
    ) -> DPRScoreAuditLog:
        """
        Persist DPR score calculation to audit log.

        Args:
            bot_id: Bot identifier
            session_id: Session identifier
            composite_score: Final composite score (0-100)
            component_scores: Dictionary with component scores
            scoring_window: Time window ("session" or "fortnight")
            is_tied: Whether score resulted in tie
            tie_break_winner: Bot ID of tie-break winner if applicable
            specialist_boost_applied: Whether specialist boost was applied
            session_concern_flag: Whether SESSION_CONCERN flag was set
            metadata: Additional context

        Returns:
            Created DPRScoreAuditLog record
        """
        audit_log = DPRScoreAuditLog(
            bot_id=bot_id,
            session_id=session_id,
            scoring_window=scoring_window,
            win_rate_score=component_scores.get("win_rate", 0.0),
            pnl_score=component_scores.get("pnl", 0.0),
            consistency_score=component_scores.get("consistency", 0.0),
            ev_per_trade_score=component_scores.get("ev_per_trade", 0.0),
            composite_score=composite_score,
            is_tied=is_tied,
            tie_break_winner=tie_break_winner,
            specialist_boost_applied=specialist_boost_applied,
            session_concern_flag=session_concern_flag,
            metadata_json=metadata or {},
        )

        self.db_session.add(audit_log)
        self.db_session.commit()
        self.db_session.refresh(audit_log)

        return audit_log

    def get_bot_scores(
        self,
        bot_id: str,
        session_id: Optional[str] = None,
        scoring_window: Optional[str] = None,
        limit: int = 10,
    ) -> List[DPRScoreHistoryRecord]:
        """
        Get score history for a bot.

        Args:
            bot_id: Bot identifier
            session_id: Optional session filter
            scoring_window: Optional scoring window filter ("session" or "fortnight")
            limit: Maximum number of records to return

        Returns:
            List of DPRScoreHistoryRecord sorted by timestamp descending
        """
        query = self.db_session.query(DPRScoreAuditLog).filter(
            DPRScoreAuditLog.bot_id == bot_id
        )

        if session_id:
            query = query.filter(DPRScoreAuditLog.session_id == session_id)

        if scoring_window:
            query = query.filter(DPRScoreAuditLog.scoring_window == scoring_window)

        query = query.order_by(DPRScoreAuditLog.timestamp_utc.desc())
        query = query.limit(limit)

        results = query.all()

        return [
            DPRScoreHistoryRecord(
                bot_id=r.bot_id,
                session_id=r.session_id,
                composite_score=r.composite_score,
                timestamp_utc=r.timestamp_utc,
            )
            for r in results
        ]

    def get_fortnight_scores(self, bot_id: str) -> List[DPRScoreHistoryRecord]:
        """
        Get fortnight scores for week-over-week delta calculation.

        Args:
            bot_id: Bot identifier

        Returns:
            List of fortnight scores
        """
        query = self.db_session.query(DPRScoreAuditLog).filter(
            DPRScoreAuditLog.bot_id == bot_id,
            DPRScoreAuditLog.scoring_window == "fortnight"
        )
        query = query.order_by(DPRScoreAuditLog.timestamp_utc.desc())
        query = query.limit(2)

        results = query.all()

        return [
            DPRScoreHistoryRecord(
                bot_id=r.bot_id,
                session_id=r.session_id,
                composite_score=r.composite_score,
                timestamp_utc=r.timestamp_utc,
            )
            for r in results
        ]

    def close(self):
        """Close database session if we created it."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
