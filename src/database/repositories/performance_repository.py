"""
Performance Repository.

Provides database operations for performance models (StrategyPerformance, PaperTradingPerformance, HouseMoneyState, StrategyFamilyState).
"""

from datetime import date, datetime
from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import (
    StrategyPerformance,
    PaperTradingPerformance,
    HouseMoneyState,
    StrategyFamilyState,
)
from src.database.models.base import TradingMode


class StrategyPerformanceRepository(BaseRepository[StrategyPerformance]):
    """Repository for StrategyPerformance database operations."""

    model = StrategyPerformance

    def get_by_strategy_name(self, strategy_name: str, limit: int = 100) -> List[StrategyPerformance]:
        """Get performance records by strategy name."""
        with self.get_session() as session:
            records = session.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_name == strategy_name
            ).order_by(StrategyPerformance.created_at.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_kelly_score(self, min_kelly: float, limit: int = 100) -> List[StrategyPerformance]:
        """Get strategies with Kelly score above threshold."""
        with self.get_session() as session:
            records = session.query(StrategyPerformance).filter(
                StrategyPerformance.kelly_score >= min_kelly
            ).order_by(StrategyPerformance.kelly_score.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_mode(self, mode: TradingMode, limit: int = 100) -> List[StrategyPerformance]:
        """Get performance records by trading mode."""
        with self.get_session() as session:
            records = session.query(StrategyPerformance).filter(
                StrategyPerformance.mode == mode
            ).order_by(StrategyPerformance.created_at.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records


class PaperTradingPerformanceRepository(BaseRepository[PaperTradingPerformance]):
    """Repository for PaperTradingPerformance database operations."""

    model = PaperTradingPerformance

    def get_by_agent_id(self, agent_id: str, limit: int = 100) -> List[PaperTradingPerformance]:
        """Get performance records by agent ID."""
        with self.get_session() as session:
            records = session.query(PaperTradingPerformance).filter(
                PaperTradingPerformance.agent_id == agent_id
            ).order_by(PaperTradingPerformance.timestamp.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_validation_status(self, status: str, limit: int = 100) -> List[PaperTradingPerformance]:
        """Get records by validation status."""
        return self.filter_by(limit=limit, validation_status=status)

    def get_validated_agents(self, limit: int = 100) -> List[PaperTradingPerformance]:
        """Get agents that have been validated."""
        with self.get_session() as session:
            records = session.query(PaperTradingPerformance).filter(
                PaperTradingPerformance.validation_status == 'validated'
            ).order_by(PaperTradingPerformance.sharpe_ratio.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_mode(self, mode: TradingMode, limit: int = 100) -> List[PaperTradingPerformance]:
        """Get performance records by trading mode."""
        with self.get_session() as session:
            records = session.query(PaperTradingPerformance).filter(
                PaperTradingPerformance.mode == mode
            ).order_by(PaperTradingPerformance.timestamp.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records


class HouseMoneyStateRepository(BaseRepository[HouseMoneyState]):
    """Repository for HouseMoneyState database operations."""

    model = HouseMoneyState

    def get_by_account_id(self, account_id: str, limit: int = 100) -> List[HouseMoneyState]:
        """Get house money state records by account ID."""
        with self.get_session() as session:
            records = session.query(HouseMoneyState).filter(
                HouseMoneyState.account_id == account_id
            ).order_by(HouseMoneyState.date.desc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_by_date(self, date_str: str) -> List[HouseMoneyState]:
        """Get house money state records by date."""
        return self.filter_by(date=date_str)

    def get_by_account_and_date(self, account_id: str, date_str: str) -> Optional[HouseMoneyState]:
        """Get house money state for a specific account and date."""
        with self.get_session() as session:
            record = session.query(HouseMoneyState).filter(
                HouseMoneyState.account_id == account_id,
                HouseMoneyState.date == date_str
            ).first()
            if record is not None:
                session.expunge(record)
            return record

    def get_in_preservation_mode(self, limit: int = 100) -> List[HouseMoneyState]:
        """Get accounts in preservation mode."""
        with self.get_session() as session:
            records = session.query(HouseMoneyState).filter(
                HouseMoneyState.is_preservation_mode == True
            ).order_by(HouseMoneyState.current_pnl.asc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records


class StrategyFamilyStateRepository(BaseRepository[StrategyFamilyState]):
    """Repository for StrategyFamilyState database operations."""

    model = StrategyFamilyState

    def get_by_family(self, family: str) -> Optional[StrategyFamilyState]:
        """Get state for a specific strategy family."""
        with self.get_session() as session:
            record = session.query(StrategyFamilyState).filter(
                StrategyFamilyState.family == family
            ).first()
            if record is not None:
                session.expunge(record)
            return record

    def get_quarantined_families(self, limit: int = 100) -> List[StrategyFamilyState]:
        """Get all quarantined strategy families."""
        with self.get_session() as session:
            records = session.query(StrategyFamilyState).filter(
                StrategyFamilyState.is_quarantined == True
            ).order_by(StrategyFamilyState.total_pnl.asc()).limit(limit).all()
            for record in records:
                session.expunge(record)
            return records

    def get_all_families(self, limit: int = 100) -> List[StrategyFamilyState]:
        """Get all strategy families."""
        return self.get_all(limit=limit)
