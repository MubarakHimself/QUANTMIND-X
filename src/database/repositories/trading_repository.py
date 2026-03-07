"""
Trading Repository.

Provides database operations for trading models (RiskTierTransition, CryptoTrade, TradeJournal).
"""

from datetime import datetime
from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import RiskTierTransition, CryptoTrade, TradeJournal
from src.database.models.base import TradingMode


class RiskTierTransitionRepository(BaseRepository[RiskTierTransition]):
    """Repository for RiskTierTransition database operations."""

    model = RiskTierTransition

    def get_by_account_id(self, account_id: int, limit: int = 100) -> List[RiskTierTransition]:
        """Get tier transitions for an account."""
        with self.get_session() as session:
            transitions = session.query(RiskTierTransition).filter(
                RiskTierTransition.account_id == account_id
            ).order_by(RiskTierTransition.transition_timestamp.desc()).limit(limit).all()
            for t in transitions:
                session.expunge(t)
            return transitions

    def get_latest_by_account(self, account_id: int) -> Optional[RiskTierTransition]:
        """Get the most recent tier transition for an account."""
        with self.get_session() as session:
            transition = session.query(RiskTierTransition).filter(
                RiskTierTransition.account_id == account_id
            ).order_by(RiskTierTransition.transition_timestamp.desc()).first()
            if transition is not None:
                session.expunge(transition)
            return transition


class CryptoTradeRepository(BaseRepository[CryptoTrade]):
    """Repository for CryptoTrade database operations."""

    model = CryptoTrade

    def get_by_order_id(self, order_id: str) -> Optional[CryptoTrade]:
        """Get a trade by order ID."""
        with self.get_session() as session:
            trade = session.query(CryptoTrade).filter(
                CryptoTrade.order_id == order_id
            ).first()
            if trade is not None:
                session.expunge(trade)
            return trade

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[CryptoTrade]:
        """Get trades by symbol."""
        with self.get_session() as session:
            trades = session.query(CryptoTrade).filter(
                CryptoTrade.symbol == symbol
            ).order_by(CryptoTrade.open_timestamp.desc()).limit(limit).all()
            for trade in trades:
                session.expunge(trade)
            return trades

    def get_by_status(self, status: str, limit: int = 100) -> List[CryptoTrade]:
        """Get trades by status."""
        return self.filter_by(limit=limit, status=status)

    def get_open_trades(self, limit: int = 100) -> List[CryptoTrade]:
        """Get all open trades."""
        return self.get_by_status('open', limit)

    def get_by_broker(self, broker_id: str, limit: int = 100) -> List[CryptoTrade]:
        """Get trades by broker ID."""
        with self.get_session() as session:
            trades = session.query(CryptoTrade).filter(
                CryptoTrade.broker_id == broker_id
            ).order_by(CryptoTrade.open_timestamp.desc()).limit(limit).all()
            for trade in trades:
                session.expunge(trade)
            return trades

    def get_by_mode(self, mode: TradingMode, limit: int = 100) -> List[CryptoTrade]:
        """Get trades by trading mode."""
        with self.get_session() as session:
            trades = session.query(CryptoTrade).filter(
                CryptoTrade.mode == mode
            ).order_by(CryptoTrade.open_timestamp.desc()).limit(limit).all()
            for trade in trades:
                session.expunge(trade)
            return trades


class TradeJournalRepository(BaseRepository[TradeJournal]):
    """Repository for TradeJournal database operations."""

    model = TradeJournal

    def get_by_bot_id(self, bot_id: str, limit: int = 100) -> List[TradeJournal]:
        """Get journal entries by bot ID."""
        with self.get_session() as session:
            entries = session.query(TradeJournal).filter(
                TradeJournal.bot_id == bot_id
            ).order_by(TradeJournal.timestamp.desc()).limit(limit).all()
            for entry in entries:
                session.expunge(entry)
            return entries

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[TradeJournal]:
        """Get journal entries by symbol."""
        with self.get_session() as session:
            entries = session.query(TradeJournal).filter(
                TradeJournal.symbol == symbol
            ).order_by(TradeJournal.timestamp.desc()).limit(limit).all()
            for entry in entries:
                session.expunge(entry)
            return entries

    def get_by_balance_zone(self, zone: str, limit: int = 100) -> List[TradeJournal]:
        """Get journal entries by balance zone."""
        return self.filter_by(limit=limit, balance_zone=zone)

    def get_by_mode(self, mode: TradingMode, limit: int = 100) -> List[TradeJournal]:
        """Get journal entries by trading mode."""
        with self.get_session() as session:
            entries = session.query(TradeJournal).filter(
                TradeJournal.mode == mode
            ).order_by(TradeJournal.timestamp.desc()).limit(limit).all()
            for entry in entries:
                session.expunge(entry)
            return entries

    def get_by_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> List[TradeJournal]:
        """Get journal entries within a date range."""
        with self.get_session() as session:
            entries = session.query(TradeJournal).filter(
                TradeJournal.timestamp >= start,
                TradeJournal.timestamp <= end
            ).order_by(TradeJournal.timestamp.desc()).limit(limit).all()
            for entry in entries:
                session.expunge(entry)
            return entries
