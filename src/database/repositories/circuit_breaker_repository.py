"""
Circuit Breaker Repository.

Provides database operations for bot circuit breakers.
"""

from datetime import datetime, timezone
from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import BotCircuitBreaker
from src.database.models.base import TradingMode


class CircuitBreakerRepository(BaseRepository[BotCircuitBreaker]):
    """Repository for BotCircuitBreaker database operations."""

    model = BotCircuitBreaker

    def get_by_bot_id(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """Get circuit breaker state for a bot."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()
            if cb is not None:
                session.expunge(cb)
            return cb

    def get_quarantined(self, mode: TradingMode = None) -> List[BotCircuitBreaker]:
        """Get all quarantined bots."""
        with self.get_session() as session:
            query = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.is_quarantined == True
            )
            if mode is not None:
                query = query.filter(BotCircuitBreaker.mode == mode)

            cbs = query.all()
            for cb in cbs:
                session.expunge(cb)
            return cbs

    def get_active_bots(self, mode: TradingMode = None) -> List[BotCircuitBreaker]:
        """Get all active (non-quarantined) bots."""
        with self.get_session() as session:
            query = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.is_quarantined == False
            )
            if mode is not None:
                query = query.filter(BotCircuitBreaker.mode == mode)

            cbs = query.all()
            for cb in cbs:
                session.expunge(cb)
            return cbs

    def create(
        self,
        bot_id: str,
        mode: TradingMode = TradingMode.LIVE
    ) -> BotCircuitBreaker:
        """Create a new circuit breaker for a bot."""
        return super().create(
            bot_id=bot_id,
            consecutive_losses=0,
            daily_trade_count=0,
            is_quarantined=False,
            mode=mode
        )

    def record_loss(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """Record a loss for a bot, incrementing consecutive losses."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if cb is None:
                return None

            cb.consecutive_losses += 1
            session.flush()
            session.refresh(cb)
            session.expunge(cb)
            return cb

    def record_trade(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """Record a trade for a bot, incrementing daily trade count."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if cb is None:
                return None

            cb.daily_trade_count += 1
            cb.last_trade_time = datetime.now(timezone.utc)
            session.flush()
            session.refresh(cb)
            session.expunge(cb)
            return cb

    def quarantine(
        self,
        bot_id: str,
        reason: str = None
    ) -> Optional[BotCircuitBreaker]:
        """Quarantine a bot."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if cb is None:
                return None

            cb.is_quarantined = True
            cb.quarantine_reason = reason
            cb.quarantine_start = datetime.now(timezone.utc)
            session.flush()
            session.refresh(cb)
            session.expunge(cb)
            return cb

    def unquarantine(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """Remove quarantine from a bot."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if cb is None:
                return None

            cb.is_quarantined = False
            cb.quarantine_reason = None
            cb.quarantine_start = None
            cb.consecutive_losses = 0
            session.flush()
            session.refresh(cb)
            session.expunge(cb)
            return cb

    def reset_daily_count(self, bot_id: str) -> Optional[BotCircuitBreaker]:
        """Reset daily trade count for a bot."""
        with self.get_session() as session:
            cb = session.query(BotCircuitBreaker).filter(
                BotCircuitBreaker.bot_id == bot_id
            ).first()

            if cb is None:
                return None

            cb.daily_trade_count = 0
            session.flush()
            session.refresh(cb)
            session.expunge(cb)
            return cb
