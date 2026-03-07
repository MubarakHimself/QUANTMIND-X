"""
Monitoring Repository.

Provides database operations for monitoring models (SymbolSubscription, TickCache, AlertHistory, WebhookLog).
"""

from typing import Optional, List
from datetime import datetime
from src.database.repositories.base_repository import BaseRepository
from src.database.models import SymbolSubscription, TickCache, AlertHistory, WebhookLog


class SymbolSubscriptionRepository(BaseRepository[SymbolSubscription]):
    """Repository for SymbolSubscription database operations."""

    model = SymbolSubscription

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[SymbolSubscription]:
        """Get subscriptions by symbol."""
        with self.get_session() as session:
            subs = session.query(SymbolSubscription).filter(
                SymbolSubscription.symbol == symbol
            ).order_by(SymbolSubscription.priority.asc()).limit(limit).all()
            for sub in subs:
                session.expunge(sub)
            return subs

    def get_by_bot(self, bot_id: str, limit: int = 100) -> List[SymbolSubscription]:
        """Get subscriptions by bot ID."""
        with self.get_session() as session:
            subs = session.query(SymbolSubscription).filter(
                SymbolSubscription.bot_id == bot_id
            ).order_by(SymbolSubscription.subscribed_at.desc()).limit(limit).all()
            for sub in subs:
                session.expunge(sub)
            return subs

    def get_by_timeframe(self, timeframe: str, limit: int = 100) -> List[SymbolSubscription]:
        """Get subscriptions by timeframe."""
        return self.filter_by(limit=limit, timeframe=timeframe)

    def get_by_priority(self, priority: int, limit: int = 100) -> List[SymbolSubscription]:
        """Get subscriptions by priority level."""
        return self.filter_by(limit=limit, priority=priority)

    def delete_by_symbol_and_bot(self, symbol: str, bot_id: str) -> bool:
        """Delete a subscription by symbol and bot ID."""
        with self.get_session() as session:
            sub = session.query(SymbolSubscription).filter(
                SymbolSubscription.symbol == symbol,
                SymbolSubscription.bot_id == bot_id
            ).first()
            if sub is not None:
                session.delete(sub)
                return True
            return False


class TickCacheRepository(BaseRepository[TickCache]):
    """Repository for TickCache database operations."""

    model = TickCache

    def get_by_symbol(self, symbol: str, limit: int = 1000) -> List[TickCache]:
        """Get ticks by symbol."""
        with self.get_session() as session:
            ticks = session.query(TickCache).filter(
                TickCache.symbol == symbol
            ).order_by(TickCache.timestamp.desc()).limit(limit).all()
            for tick in ticks:
                session.expunge(tick)
            return ticks

    def get_by_date_range(
        self, symbol: str, start: datetime, end: datetime, limit: int = 1000
    ) -> List[TickCache]:
        """Get ticks for a symbol within a date range."""
        with self.get_session() as session:
            ticks = session.query(TickCache).filter(
                TickCache.symbol == symbol,
                TickCache.timestamp >= start,
                TickCache.timestamp <= end
            ).order_by(TickCache.timestamp.desc()).limit(limit).all()
            for tick in ticks:
                session.expunge(tick)
            return ticks

    def delete_old_ticks(self, before: datetime) -> int:
        """Delete ticks older than the specified datetime."""
        with self.get_session() as session:
            deleted = session.query(TickCache).filter(
                TickCache.timestamp < before
            ).delete()
            return deleted


class AlertHistoryRepository(BaseRepository[AlertHistory]):
    """Repository for AlertHistory database operations."""

    model = AlertHistory

    def get_by_level(self, level: str, limit: int = 100) -> List[AlertHistory]:
        """Get alerts by level."""
        return self.filter_by(limit=limit, level=level)

    def get_by_tier(self, tier: int, limit: int = 100) -> List[AlertHistory]:
        """Get alerts by tier."""
        return self.filter_by(limit=limit, tier=tier)

    def get_by_source(self, source: str, limit: int = 100) -> List[AlertHistory]:
        """Get alerts by source."""
        return self.filter_by(limit=limit, source=source)

    def get_active_alerts(self, limit: int = 100) -> List[AlertHistory]:
        """Get all active alerts."""
        with self.get_session() as session:
            alerts = session.query(AlertHistory).filter(
                AlertHistory.is_active == True
            ).order_by(AlertHistory.triggered_at.desc()).limit(limit).all()
            for alert in alerts:
                session.expunge(alert)
            return alerts

    def get_cleared_alerts(self, limit: int = 100) -> List[AlertHistory]:
        """Get all cleared alerts."""
        with self.get_session() as session:
            alerts = session.query(AlertHistory).filter(
                AlertHistory.is_active == False
            ).order_by(AlertHistory.cleared_at.desc()).limit(limit).all()
            for alert in alerts:
                session.expunge(alert)
            return alerts


class WebhookLogRepository(BaseRepository[WebhookLog]):
    """Repository for WebhookLog database operations."""

    model = WebhookLog

    def get_by_source_ip(self, source_ip: str, limit: int = 100) -> List[WebhookLog]:
        """Get webhook logs by source IP."""
        return self.filter_by(limit=limit, source_ip=source_ip)

    def get_triggered_webhooks(self, limit: int = 100) -> List[WebhookLog]:
        """Get webhooks that triggered a bot."""
        with self.get_session() as session:
            logs = session.query(WebhookLog).filter(
                WebhookLog.bot_triggered == True
            ).order_by(WebhookLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs

    def get_failed_webhooks(self, limit: int = 100) -> List[WebhookLog]:
        """Get webhooks that failed to process."""
        with self.get_session() as session:
            logs = session.query(WebhookLog).filter(
                WebhookLog.error_message.isnot(None)
            ).order_by(WebhookLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs

    def get_by_date_range(
        self, start: datetime, end: datetime, limit: int = 1000
    ) -> List[WebhookLog]:
        """Get webhook logs within a date range."""
        with self.get_session() as session:
            logs = session.query(WebhookLog).filter(
                WebhookLog.timestamp >= start,
                WebhookLog.timestamp <= end
            ).order_by(WebhookLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs
