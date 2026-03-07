"""
Bot Repository.

Provides database operations for bot manifests and management.
"""

from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import BotManifest
from src.database.models.base import TradingMode


class BotRepository(BaseRepository[BotManifest]):
    """Repository for BotManifest database operations."""

    model = BotManifest

    def get_by_name(self, bot_name: str) -> Optional[BotManifest]:
        """Get a bot by name."""
        with self.get_session() as session:
            bot = session.query(BotManifest).filter(
                BotManifest.bot_name == bot_name
            ).first()
            if bot is not None:
                session.expunge(bot)
            return bot

    def get_by_status(self, status: str, limit: int = 100) -> List[BotManifest]:
        """Get bots by status."""
        return self.filter_by(limit=limit, status=status)

    def get_by_mode(self, trading_mode: TradingMode, limit: int = 100) -> List[BotManifest]:
        """Get bots by trading mode."""
        with self.get_session() as session:
            bots = session.query(BotManifest).filter(
                BotManifest.trading_mode == trading_mode
            ).order_by(BotManifest.created_at.desc()).limit(limit).all()
            for bot in bots:
                session.expunge(bot)
            return bots

    def get_active_bots(self, trading_mode: TradingMode = None) -> List[BotManifest]:
        """Get all active bots, optionally filtered by mode."""
        with self.get_session() as session:
            query = session.query(BotManifest).filter(
                BotManifest.status == 'active'
            )
            if trading_mode is not None:
                query = query.filter(BotManifest.trading_mode == trading_mode)

            bots = query.order_by(BotManifest.created_at.desc()).all()
            for bot in bots:
                session.expunge(bot)
            return bots

    def create(
        self,
        bot_name: str,
        bot_type: str,
        strategy_type: str,
        broker_type: str = 'STANDARD',
        trade_frequency: str = None,
        required_margin: float = 100.0,
        max_drawdown_pct: float = 10.0,
        min_win_rate: float = 0.55,
        target_sharpe: float = None,
        trading_mode: TradingMode = TradingMode.DEMO,
        metadata: dict = None
    ) -> BotManifest:
        """Create a new bot manifest."""
        return super().create(
            bot_name=bot_name,
            bot_type=bot_type,
            strategy_type=strategy_type,
            broker_type=broker_type,
            trade_frequency=trade_frequency,
            required_margin=required_margin,
            max_drawdown_pct=max_drawdown_pct,
            min_win_rate=min_win_rate,
            target_sharpe=target_sharpe,
            trading_mode=trading_mode,
            bot_metadata=metadata
        )

    def update_status(self, bot_id: int, status: str) -> Optional[BotManifest]:
        """Update bot status."""
        return self.update(bot_id, status=status)
