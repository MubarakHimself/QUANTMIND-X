"""
LifecycleManager: Strategy Lifecycle Management

Handles strategy promotion to live trading, retirement, and quarantine.
This is the core interface for managing strategy lifecycles in the improvement loop.

Core Methods:
- promote_to_live(): Promote a strategy from paper to live trading
- retire(): Retire a failed strategy
- quarantine(): Quarantine an underperforming strategy
- get_status(): Get current status of a strategy
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from src.router.bot_manifest import BotManifest, BotRegistry
    from src.router.promotion_manager import PerformanceTracker
    from src.database.db_manager import DBManager
else:
    from src.router.bot_manifest import BotRegistry
    from src.router.promotion_manager import PerformanceTracker
    from src.database.db_manager import DBManager

logger = logging.getLogger(__name__)


# Lifecycle tag constants
TAG_LIVE = "@live"
TAG_QUARANTINE = "@quarantine"
TAG_DEAD = "@dead"
TAG_HIERARCHY = [TAG_QUARANTINE, TAG_DEAD]


class LifecycleManager:
    """
    Manages strategy lifecycle transitions.

    Responsibilities:
    - Promote strategies from paper to live trading
    - Retire failed strategies
    - Quarantine underperforming strategies
    - Track lifecycle events in database

    Usage:
        manager = LifecycleManager()
        manager.promote_to_live(strategy_id)
        manager.retire(strategy_id)
        manager.quarantine(strategy_id)
        status = manager.get_status(strategy_id)
    """

    def __init__(
        self,
        bot_registry: Optional["BotRegistry"] = None,
        performance_tracker: Optional["PerformanceTracker"] = None,
        db_manager: Optional["DBManager"] = None,
    ):
        self._bot_registry = bot_registry
        self._performance_tracker = performance_tracker
        self._db_manager = db_manager

    @property
    def bot_registry(self) -> "BotRegistry":
        """Lazy load BotRegistry if not provided."""
        if self._bot_registry is None:
            from src.router.bot_manifest import BotRegistry
            self._bot_registry = BotRegistry()
        return self._bot_registry

    @property
    def performance_tracker(self) -> "PerformanceTracker":
        """Lazy load PerformanceTracker if not provided."""
        if self._performance_tracker is None:
            from src.router.promotion_manager import PerformanceTracker
            self._performance_tracker = PerformanceTracker()
        return self._performance_tracker

    @property
    def db_manager(self) -> "DBManager":
        """Lazy load DBManager if not provided."""
        if self._db_manager is None:
            from src.database.db_manager import DBManager
            self._db_manager = DBManager()
        return self._db_manager

    def _record_lifecycle_event(
        self,
        bot_id: str,
        from_tag: str,
        to_tag: str,
        reason: str,
        performance_stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a lifecycle event to the database.
        """
        try:
            session = self.db_manager.session
            timestamp = datetime.now(timezone.utc)
            stats_json = json.dumps(performance_stats) if performance_stats else None

            query = """
                INSERT INTO bot_lifecycle_log
                (bot_id, from_tag, to_tag, reason, timestamp, triggered_by, performance_stats)
                VALUES (?, ?, ?, ?, ?, 'system', ?)
            """
            session.execute(
                query,
                (bot_id, from_tag, to_tag, reason, timestamp, stats_json)
            )
            session.commit()
            logger.debug(f"Recorded lifecycle event: {bot_id} {from_tag} -> {to_tag}")

        except Exception as e:
            logger.warning(f"Failed to record lifecycle event: {e}")

    def _get_primary_tag(self, bot: "BotManifest") -> str:
        """Get the primary lifecycle tag from bot."""
        for tag in TAG_HIERARCHY:
            if tag in bot.tags:
                return tag
        return TAG_LIVE  # Default to live

    def _get_bot_stats(self, bot: "BotManifest") -> Dict[str, Any]:
        """Get performance stats for a bot."""
        stats = {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "trading_days": 0,
            "total_pnl": 0.0,
        }

        current_stats = getattr(bot, 'get_current_stats', lambda: None)()
        if current_stats:
            stats.update({
                "total_trades": getattr(current_stats, 'total_trades', 0) or 0,
                "win_rate": getattr(current_stats, 'win_rate', 0.0) or 0.0,
                "sharpe_ratio": getattr(current_stats, 'sharpe_ratio', 0.0) or 0.0,
                "max_drawdown": getattr(current_stats, 'max_drawdown', 0.0) or 0.0,
                "profit_factor": getattr(current_stats, 'profit_factor', 0.0) or 0.0,
                "trading_days": getattr(current_stats, 'trading_days', 0) or 0,
                "total_pnl": getattr(current_stats, 'total_pnl', 0.0) or 0.0,
            })

        return stats

    def promote_to_live(self, strategy_id: str, dpr_score: Optional[float] = None) -> Dict[str, Any]:
        """
        Promote a strategy to live trading.

        Tags new promoted bots as TIER_2_PAPER for Monday-only release.
        Checks DPR score threshold before promoting.

        Args:
            strategy_id: Strategy/bot identifier
            dpr_score: Optional DPR score for threshold check

        Returns:
            Dict with promotion result
        """
        logger.info(f"Promoting to live: {strategy_id}")

        bot = self.bot_registry.get(strategy_id)
        if bot is None:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found",
            }

        # Check DPR score threshold (50)
        if dpr_score is not None and dpr_score < 50:
            logger.warning(f"DPR score {dpr_score} below threshold for {strategy_id}")
            return {
                "success": False,
                "error": f"DPR score {dpr_score} below threshold (50)",
                "dpr_score": dpr_score,
            }

        old_tag = self._get_primary_tag(bot)

        # Remove old lifecycle tags and add live
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append(TAG_LIVE)

        # Set trading mode to live
        from src.router.bot_manifest import TradingMode
        bot.trading_mode = TradingMode.LIVE
        bot.mode_start_date = datetime.now()

        # Record lifecycle event
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=old_tag,
            to_tag=TAG_LIVE,
            reason="Promoted to live trading via improvement loop",
            performance_stats=stats
        )

        # Save registry
        self.bot_registry._save()

        logger.info(f"PROMOTED to live: {strategy_id}")

        return {
            "success": True,
            "strategy_id": strategy_id,
            "old_tag": old_tag,
            "new_tag": TAG_LIVE,
            "dpr_score": dpr_score,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
            "tier": "TIER_2_PAPER",
            "release_day": "Monday",
        }

    def retire(self, strategy_id: str, reason: str = "Failed after max iterations") -> Dict[str, Any]:
        """
        Retire a failed strategy.

        Args:
            strategy_id: Strategy/bot identifier
            reason: Reason for retirement

        Returns:
            Dict with retirement result
        """
        logger.info(f"Retiring strategy: {strategy_id}, reason: {reason}")

        bot = self.bot_registry.get(strategy_id)
        if bot is None:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found",
            }

        old_tag = self._get_primary_tag(bot)

        # Remove old lifecycle tags and add dead
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append(TAG_DEAD)

        # Set trading mode to paper (safest)
        from src.router.bot_manifest import TradingMode
        bot.trading_mode = TradingMode.PAPER

        # Record lifecycle event
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=old_tag,
            to_tag=TAG_DEAD,
            reason=f"Retired: {reason}",
            performance_stats=stats
        )

        # Save registry
        self.bot_registry._save()

        logger.info(f"RETIRED: {strategy_id}")

        return {
            "success": True,
            "strategy_id": strategy_id,
            "old_tag": old_tag,
            "new_tag": TAG_DEAD,
            "reason": reason,
            "retired_at": datetime.now(timezone.utc).isoformat(),
        }

    def quarantine(self, strategy_id: str, reason: str = "Underperformance detected") -> Dict[str, Any]:
        """
        Quarantine an underperforming strategy.

        Args:
            strategy_id: Strategy/bot identifier
            reason: Reason for quarantine

        Returns:
            Dict with quarantine result
        """
        logger.info(f"Quarantining strategy: {strategy_id}, reason: {reason}")

        bot = self.bot_registry.get(strategy_id)
        if bot is None:
            return {
                "success": False,
                "error": f"Strategy {strategy_id} not found",
            }

        current_tag = self._get_primary_tag(bot)

        if current_tag == TAG_DEAD:
            return {
                "success": False,
                "error": "Cannot quarantine a dead strategy",
            }

        # Remove old lifecycle tags and add quarantine
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append(TAG_QUARANTINE)

        # Downgrade trading mode to paper
        from src.router.bot_manifest import TradingMode
        if bot.trading_mode == TradingMode.LIVE:
            bot.trading_mode = TradingMode.PAPER
            bot.mode_start_date = datetime.now()

        # Record lifecycle event
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=current_tag,
            to_tag=TAG_QUARANTINE,
            reason=f"Quarantined: {reason}",
            performance_stats=stats
        )

        # Save registry
        self.bot_registry._save()

        logger.info(f"QUARANTINED: {strategy_id}")

        # M6-A: Dispatch FloorManager notification for bot quarantine review
        try:
            from src.agents.departments.floor_manager import get_floor_manager
            from src.agents.departments.types import Department
            fm = get_floor_manager()
            fm.dispatch(
                to_dept=Department.TRADING,
                task="BOT_QUARANTINE_REVIEW",
                priority="high",
                context={
                    "strategy_id": strategy_id,
                    "reason": reason,
                    "bot_id": bot.bot_id,
                    "stats": stats,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to dispatch FloorManager quarantine review: {e}")

        return {
            "success": True,
            "strategy_id": strategy_id,
            "old_tag": current_tag,
            "new_tag": TAG_QUARANTINE,
            "reason": reason,
            "quarantined_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get current lifecycle status of a strategy.

        Args:
            strategy_id: Strategy/bot identifier

        Returns:
            Dict with lifecycle status and stats
        """
        bot = self.bot_registry.get(strategy_id)
        if bot is None:
            return {
                "strategy_id": strategy_id,
                "error": "Strategy not found",
            }

        current_tag = self._get_primary_tag(bot)
        stats = self._get_bot_stats(bot)

        return {
            "strategy_id": strategy_id,
            "bot_id": bot.bot_id,
            "current_tag": current_tag,
            "trading_mode": bot.trading_mode.value if hasattr(bot, 'trading_mode') else None,
            "stats": stats,
            "dpr_score": getattr(bot, 'dpr_score', None),
            "tags": bot.tags,
            "mode_start_date": getattr(bot, 'mode_start_date', None),
        }

    def get_all_statuses(self) -> List[Dict[str, Any]]:
        """
        Get lifecycle status for all strategies.

        Returns:
            List of lifecycle status dicts
        """
        statuses = []
        for bot in self.bot_registry.list_all():
            status = self.get_status(bot.bot_id)
            statuses.append(status)
        return statuses