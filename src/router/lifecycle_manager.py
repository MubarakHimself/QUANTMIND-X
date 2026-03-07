"""
LifecycleManager: Automatic Tag Progression System

Handles automatic bot tag progression based on performance criteria:
- @primal → @pending → @perfect → @live (promotion)
- @live → @quarantine (underperformance)
- @quarantine → @dead (critical failure)

This is separate from PromotionManager which handles PAPER→DEMO→LIVE mode transitions.
LifecycleManager focuses on tag-based quality progression within the bot fleet.

Tag Progression Criteria (from spec):
@primal → @pending:
  - Min trades: 20
  - Min win rate: 50%
  - Min days active: 7
  - No critical errors

@pending → @perfect:
  - Min trades: 50
  - Min win rate: 55%
  - Min Sharpe ratio: 1.5
  - Min days active: 30
  - Max drawdown: < 15%

@perfect → @live:
  - Min trades: 100
  - Min win rate: 58%
  - Min Sharpe ratio: 2.0
  - Min days active: 60
  - Max drawdown: < 10%
  - Profit factor: > 1.5

@live → @quarantine (triggers):
  - Win rate drops below 45%
  - Sharpe ratio < 0.5
  - Drawdown > 20%
  - 5 consecutive losing days

@quarantine → @dead (triggers):
  - 30 days in quarantine
  - Win rate < 40%
  - Total loss > 50% of allocated capital
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import json
import asyncio

if TYPE_CHECKING:
    from src.router.bot_manifest import BotManifest, BotRegistry, ModePerformanceStats
    from src.router.promotion_manager import PerformanceTracker
    from src.database.db_manager import DBManager

logger = logging.getLogger(__name__)

# Prometheus metrics at module scope to prevent duplicate registration
# Uses registry check pattern to avoid duplicate metric exceptions on module reload
PROMETHEUS_AVAILABLE = False
lifecycle_promotions_counter = None
lifecycle_quarantines_counter = None
lifecycle_kills_counter = None
active_bots_by_tag_gauge = None

def _init_prometheus_metrics():
    """
    Initialize Prometheus metrics with registry check to prevent duplicates.
    
    This function checks the PROMETHEUS_REGISTRY before creating metrics,
    allowing safe module reload without duplicate metric exceptions.
    """
    global PROMETHEUS_AVAILABLE, lifecycle_promotions_counter, lifecycle_quarantines_counter
    global lifecycle_kills_counter, active_bots_by_tag_gauge
    
    if PROMETHEUS_AVAILABLE:
        return  # Already initialized
    
    try:
        from prometheus_client import Counter, Gauge, REGISTRY
        
        # Helper to get or create metric safely
        def _get_or_create_counter(name, documentation, registry=REGISTRY):
            """Get existing counter or create new one."""
            try:
                # Try to get existing metric from registry
                for collector in registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
            except Exception:
                pass
            # Create new counter if not found
            return Counter(name, documentation)
        
        def _get_or_create_gauge(name, documentation, labelnames=None, registry=REGISTRY):
            """Get existing gauge or create new one."""
            try:
                # Try to get existing metric from registry
                for collector in registry._names_to_collectors.values():
                    if hasattr(collector, '_name') and collector._name == name:
                        return collector
            except Exception:
                pass
            # Create new gauge if not found
            if labelnames:
                return Gauge(name, documentation, labelnames)
            return Gauge(name, documentation)
        
        # Lifecycle event counters
        lifecycle_promotions_counter = _get_or_create_counter(
            'quantmind_lifecycle_promotions_total',
            'Total number of bot promotions'
        )
        lifecycle_quarantines_counter = _get_or_create_counter(
            'quantmind_lifecycle_quarantines_total',
            'Total number of bot quarantines'
        )
        lifecycle_kills_counter = _get_or_create_counter(
            'quantmind_lifecycle_kills_total',
            'Total number of bot kills'
        )
        active_bots_by_tag_gauge = _get_or_create_gauge(
            'quantmind_active_bots_by_tag',
            'Number of active bots by tag',
            labelnames=['tag']
        )
        
        PROMETHEUS_AVAILABLE = True
        logger.debug("Prometheus metrics initialized successfully")
        
    except ImportError:
        logger.debug("prometheus_client not available, metrics disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize Prometheus metrics: {e}")

# Initialize metrics on module load
_init_prometheus_metrics()


# Tag progression hierarchy
TAG_HIERARCHY = ["@primal", "@pending", "@perfect", "@live", "@quarantine", "@dead"]

# Promotion criteria for each tag level
PROMOTION_CRITERIA = {
    "@primal": {
        # Entry level - no criteria to reach here
        "min_trades": 0,
        "min_win_rate": 0.0,
        "min_days_active": 0,
        "min_sharpe_ratio": 0.0,
        "max_drawdown": 1.0,  # 100%
        "min_profit_factor": 0.0,
    },
    "@pending": {
        "min_trades": 20,
        "min_win_rate": 0.50,
        "min_days_active": 7,
        "min_sharpe_ratio": 0.0,
        "max_drawdown": 1.0,  # 100%
        "min_profit_factor": 0.0,
        "no_critical_errors": True,
    },
    "@perfect": {
        "min_trades": 50,
        "min_win_rate": 0.55,
        "min_days_active": 30,
        "min_sharpe_ratio": 1.5,
        "max_drawdown": 0.15,  # 15%
        "min_profit_factor": 0.0,
    },
    "@live": {
        "min_trades": 100,
        "min_win_rate": 0.58,
        "min_days_active": 60,
        "min_sharpe_ratio": 2.0,
        "max_drawdown": 0.10,  # 10%
        "min_profit_factor": 1.5,
    },
}

# Quarantine triggers
QUARANTINE_TRIGGERS = {
    "min_win_rate": 0.45,  # Win rate drops below 45%
    "min_sharpe_ratio": 0.5,  # Sharpe ratio below 0.5
    "max_drawdown": 0.20,  # Drawdown exceeds 20%
    "max_consecutive_losing_days": 5,  # 5 consecutive losing days
}

# Dead bot triggers
DEAD_TRIGGERS = {
    "max_quarantine_days": 30,  # 30 days in quarantine
    "min_win_rate": 0.40,  # Win rate below 40%
    "max_capital_loss_pct": 0.50,  # Lost 50% of allocated capital
}

# Configurable thresholds by book type
# Personal Book: 5 losses, Prop Firm Book: 3 losses (tighter)
THRESHOLDS_BY_BOOK = {
    "personal": {
        "max_consecutive_losing_days": 5,
        "max_quarantine_days": 30,
    },
    "prop_firm": {
        "max_consecutive_losing_days": 3,  # Tighter for prop firms
        "max_quarantine_days": 14,          # Shorter quarantine for prop firms
    },
}


@dataclass
class LifecycleCheckResult:
    """Result of a lifecycle check for a single bot."""
    bot_id: str
    current_tag: str
    action: str  # 'promote', 'quarantine', 'kill', 'none'
    next_tag: Optional[str] = None
    missing_criteria: List[str] = field(default_factory=list)
    triggered_criteria: List[str] = field(default_factory=list)
    current_stats: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error: Optional[str] = None


@dataclass
class LifecycleReport:
    """Summary report of a daily lifecycle check."""
    timestamp: str
    total_bots_checked: int
    promotions: List[LifecycleCheckResult]
    quarantines: List[LifecycleCheckResult]
    kills: List[LifecycleCheckResult]
    no_change: List[LifecycleCheckResult]
    errors: List[LifecycleCheckResult]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "timestamp": self.timestamp,
            "total_bots_checked": self.total_bots_checked,
            "summary": {
                "promotions": len(self.promotions),
                "quarantines": len(self.quarantines),
                "kills": len(self.kills),
                "no_change": len(self.no_change),
                "errors": len(self.errors),
            },
            "promotions": [r.__dict__ for r in self.promotions],
            "quarantines": [r.__dict__ for r in self.quarantines],
            "kills": [r.__dict__ for r in self.kills],
            "no_change": [r.__dict__ for r in self.no_change[:10]],  # Limit for brevity
            "errors": [r.__dict__ for r in self.errors],
        }


class LifecycleManager:
    """
    Manages automatic bot tag progression based on performance.
    
    Responsibilities:
    - Daily performance evaluation for all bots
    - Automatic tag progression: @primal → @pending → @perfect → @live
    - Automatic quarantine for underperforming bots
    - Dead bot marking for irrecoverable bots
    - Prometheus metrics for lifecycle events
    - WebSocket notifications for UI updates
    
    Usage:
        manager = LifecycleManager()
        report = manager.run_daily_check()
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
        self._quarantine_tracker: Dict[str, datetime] = {}  # bot_id -> quarantine_start
        self._losing_streak_tracker: Dict[str, int] = {}  # bot_id -> consecutive losing days
        
        # Load persisted state on initialization
        self._load_persisted_state()
    
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
    
    def _load_persisted_state(self) -> None:
        """
        Load quarantine start times and losing streak state from database.
        
        This ensures restarts don't reset counters.
        """
        try:
            # Query bot_lifecycle_log for quarantine start times
            session = self.db_manager.session
            
            # Get latest quarantine events for each bot
            query = """
                SELECT bot_id, timestamp 
                FROM bot_lifecycle_log 
                WHERE to_tag = '@quarantine'
                ORDER BY timestamp DESC
            """
            try:
                result = session.execute(query)
                rows = result.fetchall()
                for row in rows:
                    bot_id = row[0]
                    timestamp_str = row[1]
                    if bot_id not in self._quarantine_tracker:
                        # Parse timestamp
                        if isinstance(timestamp_str, str):
                            ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        else:
                            ts = timestamp_str
                        self._quarantine_tracker[bot_id] = ts
                        logger.debug(f"Loaded quarantine start for {bot_id}: {ts}")
            except Exception as e:
                logger.debug(f"Could not query quarantine state: {e}")
            
            # Note: Losing streak is derived from trade history in _get_consecutive_losing_days
            # which is recalculated on each check, so no persistence needed for that.
            
            logger.info(f"Loaded persisted state: {len(self._quarantine_tracker)} quarantine trackers")
            
        except Exception as e:
            logger.warning(f"Failed to load persisted lifecycle state: {e}")
    
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
        
        Args:
            bot_id: Bot identifier
            from_tag: Source tag
            to_tag: Destination tag
            reason: Reason for transition
            performance_stats: Optional snapshot of performance at transition time
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
    
    def run_daily_check(self) -> LifecycleReport:
        """
        Run daily lifecycle evaluation for all bots.
        
        This is the main entry point for lifecycle management.
        Should be called once per day (e.g., via scheduler at 3:00 AM UTC).
        
        Returns:
            LifecycleReport with all check results
        """
        logger.info("=" * 60)
        logger.info("Starting daily lifecycle check")
        logger.info("=" * 60)
        
        timestamp = datetime.now(timezone.utc).isoformat()
        
        promotions = []
        quarantines = []
        kills = []
        no_change = []
        errors = []
        
        all_bots = self.bot_registry.list_all()
        total_checked = len(all_bots)
        
        logger.info(f"Checking {total_checked} bots")
        
        for bot in all_bots:
            try:
                result = self._check_bot_lifecycle(bot)
                
                if result.action == "promote":
                    promotions.append(result)
                    self._execute_promotion(bot, result.next_tag)
                elif result.action == "quarantine":
                    quarantines.append(result)
                    self._execute_quarantine(bot)
                elif result.action == "kill":
                    kills.append(result)
                    self._execute_kill(bot)
                elif result.action == "none":
                    no_change.append(result)
                else:
                    logger.warning(f"Unknown action '{result.action}' for bot {bot.bot_id}")
                    no_change.append(result)
                    
            except Exception as e:
                logger.error(f"Error checking lifecycle for bot {bot.bot_id}: {e}")
                errors.append(LifecycleCheckResult(
                    bot_id=bot.bot_id,
                    current_tag=self._get_primary_tag(bot),
                    action="error",
                    error=str(e),
                ))
        
        # Save registry after all changes
        self.bot_registry._save()
        
        # Emit metrics and notifications
        self._emit_metrics(promotions, quarantines, kills)
        self._send_notifications(promotions, quarantines, kills)
        
        report = LifecycleReport(
            timestamp=timestamp,
            total_bots_checked=total_checked,
            promotions=promotions,
            quarantines=quarantines,
            kills=kills,
            no_change=no_change,
            errors=errors,
        )
        
        logger.info("=" * 60)
        logger.info(f"Lifecycle check complete: {len(promotions)} promoted, "
                   f"{len(quarantines)} quarantined, {len(kills)} killed")
        logger.info("=" * 60)
        
        return report
    
    def _check_bot_lifecycle(self, bot: "BotManifest") -> LifecycleCheckResult:
        """
        Check lifecycle status for a single bot.
        
        Args:
            bot: Bot manifest to check
            
        Returns:
            LifecycleCheckResult with action recommendation
        """
        current_tag = self._get_primary_tag(bot)
        
        # Skip dead bots
        if current_tag == "@dead":
            return LifecycleCheckResult(
                bot_id=bot.bot_id,
                current_tag=current_tag,
                action="none",
            )
        
        # Get performance stats
        stats = self._get_bot_stats(bot)
        
        # Check for quarantine triggers first (higher priority for @live bots)
        if current_tag == "@live":
            triggered = self._check_quarantine_triggers(bot, stats)
            if triggered:
                return LifecycleCheckResult(
                    bot_id=bot.bot_id,
                    current_tag=current_tag,
                    action="quarantine",
                    next_tag="@quarantine",
                    triggered_criteria=triggered,
                    current_stats=stats,
                )
        
        # Check for dead triggers (for quarantined bots)
        if current_tag == "@quarantine":
            triggered = self._check_dead_triggers(bot, stats)
            if triggered:
                return LifecycleCheckResult(
                    bot_id=bot.bot_id,
                    current_tag=current_tag,
                    action="kill",
                    next_tag="@dead",
                    triggered_criteria=triggered,
                    current_stats=stats,
                )
        
        # Check for promotion eligibility
        next_tag = self._get_next_tag(current_tag)
        if next_tag and next_tag not in ["@quarantine", "@dead"]:
            missing = self._check_promotion_criteria(bot, stats, next_tag)
            if not missing:
                return LifecycleCheckResult(
                    bot_id=bot.bot_id,
                    current_tag=current_tag,
                    action="promote",
                    next_tag=next_tag,
                    current_stats=stats,
                )
            else:
                return LifecycleCheckResult(
                    bot_id=bot.bot_id,
                    current_tag=current_tag,
                    action="none",
                    missing_criteria=missing,
                    current_stats=stats,
                )
        
        return LifecycleCheckResult(
            bot_id=bot.bot_id,
            current_tag=current_tag,
            action="none",
            current_stats=stats,
        )
    
    def _get_primary_tag(self, bot: "BotManifest") -> str:
        """Get the primary lifecycle tag from bot."""
        for tag in TAG_HIERARCHY:
            if tag in bot.tags:
                return tag
        return "@primal"  # Default to entry level
    
    def _get_next_tag(self, current_tag: str) -> Optional[str]:
        """Get the next tag in the progression hierarchy."""
        try:
            idx = TAG_HIERARCHY.index(current_tag)
            if idx < len(TAG_HIERARCHY) - 1:
                return TAG_HIERARCHY[idx + 1]
        except ValueError:
            pass
        return None
    
    def _get_bot_stats(self, bot: "BotManifest") -> Dict[str, Any]:
        """
        Get performance stats for a bot.
        
        Combines stats from PerformanceTracker and ModePerformanceStats.
        """
        stats = {
            "total_trades": 0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "trading_days": 0,
            "total_pnl": 0.0,
        }
        
        # Get from ModePerformanceStats if available
        current_stats = bot.get_current_stats()
        if current_stats:
            stats.update({
                "total_trades": current_stats.total_trades,
                "win_rate": current_stats.win_rate,
                "sharpe_ratio": current_stats.sharpe_ratio,
                "max_drawdown": current_stats.max_drawdown,
                "profit_factor": current_stats.profit_factor,
                "trading_days": current_stats.trading_days,
                "total_pnl": current_stats.total_pnl,
            })
        
        # Calculate days active from mode_start_date
        if bot.mode_start_date:
            days_active = (datetime.now(timezone.utc) - bot.mode_start_date).days
            stats["days_active"] = days_active
        else:
            stats["days_active"] = 0
        
        return stats
    
    def _check_promotion_criteria(
        self,
        bot: "BotManifest",
        stats: Dict[str, Any],
        target_tag: str
    ) -> List[str]:
        """
        Check if bot meets promotion criteria for target tag.
        
        Returns:
            List of missing criteria (empty if all criteria met)
        """
        criteria = PROMOTION_CRITERIA.get(target_tag, {})
        missing = []
        
        if stats["total_trades"] < criteria.get("min_trades", 0):
            missing.append(
                f"trades: {stats['total_trades']}/{criteria['min_trades']}"
            )
        
        if stats["win_rate"] < criteria.get("min_win_rate", 0):
            missing.append(
                f"win_rate: {stats['win_rate']:.1%}/{criteria['min_win_rate']:.0%}"
            )
        
        if stats["days_active"] < criteria.get("min_days_active", 0):
            missing.append(
                f"days_active: {stats['days_active']}/{criteria['min_days_active']}"
            )
        
        if stats["sharpe_ratio"] < criteria.get("min_sharpe_ratio", 0):
            missing.append(
                f"sharpe: {stats['sharpe_ratio']:.2f}/{criteria['min_sharpe_ratio']}"
            )
        
        if stats["max_drawdown"] > criteria.get("max_drawdown", 1.0):
            missing.append(
                f"drawdown: {stats['max_drawdown']:.1%}>{criteria['max_drawdown']:.0%}"
            )
        
        if stats["profit_factor"] < criteria.get("min_profit_factor", 0):
            missing.append(
                f"profit_factor: {stats['profit_factor']:.2f}/{criteria['min_profit_factor']}"
            )
        
        return missing
    
    def _check_quarantine_triggers(
        self,
        bot: "BotManifest",
        stats: Dict[str, Any]
    ) -> List[str]:
        """
        Check if bot should be quarantined.

        Returns:
            List of triggered quarantine conditions
        """
        triggered = []

        if stats["win_rate"] < QUARANTINE_TRIGGERS["min_win_rate"]:
            triggered.append(
                f"win_rate_below_{QUARANTINE_TRIGGERS['min_win_rate']:.0%}"
            )

        if stats["sharpe_ratio"] < QUARANTINE_TRIGGERS["min_sharpe_ratio"]:
            triggered.append(
                f"sharpe_below_{QUARANTINE_TRIGGERS['min_sharpe_ratio']}"
            )

        if stats["max_drawdown"] > QUARANTINE_TRIGGERS["max_drawdown"]:
            triggered.append(
                f"drawdown_above_{QUARANTINE_TRIGGERS['max_drawdown']:.0%}"
            )

        # Check consecutive losing days - use configurable threshold by book type
        losing_days = self._get_consecutive_losing_days(bot.bot_id)
        # Get account book type from bot manifest
        account_book = getattr(bot, 'account_book', 'personal')
        book_thresholds = THRESHOLDS_BY_BOOK.get(account_book, THRESHOLDS_BY_BOOK["personal"])
        max_losing_days = book_thresholds["max_consecutive_losing_days"]

        if losing_days >= max_losing_days:
            triggered.append(
                f"{losing_days}_consecutive_losing_days (threshold: {max_losing_days})"
            )
            triggered.append(
                f"{losing_days}_consecutive_losing_days"
            )
        
        return triggered
    
    def _check_dead_triggers(
        self,
        bot: "BotManifest",
        stats: Dict[str, Any]
    ) -> List[str]:
        """
        Check if bot should be marked as dead.

        Returns:
            List of triggered dead conditions
        """
        triggered = []

        # Check quarantine duration - use configurable threshold by book type
        quarantine_start = self._quarantine_tracker.get(bot.bot_id)
        if quarantine_start:
            account_book = getattr(bot, 'account_book', 'personal')
            book_thresholds = THRESHOLDS_BY_BOOK.get(account_book, THRESHOLDS_BY_BOOK["personal"])
            max_quarantine_days = book_thresholds["max_quarantine_days"]

            days_in_quarantine = (
                datetime.now(timezone.utc) - quarantine_start
            ).days
            if days_in_quarantine >= max_quarantine_days:
                triggered.append(
                    f"{days_in_quarantine}_days_in_quarantine (threshold: {max_quarantine_days})"
                )

        if stats["win_rate"] < DEAD_TRIGGERS["min_win_rate"]:
            triggered.append(
                f"win_rate_below_{DEAD_TRIGGERS['min_win_rate']:.0%}"
            )
        
        # Check capital loss - only consider actual losses, not profits
        if bot.capital_allocated > 0 and stats["total_pnl"] < 0:
            capital_loss_pct = -stats["total_pnl"] / bot.capital_allocated
            if capital_loss_pct > DEAD_TRIGGERS["max_capital_loss_pct"]:
                triggered.append(
                    f"capital_loss_{capital_loss_pct:.0%}"
                )
        
        return triggered
    
    def _get_consecutive_losing_days(self, bot_id: str) -> int:
        """Get consecutive losing days for a bot from trade history."""
        try:
            trades = self.performance_tracker.get_trade_history(bot_id)
            if not trades:
                return 0
            
            # Group trades by date and calculate daily P&L
            daily_pnl = {}
            for trade in trades:
                ts = trade.get("timestamp")
                if ts:
                    date_str = ts[:10]  # YYYY-MM-DD
                    daily_pnl[date_str] = daily_pnl.get(date_str, 0) + trade.get("pnl", 0)
            
            # Count consecutive losing days from most recent
            consecutive = 0
            sorted_dates = sorted(daily_pnl.keys(), reverse=True)
            for date in sorted_dates:
                if daily_pnl[date] < 0:
                    consecutive += 1
                else:
                    break
            
            return consecutive
            
        except Exception as e:
            logger.warning(f"Failed to get losing streak for {bot_id}: {e}")
            return 0
    
    def _execute_promotion(self, bot: "BotManifest", new_tag: str) -> None:
        """Execute bot promotion to new tag."""
        old_tag = self._get_primary_tag(bot)
        
        # Remove old lifecycle tags and add new one
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append(new_tag)
        
        # Record lifecycle event to database
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=old_tag,
            to_tag=new_tag,
            reason="Promotion criteria met",
            performance_stats=stats
        )
        
        logger.info(f"PROMOTED: {bot.bot_id} from {old_tag} to {new_tag}")
    
    def _execute_quarantine(self, bot: "BotManifest") -> None:
        """Execute bot quarantine."""
        old_tag = self._get_primary_tag(bot)
        
        # Remove old lifecycle tags and add quarantine
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append("@quarantine")
        
        # Track quarantine start time (already persisted in DB via _record_lifecycle_event)
        self._quarantine_tracker[bot.bot_id] = datetime.now(timezone.utc)
        
        # Downgrade trading mode to PAPER directly (avoid bot.downgrade() which may crash)
        # Set trading_mode directly instead of calling downgrade method
        from src.router.bot_manifest import TradingMode
        if bot.trading_mode == TradingMode.LIVE or bot.trading_mode == TradingMode.DEMO:
            bot.trading_mode = TradingMode.PAPER
            bot.mode_start_date = datetime.now()
            logger.info(f"Bot {bot.bot_id} trading mode downgraded to PAPER due to quarantine")
        
        # Record lifecycle event to database
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=old_tag,
            to_tag="@quarantine",
            reason="Quarantine trigger activated",
            performance_stats=stats
        )
        
        logger.warning(f"QUARANTINED: {bot.bot_id} (was {old_tag})")
    
    def _execute_kill(self, bot: "BotManifest") -> None:
        """Execute bot kill (mark as dead)."""
        old_tag = self._get_primary_tag(bot)
        
        # Remove old lifecycle tags and add dead
        bot.tags = [t for t in bot.tags if t not in TAG_HIERARCHY]
        bot.tags.append("@dead")
        
        # Set trading mode to paper (safest)
        from src.router.bot_manifest import TradingMode
        bot.trading_mode = TradingMode.PAPER
        
        # Record lifecycle event to database
        stats = self._get_bot_stats(bot)
        self._record_lifecycle_event(
            bot_id=bot.bot_id,
            from_tag=old_tag,
            to_tag="@dead",
            reason="Dead trigger activated - bot terminated",
            performance_stats=stats
        )
        
        # Clean up quarantine tracker
        self._quarantine_tracker.pop(bot.bot_id, None)
        
        logger.error(f"KILLED: {bot.bot_id} (was {old_tag})")
    
    def _emit_metrics(
        self,
        promotions: List[LifecycleCheckResult],
        quarantines: List[LifecycleCheckResult],
        kills: List[LifecycleCheckResult]
    ) -> None:
        """Emit Prometheus metrics for lifecycle events."""
        if not PROMETHEUS_AVAILABLE:
            logger.debug("Prometheus client not available, skipping metrics")
            return
        
        try:
            # Increment counters using module-level metrics
            for _ in promotions:
                lifecycle_promotions_counter.inc()
            for _ in quarantines:
                lifecycle_quarantines_counter.inc()
            for _ in kills:
                lifecycle_kills_counter.inc()
            
            # Update gauges
            for tag in TAG_HIERARCHY:
                count = len(self.bot_registry.list_by_tag(tag))
                active_bots_by_tag_gauge.labels(tag=tag).set(count)
        except Exception as e:
            logger.warning(f"Failed to emit metrics: {e}")
    
    def _send_notifications(
        self,
        promotions: List[LifecycleCheckResult],
        quarantines: List[LifecycleCheckResult],
        kills: List[LifecycleCheckResult]
    ) -> None:
        """Send WebSocket notifications for lifecycle events."""
        notifications = []
        
        # Build notification list
        for result in promotions:
            notifications.append({
                "type": "lifecycle_promotion",
                "bot_id": result.bot_id,
                "old_tag": result.current_tag,
                "new_tag": result.next_tag,
                "timestamp": result.timestamp,
            })
        
        for result in quarantines:
            notifications.append({
                "type": "lifecycle_quarantine",
                "bot_id": result.bot_id,
                "old_tag": result.current_tag,
                "triggered_criteria": result.triggered_criteria,
                "timestamp": result.timestamp,
            })
        
        for result in kills:
            notifications.append({
                "type": "lifecycle_kill",
                "bot_id": result.bot_id,
                "old_tag": result.current_tag,
                "triggered_criteria": result.triggered_criteria,
                "timestamp": result.timestamp,
            })
        
        # Send via WebSocket broadcast
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                for notif in notifications:
                    event_type = notif["type"].replace("lifecycle_", "")
                    
                    # Map to proper event type for broadcast
                    if event_type == "promotion":
                        from_tag = notif.get("old_tag", "")
                        to_tag = notif.get("new_tag", "")
                    elif event_type == "quarantine":
                        from_tag = notif.get("old_tag", "")
                        to_tag = "@quarantine"
                    else:  # kill
                        from_tag = notif.get("old_tag", "")
                        to_tag = "@dead"
                    
                    reason = " or ".join(notif.get("triggered_criteria", [])) if notif.get("triggered_criteria") else "Automatic lifecycle transition"
                    
                    # Use the broadcast function from websocket_endpoints
                    from src.api.websocket_endpoints import broadcast_lifecycle_event
                    loop.run_until_complete(
                        broadcast_lifecycle_event(
                            event_type=event_type,
                            bot_id=notif["bot_id"],
                            from_tag=from_tag,
                            to_tag=to_tag,
                            reason=reason,
                            timestamp=notif["timestamp"]
                        )
                    )
                    logger.info(f"Broadcast lifecycle event: {notif}")
            finally:
                loop.close()
        except Exception as e:
            # Fallback to logging if WebSocket broadcast fails
            logger.warning(f"WebSocket broadcast failed, falling back to log: {e}")
            for notif in notifications:
                logger.info(f"NOTIFICATION: {json.dumps(notif)}")
    
    def get_bot_lifecycle_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get detailed lifecycle status for a specific bot.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Dict with lifecycle status, criteria, and stats
        """
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return {
                "bot_id": bot_id,
                "error": "Bot not found",
            }
        
        current_tag = self._get_primary_tag(bot)
        stats = self._get_bot_stats(bot)
        next_tag = self._get_next_tag(current_tag)
        
        # Get promotion criteria for next level
        promotion_criteria = {}
        missing_criteria = []
        if next_tag and next_tag not in ["@quarantine", "@dead"]:
            promotion_criteria = PROMOTION_CRITERIA.get(next_tag, {})
            missing_criteria = self._check_promotion_criteria(bot, stats, next_tag)
        
        # Get quarantine triggers if @live
        quarantine_triggers = {}
        if current_tag == "@live":
            quarantine_triggers = QUARANTINE_TRIGGERS.copy()
        
        return {
            "bot_id": bot_id,
            "current_tag": current_tag,
            "next_tag": next_tag,
            "stats": stats,
            "promotion_criteria": promotion_criteria,
            "missing_criteria": missing_criteria,
            "quarantine_triggers": quarantine_triggers,
            "is_promotion_eligible": len(missing_criteria) == 0,
            "trading_mode": bot.trading_mode.value,
        }
    
    def get_all_lifecycle_statuses(self) -> List[Dict[str, Any]]:
        """
        Get lifecycle status for all bots.
        
        Returns:
            List of lifecycle status dicts
        """
        statuses = []
        for bot in self.bot_registry.list_all():
            status = self.get_bot_lifecycle_status(bot.bot_id)
            statuses.append(status)
        return statuses
    
    def manually_promote_bot(self, bot_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Manually promote a bot to the next tag level.
        
        Args:
            bot_id: Bot identifier
            force: If True, bypass criteria check
            
        Returns:
            Dict with promotion result
        """
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return {
                "success": False,
                "error": f"Bot {bot_id} not found",
            }
        
        current_tag = self._get_primary_tag(bot)
        next_tag = self._get_next_tag(current_tag)
        
        if not next_tag or next_tag in ["@quarantine", "@dead"]:
            return {
                "success": False,
                "error": f"Cannot promote from {current_tag}",
            }
        
        if not force:
            stats = self._get_bot_stats(bot)
            missing = self._check_promotion_criteria(bot, stats, next_tag)
            if missing:
                return {
                    "success": False,
                    "error": "Bot does not meet promotion criteria",
                    "missing_criteria": missing,
                }
        
        self._execute_promotion(bot, next_tag)
        self.bot_registry._save()
        
        return {
            "success": True,
            "bot_id": bot_id,
            "old_tag": current_tag,
            "new_tag": next_tag,
        }
    
    def manually_quarantine_bot(self, bot_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Manually quarantine a bot.
        
        Args:
            bot_id: Bot identifier
            reason: Reason for quarantine
            
        Returns:
            Dict with quarantine result
        """
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return {
                "success": False,
                "error": f"Bot {bot_id} not found",
            }
        
        current_tag = self._get_primary_tag(bot)
        
        if current_tag == "@dead":
            return {
                "success": False,
                "error": "Cannot quarantine a dead bot",
            }
        
        self._execute_quarantine(bot)
        self.bot_registry._save()
        
        return {
            "success": True,
            "bot_id": bot_id,
            "old_tag": current_tag,
            "new_tag": "@quarantine",
            "reason": reason,
        }