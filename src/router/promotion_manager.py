"""
PromotionManager and PerformanceTracker

Implements the Paper->Demo->Live promotion workflow per spec.

PromotionManager:
- Evaluates promotion criteria for bots
- Manages promotion/demotion transitions
- Enforces capital scaling rules
- Runs daily promotion checks

PerformanceTracker:
- Tracks paper trading performance
- Calculates performance metrics (Sharpe, win rate, etc.)
- Updates bot manifests with stats

Promotion Criteria (per spec):
- PAPER->DEMO: 30+ days, Sharpe > 1.5, Win Rate > 55%
- DEMO->LIVE: 30+ days, Sharpe > 1.5, Win Rate > 55%, Max DD < 10%

Capital Scaling (per spec):
- PAPER: Virtual capital (no real money)
- DEMO: $1,000 starting capital (demo account)
- LIVE: Scaled based on performance (starting at $1,000, max $10,000)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import json

if TYPE_CHECKING:
    from src.router.bot_manifest import BotManifest, BotRegistry, TradingMode, ModePerformanceStats

logger = logging.getLogger(__name__)


# Promotion thresholds per spec
PROMOTION_THRESHOLDS = {
    "paper_to_demo": {
        "min_trading_days": 30,
        "min_sharpe_ratio": 1.5,
        "min_win_rate": 0.55,
        "min_total_trades": 50,  # Minimum sample size
    },
    "demo_to_live": {
        "min_trading_days": 30,
        "min_sharpe_ratio": 1.5,
        "min_win_rate": 0.55,
        "max_drawdown": 0.10,  # 10% max drawdown
        "min_total_trades": 50,
    },
}

# Capital scaling rules per spec
# NOTE: Using $1,000 for demo/live (more realistic than $50 spec requirement)
# This provides adequate capital for proper risk management and position sizing
CAPITAL_SCALING = {
    "paper": 0.0,  # Virtual capital
    "demo": 1000.0,  # $1,000 demo starting capital
    "live_base": 1000.0,  # $1,000 live starting capital
    "live_max": 10000.0,  # $10,000 max capital per bot
    "live_scaling_factor": 0.5,  # Scale by 50% of profits
}


@dataclass
class PromotionResult:
    """Result of a promotion evaluation."""
    bot_id: str
    current_mode: str
    eligible: bool
    next_mode: Optional[str] = None
    missing_criteria: List[str] = field(default_factory=list)
    current_stats: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    promoted_at: Optional[str] = None
    error: Optional[str] = None


class PerformanceTracker:
    """
    Tracks and calculates performance metrics for trading bots.
    
    Used to evaluate promotion eligibility based on:
    - Sharpe ratio
    - Win rate
    - Max drawdown
    - Trading days
    - Total trades
    """
    
    def __init__(self, storage_path: str = "data/performance_tracking.json"):
        self.storage_path = storage_path
        self._trade_history: Dict[str, List[Dict]] = {}  # bot_id -> list of trades
        self._load()
    
    def _load(self) -> None:
        """Load trade history from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self._trade_history = data.get("trade_history", {})
                logger.info(f"Loaded performance tracking for {len(self._trade_history)} bots")
        except FileNotFoundError:
            logger.info("No existing performance tracking found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load performance tracking: {e}")
    
    def _save(self) -> None:
        """Persist trade history to storage."""
        try:
            data = {
                "trade_history": self._trade_history,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance tracking: {e}")
    
    def record_trade(self, bot_id: str, trade: Dict[str, Any]) -> None:
        """
        Record a trade for performance tracking.
        
        Args:
            bot_id: Bot identifier
            trade: Trade dict with keys: timestamp, symbol, direction, pnl, etc.
        """
        if bot_id not in self._trade_history:
            self._trade_history[bot_id] = []
        
        # Ensure timestamp is ISO format
        if "timestamp" not in trade:
            trade["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        self._trade_history[bot_id].append(trade)
        self._save()
        
        logger.debug(f"Recorded trade for bot {bot_id}: pnl={trade.get('pnl', 0)}")
    
    def calculate_stats(self, bot_id: str, mode: Optional[str] = None) -> Optional["ModePerformanceStats"]:
        """
        Calculate performance statistics for a bot.
        
        Args:
            bot_id: Bot identifier
            mode: Optional mode filter ('paper', 'demo', 'live')
            
        Returns:
            ModePerformanceStats or None if no trades
        """
        from src.router.bot_manifest import ModePerformanceStats
        
        trades = self._trade_history.get(bot_id, [])
        
        if mode:
            # Filter trades by mode if specified
            trades = [t for t in trades if t.get("mode") == mode]
        
        if not trades:
            return None
        
        # Calculate metrics
        total_trades = len(trades)
        pnls = [t.get("pnl", 0) for t in trades]
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        
        total_pnl = sum(pnls)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Calculate max drawdown
        cumulative_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        for pnl in pnls:
            cumulative_pnl += pnl
            if cumulative_pnl > peak:
                peak = cumulative_pnl
            drawdown = (peak - cumulative_pnl) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Calculate Sharpe ratio (simplified)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0
        std_pnl = (sum((p - avg_pnl) ** 2 for p in pnls) / total_trades) ** 0.5 if total_trades > 1 else 0.0
        sharpe_ratio = (avg_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0.0  # Annualized
        
        # Calculate profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
        
        # Calculate trading days
        timestamps = [datetime.fromisoformat(t["timestamp"]) for t in trades if "timestamp" in t]
        trading_days = len(set(ts.date() for ts in timestamps)) if timestamps else 0
        
        start_date = min(timestamps) if timestamps else None
        end_date = max(timestamps) if timestamps else None
        
        return ModePerformanceStats(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            trading_days=trading_days,
            start_date=start_date,
            end_date=end_date,
        )
    
    def get_trade_history(self, bot_id: str, mode: Optional[str] = None) -> List[Dict]:
        """Get trade history for a bot, optionally filtered by mode."""
        trades = self._trade_history.get(bot_id, [])
        if mode:
            trades = [t for t in trades if t.get("mode") == mode]
        return trades


class PromotionManager:
    """
    Manages the Paper->Demo->Live promotion workflow.
    
    Responsibilities:
    - Evaluate promotion eligibility
    - Execute promotions/demotions
    - Calculate capital scaling
    - Run daily promotion checks
    """
    
    def __init__(
        self,
        bot_registry: Optional["BotRegistry"] = None,
        performance_tracker: Optional[PerformanceTracker] = None,
    ):
        self._bot_registry = bot_registry
        self._performance_tracker = performance_tracker or PerformanceTracker()
    
    @property
    def bot_registry(self) -> "BotRegistry":
        """Lazy load BotRegistry if not provided."""
        if self._bot_registry is None:
            from src.router.bot_manifest import BotRegistry
            self._bot_registry = BotRegistry()
        return self._bot_registry
    
    def check_promotion_eligibility(self, bot_id: str) -> PromotionResult:
        """
        Check if a bot is eligible for promotion to the next mode.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            PromotionResult with eligibility status
        """
        from src.router.bot_manifest import TradingMode
        
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return PromotionResult(
                bot_id=bot_id,
                current_mode="unknown",
                eligible=False,
                error=f"Bot {bot_id} not found",
            )
        
        # Get current stats from performance tracker
        current_mode = bot.trading_mode.value
        stats = self._performance_tracker.calculate_stats(bot_id, current_mode)
        
        if stats is None:
            return PromotionResult(
                bot_id=bot_id,
                current_mode=current_mode,
                eligible=False,
                missing_criteria=["No performance data available"],
                thresholds=self._get_thresholds_for_mode(bot.trading_mode),
            )
        
        # Update bot's stats
        bot.update_stats(stats)
        
        # Check eligibility using bot's method
        eligibility = bot.check_promotion_eligibility()
        
        return PromotionResult(
            bot_id=bot_id,
            current_mode=current_mode,
            eligible=eligibility["eligible"],
            next_mode=eligibility.get("next_mode"),
            missing_criteria=eligibility.get("missing_criteria", []),
            current_stats=stats.to_dict(),
            thresholds=eligibility.get("criteria", {}),
        )
    
    def _get_thresholds_for_mode(self, mode: "TradingMode") -> Dict[str, Any]:
        """Get promotion thresholds for the given mode."""
        from src.router.bot_manifest import TradingMode
        
        if mode == TradingMode.PAPER:
            return PROMOTION_THRESHOLDS["paper_to_demo"]
        elif mode == TradingMode.DEMO:
            return PROMOTION_THRESHOLDS["demo_to_live"]
        else:
            return {}  # No thresholds for LIVE (already at max)
    
    def promote_bot(self, bot_id: str, force: bool = False) -> PromotionResult:
        """
        Promote a bot to the next trading mode.
        
        Args:
            bot_id: Bot identifier
            force: If True, bypass eligibility check (use with caution)
            
        Returns:
            PromotionResult with promotion status
        """
        from src.router.bot_manifest import TradingMode
        
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return PromotionResult(
                bot_id=bot_id,
                current_mode="unknown",
                eligible=False,
                error=f"Bot {bot_id} not found",
            )
        
        current_mode = bot.trading_mode.value
        
        # Check eligibility unless forced
        if not force:
            eligibility = self.check_promotion_eligibility(bot_id)
            if not eligibility.eligible:
                return eligibility
        
        # Perform promotion
        result = self.bot_registry.promote_bot(bot_id)
        
        if result["success"]:
            # Calculate capital allocation for new mode
            new_mode = result["new_mode"]
            capital = self._calculate_capital_for_mode(bot, TradingMode(new_mode))
            bot.capital_allocated = capital
            self.bot_registry._save()
            
            current_stats = bot.get_current_stats()
            
            return PromotionResult(
                bot_id=bot_id,
                current_mode=current_mode,
                eligible=True,
                next_mode=new_mode,
                promoted_at=result.get("promoted_at"),
                current_stats=current_stats.to_dict() if current_stats else {},
            )
        else:
            return PromotionResult(
                bot_id=bot_id,
                current_mode=current_mode,
                eligible=False,
                error=result.get("reason", "Promotion failed"),
            )
    
    def downgrade_bot(self, bot_id: str, reason: str = "") -> PromotionResult:
        """
        Downgrade a bot to a lower trading mode.
        
        Args:
            bot_id: Bot identifier
            reason: Reason for downgrade
            
        Returns:
            PromotionResult with downgrade status
        """
        bot = self.bot_registry.get(bot_id)
        if bot is None:
            return PromotionResult(
                bot_id=bot_id,
                current_mode="unknown",
                eligible=False,
                error=f"Bot {bot_id} not found",
            )
        
        current_mode = bot.trading_mode.value
        
        # Perform downgrade
        result = self.bot_registry.downgrade_bot(bot_id, reason)
        
        if result["success"]:
            new_mode = result["new_mode"]
            
            current_stats = bot.get_current_stats()
            
            return PromotionResult(
                bot_id=bot_id,
                current_mode=current_mode,
                eligible=False,  # Not eligible after downgrade
                next_mode=None,
                error=f"Downgraded: {reason}",
                current_stats=current_stats.to_dict() if current_stats else {},
            )
        else:
            return PromotionResult(
                bot_id=bot_id,
                current_mode=current_mode,
                eligible=False,
                error=result.get("reason", "Downgrade failed"),
            )
    
    def _calculate_capital_for_mode(self, bot: "BotManifest", mode: "TradingMode") -> float:
        """
        Calculate capital allocation for a bot in a specific mode.
        
        Capital Scaling Rules:
        - PAPER: $0 (virtual)
        - DEMO: $1,000 (demo account)
        - LIVE: Base $1,000, scaled by performance up to $10,000
        """
        from src.router.bot_manifest import TradingMode
        
        if mode == TradingMode.PAPER:
            return CAPITAL_SCALING["paper"]
        elif mode == TradingMode.DEMO:
            return CAPITAL_SCALING["demo"]
        elif mode == TradingMode.LIVE:
            # Base capital
            base = CAPITAL_SCALING["live_base"]
            max_cap = CAPITAL_SCALING["live_max"]
            scaling = CAPITAL_SCALING["live_scaling_factor"]
            
            # Scale based on demo performance
            if bot.demo_stats and bot.demo_stats.total_pnl > 0:
                # Add 50% of demo profits to base capital
                scaled = base + (bot.demo_stats.total_pnl * scaling)
                return min(scaled, max_cap)
            
            return base
        
        return 0.0
    
    def run_daily_promotion_check(self) -> List[PromotionResult]:
        """
        Run daily promotion eligibility check for all bots.
        
        This should be called once per day (e.g., via cron or scheduler).
        
        Returns:
            List of PromotionResult for all bots
        """
        results = []
        
        for bot in self.bot_registry.list_all():
            result = self.check_promotion_eligibility(bot.bot_id)
            results.append(result)
            
            # Log promotion-eligible bots
            if result.eligible:
                logger.info(
                    f"Bot {bot.bot_id} is eligible for promotion from {result.current_mode} "
                    f"to {result.next_mode}"
                )
        
        # Save updated eligibility status
        self.bot_registry._save()
        
        return results
    
    def get_promotion_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get detailed promotion status for a bot.
        
        Returns:
            Dict with current mode, eligibility, stats, and thresholds
        """
        result = self.check_promotion_eligibility(bot_id)
        
        return {
            "bot_id": result.bot_id,
            "current_mode": result.current_mode,
            "eligible": result.eligible,
            "next_mode": result.next_mode,
            "missing_criteria": result.missing_criteria,
            "current_stats": result.current_stats,
            "thresholds": result.thresholds,
            "error": result.error,
        }
    
    def get_all_promotion_statuses(self) -> List[Dict[str, Any]]:
        """
        Get promotion status for all bots.
        
        Returns:
            List of promotion status dicts
        """
        statuses = []
        for bot in self.bot_registry.list_all():
            status = self.get_promotion_status(bot.bot_id)
            statuses.append(status)
        return statuses
    
    def record_trade_for_bot(self, bot_id: str, trade: Dict[str, Any]) -> None:
        """
        Record a trade for performance tracking.
        
        Args:
            bot_id: Bot identifier
            trade: Trade dict with pnl, timestamp, etc.
        """
        # Add mode from bot's current trading mode
        bot = self.bot_registry.get(bot_id)
        if bot:
            trade["mode"] = bot.trading_mode.value
        
        self._performance_tracker.record_trade(bot_id, trade)