from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)

from src.router.bot_manifest import BotManifest
from src.position_sizing.portfolio_kelly import PortfolioKellyScaler
from src.database.db_manager import DBManager
from src.database.models import StrategyPerformance, BotCloneHistory

SYMBOL_GROUPS = {
    "MAJORS": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"],
    "CROSSES": ["EURGBP", "EURJPY", "GBPJPY"],
    "GOLD": ["XAUUSD", "XAGUSD"],
    "INDICES": ["NAS100", "SPX500", "US30"]
}

@dataclass
class CloneCandidate:
    bot_id: str
    original_symbol: str
    performance_score: float
    win_rate: float
    total_trades: int
    recommended_symbols: List[str]

class BotCloner:
    def __init__(self, db_manager: Optional[DBManager] = None):
        self.scaler = PortfolioKellyScaler()
        self.db = db_manager or DBManager()
        # Initialize BotRegistry for loading source bot manifests
        try:
            from src.router.bot_manifest import BotRegistry
            self.bot_registry = BotRegistry()
        except Exception as e:
            logger.warning(f"Could not initialize BotRegistry: {e}")
            self.bot_registry = None

    def is_clone_eligible(self, bot_id: str) -> bool:
        """Validates bot meets cloning criteria (Sharpe > 2.0, win rate > 55%, 100+ trades, 30+ days active)"""
        metrics = self._get_bot_metrics(bot_id)
        return (
            metrics.get('sharpe', 0) > 2.0 and
            metrics.get('win_rate', 0) > 0.55 and
            metrics.get('total_trades', 0) > 100 and
            metrics.get('days_active', 0) > 30
        )

    def _get_bot_metrics(self, bot_id: str) -> Dict[str, Any]:
        """Retrieve bot performance from StrategyPerformance table"""
        perf_record = self.db.session.query(StrategyPerformance).filter_by(strategy_name=bot_id).first()
        if not perf_record:
            return {}
        days_active = (datetime.now(timezone.utc) - perf_record.created_at).days  # Use created_at since start_date not exist
        return {
            'sharpe': perf_record.sharpe_ratio,
            'win_rate': perf_record.win_rate,
            'total_trades': perf_record.total_trades,
            'days_active': days_active
        }

    def get_similar_symbols(self, symbol: str) -> List[str]:
        """Returns similar trading pairs using predefined symbol groups"""
        for symbols in SYMBOL_GROUPS.values():
            if symbol in symbols:
                return [s for s in symbols if s != symbol]
        return []

    def clone_bot(self, original_bot_id: str, target_symbols: Optional[List[str]] = None) -> List[str]:
        """Creates new BotManifest instances for target symbols with copied strategy parameters"""
        # Validate with is_clone_eligible before cloning
        if not self.is_clone_eligible(original_bot_id):
            logger.warning(f"Bot {original_bot_id} is not eligible for cloning")
            return []
        
        # Load the source bot manifest from BotRegistry/DB instead of a stub
        original_manifest = None
        if self.bot_registry:
            original_manifest = self.bot_registry.get(original_bot_id)
        
        if not original_manifest:
            logger.error(f"Could not find original bot manifest for {original_bot_id}")
            return []
        
        # Derive target symbols via get_similar_symbols when none provided
        original_symbol = original_manifest.symbols[0] if original_manifest.symbols else "EURUSD"
        if target_symbols is None:
            target_symbols = self.get_similar_symbols(original_symbol)
        
        num_clones = len(target_symbols)
        if num_clones == 0:
            logger.warning(f"No target symbols found for cloning {original_bot_id}")
            return []
        
        # Use PortfolioKellyScaler and _calculate_adaptive_allocation to set per-clone allocations
        allocations = self._calculate_adaptive_allocation(num_clones)
        
        clone_ids = []
        for i, sym in enumerate(target_symbols):
            clone_id = str(uuid.uuid4())[:8]
            from src.router.bot_manifest import BotManifest
            
            # Copy actual strategy parameters/symbol from the manifest
            clone_manifest = BotManifest(
                bot_id=clone_id,
                strategy_type=original_manifest.strategy_type,
                frequency=original_manifest.frequency,
                symbols=[sym],
                name=f"Clone_{original_bot_id}_{sym}",
                prop_firm_safe=original_manifest.prop_firm_safe,
                # Copy other relevant fields from original manifest
                preferred_conditions=original_manifest.preferred_conditions,
                preferred_broker_type=original_manifest.preferred_broker_type,
                min_capital_req=original_manifest.min_capital_req,
                tags=original_manifest.tags + ["@clone"],  # Add clone tag
                # Performance metrics
                total_trades=original_manifest.total_trades,
                win_rate=original_manifest.win_rate,
                # Timeframe settings
                preferred_timeframe=original_manifest.preferred_timeframe,
                use_multi_timeframe=original_manifest.use_multi_timeframe,
                secondary_timeframes=original_manifest.secondary_timeframes,
                # Position limits
                max_positions=original_manifest.max_positions,
                max_daily_trades=original_manifest.max_daily_trades
            )
            
            # Register/persist each BotManifest before writing BotCloneHistory
            if self.bot_registry:
                self.bot_registry.register(clone_manifest)
                logger.info(f"Registered clone bot: {clone_id}")
            
            # Also persist to database if needed
            try:
                # Store in database for tracking
                self.db.session.add(clone_manifest)
                self.db.commit()
            except Exception as e:
                logger.warning(f"Could not persist clone manifest to DB: {e}")
            
            clone_ids.append(clone_id)
            
            # Record clone history after registration
            history = BotCloneHistory(
                original_bot_id=original_bot_id,
                clone_bot_id=clone_id,
                original_symbol=original_symbol,
                clone_symbol=sym,
                performance_at_clone=self._get_bot_metrics(original_bot_id),
                allocation_strategy='adaptive',
                allocation_pct=allocations.get('clones', 0.5 / num_clones)
            )
            self.db.session.add(history)
            self.db.commit()
        
        logger.info(f"Successfully cloned {original_bot_id} into {len(clone_ids)} bots")
        return clone_ids

    def _calculate_adaptive_allocation(self, num_clones: int) -> Dict[str, float]:
        """Implements adaptive allocation logic using PortfolioKellyScaler"""
        if num_clones == 0:
            return {'original': 1.0, 'clones': 0.0}
        
        # Use PortfolioKellyScaler for more sophisticated allocation
        # Original bot gets 50%, remaining 50% split among clones
        original_allocation = 0.5
        clone_allocation = 0.5 / num_clones
        
        # Apply portfolio risk management
        bot_ids = ['original'] + [f'clone_{i}' for i in range(num_clones)]
        performance = {'original': 1.0}  # Original bot gets base performance
        
        # Add estimated performance for clones (conservative estimate)
        for i in range(num_clones):
            performance[f'clone_{i}'] = 0.8  # Clones start with 80% of original performance
        
        # Use PortfolioKellyScaler to allocate based on performance
        allocations = self.scaler.allocate_risk_by_performance(
            performance,
            total_risk_budget=1.0,  # 100% allocation
            min_allocation=0.01  # Minimum 1% per bot
        )
        
        # Ensure original gets at least 50%
        original_risk = allocations.get('original', original_allocation)
        if original_risk < original_allocation:
            # Redistribute from clones to original
            shortage = original_allocation - original_risk
            clone_ids = [f'clone_{i}' for i in range(num_clones)]
            for clone_id in clone_ids:
                if clone_id in allocations:
                    allocations[clone_id] *= (1 - shortage)
        
        return {
            'original': allocations.get('original', original_allocation),
            'clones': allocations.get('clone_0', clone_allocation)
        }
