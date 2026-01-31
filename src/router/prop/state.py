"""
Prop State Management
Responsible for persisting funded account metrics (Daily Loss, High Water Mark).
"""

import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime, date

logger = logging.getLogger(__name__)

@dataclass
class PropAccountMetrics:
    account_id: str
    daily_start_balance: float
    high_water_mark: float
    current_equity: float
    trading_days: int
    target_met: bool = False
    last_updated: str = ""

class PropState:
    """
    Manages persistent state for Prop Accounts using database-backed storage.
    
    Provides methods for:
    - Daily snapshot updates
    - Daily loss calculations
    - Quadratic throttle calculations
    - Account metrics retrieval
    """
    def __init__(self, account_id: str, db_manager=None):
        self.account_id = account_id
        self._db_manager = db_manager
        self._metrics_cache = None
        
    @property
    def db_manager(self):
        """Lazy load database manager to avoid circular imports."""
        if self._db_manager is None:
            try:
                from src.database.manager import DatabaseManager
                self._db_manager = DatabaseManager()
            except ImportError as e:
                logger.warning(f"DatabaseManager not available: {e}")
                self._db_manager = None
        return self._db_manager
    
    def get_metrics(self) -> Optional[PropAccountMetrics]:
        """
        Retrieve current account metrics from database.
        
        Returns:
            PropAccountMetrics object or None if not found
        """
        if not self.db_manager:
            logger.warning("Database manager not available, using cached metrics")
            return self._metrics_cache
            
        try:
            # Get account from database
            account = self.db_manager.get_prop_account(self.account_id)
            if not account:
                logger.warning(f"Account {self.account_id} not found in database")
                return None
            
            # Get latest snapshot
            snapshot = self.db_manager.get_latest_snapshot(self.account_id)
            if not snapshot:
                logger.warning(f"No snapshot found for account {self.account_id}")
                return None
            
            # Build metrics object
            metrics = PropAccountMetrics(
                account_id=self.account_id,
                daily_start_balance=snapshot.high_water_mark,
                high_water_mark=snapshot.high_water_mark,
                current_equity=snapshot.current_equity,
                trading_days=snapshot.trades_count,  # Using trades_count as proxy for trading days
                target_met=False,  # Calculate based on profit
                last_updated=snapshot.created_at.isoformat() if snapshot.created_at else ""
            )
            
            # Cache for fallback
            self._metrics_cache = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving metrics for account {self.account_id}: {e}")
            return self._metrics_cache

    def update_snapshot(self, equity: float, balance: float):
        """
        Called at Midnight (00:00) to reset Daily Drawdown markers.
        
        Args:
            equity: Current account equity
            balance: Current account balance
        """
        if not self.db_manager:
            logger.warning("Database manager not available, cannot update snapshot")
            return
            
        try:
            # Get account
            account = self.db_manager.get_prop_account(self.account_id)
            if not account:
                logger.error(f"Account {self.account_id} not found, cannot update snapshot")
                return
            
            # Save daily snapshot
            self.db_manager.save_daily_snapshot(
                account_id=self.account_id,
                equity=equity,
                balance=balance
            )
            
            logger.info(f"Updated snapshot for account {self.account_id}: equity={equity}, balance={balance}")
            
            # Invalidate cache
            self._metrics_cache = None
            
        except Exception as e:
            logger.error(f"Error updating snapshot for account {self.account_id}: {e}")

    def check_daily_loss(self, current_equity: float) -> float:
        """
        Returns drawdown percentage from Daily Start Balance.
        
        Args:
            current_equity: Current account equity
            
        Returns:
            Drawdown percentage (0.0 to 1.0)
        """
        metrics = self.get_metrics()
        if not metrics or not metrics.daily_start_balance:
            logger.warning("No metrics available for daily loss calculation")
            return 0.0
        
        loss = metrics.daily_start_balance - current_equity
        if loss <= 0:
            return 0.0
        
        drawdown_pct = loss / metrics.daily_start_balance
        return drawdown_pct

    def get_quadratic_throttle(self, current_balance: float, limit_pct: float = 0.05) -> float:
        """
        Calculates the 'Distance to Ruin' multiplier using quadratic formula.
        
        Formula: Throttle = 1.0 - (CurrentLoss / EffectiveLimit)^2
        
        Args:
            current_balance: Current account balance
            limit_pct: Daily loss limit percentage (default: 0.05 = 5%)
            
        Returns:
            Throttle multiplier (0.0 to 1.0)
        """
        metrics = self.get_metrics()
        if not metrics or not metrics.daily_start_balance:
            logger.warning("No metrics available for throttle calculation")
            return 1.0
        
        start_balance = metrics.daily_start_balance
        current_loss = start_balance - current_balance
        
        # If in profit, no throttle
        if current_loss <= 0:
            return 1.0
        
        loss_pct = current_loss / start_balance
        
        # Hard stop buffer (1%)
        effective_limit = limit_pct - 0.01
        
        # If breach effective limit, kill switch
        if loss_pct >= effective_limit:
            return 0.0
        
        # Quadratic curve
        throttle = 1.0 - (loss_pct / effective_limit) ** 2
        return max(0.0, throttle)
