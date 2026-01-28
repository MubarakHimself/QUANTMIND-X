"""
The Prop Commander (Extension)
Inherits from Base Commander and adds Goal-Oriented Logic.
"""

from typing import List
from src.router.commander import Commander
from src.router.prop.state import PropState

class PropCommander(Commander):
    """
    Extensions:
    1. Preservation Mode (When Target Reached).
    2. Coin Flip Bot (Minimum Trading Days).
    """
    def __init__(self, account_id: str):
        super().__init__()
        self.prop_state = PropState(account_id)
        self.target_profit_pct = 0.08  # 8% Target
        self.min_trading_days = 5

    def run_auction(self, regime_report) -> List[dict]:
        """
        Overrides Base Auction to filter for Goal Preservation.
        """
        # 1. Check State
        metrics = self.prop_state.get_metrics() # Hypothetical accessor
        
        # 2. Determine Mode
        is_preservation_mode = self._check_preservation_mode(metrics)
        
        # 3. Modify Auction Logic
        if is_preservation_mode:
            # A. Check if we need Coin Flip (Trading Days)
            if self._needs_trading_days(metrics):
                return [self._get_coin_flip_bot()]
            
            # B. Filter for A+ Setups Only (High Score)
            # Run base auction first
            base_bots = super().run_auction(regime_report)
            
            # Filter: Only allow bots with Kelly Score >= 0.8 (A+ setups)
            filtered_bots = [b for b in base_bots if b.get('kelly_score', 0) >= 0.8]
            return filtered_bots
            
        else:
            # Standard Mode: Just call Parent
            return super().run_auction(regime_report)

    def _check_preservation_mode(self, metrics) -> bool:
        """
        Returns True if we have passed the Profit Target.
        """
        if not metrics:
            return False
        # Calculate Gain
        start = metrics.daily_start_balance # simplified
        if not start: return False
        
        gain_pct = (metrics.current_equity - start) / start
        return gain_pct >= self.target_profit_pct

    def _needs_trading_days(self, metrics) -> bool:
        """
        Returns True if we passed target but haven't traded enough days.
        """
        if not metrics: return False
        return metrics.trading_days < self.min_trading_days

    def _get_coin_flip_bot(self) -> dict:
        """
        Returns the special 'CoinFlip' bot definition.
        """
        return {
            "name": "CoinFlip_Bot",
            "strategy": "MIN_DAYS_TICKER",
            "risk_mode": "MINIMAL", # 0.01 Lots
            "instruction": "OPEN_RANDOM_CLOSE_FAST"
        }
