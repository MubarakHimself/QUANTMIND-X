"""
The Commander (Execution Layer)
Responsible for selecting and dispatching Bots based on Regime.
"""

from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class Commander:
    """
    The Base Commander implementing the Strategy Auction logic.
    """
    def __init__(self):
        self.active_bots: Dict = {}
        self.regime_map: Dict = {}  # Load from config

    def run_auction(self, regime_report) -> List[dict]:
        """
        Conducts the Bot Auction for the current Regime.
        
        Args:
            regime_report: Data from Sentinel (Regime Enum, Chaos Score)
            
        Returns:
            List of authorized Bot Dispatches.
        """
        # 1. Identify Eligible Bots
        eligible_bots = self._get_bots_for_regime(regime_report.regime)
        
        # 2. Rank by Performance (Base Logic)
        ranked_bots = sorted(eligible_bots, key=lambda x: x['score'], reverse=True)
        
        # 3. Select Top N
        top_bots = ranked_bots[:3]
        
        return top_bots

    def _get_bots_for_regime(self, regime) -> List[dict]:
        """
        Returns bots that are authorized to trade in this physics state.
        For V1: All bots are authorize for all regimes (except high chaos).
        """
        if regime == "HIGH_CHAOS":
            return []
            
        # Return all active bots as candidates
        return list(self.active_bots.values())

    def dispatch(self, bot_id: str, instruction: str):
        """
        Sends JSON command to the Interface.
        """
        pass
