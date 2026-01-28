"""
QuantMindX Strategy Router Engine
The coordination layer for the Sentient Loop.
"""

import logging
from typing import Dict, List, Optional
import time

from src.router.sentinel import Sentinel
from src.router.governor import Governor, RiskMandate
from src.router.commander import Commander

logger = logging.getLogger(__name__)

class StrategyRouter:
    """
    The Orchestrator of the Strategy Router system.
    Ties together the Intelligence, Compliance, and Decision layers.
    """
    def __init__(self):
        self.sentinel = Sentinel()
        self.governor = Governor()
        self.commander = Commander()
        
        # In-memory registration of bots
        self.registered_bots = {}

    def process_tick(self, symbol: str, price: float, account_data: Optional[Dict] = None) -> Dict:
        """
        The Full Sentient Loop execution.
        """
        # 1. Observe (Sentinel)
        report = self.sentinel.on_tick(symbol, price)
        logger.debug(f"Sentinel Report: {report.regime} (Quality: {report.regime_quality:.2f})")

        # 2. Govern (Governor)
        # We issue a "Swarm Mandate" for the current physics
        # pass account_data for Tier 3 Prop rules
        proposal = {"symbol": symbol}
        if account_data:
            proposal.update(account_data)

        mandate = self.governor.calculate_risk(report, proposal)
        
        # 3. Command (Commander)
        # Identify which bots are authorized to trade in this regime
        dispatches = self.commander.run_auction(report)
        
        # Apply Governor's Mandate to dispatches
        for bot in dispatches:
            bot['authorized_risk_scalar'] = mandate.allocation_scalar
            bot['risk_mode'] = mandate.risk_mode
            
        return {
            "regime": report.regime,
            "quality": report.regime_quality,
            "mandate": mandate,
            "dispatches": dispatches
        }

    def register_bot(self, bot_instance):
        """Register a new bot into the factory pipeline."""
        self.registered_bots[bot_instance.bot_id] = bot_instance
        # Also register in commander's auction pool
        self.commander.active_bots[bot_instance.bot_id] = bot_instance.to_dict()
        logger.info(f"Bot {bot_instance.bot_id} (@{bot_instance.magic_number}) registered in Strategy Router.")
