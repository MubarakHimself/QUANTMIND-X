"""
QuantMindX Base Bot (Tier 1 Integration)
Provides the standardized interface for all Expert Advisors in the ecosystem.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional
import logging

from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

logger = logging.getLogger(__name__)

class BaseBot(ABC):
    """
    Abstract Base Class for all Bots.
    All registered bots must inherit from this to ensure compatibility
    with the Strategy Router's Risk Stacking formula.
    """
    def __init__(self, bot_id: str, magic_number: int):
        self.bot_id = bot_id
        self.magic_number = magic_number
        self.risk_engine = EnhancedKellyCalculator()
        
        # Performance metrics (stub)
        self.expected_win_rate = 0.55
        self.avg_win = 400.0
        self.avg_loss = 200.0

    @abstractmethod
    def get_signal(self, data: Dict) -> float:
        """Return signal strength (-1.0 to 1.0)."""
        pass

    def calculate_entry_size(
        self, 
        account_balance: float, 
        sentinel_report: Optional[Dict] = None
    ) -> float:
        """
        Calculates the Tier 1 position size.
        Injects market physics (Regime Quality) into the Kelly engine.
        """
        # 1. Default quality if no sentinel report
        regime_quality = 1.0
        current_atr = 0.0010
        avg_atr = 0.0010
        
        if sentinel_report:
            # Physics Injection: Stability (1.0) -> Chaos (0.0)
            regime_quality = sentinel_report.get('regime_quality', 1.0)
            current_atr = sentinel_report.get('current_atr', 0.0010)
            avg_atr = sentinel_report.get('average_atr', 0.0010)

        # 2. Ask Kelly for Size
        # This is the "Bot's Opinion" before the Governor's Compliance check.
        result = self.risk_engine.calculate(
            account_balance=account_balance,
            win_rate=self.expected_win_rate,
            avg_win=self.avg_win,
            avg_loss=self.avg_loss,
            current_atr=current_atr,
            average_atr=avg_atr,
            stop_loss_pips=20.0,  # Generic SL for Tier 1 calculation
            regime_quality=regime_quality
        )
        
        logger.info(f"Bot {self.bot_id} Tier 1 Risk: {result.kelly_f:.2%} ({result.position_size} lots)")
        return result.position_size

    def to_dict(self) -> Dict:
        """Metadata for the Strategy Auction."""
        return {
            "bot_id": self.bot_id,
            "magic_number": self.magic_number,
            "score": self.expected_win_rate * (self.avg_win / self.avg_loss), # Simple Ranking Score
            "win_rate": self.expected_win_rate
        }
