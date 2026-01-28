"""
The Sentinel (Intelligence Layer)
Aggregates Sensor Data into a Unified Regime Report.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import logging

from src.router.sensors.chaos import ChaosSensor
from src.router.sensors.regime import RegimeSensor
from src.router.sensors.correlation import CorrelationSensor
from src.router.sensors.news import NewsSensor

logger = logging.getLogger(__name__)

@dataclass
class RegimeReport:
    regime: str             # TREND_STABLE, RANGE_STABLE, BREAKOUT, ETC
    chaos_score: float      # 0.0 - 1.0
    regime_quality: float   # 1.0 - chaos_score
    susceptibility: float   # 0.0 - 1.0
    is_systemic_risk: bool  
    news_state: str         # SAFE, KILL_ZONE
    timestamp: float

class Sentinel:
    """
    The Intelligence Engine. 
    Ingests Ticks -> Updates Sensors -> Classifies Regime.
    """
    def __init__(self):
        self.chaos = ChaosSensor()
        self.regime = RegimeSensor()
        self.correlation = CorrelationSensor()
        self.news = NewsSensor()
        
        self.current_report: Optional[RegimeReport] = None

    def on_tick(self, symbol: str, price: float) -> RegimeReport:
        """
        Main Loop: Called on every tick.
        """
        # 1. Update Sensors
        c_report = self.chaos.update(price)
        r_report = self.regime.update(price) # Simplified: Passing price instead of delta
        # co_report = self.correlation.update(...) # Requires multi-symbol feed
        n_state = self.news.check_state()
        
        # 2. Classify Regime (The Matrix)
        regime_label = self._classify(c_report, r_report, n_state)
        
        # 3. Compile Report
        self.current_report = RegimeReport(
            regime=regime_label,
            chaos_score=c_report.score,
            regime_quality=1.0 - c_report.score,
            susceptibility=r_report.susceptibility,
            is_systemic_risk=False, # Placeholder
            news_state=n_state,
            timestamp=0.0
        )
        return self.current_report

    def _classify(self, c, r, n_state) -> str:
        """
        Maps Sensor Outputs to Regime Enum.
        """
        if n_state == "KILL_ZONE":
            return "NEWS_EVENT"
            
        if c.score > 0.6:
            return "HIGH_CHAOS"
            
        if r.state == "CRITICAL":
            return "BREAKOUT_PRIME"
            
        if r.state == "ORDERED" and c.score < 0.3:
            return "TREND_STABLE"
            
        if r.state == "DISORDERED" and c.score < 0.3:
            return "RANGE_STABLE"
            
        return "UNCERTAIN"
