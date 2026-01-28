"""
Correlation Sensor (RMT)
Measures Systemic Risk via Random Matrix Theory (Max Eigenvalue).
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class CorrelationReport:
    max_eigenvalue: float # > 2.0 means Swarm
    score: float          # 0.0 to 1.0 (Normalized Risk)
    is_systemic: bool

class CorrelationSensor:
    def __init__(self):
        # Stores recent returns for all symbols
        self.market_returns: Dict[str, list] = {}
        self.window_size = 50

    def update(self, symbol: str, price_change_pct: float) -> CorrelationReport:
        if symbol not in self.market_returns:
            self.market_returns[symbol] = []
            
        self.market_returns[symbol].append(price_change_pct)
        if len(self.market_returns[symbol]) > self.window_size:
             self.market_returns[symbol].pop(0)

        # Only calculate if we have enough data across enough assets
        if len(self.market_returns) < 3: 
             return CorrelationReport(1.0, 0.0, False)

        return self._calculate_rmt()

    def _calculate_rmt(self) -> CorrelationReport:
        # Placeholder for meaningful RMT calc
        # In prod, this builds a Correlation Matrix and runs Eigen decomp.
        
        # Mock logic:
        # We assume if we have data, we are 'safe' for now unless external feed says otherwise.
        
        lambda_max = 1.0
        score = 0.1
        return CorrelationReport(lambda_max, score, False)
