"""
Chaos Sensor (Lyapunov)
Measures market turbulence/predictability using Lyapunov Exponents.
"""

import numpy as np
from dataclasses import dataclass

@dataclass
class ChaosReport:
    score: float  # 0.0 to 1.0
    label: str    # STABLE, CHOP, CHAOTIC

class ChaosSensor:
    def __init__(self, window_size=300):
        self.window_size = window_size
        self.prices = []

    def update(self, price: float) -> ChaosReport:
        self.prices.append(price)
        if len(self.prices) > self.window_size:
            self.prices.pop(0)
            
        return self._calculate_lyapunov()

    def _calculate_lyapunov(self) -> ChaosReport:
        if len(self.prices) < self.window_size:
            return ChaosReport(0.0, "STABLE") # Not enough data
            
        # Simplified Lyapunov proxy for real-time speed
        # Real implementation would use nolds or similar lib
        # Here we use Log Variance of Returns as a proxy
        
        returns = np.diff(self.prices)
        if len(returns) == 0:
             return ChaosReport(0.0, "STABLE")

        volatility = np.std(returns)
        
        # Arbitrary scaling for prototype
        # 0.0001 vol = 0.1 score
        # 0.0010 vol = 0.8 score
        score = min(1.0, volatility * 1000)
        
        label = "STABLE"
        if score > 0.3: label = "NOISY"
        if score > 0.6: label = "CHAOTIC"
        
        return ChaosReport(score, label)
