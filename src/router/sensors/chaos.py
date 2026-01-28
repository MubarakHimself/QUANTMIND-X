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
        if len(self.prices) < 20:
             return ChaosReport(0.0, "STABLE")
             
        # Lyapunov Proxy: Rate of separation of nearby points in Phase Space
        # Calculate log-divergence of returns over the window
        returns = np.diff(self.prices)
        if len(returns) < 5:
            return ChaosReport(0.0, "STABLE")
            
        # Reshape into "delay coordinates" to see phase space (simplified)
        # We look at how |x(t+1) - x(t)| grows
        dists = np.abs(np.diff(returns))
        # Rate of growth of divergence
        # Small noise floor to avoid log(0)
        log_dists = np.log(dists + 1e-9)
        
        # Mean of log-divergence is an estimate of the largest Lyapunov exponent (LLE)
        # λ > 0 implies chaos
        lambda_proxy = np.mean(log_dists)
        
        # Normalize score: lambda_proxy often ranges from -10 to 0 for stable systems
        # and moves towards positive for chaotic ones.
        # We'll map -10 -> 0.0 and -2 -> 1.0 (empirical mapping for FX)
        score = np.clip((lambda_proxy + 10) / 8, 0.0, 1.0)
        
        label = "STABLE"
        if score > 0.4: label = "NOISY"
        if score > 0.7: label = "CHAOTIC"
        
        return ChaosReport(score, label)
