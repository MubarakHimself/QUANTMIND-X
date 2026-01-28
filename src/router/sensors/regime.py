"""
Regime Sensor (Ising Model)
Measures Phase Transitions (Susceptibility) and Market Temperature.
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class RegimeReport:
    magnetization: float # -1.0 to 1.0 (Trend Strength)
    susceptibility: float # Variance (Criticality)
    energy: float         # Frustration (Rangeiness)
    state: str            # ORDERED, CRITICAL, DISORDERED

class RegimeSensor:
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.spins = [] # +1 (Up tick), -1 (Down tick)

    def update(self, price_change: float) -> RegimeReport:
        spin = 1 if price_change >= 0 else -1
        self.spins.append(spin)
        if len(self.spins) > self.window_size:
            self.spins.pop(0)
            
        return self._calculate_ising()

    def _calculate_ising(self) -> RegimeReport:
        if not self.spins:
            return RegimeReport(0.0, 0.0, 0.0, "DISORDERED")
            
        # Magnetization: Average Spin (Trend Direction & Strength)
        M = np.mean(self.spins)
        
        # Susceptibility: Variance of Spins (Sensitivity to change)
        # High Variance = System is flickering = Critical Phase Transition
        chi = np.var(self.spins) 
        
        # Energy: Neighbor interactions (Frustration)
        # E = -Sum(s_i * s_i+1)
        energy = 0
        arr = np.array(self.spins)
        for i in range(len(arr)-1):
            energy -= arr[i] * arr[i+1]
        
        # Normalize Energy
        E = energy / len(arr) 

        state = "DISORDERED"
        if abs(M) > 0.7: state = "ORDERED" (Trend)
        if chi > 0.24: state = "CRITICAL" # Phase Transition (0.25 is max var for binary)

        return RegimeReport(M, chi, E, state)
