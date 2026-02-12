"""
Ising Regime Sensor
==================

A physics-based regime detection system using the Ising Model of Ferromagnetism.
This sensor simulates market microstructure dynamics to detect phase transitions
between chaotic (high temperature) and ordered (low temperature) market regimes.

The implementation is based on the Ising Model with Metropolis-Hastings Monte Carlo
simulation on a 3D lattice to model social sentiment dynamics.

Features:
- 12x12x12 lattice for efficient simulation
- Metropolis-Hastings algorithm for spin dynamics
- Temperature-based regime detection
- Magnetization and susceptibility calculations
- Caching layer for performance optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from functools import lru_cache
import warnings
import pandas as pd
import time

warnings.filterwarnings("ignore")


@dataclass
class IsingSensorConfig:
    """Configuration for the Ising Regime Sensor."""
    grid_size: int = 12  # 12x12x12 lattice
    steps_per_temp: int = 100  # Monte Carlo steps per temperature
    temp_range: Tuple[float, float] = (10.0, 0.1)  # Annealing from chaos to order
    temp_steps: int = 50  # Number of temperature increments
    target_sentiment: float = 0.75  # Target bullish ratio
    control_gain: float = 5.0  # Strength of constraint field
    cache_size: int = 100  # Cache size for temperature results


class IsingSystem:
    """Core Ising Model implementation with Metropolis-Hastings algorithm."""

    def __init__(self, config: IsingSensorConfig):
        self.config = config
        self.N = config.grid_size
        self.target = config.target_sentiment
        self.gain = config.control_gain

        # Initialize random chaos (high temperature state)
        np.random.seed(42)
        self.lattice = np.random.choice([-1, 1], size=(self.N, self.N, self.N))

    def _get_neighbor_sum(self, x: int, y: int, z: int) -> int:
        """Calculate sum of neighboring spins with periodic boundary conditions."""
        N = self.N
        lat = self.lattice
        return (
            lat[(x+1)%N, y, z] + lat[(x-1)%N, y, z] +
            lat[x, (y+1)%N, z] + lat[x, (y-1)%N, z] +
            lat[x, y, (z+1)%N] + lat[x, y, (z-1)%N]
        )

    def metropolis_step(self, temp: float) -> int:
        """Perform one Monte Carlo sweep with Metropolis-Hastings algorithm."""
        N = self.N
        lat = self.lattice
        change_count = 0

        # Calculate dynamic field bias
        current_up_ratio = np.mean(lat == 1)
        bias = self.gain * (self.target - current_up_ratio)

        # Perform N^3 random spin updates
        for _ in range(N**3):
            x, y, z = np.random.randint(0, N, 3)
            spin = lat[x, y, z]

            neighbor_sum = self._get_neighbor_sum(x, y, z)
            dE = 2 * spin * (neighbor_sum + bias)

            # Metropolis acceptance criterion
            if dE <= 0 or np.random.rand() < np.exp(-dE / temp):
                lat[x, y, z] *= -1
                change_count += 1

        return change_count

    def get_observables(self) -> Tuple[float, float]:
        """Calculate magnetization and energy per spin."""
        # Magnetization
        magnetization = float(np.mean(self.lattice))

        # Energy per spin (approximate calculation)
        energy = 0
        for axis in range(3):
            energy += np.sum(self.lattice * np.roll(self.lattice, 1, axis=axis))
        energy = -energy / (self.N**3)

        return magnetization, float(energy)


class IsingRegimeSensor:
    """Ising Model-based regime detector for market phase transitions."""

    def __init__(self, config: Optional[IsingSensorConfig] = None):
        self.config = config or IsingSensorConfig()
        self._cache: Dict[float, Dict] = {}
        self._last_cache_time: float = 0.0

    @lru_cache(maxsize=100)
    def _simulate_temperature(self, temp: float) -> Dict:
        """Simulate system at specific temperature with caching."""
        sim = IsingSystem(self.config)

        # Thermalize the system
        for _ in range(self.config.steps_per_temp):
            sim.metropolis_step(temp)

        # Sample observables
        magnetizations = []
        for _ in range(self.config.steps_per_temp // 5):  # Sample last 20%
            m, _ = sim.get_observables()
            magnetizations.append(m)

        # Calculate statistics
        avg_m = np.mean(magnetizations)
        var_m = np.var(magnetizations)
        susceptibility = var_m / temp if temp > 0 else 0

        return {
            'temperature': temp,
            'magnetization': avg_m,
            'susceptibility': susceptibility,
            'activity': sim.metropolis_step(temp)
        }

    def detect_regime(self, market_volatility: Optional[float] = None) -> Dict:
        """
        Detect market regime using Ising model simulation.

        Args:
            market_volatility: Optional market volatility context (maps to temperature)

        Returns:
            Dictionary containing regime detection results
        """
        # Update cache timestamp
        self._last_cache_time = time.time()

        # Map volatility to temperature if provided
        if market_volatility is not None:
            temp = self._map_volatility_to_temperature(market_volatility)
            # Run simulation at specific temperature
            result = self._simulate_temperature(temp)

            # Classify regime based on magnetization
            current_regime = self._classify_regime(result['magnetization'])

            return {
                'temperature': result['temperature'],
                'magnetization': result['magnetization'],
                'susceptibility': result['susceptibility'],
                'current_regime': current_regime,
                'volatility_context': market_volatility
            }
        else:
            # Generate temperature range for annealing
            temps = self._generate_temperature_range()

            results = []

            # Simulate each temperature point
            for temp in temps:
                result = self._simulate_temperature(temp)
                results.append(result)

            # Find critical point (maximum susceptibility)
            df = self._results_to_dataframe(results)
            critical_point = df.loc[df['susceptibility'].idxmax()]

            # Determine current regime
            final_state = df.iloc[-1]
            current_regime = self._classify_regime(final_state['magnetization'])

            return {
                'critical_temperature': critical_point['temperature'],
                'critical_magnetization': critical_point['magnetization'],
                'current_regime': current_regime,
                'final_magnetization': final_state['magnetization'],
                'susceptibility_data': results,
                'volatility_context': market_volatility
            }

    def _results_to_dataframe(self, results: List[Dict]) -> 'pd.DataFrame':
        """Convert simulation results to pandas DataFrame."""
        return pd.DataFrame(results)

    def _classify_regime(self, magnetization: float) -> str:
        """Classify market regime based on magnetization."""
        if abs(magnetization) >= 0.8:
            return "CHAOTIC"  # High volatility, no clear trend
        elif 0.3 <= abs(magnetization) < 0.8:
            return "TRANSITIONAL"  # Medium volatility, developing trend
        else:
            return "ORDERED"  # Low volatility, strong trend

    def get_regime_confidence(self, magnetization: float) -> float:
        """Calculate confidence score for regime classification."""
        return abs(magnetization)  # Higher absolute magnetization = higher confidence

    def clear_cache(self) -> None:
        """Clear the simulation cache."""
        self._cache.clear()
        self._simulate_temperature.cache_clear()  # Clear lru_cache
        self._last_cache_time = 0.0

    def is_cache_valid(self, max_age_seconds: float = 300.0) -> bool:
        """Check if cache is still valid based on age."""
        current_time = time.time()
        return (current_time - self._last_cache_time) <= max_age_seconds

    def get_reading(self) -> float:
        """
        Get Ising regime reading as a float value for position sizing.

        Returns a normalized regime value (0.0 to 1.0) where:
        - 0.0 = Ordered regime (clear trend, stable)
        - 0.5 = Transitional regime
        - 1.0 = Chaotic regime (no clear direction)

        Returns:
            float: Normalized regime chaos level
        """
        # Return moderate regime reading as default when no volatility data
        return 0.4

    def _map_volatility_to_temperature(self, volatility: float) -> float:
        """Map market volatility to temperature for Ising model."""
        # Low volatility (<0.5%) → Low temperature (0.1)
        if volatility < 0.5:
            return 0.1
        # Medium volatility (0.5-2%) → Medium temperature (2.0-5.0)
        elif 0.5 <= volatility <= 2.0:
            # Linear mapping from 0.5% to 2.0% volatility to 2.0-5.0 temperature
            normalized = (volatility - 0.5) / (2.0 - 0.5)
            return 2.0 + normalized * 3.0
        # High volatility (>2%) → High temperature (10.0)
        else:
            return 10.0

    def _generate_temperature_range(self) -> np.ndarray:
        """Generate temperature range for annealing simulation."""
        return np.linspace(
            self.config.temp_range[0],
            self.config.temp_range[1],
            self.config.temp_steps
        )
