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


@dataclass
class RegimePersistenceTimer:
    """Tracks regime persistence before confirmation to prevent whipsaw false flips.

    Before any Layer 2 action is triggered by a regime shift, the new regime
    must persist for N consecutive bars. Default: 3 M5 bars (15 min) for scalping,
    2 H1 bars (2 hours) for ORB.
    """
    current_regime: str = "ORDERED"
    persistence_counter: int = 0
    persistence_target: int = 3
    last_regime: str = "ORDERED"
    is_confirmed: bool = True

    # Timeframe-specific persistence targets
    TARGETS: Dict[str, int] = None

    def __post_init__(self):
        if RegimePersistenceTimer.TARGETS is None:
            RegimePersistenceTimer.TARGETS = {"M5": 3, "H1": 2}

    def check_regime_change(self, new_regime: str, timeframe: str = "M5") -> Tuple[bool, str]:
        """
        Check if regime has changed and update persistence tracking.

        Args:
            new_regime: The newly observed regime from detect_regime()
            timeframe: "M5" (default, target 3 bars) or "H1" (target 2 bars)

        Returns:
            Tuple of (is_confirmed, current_regime):
            - is_confirmed: True only if regime has persisted for N bars
            - current_regime: The confirmed regime (same as new_regime if confirmed, else last confirmed regime)
        """
        self.persistence_target = self.TARGETS.get(timeframe, 3)

        if new_regime == self.last_regime:
            # Same regime observed — increment counter
            self.persistence_counter += 1
            if self.persistence_counter >= self.persistence_target:
                self.is_confirmed = True
                self.current_regime = new_regime
        else:
            # Regime changed — reset and start fresh
            self.persistence_counter = 1
            self.last_regime = new_regime
            self.is_confirmed = False
            # current_regime stays as the last confirmed regime

        return self.is_confirmed, self.current_regime

    def get_confirmed_regime(self, raw_regime: str, timeframe: str = "M5") -> str:
        """
        Convenience method to get the confirmed regime given a raw detection.

        Wraps check_regime_change. Returns the confirmed regime after persistence
        check, or the last confirmed regime if persistence not yet met.

        Args:
            raw_regime: Regime from detect_regime()
            timeframe: "M5" or "H1"

        Returns:
            The confirmed regime string
        """
        _, confirmed = self.check_regime_change(raw_regime, timeframe)
        return confirmed


class IsingRegimeSensor:
    """Ising Model-based regime detector for market phase transitions."""

    def __init__(self, config: Optional[IsingSensorConfig] = None):
        self.config = config or IsingSensorConfig()
        self._cache: Dict[float, Dict] = {}
        self._last_cache_time: float = 0.0
        self._persistence_timer: RegimePersistenceTimer = RegimePersistenceTimer()

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

    def get_confirmed_regime(
        self,
        market_volatility: Optional[float] = None,
        timeframe: str = "M5"
    ) -> Dict:
        """
        Detect regime and apply persistence filtering to prevent whipsaw false flips.

        This wraps detect_regime() with the RegimePersistenceTimer. A newly detected
        regime is only confirmed after N consecutive bars (3 for M5, 2 for H1).
        Until confirmed, the last confirmed regime is returned.

        Args:
            market_volatility: Optional market volatility context (maps to temperature)
            timeframe: "M5" (default) or "H1" — controls persistence target

        Returns:
            Dictionary with regime detection results plus confirmed regime info
        """
        raw_result = self.detect_regime(market_volatility)
        raw_regime = raw_result.get('current_regime', 'ORDERED')

        is_confirmed, confirmed_regime = self._persistence_timer.check_regime_change(
            raw_regime, timeframe
        )

        raw_result['is_regime_confirmed'] = is_confirmed
        raw_result['confirmed_regime'] = confirmed_regime
        raw_result['persistence_counter'] = self._persistence_timer.persistence_counter
        raw_result['persistence_target'] = self._persistence_timer.persistence_target
        raw_result['persistence_timeframe'] = timeframe

        return raw_result

    def clear_cache(self) -> None:
        """Clear the simulation cache."""
        self._cache.clear()
        self._simulate_temperature.cache_clear()  # Clear lru_cache
        self._last_cache_time = 0.0
        # Reset persistence timer on cache clear — new context may have new regime
        self._persistence_timer = RegimePersistenceTimer()

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
