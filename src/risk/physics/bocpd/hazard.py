"""Hazard functions for BOCPD.

Hazard functions determine the prior probability of a changepoint at each time step.
H(t) = P(changepoint at time t | run_length = t)
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

logger = logging.getLogger(__name__)


class BaseHazard(ABC):
    """Abstract base class for hazard functions."""

    @abstractmethod
    def __call__(self, run_length: int) -> float:
        """Return P(changepoint | run_length).

        Args:
            run_length: Current run length (bars since last changepoint)

        Returns:
            Hazard rate (probability of changepoint) in [0, 1]
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Return hazard parameters as dictionary."""
        pass


class ConstantHazard(BaseHazard):
    """Constant hazard rate: H(tau) = 1/lambda.

    This is a geometric prior on run length.
    lambda = expected run length (average bars between regime changes).

    For forex M5 data, typical lambda = 200-500 (roughly 1-3 trading days).
    lambda = 250 means on average a changepoint every ~250 M5 bars (~20 hours).

    Properties:
        - Simple and tractable
        - Assumes memoryless arrival of changepoints (Poisson process)
        - Constant hazard is independent of run length
    """

    def __init__(self, lam: float = 250.0):
        """Initialize constant hazard.

        Args:
            lam: Expected run length (must be > 0)
        """
        if lam <= 0:
            raise ValueError(f"Lambda must be > 0, got {lam}")
        self.lam = float(lam)

    def __call__(self, run_length: int) -> float:
        """Return constant hazard rate 1/lambda."""
        return 1.0 / self.lam

    def get_params(self) -> Dict:
        """Return hazard parameters."""
        return {"type": "ConstantHazard", "lambda": self.lam}


class LogisticHazard(BaseHazard):
    """Logistic hazard: hazard increases with run length.

    Longer runs become increasingly likely to end (not memoryless).
    H(tau) = 1 / (1 + exp(-k*(tau - tau_0)))

    Parameters:
        tau_0: Inflection point (where hazard = 0.5)
        k: Steepness/slope parameter (higher k = steeper transition)

    Example: tau_0=200, k=0.02 means:
        - At tau=0: hazard ≈ 0.0000135
        - At tau=200: hazard = 0.5
        - At tau=400: hazard ≈ 0.99998

    This models the intuition that markets stay in regime for a while,
    then become increasingly unstable.
    """

    def __init__(self, tau_0: float = 200.0, k: float = 0.02):
        """Initialize logistic hazard.

        Args:
            tau_0: Inflection point (run length at 50% hazard)
            k: Steepness parameter (must be > 0)
        """
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        self.tau_0 = float(tau_0)
        self.k = float(k)

    def __call__(self, run_length: int) -> float:
        """Return logistic hazard rate."""
        return 1.0 / (1.0 + np.exp(-self.k * (run_length - self.tau_0)))

    def get_params(self) -> Dict:
        """Return hazard parameters."""
        return {"type": "LogisticHazard", "tau_0": self.tau_0, "k": self.k}
