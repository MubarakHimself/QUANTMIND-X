"""Observation models for BOCPD.

Observation models define the likelihood of data given a run length.
Uses conjugate priors (Normal-InverseGamma) for computational efficiency.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional
import numpy as np
from scipy import special, stats

logger = logging.getLogger(__name__)


class BaseObservation(ABC):
    """Abstract base class for observation models."""

    @abstractmethod
    def pdf(self, x: float) -> float:
        """Probability density of observation x under current sufficient stats.

        Args:
            x: Observation value

        Returns:
            Probability density (must be > 0)
        """
        pass

    @abstractmethod
    def update(self, x: float) -> None:
        """Update sufficient statistics with new observation.

        Args:
            x: New observation
        """
        pass

    @abstractmethod
    def reset(self) -> "BaseObservation":
        """Return a fresh copy with initial priors (for new run).

        Returns:
            New observation model with reset statistics
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict:
        """Return model parameters and current sufficient statistics."""
        pass


class GaussianObservation(BaseObservation):
    """Gaussian observation model with Normal-InverseGamma conjugate prior.

    Uses sufficient statistics to update in O(1):
        - n: number of observations
        - sum_x: sum of observations
        - sum_x2: sum of x^2

    Predictive distribution is Student-t (marginalizing over unknown variance).
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        kappa_0: float = 1.0,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
    ):
        """Initialize Gaussian observation model with conjugate prior.

        Args:
            mu_0: Prior mean
            kappa_0: Prior precision weight (how strong is prior on mean)
            alpha_0: Shape parameter for InverseGamma on variance
            beta_0: Scale parameter for InverseGamma on variance

        Prior interpretation:
            - (mu_0, kappa_0): N(mu_0, beta_0/kappa_0) on mean
            - (alpha_0, beta_0): InvGamma(alpha_0, beta_0) on variance
        """
        self.mu_0 = float(mu_0)
        self.kappa_0 = float(kappa_0)
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)

        # Sufficient statistics
        self.n = 0  # Count
        self.sum_x = 0.0  # Sum of x
        self.sum_x2 = 0.0  # Sum of x^2

    def update(self, x: float) -> None:
        """Update sufficient statistics with observation x.

        O(1) update: just increment counters.
        """
        x = float(x)
        if not np.isfinite(x):
            logger.warning(f"Non-finite observation {x}, skipping")
            return
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x

    def pdf(self, x: float) -> float:
        """Compute predictive probability using Student-t.

        When both mean and variance are unknown, the marginal
        predictive distribution is Student-t with parameters
        derived from the posterior.

        Returns:
            Probability density at x
        """
        x = float(x)
        if not np.isfinite(x):
            return 1e-10

        # Posterior parameters after n observations
        kappa_n = self.kappa_0 + self.n
        mu_n = (self.kappa_0 * self.mu_0 + self.sum_x) / kappa_n
        alpha_n = self.alpha_0 + self.n / 2.0
        beta_n = (
            self.beta_0
            + 0.5 * self.sum_x2
            + 0.5 * self.kappa_0 * self.mu_0**2
            - 0.5 * (self.kappa_0 * self.mu_0 + self.sum_x) ** 2 / kappa_n
        )

        # Avoid division by zero
        if beta_n <= 0:
            beta_n = 1e-10

        # Student-t distribution parameters
        # df = 2 * alpha_n
        # loc = mu_n
        # scale = sqrt(beta_n * (kappa_n + 1) / (alpha_n * kappa_n))
        df = 2.0 * alpha_n
        loc = mu_n
        scale = np.sqrt(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n))

        # Clamp scale to avoid numerical issues
        if scale < 1e-10:
            scale = 1e-10

        # Evaluate Student-t pdf
        try:
            return stats.t.pdf(x, df=df, loc=loc, scale=scale)
        except Exception as e:
            logger.warning(f"Error computing Student-t pdf: {e}")
            return 1e-10

    def reset(self) -> "GaussianObservation":
        """Return fresh Gaussian observation with same prior."""
        return GaussianObservation(
            mu_0=self.mu_0, kappa_0=self.kappa_0, alpha_0=self.alpha_0, beta_0=self.beta_0
        )

    def get_params(self) -> Dict:
        """Return model parameters and current statistics."""
        return {
            "type": "GaussianObservation",
            "mu_0": self.mu_0,
            "kappa_0": self.kappa_0,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "n": self.n,
            "sum_x": self.sum_x,
            "sum_x2": self.sum_x2,
        }


class StudentTObservation(BaseObservation):
    """Student-t observation model for heavy-tailed financial data.

    Uses Normal-InverseGamma prior like Gaussian, but explicitly
    models Student-t predictive to account for fat tails.
    Better for forex returns which have excess kurtosis.

    Parameters:
        mu_0, kappa_0, alpha_0, beta_0: Same as GaussianObservation
        df: Degrees of freedom (lower = fatter tails)
            df=5 is typical for currency pair returns
    """

    def __init__(
        self,
        mu_0: float = 0.0,
        kappa_0: float = 0.1,
        alpha_0: float = 2.0,
        beta_0: float = 1.0,
        df: float = 5.0,
    ):
        """Initialize Student-t observation model.

        Args:
            mu_0: Prior mean
            kappa_0: Prior precision weight
            alpha_0: InverseGamma shape
            beta_0: InverseGamma scale
            df: Degrees of freedom for Student-t (> 0)
        """
        self.mu_0 = float(mu_0)
        self.kappa_0 = float(kappa_0)
        self.alpha_0 = float(alpha_0)
        self.beta_0 = float(beta_0)
        self.df = float(df)

        # Sufficient statistics
        self.n = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0

    def update(self, x: float) -> None:
        """Update sufficient statistics with observation x."""
        x = float(x)
        if not np.isfinite(x):
            logger.warning(f"Non-finite observation {x}, skipping")
            return
        self.n += 1
        self.sum_x += x
        self.sum_x2 += x * x

    def pdf(self, x: float) -> float:
        """Compute predictive probability.

        For Student-t observation model, we use a robust estimate
        of variance that downweights outliers.

        Returns:
            Probability density at x
        """
        x = float(x)
        if not np.isfinite(x):
            return 1e-10

        if self.n == 0:
            # No data yet; return weakly informative density
            return stats.t.pdf(x, df=self.df, loc=self.mu_0, scale=1.0)

        # Posterior parameters
        kappa_n = self.kappa_0 + self.n
        mu_n = (self.kappa_0 * self.mu_0 + self.sum_x) / kappa_n

        # Robust variance estimate
        mean_x = self.sum_x / self.n
        var_x = (self.sum_x2 / self.n) - mean_x**2
        if var_x < 1e-10:
            var_x = 1e-10

        # Combine prior and observed variance
        alpha_n = self.alpha_0 + self.n / 2.0
        beta_n = (
            self.beta_0
            + 0.5 * self.n * var_x
            + 0.5 * self.kappa_0 * self.n * (mean_x - self.mu_0) ** 2 / kappa_n
        )

        if beta_n <= 0:
            beta_n = 1e-10

        # Student-t with posterior variance estimate
        df = 2.0 * alpha_n
        scale = np.sqrt(beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n))

        if scale < 1e-10:
            scale = 1e-10

        try:
            return stats.t.pdf(x, df=df, loc=mu_n, scale=scale)
        except Exception as e:
            logger.warning(f"Error computing Student-t pdf: {e}")
            return 1e-10

    def reset(self) -> "StudentTObservation":
        """Return fresh Student-t observation with same prior."""
        return StudentTObservation(
            mu_0=self.mu_0,
            kappa_0=self.kappa_0,
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            df=self.df,
        )

    def get_params(self) -> Dict:
        """Return model parameters and current statistics."""
        return {
            "type": "StudentTObservation",
            "mu_0": self.mu_0,
            "kappa_0": self.kappa_0,
            "alpha_0": self.alpha_0,
            "beta_0": self.beta_0,
            "df": self.df,
            "n": self.n,
            "sum_x": self.sum_x,
            "sum_x2": self.sum_x2,
        }
