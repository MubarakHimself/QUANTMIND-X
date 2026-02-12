"""
Portfolio Kelly Scaler

Handles portfolio-level position scaling to prevent over-leverage
when multiple bots are trading simultaneously.
"""

import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PortfolioStatus:
    """Portfolio risk status."""
    total_raw_risk: float
    total_scaled_risk: float
    risk_utilization: float
    bot_count: int
    status: str
    scale_factor: float
    recommendation: str


class PortfolioKellyScaler:
    """
    Scales Kelly factors across multiple bots to manage portfolio risk.

    This class:
    1. Tracks total portfolio risk across all bots
    2. Applies correlation-based scaling
    3. Enforces maximum portfolio risk limits
    4. Provides portfolio status monitoring

    Example:
        >>> scaler = PortfolioKellyScaler()
        >>> factors = scaler.scale_bot_positions(
        ...     {"bot1": 0.02, "bot2": 0.03},
        ...     {"bot1-bot2": 0.8}
        ... )
    """

    def __init__(
        self,
        max_portfolio_risk: float = 0.15,
        correlation_threshold: float = 0.7
    ):
        """
        Initialize portfolio scaler.

        Args:
            max_portfolio_risk: Maximum total portfolio risk (default 15%)
            correlation_threshold: Threshold for high correlation (default 0.7)
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_threshold = correlation_threshold

    def scale_bot_positions(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]:
        """
        Scale bot Kelly factors based on portfolio risk.

        Args:
            bot_kelly_factors: Dictionary of bot_id -> Kelly factor
            bot_correlations: Optional pairwise correlations between bots

        Returns:
            Dictionary of bot_id -> scaled Kelly factor
        """
        if not bot_kelly_factors:
            return {}

        # Calculate total raw risk
        total_raw_risk = sum(bot_kelly_factors.values())

        # Start with no scaling
        scaled_factors = bot_kelly_factors.copy()
        scale_factor = 1.0

        # Apply portfolio scaling if total risk exceeds max
        if total_raw_risk > self.max_portfolio_risk:
            scale_factor = self.max_portfolio_risk / total_raw_risk
            for bot_id in scaled_factors:
                scaled_factors[bot_id] *= scale_factor

        # Apply correlation penalty if needed
        if bot_correlations:
            correlation_penalty = self._calculate_correlation_penalty(
                bot_correlations
            )
            if correlation_penalty < 1.0:
                for bot_id in scaled_factors:
                    scaled_factors[bot_id] *= correlation_penalty
                scale_factor *= correlation_penalty

        logger.info(
            "Portfolio scaling applied",
            extra={
                "total_raw_risk": total_raw_risk,
                "scale_factor": scale_factor,
                "bot_count": len(bot_kelly_factors)
            }
        )

        return scaled_factors

    def _calculate_correlation_penalty(
        self,
        bot_correlations: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate penalty based on pairwise correlations."""
        if not bot_correlations:
            return 1.0

        # Count high correlations
        high_corr_count = sum(
            1 for corr in bot_correlations.values()
            if abs(corr) > self.correlation_threshold
        )

        if high_corr_count == 0:
            return 1.0

        # Apply scaling based on number of high correlations
        # More high correlations = more scaling
        penalty = max(0.5, 1.0 - (high_corr_count * 0.1))
        return penalty

    def get_portfolio_status(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> PortfolioStatus:
        """
        Get current portfolio risk status.

        Args:
            bot_kelly_factors: Dictionary of bot_id -> Kelly factor
            bot_correlations: Optional pairwise correlations

        Returns:
            PortfolioStatus object with current risk metrics
        """
        total_raw_risk = sum(bot_kelly_factors.values())
        scaled_factors = self.scale_bot_positions(
            bot_kelly_factors, bot_correlations
        )
        total_scaled_risk = sum(scaled_factors.values())

        risk_utilization = total_scaled_risk / self.max_portfolio_risk

        # Determine status
        if risk_utilization > 0.9:
            status = "CRITICAL"
            recommendation = "Reduce position sizes immediately"
        elif risk_utilization > 0.7:
            status = "WARNING"
            recommendation = "Consider reducing exposure"
        elif risk_utilization > 0.5:
            status = "MODERATE"
            recommendation = "Monitor portfolio risk"
        else:
            status = "HEALTHY"
            recommendation = "Portfolio risk within limits"

        scale_factor = total_scaled_risk / total_raw_risk if total_raw_risk > 0 else 1.0

        return PortfolioStatus(
            total_raw_risk=total_raw_risk,
            total_scaled_risk=total_scaled_risk,
            risk_utilization=risk_utilization,
            bot_count=len(bot_kelly_factors),
            status=status,
            scale_factor=scale_factor,
            recommendation=recommendation
        )
