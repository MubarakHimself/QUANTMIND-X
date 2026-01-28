"""
Portfolio Kelly Scaler

Manages position sizing across multiple bots to prevent portfolio over-leverage.

Key insight: 10 bots × 1% each = 10% total daily risk (dangerous!)
This module scales down individual bot positions to maintain safe portfolio risk.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class PortfolioStatus:
    """Portfolio risk status report."""
    total_raw_risk: float        # Sum of all bot Kelly fractions
    total_scaled_risk: float     # After portfolio scaling
    risk_utilization: float      # Percentage of max allowed
    bot_count: int
    status: str                  # 'safe', 'caution', 'danger'
    scale_factor: float          # How much each bot was scaled
    recommendation: str


class PortfolioKellyScaler:
    """
    Manages position sizing across multiple bots to prevent portfolio over-leverage.
    
    When multiple EAs trade the same account, their individual Kelly fractions
    can sum to dangerous levels. This scaler automatically adjusts each bot's
    position size to keep total portfolio risk within safe bounds.
    
    Example:
        If 5 bots each want 1.5% risk:
        - Raw total = 7.5%
        - If max_portfolio_risk = 3%
        - Scale factor = 3% / 7.5% = 0.4
        - Each bot gets: 1.5% × 0.4 = 0.6% (scaled)
    """

    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.03,  # 3% total daily risk target
        correlation_adjustment: float = 1.5    # Additional safety for correlated bots
    ):
        """
        Initialize the portfolio scaler.
        
        Args:
            max_portfolio_risk_pct: Maximum total risk across all bots
            correlation_adjustment: Penalty multiplier for correlated positions
        """
        self.max_portfolio_risk = max_portfolio_risk_pct
        self.correlation_adjustment = correlation_adjustment

    def scale_bot_positions(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]:
        """
        Scale down position sizes when multiple bots are active.

        Args:
            bot_kelly_factors: {bot_id: kelly_f} for each active bot
            bot_correlations: {(bot1, bot2): correlation} optional correlation matrix

        Returns:
            {bot_id: scaled_kelly_f} maintaining proportional allocation
        """
        if not bot_kelly_factors:
            return {}

        raw_total_risk = sum(bot_kelly_factors.values())

        # If total risk is within limits, no scaling needed
        if raw_total_risk <= self.max_portfolio_risk:
            return bot_kelly_factors.copy()

        # Calculate base scale factor
        scale_factor = self.max_portfolio_risk / raw_total_risk

        # Additional adjustment for correlated bots
        if bot_correlations:
            avg_correlation = self._calculate_average_correlation(bot_correlations)
            # More correlation = less diversification = more scaling needed
            # correlation_penalty ranges from 1.0 (no correlation) to 1.5 (full correlation)
            correlation_penalty = 1.0 + (avg_correlation * (self.correlation_adjustment - 1.0))
            scale_factor /= correlation_penalty

        # Scale all bots proportionally
        return {
            bot_id: kelly_f * scale_factor
            for bot_id, kelly_f in bot_kelly_factors.items()
        }

    def _calculate_average_correlation(
        self,
        correlations: Dict[Tuple[str, str], float]
    ) -> float:
        """Calculate average correlation from correlation matrix."""
        if not correlations:
            return 0.0
        return sum(abs(c) for c in correlations.values()) / len(correlations)

    def get_portfolio_status(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> PortfolioStatus:
        """
        Return portfolio risk metrics for monitoring.

        Returns:
            PortfolioStatus with detailed risk analysis
        """
        if not bot_kelly_factors:
            return PortfolioStatus(
                total_raw_risk=0.0,
                total_scaled_risk=0.0,
                risk_utilization=0.0,
                bot_count=0,
                status='safe',
                scale_factor=1.0,
                recommendation='No active bots'
            )

        raw_total = sum(bot_kelly_factors.values())
        scaled = self.scale_bot_positions(bot_kelly_factors, bot_correlations)
        scaled_total = sum(scaled.values())

        # Calculate utilization
        utilization = scaled_total / self.max_portfolio_risk

        # Determine status
        if utilization < 0.7:
            status = 'safe'
            recommendation = 'Portfolio risk is healthy'
        elif utilization < 0.9:
            status = 'caution'
            recommendation = 'Approaching risk limit - monitor closely'
        else:
            status = 'danger'
            recommendation = 'At risk limit - consider reducing bot count'

        # Calculate effective scale factor
        scale_factor = scaled_total / raw_total if raw_total > 0 else 1.0

        return PortfolioStatus(
            total_raw_risk=raw_total,
            total_scaled_risk=scaled_total,
            risk_utilization=utilization,
            bot_count=len(bot_kelly_factors),
            status=status,
            scale_factor=scale_factor,
            recommendation=recommendation
        )

    def recommend_bot_limit(self, target_risk_per_bot: float = 0.01) -> int:
        """
        Recommend maximum number of bots for given individual risk.
        
        Args:
            target_risk_per_bot: Desired risk per bot (e.g., 0.01 = 1%)
            
        Returns:
            Maximum recommended bot count
        """
        if target_risk_per_bot <= 0:
            return 0
        return int(self.max_portfolio_risk / target_risk_per_bot)

    def allocate_risk_equally(
        self,
        bot_ids: List[str],
        total_risk_budget: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Allocate risk equally across all bots.
        
        Args:
            bot_ids: List of active bot identifiers
            total_risk_budget: Total risk to allocate (defaults to max_portfolio_risk)
            
        Returns:
            {bot_id: kelly_f} with equal allocations
        """
        if not bot_ids:
            return {}

        budget = total_risk_budget or self.max_portfolio_risk
        risk_per_bot = budget / len(bot_ids)

        return {bot_id: risk_per_bot for bot_id in bot_ids}

    def allocate_risk_by_performance(
        self,
        bot_performance: Dict[str, float],
        total_risk_budget: Optional[float] = None,
        min_allocation: float = 0.005  # Minimum 0.5% per bot
    ) -> Dict[str, float]:
        """
        Allocate risk based on bot performance (higher performers get more risk).
        
        Args:
            bot_performance: {bot_id: sharpe_ratio} or other performance metric
            total_risk_budget: Total risk to allocate
            min_allocation: Minimum risk allocation per bot
            
        Returns:
            {bot_id: kelly_f} weighted by performance, preserving minimums.
        """
        if not bot_performance:
            return {}

        budget = total_risk_budget or self.max_portfolio_risk
        
        # Determine number of bots and total minimum required
        n_bots = len(bot_performance)
        total_min_required = min_allocation * n_bots
        
        # Separate positive performers
        positive_bots = {k: v for k, v in bot_performance.items() if v > 0}
        
        # Case 1: All bots negative or budget too small for all minimums
        if not positive_bots or total_min_required >= budget:
            if total_min_required <= budget:
                return {bot_id: min_allocation for bot_id in bot_performance}
            else:
                # Scale min_allocation to fit budget
                scale = budget / total_min_required if total_min_required > 0 else 1.0
                return {bot_id: min_allocation * scale for bot_id in bot_performance}

        # Case 2: Distribute remaining budget among positive performers
        # Start everyone at min_allocation
        allocations = {bot_id: min_allocation for bot_id in bot_performance}
        remaining_budget = budget - total_min_required
        
        total_positive_perf = sum(positive_bots.values())
        if total_positive_perf > 0:
            for bot_id, perf in positive_bots.items():
                weight = perf / total_positive_perf
                allocations[bot_id] += remaining_budget * weight
                
        # Final safety check for float precision
        total = sum(allocations.values())
        if total > budget:
            scale = budget / total
            for k in allocations:
                allocations[k] *= scale
                
        return allocations
