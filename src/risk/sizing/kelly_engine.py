"""
Physics-Aware Kelly Engine for Enhanced Kelly Position Sizing.

Implements Kelly criterion with econophysics-based risk adjustments:
- Base Kelly: f* = (p*b - q) / b
- Half-Kelly: f_base = f* * 0.5
- Physics multipliers from Lyapunov, Ising, and Eigenvalue indicators
- Weakest link aggregation: M_physics = min(P_λ, P_χ, P_E)

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

import logging
from typing import Optional

from src.risk.models.market_physics import MarketPhysics
from src.risk.models.strategy_performance import StrategyPerformance
from src.risk.models.sizing_recommendation import SizingRecommendation
from src.risk.sizing.monte_carlo_validator import MonteCarloValidator


logger = logging.getLogger(__name__)


class PhysicsAwareKellyEngine:
    """
    Physics-Aware Kelly Engine for position sizing.

    Combines traditional Kelly criterion with econophysics indicators
    for dynamic risk adjustment based on market regime.

    Formula:
        f* = (p*b - q) / b
        f_base = f* * 0.5
        M_physics = min(P_λ, P_χ, P_E)
        final_risk_pct = f_base * M_physics

    Where:
        p = win_rate
        b = payoff_ratio (avg_win / avg_loss)
        q = 1 - p
        P_λ = Lyapunov multiplier
        P_χ = Ising multiplier
        P_E = Eigenvalue multiplier

    Example:
        >>> engine = PhysicsAwareKellyEngine()
        >>> perf = StrategyPerformance(win_rate=0.55, avg_win=100, avg_loss=50, total_trades=100)
        >>> physics = MarketPhysics(lyapunov_exponent=-0.5, ising_susceptibility=0.5, ...)
        >>> result = engine.calculate_position_size(perf, physics)
        >>> print(f"Risk: {result.final_risk_pct:.2%}")
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,
        enable_monte_carlo: bool = True,
        monte_carlo_threshold: float = 0.005
    ):
        """
        Initialize the Physics-Aware Kelly Engine.

        Args:
            kelly_fraction: Fraction of full Kelly to use (default: 0.5 for half-Kelly)
            enable_monte_carlo: Whether to enable Monte Carlo validation
            monte_carlo_threshold: Minimum risk percentage to trigger MC validation (default: 0.5%)

        Raises:
            ValueError: If parameters are invalid
        """
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(f"kelly_fraction must be between 0 and 1, got {kelly_fraction}")
        if not 0 < monte_carlo_threshold <= 1.0:
            raise ValueError(f"monte_carlo_threshold must be between 0 and 1, got {monte_carlo_threshold}")

        self.kelly_fraction = kelly_fraction
        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_threshold = monte_carlo_threshold

        logger.info(
            "PhysicsAwareKellyEngine initialized",
            extra={
                "kelly_fraction": kelly_fraction,
                "enable_monte_carlo": enable_monte_carlo,
                "monte_carlo_threshold": monte_carlo_threshold
            }
        )

    def calculate_position_size(
        self,
        perf: StrategyPerformance,
        physics: MarketPhysics,
        validator: Optional[MonteCarloValidator] = None
    ) -> SizingRecommendation:
        """
        Calculate position size using Physics-Aware Kelly criterion.

        Calculation steps:
        1. Calculate base Kelly: f* = (p*b - q) / b
        2. Apply Kelly fraction: f_base = f* * kelly_fraction
        3. Calculate physics multipliers:
           - P_λ = max(0, 1.0 - (2.0 × λ))
           - P_χ = 0.5 if χ > 0.8 else 1.0
           - P_E = min(1.0, 1.5 / λ_max) if λ_max > 1.5 else 1.0
        4. Aggregate using weakest link: M_physics = min(P_λ, P_χ, P_E)
        5. Apply physics adjustment: final_risk_pct = f_base * M_physics
        6. Optional Monte Carlo validation if final_risk_pct > threshold

        Args:
            perf: Strategy performance metrics (win_rate, avg_win, avg_loss)
            physics: Market physics indicators (Lyapunov, Ising, Eigenvalue)
            validator: Optional Monte Carlo validator for risk assessment

        Returns:
            SizingRecommendation with:
            - raw_kelly: Base Kelly fraction
            - physics_multiplier: Aggregated multiplier M_physics
            - final_risk_pct: Final risk percentage
            - position_size_lots: 0.0 (must be calculated separately with stop loss)
            - adjustments_applied: List of calculation steps
            - validation_passed: True if MC validation passed or not run

        Raises:
            ValueError: If inputs are invalid
        """
        adjustments = []

        # Step 1: Calculate payoff ratio (b)
        b = perf.payoff_ratio
        adjustments.append(f"Payoff ratio (b): {b:.2f}")

        # Step 2: Calculate base Kelly: f* = (p*b - q) / b
        p = perf.win_rate
        q = 1.0 - p
        f_star = (p * b - q) / b
        adjustments.append(f"Base Kelly (f*): {f_star:.4f} ({f_star*100:.2f}%)")

        # Handle negative Kelly (no edge)
        if f_star <= 0:
            logger.warning(
                "Negative Kelly detected - no edge",
                extra={
                    "win_rate": p,
                    "payoff_ratio": b,
                    "kelly": f_star
                }
            )
            adjustments.append("Negative expectancy - returning zero position")
            return SizingRecommendation(
                raw_kelly=0.0,
                physics_multiplier=0.0,
                final_risk_pct=0.0,
                position_size_lots=0.0,
                constraint_source="negative_expectancy",
                validation_passed=False,
                adjustments_applied=adjustments
            )

        # Step 3: Apply half-Kelly (or configured fraction)
        f_base = f_star * self.kelly_fraction
        adjustments.append(f"Half-Kelly (f_base = f* × {self.kelly_fraction}): {f_base:.4f} ({f_base*100:.2f}%)")

        # Step 4: Calculate Lyapunov multiplier
        # P_λ = max(0, 1.0 - (2.0 × λ))
        lyapunov_lambda = physics.lyapunov_exponent
        p_lambda = max(0.0, 1.0 - (2.0 * lyapunov_lambda))
        adjustments.append(f"Lyapunov multiplier (P_λ = max(0, 1 - 2×{lyapunov_lambda:.2f})): {p_lambda:.2f}")

        # Step 5: Calculate Ising multiplier
        # P_χ = 0.5 if χ > 0.8 else 1.0
        ising_chi = physics.ising_susceptibility
        p_chi = 0.5 if ising_chi > 0.8 else 1.0
        adjustments.append(f"Ising multiplier (P_χ = 0.5 if χ={ising_chi:.2f} > 0.8): {p_chi:.2f}")

        # Step 6: Calculate Eigenvalue multiplier
        # P_E = min(1.0, 1.5 / λ_max) if λ_max > 1.5 else 1.0
        if physics.rmt_max_eigenvalue is not None:
            lambda_max = physics.rmt_max_eigenvalue
            if lambda_max > 1.5:
                p_eigen = min(1.0, 1.5 / lambda_max)
                adjustments.append(f"Eigen multiplier (P_E = min(1, 1.5/{lambda_max:.2f})): {p_eigen:.2f}")
            else:
                p_eigen = 1.0
                adjustments.append(f"Eigen multiplier (P_E = 1.0 since λ_max={lambda_max:.2f} ≤ 1.5): {p_eigen:.2f}")
        else:
            # No RMT data available - assume no reduction
            p_eigen = 1.0
            adjustments.append("Eigen multiplier (P_E = 1.0, no RMT data): {p_eigen:.2f}")

        # Step 7: Weakest link aggregation
        # M_physics = min(P_λ, P_χ, P_E)
        m_physics = min(p_lambda, p_chi, p_eigen)
        adjustments.append(f"Weakest link (M_physics = min({p_lambda:.2f}, {p_chi:.2f}, {p_eigen:.2f})): {m_physics:.2f}")

        # Step 8: Apply physics adjustment
        final_risk_pct = f_base * m_physics
        adjustments.append(f"Final risk (f_base × M_physics = {f_base:.4f} × {m_physics:.2f}): {final_risk_pct:.4f} ({final_risk_pct*100:.2f}%)")

        # Step 8.5: Apply hard cap at 10% (conservative limit)
        max_risk_cap = 0.10
        if final_risk_pct > max_risk_cap:
            final_risk_pct = max_risk_cap
            adjustments.append(f"Risk capped at {max_risk_cap*100:.1f}%: {final_risk_pct:.4f}")

        # Step 9: Optional Monte Carlo validation
        validation_passed = True
        if self.enable_monte_carlo and final_risk_pct > self.monte_carlo_threshold:
            if validator is None:
                logger.warning(
                    "Monte Carlo validation requested but no validator provided - skipping",
                    extra={"risk_pct": final_risk_pct}
                )
                adjustments.append(f"MC validation skipped (no validator, risk={final_risk_pct:.2%})")
            else:
                adjustments.append(f"Running MC validation (risk={final_risk_pct:.2%} > {self.monte_carlo_threshold:.2%})")
                try:
                    mc_result = validator.validate_risk(perf, risk_pct=final_risk_pct, runs=1000)
                    validation_passed = mc_result.passed

                    if not mc_result.passed:
                        # Apply Monte Carlo adjustment
                        final_risk_pct = final_risk_pct * mc_result.adjusted_risk
                        adjustments.append(
                            f"MC validation failed (ruin={mc_result.risk_of_ruin:.2%}) - "
                            f"adjusting by ×{mc_result.adjusted_risk:.2f}"
                        )
                    else:
                        adjustments.append(f"MC validation passed (ruin={mc_result.risk_of_ruin:.2%})")

                    logger.info(
                        "Monte Carlo validation complete",
                        extra={
                            "risk_of_ruin": mc_result.risk_of_ruin,
                            "passed": mc_result.passed,
                            "adjusted_risk_pct": final_risk_pct
                        }
                    )
                except Exception as e:
                    logger.error(
                        "Monte Carlo validation failed",
                        extra={"error": str(e)},
                        exc_info=True
                    )
                    adjustments.append(f"MC validation error: {str(e)} - proceeding without adjustment")
        elif self.enable_monte_carlo:
            adjustments.append(f"MC validation not needed (risk={final_risk_pct:.2%} ≤ {self.monte_carlo_threshold:.2%})")

        # Determine constraint source
        constraint_source = None
        if m_physics < 1.0:
            if p_lambda == m_physics:
                constraint_source = "physics_lyapunov"
            elif p_chi == m_physics:
                constraint_source = "physics_ising"
            elif p_eigen == m_physics:
                constraint_source = "physics_eigen"

        # Create recommendation
        # Note: position_size_lots requires stop_loss_pips and pip_value, so set to 0.0 here
        # The caller can calculate lots as: position_size_lots = (account_balance * final_risk_pct) / (stop_loss_pips * pip_value)
        recommendation = SizingRecommendation(
            raw_kelly=f_base,
            physics_multiplier=m_physics,
            final_risk_pct=final_risk_pct,
            position_size_lots=0.0,  # To be calculated by caller with stop loss
            constraint_source=constraint_source,
            validation_passed=validation_passed,
            adjustments_applied=adjustments
        )

        logger.info(
            "Physics-Aware Kelly calculation complete",
            extra={
                "raw_kelly": f_base,
                "physics_multiplier": m_physics,
                "final_risk_pct": final_risk_pct,
                "constraint_source": constraint_source,
                "validation_passed": validation_passed
            }
        )

        return recommendation

    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate raw Kelly fraction without adjustments.

        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade profit
            avg_loss: Average losing trade loss (positive)

        Returns:
            Kelly fraction f* = (p*b - q) / b

        Raises:
            ValueError: If inputs are invalid
        """
        if not 0 < win_rate < 1:
            raise ValueError(f"win_rate must be between 0 and 1 (exclusive), got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")

        b = avg_win / avg_loss  # Payoff ratio
        p = win_rate
        q = 1.0 - p

        return (p * b - q) / b

    def calculate_lyapunov_multiplier(self, lyapunov_exponent: float) -> float:
        """
        Calculate Lyapunov physics multiplier.

        Formula: P_λ = max(0, 1.0 - (2.0 × λ))

        Args:
            lyapunov_exponent: Lyapunov exponent value

        Returns:
            Multiplier from 0 to 1
        """
        return max(0.0, 1.0 - (2.0 * lyapunov_exponent))

    def calculate_ising_multiplier(self, ising_susceptibility: float) -> float:
        """
        Calculate Ising physics multiplier.

        Formula: P_χ = 0.5 if χ > 0.8 else 1.0

        Args:
            ising_susceptibility: Ising susceptibility value

        Returns:
            0.5 if high susceptibility (> 0.8), else 1.0
        """
        return 0.5 if ising_susceptibility > 0.8 else 1.0

    def calculate_eigen_multiplier(
        self,
        max_eigenvalue: Optional[float]
    ) -> float:
        """
        Calculate Eigenvalue physics multiplier.

        Formula: P_E = min(1.0, 1.5 / λ_max) if λ_max > 1.5 else 1.0

        Args:
            max_eigenvalue: Maximum eigenvalue from correlation matrix

        Returns:
            Multiplier from 0 to 1
        """
        if max_eigenvalue is None:
            # No RMT data - assume no reduction
            return 1.0

        if max_eigenvalue > 1.5:
            return min(1.0, 1.5 / max_eigenvalue)
        else:
            return 1.0
