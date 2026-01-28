"""
Physics-Aware Kelly Engine

Implements the physics-aware Kelly position sizing engine that combines:
1. Base Kelly formula: f* = (p*b - q) / b
2. Physics multipliers (Lyapunov, Ising, Eigen)
3. Weakest link aggregation: M_physics = min(P_λ, P_χ, P_E)

The formula: S = B × f_base × M_physics where M_physics = min(P_λ, P_χ, P_E)

Physics multipliers:
- P_λ (Lyapunov): 0.25 (EXTREME), 0.5 (HIGH), 0.75 (MODERATE), 1.0 (LOW)
- P_χ (Ising): Same scale based on susceptibility
- P_E (Eigen): Signal strength multiplier based on RMT

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""
import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from src.position_sizing.kelly_config import EnhancedKellyConfig
from src.risk.models.market_physics import MarketPhysics, RiskLevel


logger = logging.getLogger(__name__)


@dataclass
class PhysicsMultiplier:
    """Physics multiplier configuration for each indicator."""
    lyapunov: float = 1.0  # Lyapunov exponent multiplier
    ising: float = 1.0     # Ising model multiplier
    eigen: float = 1.0     # Eigenvalue/RMT multiplier


@dataclass
class PhysicsAwareKellyResult:
    """
    Result of Physics-Aware Kelly calculation with physics breakdown.

    Attributes:
        position_size: Final position size in lots (rounded to broker precision)
        base_kelly_f: Raw Kelly fraction before physics adjustments
        physics_multipliers: Applied physics multipliers
        aggregated_multiplier: Final physics multiplier (min of individual multipliers)
        risk_amount: Dollar amount at risk (account_balance × kelly_f)
        adjustments_applied: List of adjustment descriptions for audit trail
        status: Calculation status - 'calculated', 'fallback', or 'zero'
        physics_risk_level: Overall physics risk level
    """
    position_size: float
    base_kelly_f: float
    physics_multipliers: PhysicsMultiplier
    aggregated_multiplier: float
    risk_amount: float
    adjustments_applied: List[str] = field(default_factory=list)
    status: str = 'calculated'
    physics_risk_level: RiskLevel = RiskLevel.LOW

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "position_size": self.position_size,
            "base_kelly_fraction": self.base_kelly_f,
            "physics_multipliers": {
                "lyapunov": self.physics_multipliers.lyapunov,
                "ising": self.physics_multipliers.ising,
                "eigen": self.physics_multipliers.eigen
            },
            "aggregated_multiplier": self.aggregated_multiplier,
            "risk_amount": self.risk_amount,
            "adjustments": self.adjustments_applied,
            "status": self.status,
            "physics_risk_level": self.physics_risk_level.value
        }


class PhysicsAwareKellyEngine:
    """
    Physics-Aware Kelly Position Sizing Engine.

    Implements the physics-aware Kelly formula:
    S = B × f_base × M_physics where M_physics = min(P_λ, P_χ, P_E)

    Physics multipliers:
    - P_λ (Lyapunov): 0.25 (EXTREME), 0.5 (HIGH), 0.75 (MODERATE), 1.0 (LOW)
    - P_χ (Ising): Same scale based on susceptibility
    - P_E (Eigen): Signal strength multiplier based on RMT

    Attributes:
        config: EnhancedKellyConfig instance with all parameters
        physics_config: Physics multiplier configuration

    Example:
        >>> engine = PhysicsAwareKellyEngine()
        >>> result = engine.calculate(
        ...     account_balance=10000.0,
        ...     win_rate=0.55,
        ...     avg_win=400.0,
        ...     avg_loss=200.0,
        ...     market_physics=MarketPhysics(...),
        ...     stop_loss_pips=20.0,
        ...     pip_value=10.0
        ... )
        >>> print(f"Position: {result.position_size} lots with physics multiplier {result.aggregated_multiplier}")
    """

    def __init__(
        self,
        config: Optional[EnhancedKellyConfig] = None,
        physics_config: Optional[PhysicsMultiplier] = None
    ):
        """
        Initialize Physics-Aware Kelly Engine.

        Args:
            config: EnhancedKellyConfig instance. Uses defaults if None.
            physics_config: PhysicsMultiplier configuration. Uses defaults if None.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config or EnhancedKellyConfig()
        self.config.validate()
        self.physics_config = physics_config or PhysicsMultiplier()
        self._last_kelly_f: Optional[float] = None

        logger.info(
            "PhysicsAwareKellyEngine initialized",
            extra={
                "kelly_fraction": self.config.kelly_fraction,
                "max_risk_pct": self.config.max_risk_pct,
                "physics_multipliers": {
                    "lyapunov": self.physics_config.lyapunov,
                    "ising": self.physics_config.ising,
                    "eigen": self.physics_config.eigen
                }
            }
        )

    def calculate(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        market_physics: MarketPhysics,
        stop_loss_pips: float,
        pip_value: float = 10.0
    ) -> PhysicsAwareKellyResult:
        """
        Calculate optimal position size using Physics-Aware Kelly.

        The calculation follows these steps:

        1. **Validate Inputs**: Ensure all parameters are valid
        2. **Calculate Risk-Reward Ratio**: B = avg_win / avg_loss
        3. **Calculate Base Kelly**: f* = (B + 1)P - 1) / B
        4. **Apply Kelly Fraction**: Multiply by safety factor (default 50%)
        5. **Calculate Physics Multipliers**:
           - P_λ from Lyapunov exponent
           - P_χ from Ising susceptibility
           - P_E from RMT eigenvalue
        6. **Aggregate Multipliers**: M_physics = min(P_λ, P_χ, P_E)
        7. **Apply Physics Adjustment**: f_final = f_base × M_physics
        8. **Apply Hard Risk Cap**: Cap at maximum risk percentage
        9. **Calculate Risk Amount**: risk = balance × f_final
        10. **Calculate Position Size**: lots = risk / (SL × pip_value)
        11. **Round to Broker Precision**: Round down to lot step

        Args:
            account_balance: Current account balance in base currency
            win_rate: Historical win rate from 0 to 1
            avg_win: Average winning trade profit in currency
            avg_loss: Average losing trade loss in currency (positive)
            market_physics: MarketPhysics instance with physics indicators
            stop_loss_pips: Stop loss distance in pips/points
            pip_value: Value per pip for 1 standard lot (default $10)

        Returns:
            PhysicsAwareKellyResult containing:
            - position_size: Final position size in lots
            - base_kelly_f: Raw Kelly before adjustments
            - physics_multipliers: Applied physics multipliers
            - aggregated_multiplier: Final physics multiplier
            - risk_amount: Dollar amount at risk
            - adjustments_applied: List of calculation steps
            - status: 'calculated', 'fallback', or 'zero'
            - physics_risk_level: Overall physics risk level

        Raises:
            ValueError: If inputs are invalid
        """
        adjustments = []

        # Step 1: Input Validation
        self._validate_inputs(win_rate, avg_win, avg_loss, stop_loss_pips, market_physics)

        # Step 2: Calculate Risk-Reward Ratio (B)
        risk_reward_ratio = avg_win / avg_loss
        adjustments.append(f"R:R ratio = {risk_reward_ratio:.2f}")

        # Step 3: Calculate Base Kelly Formula
        # f = ((B + 1) × P - 1) / B
        base_kelly_f = ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio
        adjustments.append(f"Base Kelly = {base_kelly_f:.4f} ({base_kelly_f*100:.2f}%)")

        # Handle negative expectancy
        if base_kelly_f <= 0:
            adjustments.append("NEGATIVE EXPECTANCY - No edge detected")
            logger.warning(
                "Negative expectancy detected",
                extra={
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "base_kelly": base_kelly_f
                }
            )
            if self.config.allow_zero_position:
                return PhysicsAwareKellyResult(
                    position_size=0.0,
                    base_kelly_f=base_kelly_f,
                    physics_multipliers=PhysicsMultiplier(),
                    aggregated_multiplier=0.0,
                    risk_amount=0.0,
                    adjustments_applied=adjustments,
                    status='zero',
                    physics_risk_level=market_physics.risk_level()
                )
            else:
                # Use fallback risk
                base_kelly_f = self.config.fallback_risk_pct
                adjustments.append(f"Using fallback: {base_kelly_f*100:.1f}%")

        kelly_f = base_kelly_f

        # Step 4: Apply Kelly Fraction
        kelly_f = kelly_f * self.config.kelly_fraction
        adjustments.append(f"Kelly fraction (×{self.config.kelly_fraction}): {kelly_f:.4f}")

        # Step 5: Calculate Physics Multipliers
        physics_multipliers = self._calculate_physics_multipliers(market_physics)
        adjustments.append(
            f"Physics multipliers: Lyapunov={physics_multipliers.lyapunov:.2f}, "
            f"Ising={physics_multipliers.ising:.2f}, "
            f"Eigen={physics_multipliers.eigen:.2f}"
        )

        # Step 6: Aggregate Multipliers (Weakest Link Principle)
        aggregated_multiplier = min(
            physics_multipliers.lyapunov,
            physics_multipliers.ising,
            physics_multipliers.eigen
        )
        adjustments.append(f"Aggregated multiplier (min): {aggregated_multiplier:.2f}")

        # Step 7: Apply Physics Adjustment
        kelly_f = kelly_f * aggregated_multiplier
        adjustments.append(f"Physics adjustment (×{aggregated_multiplier}): {kelly_f:.4f}")

        # Step 8: Apply Hard Risk Cap
        if kelly_f > self.config.max_risk_pct:
            kelly_f = self.config.max_risk_pct
            adjustments.append(f"Risk cap (capped at {self.config.max_risk_pct*100:.1f}%): {kelly_f:.4f}")
        else:
            adjustments.append(f"Risk cap (no cap needed): {kelly_f:.4f}")

        # Store for portfolio scaling
        self._last_kelly_f = kelly_f

        # Step 9: Calculate Risk Amount
        risk_amount = account_balance * kelly_f
        adjustments.append(f"Risk amount = ${risk_amount:.2f}")

        # Step 10: Calculate Position Size
        risk_per_lot = stop_loss_pips * pip_value
        position_size = risk_amount / risk_per_lot
        adjustments.append(f"Raw position = {position_size:.4f} lots")

        # Step 11: Round to Broker Precision
        position_size = self._round_to_lot_size(position_size)
        adjustments.append(f"Rounded position = {position_size:.2f} lots")

        logger.info(
            "Physics-Aware Kelly calculation complete",
            extra={
                "position_size": position_size,
                "kelly_fraction": kelly_f,
                "risk_amount": risk_amount,
                "account_balance": account_balance,
                "physics_risk_level": market_physics.risk_level().value,
                "aggregated_multiplier": aggregated_multiplier
            }
        )

        return PhysicsAwareKellyResult(
            position_size=position_size,
            base_kelly_f=base_kelly_f,
            physics_multipliers=physics_multipliers,
            aggregated_multiplier=aggregated_multiplier,
            risk_amount=risk_amount,
            adjustments_applied=adjustments,
            status='calculated',
            physics_risk_level=market_physics.risk_level()
        )

    def _calculate_physics_multipliers(self, market_physics: MarketPhysics) -> PhysicsMultiplier:
        """
        Calculate individual physics multipliers based on market physics indicators.

        Multiplier logic:
        - EXTREME: 0.25 (75% reduction)
        - HIGH: 0.5 (50% reduction)
        - MODERATE: 0.75 (25% reduction)
        - LOW: 1.0 (no reduction)

        Args:
            market_physics: MarketPhysics instance with indicators

        Returns:
            PhysicsMultiplier with individual multipliers
        """
        # Get individual risk levels
        lyapunov_risk = self._get_risk_level_from_lyapunov(market_physics.lyapunov_exponent)
        ising_risk = self._get_risk_level_from_ising(market_physics.ising_susceptibility)
        eigen_risk = self._get_risk_level_from_eigen(market_physics)

        # Convert risk levels to multipliers
        multipliers = PhysicsMultiplier()
        multipliers.lyapunov = self._risk_level_to_multiplier(lyapunov_risk)
        multipliers.ising = self._risk_level_to_multiplier(ising_risk)
        multipliers.eigen = self._risk_level_to_multiplier(eigen_risk)

        return multipliers

    def _get_risk_level_from_lyapunov(self, lyapunov_exponent: float) -> RiskLevel:
        """Determine risk level from Lyapunov exponent."""
        if lyapunov_exponent > 2.0:
            return RiskLevel.EXTREME
        elif lyapunov_exponent > 0:
            return RiskLevel.HIGH
        elif lyapunov_exponent > -1.0:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _get_risk_level_from_ising(self, ising_susceptibility: float) -> RiskLevel:
        """Determine risk level from Ising susceptibility."""
        if ising_susceptibility > 5.0:
            return RiskLevel.EXTREME
        elif ising_susceptibility > 2.0:
            return RiskLevel.HIGH
        elif ising_susceptibility > 1.0:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _get_risk_level_from_eigen(self, market_physics: MarketPhysics) -> RiskLevel:
        """Determine risk level from Eigenvalue (RMT) analysis."""
        if market_physics.rmt_max_eigenvalue is None or market_physics.rmt_noise_threshold is None:
            # No RMT data - assume low risk (no reduction)
            return RiskLevel.LOW

        if market_physics.rmt_max_eigenvalue > market_physics.rmt_noise_threshold * 1.5:
            # Strong signal - low risk
            return RiskLevel.LOW
        elif market_physics.rmt_max_eigenvalue > market_physics.rmt_noise_threshold:
            # Weak signal - moderate risk
            return RiskLevel.MODERATE
        else:
            # No signal (noise) - high risk
            return RiskLevel.HIGH

    def _risk_level_to_multiplier(self, risk_level: RiskLevel) -> float:
        """Convert risk level to multiplier."""
        multipliers = {
            RiskLevel.EXTREME: 0.25,
            RiskLevel.HIGH: 0.5,
            RiskLevel.MODERATE: 0.75,
            RiskLevel.LOW: 1.0
        }
        return multipliers.get(risk_level, 1.0)

    def _validate_inputs(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        stop_loss_pips: float,
        market_physics: MarketPhysics
    ) -> None:
        """Validate all input parameters."""
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        if stop_loss_pips <= 0:
            raise ValueError(f"stop_loss_pips must be positive, got {stop_loss_pips}")
        if not isinstance(market_physics, MarketPhysics):
            raise ValueError(f"market_physics must be MarketPhysics instance, got {type(market_physics)}")

    def _round_to_lot_size(self, position_size: float) -> float:
        """Round position size to broker lot step."""
        if position_size < self.config.min_lot_size:
            return self.config.min_lot_size if not self.config.allow_zero_position else 0.0

        if position_size > self.config.max_lot_size:
            return self.config.max_lot_size

        # Round down to nearest lot step
        steps = math.floor(position_size / self.config.lot_step)
        return round(steps * self.config.lot_step, 2)

    def get_current_f(self) -> float:
        """Get the last calculated Kelly fraction (for portfolio scaling)."""
        return getattr(self, '_last_kelly_f', self.config.fallback_risk_pct)

    def set_fraction(self, scaled_f: float) -> None:
        """Set Kelly fraction from portfolio scaler."""
        self._scaled_kelly_f = scaled_f

    @staticmethod
    def calculate_base_kelly(
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate base Kelly fraction without any adjustments.

        Args:
            win_rate: Historical win rate from 0 to 1
            avg_win: Average winning trade profit
            avg_loss: Average losing trade loss

        Returns:
            Base Kelly fraction (f*)
        """
        if avg_loss <= 0:
            raise ValueError("avg_loss must be positive")
        if avg_win <= 0:
            raise ValueError("avg_win must be positive")
        if not 0 <= win_rate <= 1:
            raise ValueError("win_rate must be between 0 and 1")

        risk_reward_ratio = avg_win / avg_loss
        return ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio